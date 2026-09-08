import torch
import torch.nn.functional as F
from collections import deque

from Config.config import Config
from Tokenizers.tokenizer_fast import Tokenizer
from block_table import block_table,User
from main_models_new import Transformer

cfg=Config()

EOS=48000
CKPT_PATH="models/checkpoint_step16800.pt"
TOKENIZER_PATH="Tokenizers/tokenizer_vocab.titoken"

POOL_BYTES=4*1024*1024*1024
TOKENS_PER_BLOCK=16
MAX_BATCH=8
DEVICE="cuda:0"


def process_logits(logits,temperature,top_k=0,top_p=0,rep_penalty=1.0,ctx=None):
    #penalty -> temperature -> top_k -> top_p -> softmax. draft and verifier dists go through
    #this identically (same ctx snapshot) so the acceptance ratio compares matching dists
    logits=logits.float().clone()
    vocab=logits.shape[-1]

    if rep_penalty!=1.0 and ctx:
        idx=torch.tensor(list(set(ctx)),device=logits.device)
        sel=logits[idx]
        logits[idx]=torch.where(sel>0,sel/rep_penalty,sel*rep_penalty)

    logits=logits/temperature

    if top_k>0:
        k=torch.topk(-logits,vocab-top_k)
        logits[k[1]]=float("-inf")

    if top_p>0:
        sorted_logits,sorted_idx=torch.sort(logits,descending=True)
        cum_prob=torch.cumsum(F.softmax(sorted_logits,dim=-1),dim=-1)
        remove=cum_prob>top_p
        remove[1:]=remove[:-1].clone()
        remove[0]=False
        sorted_logits[remove]=float("-inf")
        logits=torch.full_like(logits,float("-inf")).scatter_(0,sorted_idx,sorted_logits)

    return F.softmax(logits,dim=-1)


class HidBuf:
    #contiguous hidden rows for positions [start, start+len). producers push, the next stage
    #takes; entries survive until a later take passes them (so a rolled-back head can reprocess)
    def __init__(self):
        self.start=0
        self.rows=[]

    def push(self,h):
        self.rows+=list(h)

    def take(self,a,b):
        self.rows=self.rows[a-self.start:]
        self.start=a
        return torch.stack(self.rows[:b-a+1])

    def truncate(self,pos):
        self.rows=self.rows[:max(0,pos-self.start)]


class Request:
    _next_id=0

    def __init__(self,prompt_ids,max_new_tokens,temperature=1.0,top_k=0,top_p=0,rep_penalty=1.0):
        self.id=Request._next_id; Request._next_id+=1
        self.ids=list(prompt_ids)  #committed tokens: prompt + generated
        self.prompt_len=len(prompt_ids)
        self.max_new_tokens=max_new_tokens
        self.temperature=temperature
        self.top_k=top_k
        self.top_p=top_p
        self.rep_penalty=rep_penalty
        self.user=None
        self.done=False
        self.reset()

    def reset(self):
        self.fed=0        #positions with valid trunk kv
        self.head_fed=[]  #per mtp head: next position it must process
        self.stash=[]     #stash[i]: hiddens feeding head i (stash[0] = trunk outputs)

    def out_ids(self):
        return self.ids[self.prompt_len:]

    def sample_prob(self,logits,ctx):
        return process_logits(logits,self.temperature,self.top_k,self.top_p,self.rep_penalty,ctx)


class Scheduler:
    #continuous batching over ROWS (request,position,token). prefill, decode, mtp draft chunks
    #and speculative verify all share one launch path: kv for every row is written first, then
    #each row attends kv_len=pos+1, so multi-row chunks are causal automatically.

    def __init__(self,model,pool,max_batch,max_context,device,spec=True):
        self.model=model
        self.pool=pool
        self.max_batch=max_batch
        self.max_context=max_context
        self.device=device
        self.M=model.num_mtp_heads if spec else 0

        self.queue=deque()
        self.active=[]
        self.finished=[]
        self.accepted=self.drafted=0  #speculative acceptance stats

    def submit(self,req):
        #request into job queue
        self.queue.append(req)

    def free_blocks(self):
        return len(self.pool.free_block_queue)

    def admit(self):
        #If scheduler has free space then push request from queue to scheduler
        #Initallize user for request and mtp buffers

        tpb=self.pool.tokens_per_block
        while self.queue and len(self.active)<self.max_batch:
            pages_needed=(len(self.queue[0].ids)+self.M+2)//tpb+1
            if pages_needed>self.free_blocks() and self.active:
                break
            req=self.queue.popleft()
            req.user=User(self.pool)
            req.reset()
            req.head_fed=[0]*self.M
            req.stash=[HidBuf() for _ in range(self.M)]
            self.active.append(req)

    def preempt(self):
        req=self.active.pop()
        req.user.free_cache()
        req.user=None
        self.queue.appendleft(req)

    def launch(self,rows,fn):
        #rows: list of (req,pos,token[,hid]) with a request's rows contiguous & increasing.
        #allocates slots, builds the flat page table + per-row metadata, runs fn -> hiddens (n,D)
        dev=self.device

        # get write_ptrs for eqch row for their positions
        write_ptrs=[r.user.slot(pos) for r,pos,*_ in rows]

        #Build flat page table given requests contains table contiguous and starting points for each request
        #page table is allocated pages for user

        page_table,page_starts=[],[]
        prev=None
        for r,pos,*_ in rows:
            if r is not prev:
                start=len(page_table)
                page_table+=r.user.block_ids
                prev=r
            page_starts.append(start)

        #tokens and position for given rows
        positions=[pos for _,pos,*_ in rows]
        tokens=torch.tensor([t for _,_,t,*_ in rows],device=dev).long()

        #Runs fn with these arguments
        h=fn(tokens,self.pool,
             torch.tensor(page_table,dtype=torch.int32,device=dev),
             torch.tensor(page_starts,dtype=torch.int32,device=dev),
             torch.tensor(positions,device=dev).long(),
             torch.tensor([p+1 for p in positions],dtype=torch.int32,device=dev),
             torch.tensor(write_ptrs,device=dev).long(),
             max(positions)+1)
        return h

    def finish_check(self,r):
        #truncate at EOS / caps, mark done
        out=r.out_ids()
        if EOS in out:
            r.ids=r.ids[:r.prompt_len+out.index(EOS)+1]
            r.done=True
        if len(r.out_ids())>=r.max_new_tokens:
            r.ids=r.ids[:r.prompt_len+r.max_new_tokens]
            r.done=True
        if len(r.ids)>=self.max_context-self.M-2:
            r.done=True

    def step(self):
        A=self.active
        M=self.M
        tpb=self.pool.tokens_per_block
        if not A:
            return

        #cache-full: preempt newest until this iteration's high-water pages fit
        while True:
            #find how many pages needed. Done via subtracting pages to fit tokens - pages already acquired
            need=sum(max(0,(len(r.ids)+M+tpb-1)//tpb-len(r.user.block_ids)) for r in self.active)
            if need<=self.free_blocks():
                break
            if len(self.active)==1:
                raise RuntimeError("kv pool too small for a single request")
            self.preempt()

        A=self.active
        if not A:
            return

        #route: fresh/replayed requests (whole sequence unfed, cache empty) take the intradoc
        #prefill path; steady-state requests (one unfed token) take the speculative decode path
        pre=[r for r in A if len(r.ids)-r.fed>1]
        dec=[r for r in A if len(r.ids)-r.fed==1]
        if pre:
            self.prefill_step(pre)
        if dec:
            self.spec_step(dec)

        #evict finished, free their pages
        for r in A:
            if r.done:
                r.user.free_cache()
                r.user=None
                self.finished.append(r)
        self.active=[r for r in A if not r.done]

    def prefill_launch(self,rows,fn):
        #like launch() but for the intradoc kernel: cuseq is cumulative TOKEN counts, no page
        #table needed (attention runs on the dense k,v computed in-pass; kv still written to pages)
        dev=self.device
        write_ptrs=[r.user.slot(pos) for r,pos,*_ in rows]

        counts=[]
        prev=None
        for r,_,_ in rows:
            if r is not prev:
                counts.append(0)
                prev=r
            counts[-1]+=1
        cuseq=[0]
        for c in counts:
            cuseq.append(cuseq[-1]+c)

        h=fn(torch.tensor([t for _,_,t in rows],device=dev).long(),self.pool,
             torch.tensor([p for _,p,_ in rows],device=dev).long(),
             torch.tensor(write_ptrs,device=dev).long(),
             torch.tensor(cuseq,dtype=torch.int32,device=dev),
             max(counts))
        return h

    def prefill_step(self,pre):
        #batched prefill through the intradoc kernel: trunk + mtp head caches in one iteration,
        #sample the first token from each tip, speculation starts next iteration. exactness needs
        #an empty cache per request (positions restart at 0), true for fresh and replayed requests
        M=self.M
        P={r:len(r.ids) for r in pre}
        assert all(r.fed==0 for r in pre)

        rows=[(r,p,r.ids[p]) for r in pre for p in range(P[r])]
        h=self.prefill_launch(rows,self.model.prefill_rows)

        #head i covers positions [0, P-2-i] (embed of token t+1+i must lie inside the prompt);
        #its frontier lands at P-1-i and the next spec iteration's catch-up takes it from there.
        #stash[i] keeps the previous stage's hiddens the head has not consumed yet
        prev_h,prev_cnt=h,[P[r] for r in pre]
        for i in range(M):
            rows,hids=[],[]
            off=0
            for ri,r in enumerate(pre):
                n=max(0,P[r]-1-i)
                rows+=[(r,p,r.ids[p+1+i]) for p in range(n)]
                hids.append(prev_h[off:off+n])

                r.head_fed[i]=max(0,P[r]-1-i)
                r.stash[i]=HidBuf()
                r.stash[i].start=r.head_fed[i]
                r.stash[i].push(prev_h[off+r.head_fed[i]:off+prev_cnt[ri]])
                off+=prev_cnt[ri]

            if rows:
                h_in=torch.cat(hids)
                prev_h=self.prefill_launch(rows,lambda t,*a:self.model.mtp_prefill_rows(i,h_in,t,*a))
            else:
                prev_h=prev_h[:0]
            prev_cnt=[max(0,P[r]-1-i) for r in pre]

        #sample the first new token from each request's tip row
        off=0
        for r in pre:
            prob=r.sample_prob(self.model.logits(h[off+P[r]-1]),r.ids)
            r.ids.append(int(torch.multinomial(prob,1)))
            r.fed=P[r]
            off+=P[r]
            self.finish_check(r)

    def spec_step(self,A):
        M=self.M

        #---- trunk: feed the unfed tip token, sample the next from each request's tip row
        rows=[(r,p,r.ids[p]) for r in A for p in range(r.fed,len(r.ids))]
        h=self.launch(rows,self.model.trunk_rows)

        tip={}
        off=0
        for r in A:
            n=len(r.ids)-r.fed
            if M>0:
                r.stash[0].push(h[off:off+n])
            tip[r]=len(r.ids)-1
            prob=r.sample_prob(self.model.logits(h[off+n-1]),r.ids)
            r.ids.append(int(torch.multinomial(prob,1)))
            r.fed=len(r.ids)-1
            off+=n
            self.finish_check(r)

        live=[r for r in A if not r.done]
        snap={r:list(r.ids) for r in live}  #penalty ctx frozen here so draft & verify dists match

        #---- mtp draft: head i processes [head_fed[i] .. tip] (catch-up fills the cache holes
        #---- left by multi-token commits, tip row drafts). embed input at pos p is token p+1+i
        draft={r:[] for r in live}
        draft_prob={r:[] for r in live}
        if M>0 and live:
            for i in range(M):
                rows,hids=[],[]
                for r in live:
                    for p in range(r.head_fed[i],tip[r]+1):
                        idx=p+1+i
                        tok=r.ids[idx] if idx<len(r.ids) else draft[r][idx-len(r.ids)]
                        rows.append((r,p,tok))
                    hids.append(r.stash[i].take(r.head_fed[i],tip[r]))
                h_in=torch.cat(hids)

                h=self.launch(rows,lambda t,*a:self.model.mtp_rows(i,h_in,t,*a))

                off=0
                for r in live:
                    n=tip[r]+1-r.head_fed[i]
                    if i+1<M:
                        r.stash[i+1].push(h[off:off+n])
                    prob=r.sample_prob(self.model.logits(h[off+n-1]),snap[r])
                    draft[r].append(int(torch.multinomial(prob,1)))
                    draft_prob[r].append(prob)
                    r.head_fed[i]=tip[r]+1
                    off+=n

            #---- verify: one trunk pass over [main, d1..d_{M-1}] checks d1..dM in parallel
            rows=[]
            for r in live:
                chain=[r.ids[tip[r]+1]]+draft[r][:M-1]
                rows+=[(r,tip[r]+1+k,chain[k]) for k in range(M)]
            h=self.launch(rows,self.model.trunk_rows)

            for bi,r in enumerate(live):
                hv=h[bi*M:(bi+1)*M]
                p=tip[r]
                j=M+1  #first rejected draft (1-based), M+1 = all accepted
                for k in range(M):
                    q=r.sample_prob(self.model.logits(hv[k]),snap[r])
                    d=draft_prob[r][k]
                    tok=draft[r][k]
                    if not r.done and torch.rand(1).item()<(q[tok]/(d[tok]+1e-10)).item():
                        r.ids.append(tok)  #accept
                        self.finish_check(r)
                    else:
                        j=k+1
                        if not r.done:  #reject: resample from the residual
                            res=torch.clamp(q-d,min=0)+1e-12
                            r.ids.append(int(torch.multinomial(res/res.sum(),1)))
                            self.finish_check(r)
                        break
                self.accepted+=min(j-1,M); self.drafted+=M

                #stale kv of rejected drafts: verify wrote positions p+1..p+M, valid through p+j.
                #position-addressed slots mean rollback is just rewinding fed lengths
                r.fed=p+min(j,M)+1
                r.stash[0].push(hv[:r.fed-(p+1)])
                for i in range(1,M):
                    if i>=j:  #head i's tip entry used draft d_i, which was rejected
                        r.head_fed[i]=p
                        if i+1<M:
                            r.stash[i+1].truncate(p)

    def run(self):
        while self.queue or self.active:
            self.admit()
            self.step()
        return self.finished


if __name__=="__main__":
    torch.manual_seed(0)

    model=Transformer(vocab_size=cfg.VOCAB_SIZE,max_context=cfg.MAX_CONTEXT,max_freq=cfg.MAX_FREQ,
                      d_model=cfg.D_MODEL,n_heads=cfg.N_HEAD,num_layers=cfg.NUM_LAYERS,
                      ffn_hidden_dim=cfg.FFN_HIDDEN_DIM,group_size=cfg.NUM_GROUPS,mtp_heads=cfg.MTP_HEADS)
    opt_step=model.load_ckpt(CKPT_PATH,DEVICE)
    print(f"MODEL LOADED (opt_step={opt_step})")

    tokenizer=Tokenizer(vocab_size=48000)
    tokenizer.load_tokenizer(TOKENIZER_PATH)
    print("TOKENIZER LOADED")

    pool=block_table(cfg.NUM_LAYERS+cfg.MTP_HEADS,cfg.N_HEAD//cfg.NUM_GROUPS,cfg.D_MODEL//cfg.N_HEAD,
                     POOL_BYTES,TOKENS_PER_BLOCK,DEVICE)
    print(f"POOL: {pool.num_blocks} blocks of {TOKENS_PER_BLOCK} tokens")

    scheduler=Scheduler(model,pool,MAX_BATCH,cfg.MAX_CONTEXT,DEVICE,spec=True)

    PROMPTS=[
        "The capital of France is",
        "Hey, what are you doing?",
        "In 1969, humans first landed on the",
        "Water is made of hydrogen and",
        "Once upon a time, there was a small robot who",
        "The three primary colors are",
        "Question: What is the largest planet in our solar system?\nAnswer:",
        "def add(a, b):\n    return",
        "My favorite thing about winter is",
        "How to stump a model?",
    ]

    for p in PROMPTS:
        scheduler.submit(Request(tokenizer.encode(p),max_new_tokens=200,
                                 temperature=0.9,top_k=50,top_p=0.95,rep_penalty=1.1))

    import time
    t0=time.time()
    finished=scheduler.run()
    dt=time.time()-t0

    total_new=sum(len(r.out_ids()) for r in finished)
    acc=scheduler.accepted/max(scheduler.drafted,1)
    print(f"\n{total_new} tokens in {dt:.1f}s ({total_new/dt:.1f} tok/s), draft acceptance {acc:.0%}\n")

    for r in sorted(finished,key=lambda r:r.id):
        print("="*100)
        print("PROMPT:",tokenizer.decode(r.ids[:r.prompt_len]))
        print("OUTPUT:",tokenizer.decode(r.out_ids()))
