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


def batch_probs(logits,temps,top_ks,top_ps,pens,ctxs):
    #(n,V) logits -> (n,V) probs, every row at once. penalty -> temperature -> top_k -> top_p -> softmax
    #one sort does both top_k (rank mask) and top_p (cumsum mask) instead of topk+sort per row.
    #no host syncs in here, caller does one .tolist() after multinomial
    n,V=logits.shape; dev=logits.device
    logits=logits.float()

    if any(pn!=1.0 for pn in pens):
        #penalty mask built from the ctx token ids, rows with pen 1 get no entries
        rows=[i for i in range(n) if pens[i]!=1.0 for _ in set(ctxs[i])]
        cols=[t for i in range(n) if pens[i]!=1.0 for t in set(ctxs[i])]
        mask=torch.zeros(n,V,dtype=torch.bool,device=dev)
        mask[torch.tensor(rows,dtype=torch.long,device=dev),torch.tensor(cols,dtype=torch.long,device=dev)]=True
        pen=torch.tensor(pens,device=dev)[:,None]
        logits=torch.where(mask,torch.where(logits>0,logits/pen,logits*pen),logits)

    logits=logits/torch.tensor(temps,device=dev)[:,None]

    srt,idx=torch.sort(logits,dim=-1,descending=True)
    k=torch.tensor([kk if kk>0 else V for kk in top_ks],device=dev)[:,None]  #0 = off
    srt=srt.masked_fill(torch.arange(V,device=dev)[None,:]>=k,float("-inf"))

    pp=torch.tensor([x if x>0 else 2.0 for x in top_ps],device=dev)[:,None]  #0 = off, cum never > 2
    remove=torch.cumsum(F.softmax(srt,dim=-1),dim=-1)>pp
    remove[:,1:]=remove[:,:-1].clone()
    remove[:,0]=False
    srt=srt.masked_fill(remove,float("-inf"))

    return F.softmax(torch.full_like(logits,float("-inf")).scatter_(1,idx,srt),dim=-1)

def process_logits(logits,temperature,top_k=0,top_p=0,rep_penalty=1.0,ctx=None):
    #single row, kept for tests
    return batch_probs(logits[None],[temperature],[top_k],[top_p],[rep_penalty],[ctx or []])[0]


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


class Scheduler:
    #continuous batching over ROWS (request,position,token). prefill, decode, mtp draft chunks
    #and speculative verify all share one launch path: kv for every row is written first, then
    #each row attends kv_len=pos+1, so multi-row chunks are causal automatically.

    def __init__(self,model,pool,max_batch,max_context,device,spec=True,static=None,plan="final"):
        #two independent policies. STORAGE: static=None pages on demand, static="ctx"/"req"/<int>
        #reserves that many tokens up front (the pre-paged way). ADMISSION: plan="cur" admits while
        #the pool fits what requests hold NOW, plan="final" while it fits what they will need by
        #the time they finish. plan="cur" fits more requests and preempts as they grow
        self.static=static
        self.plan=plan
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
        self.head_acc=[0]*self.M   #per head accept count, alpha_k=head_acc/head_try
        self.head_try=[0]*self.M   #times draft k was tested (only when d1..d_k-1 accepted)
        self.trunk_passes=0        #prefill + spec launches, tokens/trunk pass is the metric that matters
        self.preempts=0
        self.replays=0             #preempted after doing work, kv thrown away and prefilled again
        self.peak_running=0        #most requests that ran in one iteration, after preemption

    def submit(self,req):
        #request into job queue
        self.queue.append(req)

    def free_blocks(self):
        return len(self.pool.free_block_queue)

    def admit(self):
        #If scheduler has free space then push request from queue to scheduler
        #Initallize user for request and mtp buffers

        tpb=self.pool.tokens_per_block
        #paged requests only take pages at launch, so free_blocks() does not move while we admit.
        #track a budget instead: free pages minus what active requests grow into this step, minus
        #what each request admitted in this loop will take. without it one fitting request lets
        #the whole queue in and step() preempts them all again before they run
        budget=self.free_blocks()-sum(self.grow_pages(r) for r in self.active)
        while self.queue and len(self.active)<self.max_batch:
            q=self.queue[0]
            pages_needed=(self.reserve_tokens(q)+self.M+2)//tpb+1
            if self.static and pages_needed>self.pool.num_blocks:
                raise RuntimeError(f"static reservation {pages_needed} pages > pool {self.pool.num_blocks}")
            if self.plan=="final" and not self.static:
                #pages everyone will hold at their longest, so growth cannot outrun the pool
                proj=sum(self.final_pages(r) for r in self.active)+self.final_pages(q)
                if proj>self.pool.num_blocks and self.active:
                    break
            elif pages_needed>budget and (self.active or self.static):
                break
            budget-=pages_needed
            req=self.queue.popleft()
            req.user=User(self.pool)
            if self.static:  #grab every page now, so this request can never need more later
                for i in range(0,self.reserve_tokens(req)+self.M+1,tpb): req.user.slot(i)
            req.reset()
            req.head_fed=[0]*self.M
            req.stash=[HidBuf() for _ in range(self.M)]
            self.active.append(req)

    def grow_pages(self,r):
        #pages r needs beyond what it holds to run its next step (writes up to len+M-1)
        tpb=self.pool.tokens_per_block
        return max(0,(len(r.ids)+self.M+tpb-1)//tpb-len(r.user.block_ids))

    def final_pages(self,r):
        #pages this request will hold once it has generated everything it is allowed to
        tpb=self.pool.tokens_per_block
        return (r.prompt_len+r.max_new_tokens+self.M+2)//tpb+1

    def reserve_tokens(self,r):
        #tokens admission must plan for: current length under paging, final length under static
        if self.static=="ctx": return self.max_context
        if self.static=="req": return r.prompt_len+r.max_new_tokens
        if self.static: return self.static
        return len(r.ids)

    def preempt(self):
        self.preempts+=1
        req=self.active.pop()
        if req.fed>0: self.replays+=1
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
        tpb=self.pool.tokens_per_block
        maxpos={}
        for r,pos,*_ in rows: maxpos[r]=pos  #rows per request are increasing so the last one wins

        #Get the page_starts arr for each requests. Assume request contiguous of form r1,r1,r1,r2,r2..=[0,0,0,3,3]
        #Can get cuseq easily from here for intradoc and page atten. 
        for r,pos,*_ in rows:
            if r is not prev:
                start=len(page_table)
                page_table+=r.user.block_ids[:maxpos[r]//tpb+1]  #kernel reads cdiv(kv_len,tpb) pages, no more
                prev=r
            page_starts.append(start)

        #tokens and position for given rows
        positions=[pos for _,pos,*_ in rows]
        tokens=torch.tensor([t for _,_,t,*_ in rows],device=dev).long()

        #give page_table,starts for paged atten, write_ptrs for kv cache writing and positions for rope
        #Runs fn with these arguments
        h=fn(tokens,self.pool,
             torch.tensor(page_table,dtype=torch.int32,device=dev),
             torch.tensor(page_starts,dtype=torch.int32,device=dev),
             torch.tensor(positions,device=dev).long(),
             torch.tensor([p+1 for p in positions],dtype=torch.int32,device=dev),
             torch.tensor(write_ptrs,device=dev).long(),
             max(positions)+1)
        return h

    def sample(self,logits,reqs,ctxs):
        #(n,V) logits for n tip rows -> probs (n,V) and sampled tokens. one multinomial, one sync
        probs=batch_probs(logits,[r.temperature for r in reqs],[r.top_k for r in reqs],
                          [r.top_p for r in reqs],[r.rep_penalty for r in reqs],ctxs)
        return probs,torch.multinomial(probs,1).squeeze(1).tolist()

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
            need=sum(self.grow_pages(r) for r in self.active)
            if need<=self.free_blocks():
                break
            if len(self.active)==1:
                raise RuntimeError("kv pool too small for a single request")
            self.preempt()

        A=self.active
        if not A:
            return
        self.peak_running=max(self.peak_running,len(A))

        #route: fresh/replayed requests (cache empty) take the intradoc prefill path — even a
        #1-token prompt, since spec_step needs the tip hidden prefill stashes; steady-state
        #requests (one unfed token, hidden stashed) take the speculative decode path
        pre=[r for r in A if r.fed==0]
        dec=[r for r in A if r.fed>0]
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
        #kv write ptrs
        write_ptrs=[r.user.slot(pos) for r,pos,*_ in rows]

        counts=[]
        prev=None
        for r,_,_ in rows:
            #cuseq calculation by finding counts in each req
            if r is not prev:
                counts.append(0)
                prev=r
            counts[-1]+=1
        cuseq=[0]
        for c in counts:
            cuseq.append(cuseq[-1]+c) #pref+curr_count

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

        #prefill trunk
        rows=[(r,p,r.ids[p]) for r in pre for p in range(P[r])]
        h=self.prefill_launch(rows,self.model.prefill_rows)
        self.trunk_passes+=1

        #head i covers positions [0, P-2-i] (embed of token t+1+i must lie inside the prompt);
        #its frontier lands at P-1-i and the next spec iteration's catch-up takes it from there.
        #stash[i] keeps the previous stage's hiddens the head has not consumed yet
        prev_h,prev_cnt=h,[P[r] for r in pre]
        for i in range(M):
            rows,hids=[],[]
            off=0
            for ri,r in enumerate(pre):
                n=max(0,P[r]-1-i) #valid accepted tokens for given mtp head
                rows+=[(r,p,r.ids[p+1+i]) for p in range(n)] #create request position, ith head predicts takes p+i+1 embedding
                hids.append(prev_h[off:off+n])

                #Track which position verified and which are not 
                r.head_fed[i]=max(0,P[r]-1-i)
                r.stash[i]=HidBuf()
                r.stash[i].start=r.head_fed[i]
                r.stash[i].push(prev_h[off+r.head_fed[i]:off+prev_cnt[ri]])

                #move to next request ,prev cnt says where next request starts
                off+=prev_cnt[ri]

            if rows:
                h_in=torch.cat(hids)
                prev_h=self.prefill_launch(rows,lambda t,*a:self.model.mtp_prefill_rows(i,h_in,t,*a))
            else:
                prev_h=prev_h[:0]
            prev_cnt=[max(0,P[r]-1-i) for r in pre]

        #sample the first new token from each request's tip row, all requests in one shot
        tips,off=[],0
        for r in pre:
            tips.append(off+P[r]-1); off+=P[r]
        _,toks=self.sample(self.model.logits(h[tips]),pre,[r.ids for r in pre])
        for r,t in zip(pre,toks):
            r.ids.append(t)
            r.fed=P[r]
            self.finish_check(r)

    def spec_step(self,A):
        #fused iteration, ONE trunk pass: heads draft first — the tip hidden is already stashed
        #(from prefill or the previous verify) and the embed is the pending token ids[-1] — then
        #the trunk pass feeds [pending, d1..dM]: row k verifies d_{k+1} and on full accept the
        #last row's dist samples the next token for free (bonus). with M=0 this degenerates to
        #the plain decode step (1 row, bonus = the sampled token)
        M=self.M
        snap={r:list(r.ids) for r in A}  #penalty ctx frozen here so draft & verify dists match
        tip={r:len(r.ids)-2 for r in A}  #draft position: hidden stashed, embed is pending ids[-1]

        #---- mtp draft: head i processes [head_fed[i] .. tip] (catch-up fills the cache holes
        #---- left by multi-token commits, tip row drafts). embed input at pos p is token p+1+i
        B=len(A)
        draft={r:[] for r in A}
        draft_prob=[]  #per head: (B,V) dists the drafts were sampled from
        for i in range(M):
            rows,hids,tips=[],[],[]
            for r in A:
                for p in range(r.head_fed[i],tip[r]+1):
                    idx=p+1+i
                    tok=r.ids[idx] if idx<len(r.ids) else draft[r][idx-len(r.ids)]
                    rows.append((r,p,tok))
                hids.append(r.stash[i].take(r.head_fed[i],tip[r]))
                tips.append(len(rows)-1)  #this request's tip row, the only one we sample from
            h_in=torch.cat(hids)

            h=self.launch(rows,lambda t,*a:self.model.mtp_rows(i,h_in,t,*a))

            #logits for the B tip rows in one matmul, one sample over all of them
            probs,toks=self.sample(self.model.logits(h[tips]),A,[snap[r] for r in A])
            draft_prob.append(probs)
            off=0
            for bi,r in enumerate(A):
                n=tip[r]+1-r.head_fed[i]
                if i+1<M:
                    r.stash[i+1].push(h[off:off+n])
                draft[r].append(toks[bi])
                r.head_fed[i]=tip[r]+1
                off+=n

        #---- single trunk pass over [pending, d1..dM]: verify + bonus in one launch
        rows=[]
        for r in A:
            chain=[r.ids[-1]]+draft[r]
            rows+=[(r,tip[r]+1+k,chain[k]) for k in range(M+1)]
        h=self.launch(rows,self.model.trunk_rows)
        self.trunk_passes+=1
        logits=self.model.logits(h)  #(B*(M+1),V), embedding matrix read once for every row
        dev=logits.device

        #accept test for every (request, draft) on gpu, one sync brings back the decisions.
        #residual resamples are drawn for all of them too, only the first reject per row is used
        if M>0:
            vidx=[bi*(M+1)+k for bi in range(B) for k in range(M)]
            q=batch_probs(logits[vidx],[r.temperature for r in A for _ in range(M)],[r.top_k for r in A for _ in range(M)],
                          [r.top_p for r in A for _ in range(M)],[r.rep_penalty for r in A for _ in range(M)],
                          [snap[r] for r in A for _ in range(M)]).view(B,M,-1)
            d=torch.stack(draft_prob,1)  #(B,M,V)
            toks=torch.tensor([draft[r] for r in A],device=dev)
            qv=q.gather(2,toks[...,None]).squeeze(2)
            dv=d.gather(2,toks[...,None]).squeeze(2)
            acc=(torch.rand(B,M,device=dev)<qv/(dv+1e-10)).tolist()
            res=torch.clamp(q-d,min=0)+1e-12
            res_tok=torch.multinomial((res/res.sum(-1,keepdim=True)).view(B*M,-1),1).view(B,M).tolist()

        J=[]
        for bi,r in enumerate(A):
            j=M+1  #first rejected draft (1-based), M+1 = all accepted
            for k in range(M):
                self.head_try[k]+=1
                if not r.done and acc[bi][k]:
                    r.ids.append(draft[r][k])  #accept
                    self.head_acc[k]+=1
                    self.finish_check(r)
                else:
                    j=k+1
                    if not r.done:  #reject: token from the residual dist
                        r.ids.append(res_tok[bi][k])
                        self.finish_check(r)
                    break
            self.accepted+=min(j-1,M); self.drafted+=M
            J.append(j)

        #all accepted: bonus token from the last row's dist. with M=0 every row is a bonus row
        bon=[bi for bi,r in enumerate(A) if J[bi]==M+1 and not r.done]
        if bon:
            _,toks=self.sample(logits[[bi*(M+1)+M for bi in bon]],[A[bi] for bi in bon],[A[bi].ids for bi in bon])
            for bi,t in zip(bon,toks):
                A[bi].ids.append(t)
                self.finish_check(A[bi])

        for bi,r in enumerate(A):
            p,j=tip[r],J[bi]
            #stale kv of rejected drafts: the pass wrote positions p+1..p+M+1, valid through p+j.
            #position-addressed slots mean rollback is just rewinding fed lengths
            r.fed=p+1+j
            if M>0:
                r.stash[0].push(h[bi*(M+1):bi*(M+1)+j])
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
