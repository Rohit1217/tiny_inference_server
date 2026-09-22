#correctness tests on a tiny random model. python test_server.py
#1 prefill vs dense reference, 2 decode over prefill kv, 3 greedy spec == greedy plain (reject-only
#and mixed-accept vocabs, with preemption), 4 rep penalty, 5 spec output DISTRIBUTION == plain
import sys,collections
import torch
import torch.nn.functional as F

from main_models_new import Transformer,rotate_half
from block_table import block_table,User
from server import Scheduler,Request,process_logits

torch.manual_seed(0)
dev="cuda:0"
MAX_CTX,D,NH,NL,FFN,GROUP,MTP=128,256,4,2,512,2,2
KVH,HD=NH//GROUP,D//NH

def make_model(vocab):
    torch.manual_seed(0)
    m=Transformer(vocab,MAX_CTX,10000,D,NH,NL,FFN,GROUP,mtp_heads=MTP).to(dev).to(torch.bfloat16)
    m.buffers_to_float(); m.eval()
    return m

@torch.no_grad()
def dense_forward(model,ids):
    #full causal attention with the same weights, logits at every position
    x=model.embedding(ids)[None]
    T=ids.shape[0]
    cos,sin=model.cos[:T][None,None],model.sin[:T][None,None]
    for blk in model.transformer_block_list:
        att=blk.att
        res=x
        h=att.rms_norm_att(x)
        q=att.q_proj(h).view(1,T,NH,HD).transpose(1,2)
        kv=att.kv_proj(h).view(1,T,2,KVH,HD)
        k,v=kv.unbind(2)
        k,v=k.transpose(1,2),v.transpose(1,2)
        qf,kf=q.float(),k.float()
        q=(qf*cos+rotate_half(qf)*sin).to(x.dtype)
        k=(kf*cos+rotate_half(kf)*sin).to(x.dtype)
        k=k.repeat_interleave(GROUP,dim=1)
        v=v.repeat_interleave(GROUP,dim=1)
        o=F.scaled_dot_product_attention(q,k,v,is_causal=True)
        x=res+att.linear_proj(o.transpose(1,2).reshape(1,T,D))
        x=x+blk.ffn(blk.rms_norm_ffn(x))
    return model.logits(model.rms_out(x))[0]

def pool_for(model,nblocks=None):
    page=16*2*2*KVH*(NL+MTP)*HD
    return block_table(NL+MTP,KVH,HD,(nblocks or 512)*page,16,dev)

def run_sched(model,seqs,spec,maxnew,max_batch=8,nblocks=None,seed=1,plan="final",**samp):
    torch.manual_seed(seed)
    p=pool_for(model,nblocks)
    s=Scheduler(model,p,max_batch,MAX_CTX,dev,spec=spec,plan=plan)
    for i,seq in enumerate(seqs):
        r=Request(list(seq),maxnew,**samp); r.tag=i
        s.submit(r)
    fin=s.run()
    return {r.tag:r.out_ids() for r in fin},p,s

ok=True
def check(name,cond,extra=""):
    global ok; ok&=bool(cond)
    print(f"  [{'ok' if cond else 'FAIL'}] {name} {extra}")

#---- 1. intradoc batched prefill vs dense, ragged lengths incl P=1,2, one launch
VOCAB=100
model=make_model(VOCAB)
lens=[7,33,18,1,2]
seqs=[torch.randint(0,VOCAB,(L,)).tolist() for L in lens]
pool=pool_for(model)
users=[User(pool) for _ in lens]
rows=[(users[i],p,seqs[i][p]) for i in range(len(lens)) for p in range(lens[i])]
cuseq=[0]
for L in lens: cuseq.append(cuseq[-1]+L)
h=model.prefill_rows(torch.tensor([t for _,_,t in rows],device=dev).long(),pool,
                     torch.tensor([p for _,p,_ in rows],device=dev).long(),
                     torch.tensor([u.slot(p) for u,p,_ in rows],device=dev).long(),
                     torch.tensor(cuseq,dtype=torch.int32,device=dev),max(lens))
logits=model.logits(h)
print("prefill vs dense:")
off=0
for i,L in enumerate(lens):
    ref=dense_forward(model,torch.tensor(seqs[i],device=dev).long())
    got=logits[off:off+L]; off+=L
    check(f"len={L}",torch.allclose(ref.float(),got.float(),rtol=0.05,atol=0.5),
          f"argmax agree {(ref.argmax(-1)==got.argmax(-1)).float().mean():.2f}")

#---- 2. one decode row on top of prefill-written pages
nxt=[torch.randint(0,VOCAB,(1,)).item() for _ in lens]
drows=[(users[i],lens[i],nxt[i]) for i in range(len(lens))]
pt,ps=[],[]
for u,_,_ in drows: ps.append(len(pt)); pt+=u.block_ids
h=model.trunk_rows(torch.tensor([t for _,_,t in drows],device=dev).long(),pool,
                   torch.tensor(pt,dtype=torch.int32,device=dev),torch.tensor(ps,dtype=torch.int32,device=dev),
                   torch.tensor([p for _,p,_ in drows],device=dev).long(),
                   torch.tensor([p+1 for _,p,_ in drows],dtype=torch.int32,device=dev),
                   torch.tensor([u.slot(p) for u,p,_ in drows],device=dev).long(),max(lens)+1)
dl=model.logits(h)
good=all(torch.allclose(dense_forward(model,torch.tensor(seqs[i]+[nxt[i]],device=dev).long())[-1].float(),
                        dl[i].float(),rtol=0.05,atol=0.5) for i in range(len(lens)))
check("decode over prefill kv",good)

#---- 3. greedy spec == greedy plain. one-hot dists: accept iff argmax match, residual = argmax,
#---- so any stale kv / rollback / stash bug shows up as a token mismatch
print("greedy equivalence:")
for vocab,label in ((100,"reject-only"),(6,"mixed-accept")):
    m=make_model(vocab)
    sq=[torch.randint(0,vocab,(L,)).tolist() for L in lens]
    g=dict(temperature=1.0,top_k=1)
    plain,_,_=run_sched(m,sq,False,20,**g)
    spec,pool_sp,sch=run_sched(m,sq,True,20,**g)
    check(f"vocab={vocab} ({label})",plain==spec,f"acceptance {sch.accepted}/{sch.drafted}")
    #plan="cur" admits on current need, which is what makes the pool fill and preemption fire.
    #the default "final" admission never preempts, so this path needs it forced
    small,pool_s,ss=run_sched(m,sq,True,20,max_batch=2,nblocks=5,plan="cur",**g)
    check(f"vocab={vocab} + preemption",plain==small,f"preempts {ss.preempts}")
    check("pools drained",len(pool_sp.free_block_queue)==pool_sp.num_blocks and len(pool_s.free_block_queue)==pool_s.num_blocks)

#---- 4. rep penalty pushes an in-context token below an out-of-context one
lg=torch.zeros(VOCAB,device=dev); lg[5]=5.0; lg[7]=4.9
check("rep penalty",process_logits(lg,1.0).argmax().item()==5 and process_logits(lg,1.0,rep_penalty=2.0,ctx=[5]).argmax().item()==7)

#---- 5. distribution test. greedy cannot see the residual path (relu(q-d) is trivial for one-hot),
#---- so sample many 3-token continuations and compare tuple histograms spec vs plain. the null
#---- baseline is plain vs plain with another seed, spec must be within 1.5x of that
print("distribution (N=4096 x 3 tokens, vocab 6):")
m=make_model(6)
prompt=[[1,4,2,5]]*4096
def hist(out): return collections.Counter(tuple(v) for v in out.values())
def tv(a,b):
    keys=set(a)|set(b); n=sum(a.values())
    return sum(abs(a.get(k,0)-b.get(k,0)) for k in keys)/(2*n)
for label,samp in (("temp=1",dict(temperature=1.0)),
                   ("top_k=3 top_p=0.9",dict(temperature=1.0,top_k=3,top_p=0.9)),
                   ("rep_penalty=1.3",dict(temperature=1.0,rep_penalty=1.3))):
    pa,_,_=run_sched(m,prompt,False,3,max_batch=256,seed=1,**samp)
    pb,_,_=run_sched(m,prompt,False,3,max_batch=256,seed=2,**samp)
    sp,_,sch=run_sched(m,prompt,True,3,max_batch=256,seed=3,**samp)
    null=tv(hist(pa),hist(pb)); got=tv(hist(pa),hist(sp))
    check(label,got<1.5*null+0.01,f"tv spec-vs-plain {got:.3f}, plain-vs-plain {null:.3f}, acceptance {sch.accepted/max(sch.drafted,1):.0%}")

print("ALL OK" if ok else "FAILURES")
sys.exit(0 if ok else 1)
