#perf evals for the server. python evals.py perf | load | naive | mem
#perf runs in process: latency, throughput vs batch, spec stats, memory, roofline
#load goes through serve.py over http with poisson arrivals, throughput vs p99 sweep
#naive is the ladder: no kv cache -> kv cache -> batching -> speculation, short and long prompts
import sys,time,random,asyncio,subprocess,statistics,gc
import torch

import server as S
from server import Scheduler,Request,User,batch_probs,EOS
from main_models_new import Transformer
from block_table import block_table
from Tokenizers.tokenizer_fast import Tokenizer

cfg=S.cfg
HBM_GBPS=768  #a6000 gddr6 peak. decode is bandwidth bound so weight read time is the floor per iteration
MAXNEW=100
PORT=8000

PROSE=["The capital of France is","Hey, what are you doing?","In 1969, humans first landed on the",
       "Water is made of hydrogen and","Once upon a time, there was a small robot who",
       "The three primary colors are","Question: What is the largest planet in our solar system?\nAnswer:",
       "My favorite thing about winter is","How to stump a model?","The history of the Roman Empire begins",
       "A healthy breakfast should include","The best way to learn a new language is",
       "Climate change is affecting","In the year 2050, cities will","She opened the letter and read:",
       "The recipe calls for two cups of"]
CODE=["def add(a, b):\n    return","class Stack:\n    def __init__(self):",
      "import numpy as np\n\ndef softmax(x):","for i in range(10):\n    if i % 2 == 0:",
      "def fibonacci(n):\n    if n <= 1:","SELECT name, age FROM users WHERE"]


def pct(xs,p):
    #percentile w/o numpy, linear interp
    if not xs: return float("nan")
    xs=sorted(xs); k=(len(xs)-1)*p/100; f=int(k); c=min(f+1,len(xs)-1)
    return xs[f]+(xs[c]-xs[f])*(k-f)

def p3(xs,scale=1e3):
    #p50/p95/p99 string in ms
    return f"{pct(xs,50)*scale:7.1f} {pct(xs,95)*scale:7.1f} {pct(xs,99)*scale:7.1f}"


class Probe:
    #wraps scheduler methods to timestamp every step and every committed token
    #scheduler logic untouched, we only read state after each call
    def __init__(self,s):
        self.s=s
        self.dec_s=[]; self.pre_s=[]   #per step durations, decode and prefill separate since roofline is about decode
        self.free=[]; self.waste=[]; self.kv_tok=[]
        step,admit,pre,spec=s.step,s.admit,s.prefill_step,s.spec_step

        def timed(fn,lst):
            def w(*a):
                torch.cuda.synchronize(); t0=time.perf_counter()
                out=fn(*a)
                torch.cuda.synchronize(); lst.append(time.perf_counter()-t0)
                return out
            return w
        s.prefill_step=timed(pre,self.pre_s)
        s.spec_step=timed(spec,self.dec_s)

        def wstep():
            nfin=len(s.finished)
            for r in s.active:
                if r.fed>0: r.nsteps+=1  #decode iterations this request took part in, prefill excluded
            step()
            t=time.perf_counter()
            #new tokens since last step get this timestamp. finished this step are also checked
            for r in s.active+s.finished[nfin:]:
                n=len(r.out_ids())
                r.t_tok+=[t]*(n-r.seen); r.seen=n
            #memory samples per step
            self.free.append(s.free_blocks())
            tpb=s.pool.tokens_per_block
            self.waste+=[1-len(r.ids)/(len(r.user.block_ids)*tpb) for r in s.active]
            self.kv_tok.append(sum(len(r.ids) for r in s.active))
        s.step=wstep

        def wadmit():
            admit()
            t=time.perf_counter()
            for r in s.active:
                if r.t_adm is None: r.t_adm=t  #first admit only, queue wait = t_adm-t_sub
        s.admit=wadmit


def best(fn,n=3):
    #run n times, keep the fastest. throughput here is bimodal (~1.5x) run to run for reasons
    #outside the code (not numa, not core migration, not a leak), and the noise only ever
    #slows a run down, so the fastest run is what the code can do
    rs=[fn() for _ in range(n)]
    return max(rs,key=lambda r:r["tps"])

def load_model():
    torch.manual_seed(0)
    m=Transformer(cfg.VOCAB_SIZE,cfg.MAX_CONTEXT,cfg.MAX_FREQ,cfg.D_MODEL,cfg.N_HEAD,cfg.NUM_LAYERS,
                  cfg.FFN_HIDDEN_DIM,cfg.NUM_GROUPS,mtp_heads=cfg.MTP_HEADS)
    m.load_ckpt(S.CKPT_PATH,S.DEVICE)
    t=Tokenizer(48000); t.load_tokenizer(S.TOKENIZER_PATH)
    return m,t


def run(model,tok,prompts,spec,max_batch,temp=0.9,pool_bytes=S.POOL_BYTES,seed=0):
    #one workload -> stats dict. all prompts submitted at t=0 so ttft includes queue wait past max_batch
    torch.manual_seed(seed); random.seed(seed)
    pool=block_table(model.num_layers+model.num_mtp_heads,model.nkv_heads,model.head_dim,
                     pool_bytes,S.TOKENS_PER_BLOCK,S.DEVICE)
    s=Scheduler(model,pool,max_batch,cfg.MAX_CONTEXT,S.DEVICE,spec=spec)
    pr=Probe(s)

    torch.cuda.synchronize(); t0=time.perf_counter()
    for p in prompts:
        #prompts are strings or ready token lists (long prompt case)
        r=Request(p if isinstance(p,list) else tok.encode(p),MAXNEW,temperature=temp,top_k=50,top_p=0.95,rep_penalty=1.1)
        r.t_sub=time.perf_counter(); r.t_adm=None; r.t_tok=[]; r.seen=0; r.nsteps=0
        s.submit(r)
    fin=s.run()
    torch.cuda.synchronize(); dt=time.perf_counter()-t0

    ntok=sum(len(r.out_ids()) for r in fin)
    itl=[b-a for r in fin for a,b in zip(r.t_tok,r.t_tok[1:])]  #per token gaps, zeros = clumped commits from spec
    st=dict(ntok=ntok,dt=dt,tps=ntok/dt,nreq=len(fin),
            ttft=[r.t_tok[0]-r.t_sub for r in fin if r.t_tok],
            queue=[r.t_adm-r.t_sub for r in fin],
            itl=itl,dec_s=pr.dec_s,pre_s=pr.pre_s,
            #per request: tokens a request gets per decode iteration it sits in. first token is from prefill
            tok_per_it=sum(len(r.out_ids())-1 for r in fin)/max(sum(r.nsteps for r in fin),1),
            #same per trunk pass. fused design has 1 pass per decode iteration so equal, 2-pass design halves it
            tok_per_pass=sum(len(r.out_ids())-1 for r in fin)/max(sum(r.nsteps for r in fin),1)
                         *len(pr.dec_s)/max(s.trunk_passes-len(pr.pre_s),1),
            alpha=[a/max(t,1) for a,t in zip(s.head_acc,s.head_try)],
            acc=s.accepted/max(s.drafted,1),
            preempts=s.preempts,nblocks=pool.num_blocks,
            util=1-statistics.mean(pr.free)/pool.num_blocks if pr.free else 0,
            util_max=1-min(pr.free)/pool.num_blocks if pr.free else 0,
            waste=statistics.mean(pr.waste) if pr.waste else 0,
            kv_tok=statistics.mean(pr.kv_tok) if pr.kv_tok else 0)
    #Probe puts closures that capture s onto s itself, a cycle. without gc.collect() the scheduler
    #and its 4 GB pool live until python's cycle collector happens to run, and repeated runs OOM
    del pool,s,pr; gc.collect(); torch.cuda.empty_cache()
    return st


def roofline(model,st,spec):
    #bytes each decode iteration must read: weights once + kv of every active request
    #plain never touches the heads so only trunk weights count
    trunk=sum(p.numel() for n,p in model.named_parameters() if "mtp" not in n)*2
    heads=sum(p.numel() for n,p in model.named_parameters() if "mtp" in n)*2
    planes=model.num_layers+(model.num_mtp_heads if spec else 0)
    kv_per_tok=planes*model.nkv_heads*2*model.head_dim*2
    byts=trunk+(heads if spec else 0)+st["kv_tok"]*kv_per_tok
    floor=byts/(HBM_GBPS*1e9)
    meas=statistics.mean(st["dec_s"]) if st["dec_s"] else float("nan")
    return floor,meas,floor/meas


def perf():
    model,tok=load_model()
    print("MODEL LOADED")

    #warm both modes first. triton autotunes per kv_len bucket and the first mode measured
    #in a process pays the whole sweep, this is what corrupted the earlier benchmarks
    for spec in (False,True): run(model,tok,PROSE,spec,8)
    print("WARM\n")

    #---- latency, what a client feels. batch 8 both modes
    print("LATENCY  batch=8, 16 prompts, all submitted at t=0, fastest of 3   p50     p95     p99  (ms)")
    for spec in (False,True):
        st=best(lambda:run(model,tok,PROSE,spec,8))
        tag="spec " if spec else "plain"
        print(f"  {tag} ttft (queue+prefill)      {p3(st['ttft'])}")
        print(f"  {tag} queue wait                {p3(st['queue'])}")
        print(f"  {tag} itl per token             {p3(st['itl'])}   <- zeros are clumped commits")
        print(f"  {tag} itl per iteration         {p3(st['dec_s'])}")
        print(f"  {tag} prefill step              {p3(st['pre_s'])}")
    print()

    #---- throughput vs batch, where it flattens says compute vs overhead bound
    print("THROUGHPUT vs BATCH (fastest of 3)  tok/s   it/s  ms/it   tok/it  floor ms  roofline%")
    for spec in (False,True):
        for b in (1,4,16,32,64):
            P=PROSE*(max(1,b//len(PROSE)))  #enough prompts to fill the batch
            st=best(lambda:run(model,tok,P,spec,b))
            fl,ms,ut=roofline(model,st,spec)
            its=len(st["dec_s"])/st["dt"]
            print(f"  {'spec ' if spec else 'plain'} batch={b:2d}   {st['tps']:9.1f} {its:6.1f} {ms*1e3:6.1f}   {st['tok_per_it']:5.2f}    {fl*1e3:5.2f}     {ut*100:5.1f}")
    print()

    #---- spec decode stats. alpha_k conditional on earlier heads accepted, tells if head k pays
    print("SPEC   alpha per head is conditional: alpha2 = P(d2 ok | d1 ok)")
    print("                              alpha1  alpha2   tok/it  tok/trunk_pass   tok/s")
    def row(tag,st):
        a=st["alpha"]+[float("nan")]*(2-len(st["alpha"]))
        print(f"  {tag:26s}  {a[0]:5.2f}   {a[1]:5.2f}    {st['tok_per_it']:5.2f}      {st['tok_per_pass']:5.2f}      {st['tps']:7.1f}")
    row("plain (reference)",best(lambda:run(model,tok,PROSE,False,8)))
    for t in (0.6,0.9,1.2):
        row(f"spec temp={t}",best(lambda:run(model,tok,PROSE,True,8,temp=t)))
    row("spec code prompts",best(lambda:run(model,tok,CODE*3,True,8)))
    print()

    #---- memory. starved pool forces preemption, waste = allocated slots never filled (tpb=16)
    print("MEMORY                          blocks  util_mean  util_max  preempts  page_waste")
    page=S.TOKENS_PER_BLOCK*2*2*model.nkv_heads*(model.num_layers+model.num_mtp_heads)*model.head_dim  #same as block_table page_size
    for tag,pb in (("4GB pool",S.POOL_BYTES),("starved pool (40 blocks)",40*page)):
        st=run(model,tok,PROSE,True,8,pool_bytes=pb)
        print(f"  {tag:28s}  {st['nblocks']:5d}    {st['util']*100:5.1f}%    {st['util_max']*100:5.1f}%     {st['preempts']:3d}      {st['waste']*100:5.1f}%")
    print()


#---- static vs paged at equal memory
def staggered(model,tok,reqs,static,pool_bytes,max_batch=512,every=4,plan="final"):
    #admit a few requests per iteration instead of all at once, so at any instant requests sit at
    #different lengths. submitted all at once they grow in lockstep and paging shows no gain
    pool=block_table(model.num_layers+model.num_mtp_heads,model.nkv_heads,model.head_dim,
                     pool_bytes,S.TOKENS_PER_BLOCK,S.DEVICE)
    s=Scheduler(model,pool,max_batch,cfg.MAX_CONTEXT,S.DEVICE,spec=True,static=static,plan=plan)
    pend=list(reqs)
    torch.cuda.synchronize(); t0=time.perf_counter()
    while pend or s.queue or s.active:
        for _ in range(every):
            if pend: s.submit(pend.pop(0))
        s.admit()
        s.step()
    torch.cuda.synchronize(); dt=time.perf_counter()-t0
    ntok=sum(len(r.out_ids()) for r in s.finished)
    mean_len=statistics.mean(len(r.ids) for r in s.finished)
    #peak counts requests that actually ran together (after preemption), replays counts requests
    #preempted after doing work, i.e. kv thrown away. admitted-then-dropped churn is neither
    res=dict(peak=s.peak_running,tps=ntok/dt,mean_len=mean_len,preempts=s.preempts,replays=s.replays,nblocks=pool.num_blocks)
    del pool,s; gc.collect(); torch.cuda.empty_cache()
    return res

def mem():
    model,tok=load_model()
    tpb=S.TOKENS_PER_BLOCK
    kv_tok=(model.num_layers+model.num_mtp_heads)*model.nkv_heads*2*model.head_dim*2
    print(f"MODEL LOADED. kv {kv_tok/1024:.0f} KiB/token, page = {tpb} tokens = {tpb*kv_tok/1024:.0f} KiB\n")

    #---- capacity is pure arithmetic: how many requests fit at once, before any timing
    print("CAPACITY (pages needed per request -> how many fit)")
    print(f"{'pool':>8s} {'pages':>7s}   {'request':>14s}   {'paged (grown)':>14s} {'static @ len':>13s} {'static @ 8192':>14s}")
    for mb in (256,1024,4096):
        nb=mb*1024*1024//(tpb*kv_tok)
        for L in (110,1510):
            pg=lambda n:max(1,-(-n//tpb))
            print(f"{mb:6d}MB {nb:7d}   {L:11d} tok   {nb//pg(L):14d} {nb//pg(L):13d} {nb//pg(8192) if nb>=pg(8192) else 0:14d}")
    print("  paged and static@len tie at full length, the gap is that paged only holds what the")
    print("  request has grown into so far, and frees it at EOS. measured below\n")

    #---- measured, at a pool where all three allocators are actually memory bound
    random.seed(0)
    base=tok.encode(PROSE[0])  #~6 token prompt for every request, only the generation length varies
    regimes=[("uniform 100 tok",[100]*160),
             ("spread 30-800 tok",[int(min(800,max(30,random.lognormvariate(4.4,0.9)))) for _ in range(160)]),
             ("all long 1200 tok",[1200]*24)]

    POOL_MB=256
    nb=POOL_MB*1024*1024//(tpb*kv_tok)
    print(f"MEASURED   pool {POOL_MB} MB = {nb} pages = {nb*tpb} tokens, MAX_BATCH lifted to 512")
    print("  5 interleaved rounds per config. tok/s best and median (run to run spread is ~1.5x on this box)")
    print(f"{'workload':20s} {'allocator':26s} {'running':>7s} {'best':>7s} {'median':>7s} {'mean len':>9s}  preempts  replays")
    CFGS=(("paged, admit on current",None,"cur"),("paged, admit on final",None,"final"),
          ("static @ prompt+max_new","req","final"),("static @ max_context 8192","ctx","final"))
    for name,lens in regimes:
        mk=lambda:[Request(base,l,temperature=0.9,top_k=50,top_p=0.95) for l in lens]
        res={c[0]:[] for c in CFGS}
        #interleaved rounds A,B,C,A,B,C.. so a slow stretch of time lands on every config equally
        for rnd in range(6):
            for label,st,pl in CFGS:
                try:
                    r=staggered(model,tok,mk(),st,POOL_MB*1024*1024,plan=pl)
                    if rnd>0: res[label].append(r)  #round 0 is warmup
                except RuntimeError:
                    res[label]=None
        for label,_,_ in CFGS:
            rs=res[label]
            if rs is None:
                print(f"  {name:18s} {label:26s} {'--':>7s} {'--':>7s} {'--':>7s} {'--':>9s}  does not fit")
                continue
            t=[x["tps"] for x in rs]; r=rs[0]
            print(f"  {name:18s} {label:26s} {max(x['peak'] for x in rs):7d} {max(t):7.0f} {statistics.median(t):7.0f} {r['mean_len']:9.0f}  {r['preempts']:8d} {r['replays']:8d}")
        print()

#---- naive ladder
def nocache(model,tok,prompts):
    #textbook generation without a kv cache: recompute the whole sequence for every new token,
    #one request at a time. prefill_rows is a full causal forward so it is exactly that loop,
    #the pages it writes are just scratch
    pool=block_table(model.num_layers+model.num_mtp_heads,model.nkv_heads,model.head_dim,
                     S.POOL_BYTES,S.TOKENS_PER_BLOCK,S.DEVICE)
    dev=S.DEVICE; n=0
    torch.cuda.synchronize(); t0=time.perf_counter()
    for p in prompts:
        ids=list(p) if isinstance(p,list) else tok.encode(p)
        u=User(pool)
        for _ in range(MAXNEW):
            T=len(ids)
            h=model.prefill_rows(torch.tensor(ids,device=dev).long(),pool,torch.arange(T,device=dev),
                                 torch.tensor([u.slot(i) for i in range(T)],device=dev).long(),
                                 torch.tensor([0,T],dtype=torch.int32,device=dev),T)
            probs=batch_probs(model.logits(h[-1:]),[0.9],[50],[0.95],[1.1],[ids])
            t=int(torch.multinomial(probs,1)); ids.append(t); n+=1
            if t==EOS: break
        u.free_cache()
    torch.cuda.synchronize(); dt=time.perf_counter()-t0
    del pool; torch.cuda.empty_cache()
    return n/dt

def naive():
    model,tok=load_model()
    print("MODEL LOADED")
    #long prompts: a paragraph repeated to 1000 tokens, so the no-cache recompute is O(T^2) for real
    para=tok.encode(" ".join(PROSE))
    LONG=[(para*(1000//len(para)+1))[:1000] for _ in range(64)]
    for name,short,long in (("short (~10 tok)",PROSE*4,None),("long (1000 tok)",None,LONG)):
        P=short or long
        for spec in (False,True): run(model,tok,P[:16],spec,16)  #warm
        nocache(model,tok,P[:1])
        print(f"\nLADDER {name}                            tok/s   ms/token")
        rows=[("no kv cache, batch 1",              lambda:nocache(model,tok,P[:4])),
              ("kv cache, batch 1",                 lambda:run(model,tok,P[:4],False,1)["tps"]),
              ("+ continuous batching, batch 16",  lambda:run(model,tok,P[:16],False,16)["tps"]),
              ("+ speculative decoding, batch 16", lambda:run(model,tok,P[:16],True,16)["tps"]),
              ("+ batch 64",                       lambda:run(model,tok,P[:64],True,64)["tps"])]
        for label,fn in rows:
            tps=max(fn() for _ in range(3))  #fastest of 3, same reason as best()
            print(f"  {label:36s} {tps:8.1f}   {1e3/tps:7.2f}")

#---- load test over http, poisson arrivals through serve.py
async def one(sess,prompt):
    import aiohttp
    t0=time.perf_counter(); first=None; buf=b""
    async with sess.post(f"http://localhost:{PORT}/generate",
                         json={"prompt":prompt,"max_new_tokens":MAXNEW,"stream":True}) as resp:
        async for chunk in resp.content.iter_any():
            if first is None: first=time.perf_counter()  #first byte = ttft as the client sees it
            buf+=chunk
    return t0,first,time.perf_counter(),buf.decode(errors="ignore")

async def sweep(rate,n,tok):
    import aiohttp
    tasks=[]
    async with aiohttp.ClientSession() as sess:
        for i in range(n):
            tasks.append(asyncio.create_task(one(sess,PROSE[i%len(PROSE)])))
            await asyncio.sleep(random.expovariate(rate))  #poisson arrivals at rate req/s
        res=await asyncio.gather(*tasks)
    ntok=sum(len(tok.encode(t)) for *_,t in res)
    span=max(r[2] for r in res)-min(r[0] for r in res)
    return dict(tps=ntok/span,ttft=[r[1]-r[0] for r in res],e2e=[r[2]-r[0] for r in res],rps=n/span)

def load():
    import urllib.request
    tok=Tokenizer(48000); tok.load_tokenizer(S.TOKENIZER_PATH)

    def up():
        try: urllib.request.urlopen(f"http://localhost:{PORT}/health",timeout=2); return True
        except Exception: return False

    proc=None
    if not up():
        proc=subprocess.Popen([sys.executable,"serve.py"],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        t0=time.time()
        while not up():
            time.sleep(5)
            if time.time()-t0>400: raise RuntimeError("serve.py did not come up")
    print("SERVER UP")

    random.seed(0)
    asyncio.run(sweep(2.0,16,tok))  #warm, same autotune reason as perf()
    print("WARM\n")
    print("LOAD   poisson arrivals, 24 req each point       tok/s   ttft p50    p99   e2e p50    p99  (ms)")
    base=None
    for rate in (0.5,1,2,4,8,16):
        st=asyncio.run(sweep(rate,24,tok))
        base=base or pct(st["e2e"],99)
        knee="  <- knee" if pct(st["e2e"],99)>2*base else ""  #p99 blew past 2x the unloaded p99
        print(f"  rate={rate:4.1f} req/s (got {st['rps']:4.1f})   {st['tps']:7.1f}   {pct(st['ttft'],50)*1e3:7.1f} {pct(st['ttft'],99)*1e3:7.1f}   {pct(st['e2e'],50)*1e3:7.1f} {pct(st['e2e'],99)*1e3:7.1f}{knee}")
        if knee: base=float("inf")  #mark once
    if proc: proc.terminate()


if __name__=="__main__":
    {"perf":perf,"load":load,"naive":naive,"mem":mem}[sys.argv[1] if len(sys.argv)>1 else "perf"]()
