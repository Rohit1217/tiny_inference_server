#async serving on top of the scheduler: live submission, continuous batching, token streaming.
#single-threaded asyncio so no locking: the engine loop runs one scheduler step, flushes newly
#committed tokens to each request's stream, then yields so http handlers can run between steps.
import asyncio
import torch
from aiohttp import web

from Config.config import Config
from Tokenizers.tokenizer_fast import Tokenizer
from block_table import block_table
from main_models_new import Transformer
from server import Scheduler,Request,CKPT_PATH,TOKENIZER_PATH,POOL_BYTES,TOKENS_PER_BLOCK,MAX_BATCH,DEVICE

cfg=Config()
PORT=8000


class Engine:
    def __init__(self,scheduler,tokenizer):
        self.sched=scheduler
        self.tok=tokenizer
        self.open=set()
        self.wake=asyncio.Event()

    def submit(self,req):
        req.emitted=0
        req.stream=asyncio.Queue()
        self.open.add(req)
        self.sched.submit(req)
        self.wake.set()

    async def loop(self):
        while True:
            if not (self.sched.queue or self.sched.active):
                self.wake.clear()
                await self.wake.wait()
            self.sched.admit()
            self.sched.step()

            #flush newly committed tokens. a step never shrinks out_ids below what was already
            #emitted (eos truncation only cuts tokens committed within the same step)
            for r in list(self.open):
                out=r.out_ids()
                for t in out[r.emitted:]:
                    r.stream.put_nowait(t)
                r.emitted=len(out)
                if r.done:
                    r.stream.put_nowait(None)
                    self.open.discard(r)

            await asyncio.sleep(0)  #let handlers accept/respond between steps


async def generate(request):
    engine=request.app["engine"]
    body=await request.json()

    req=Request(engine.tok.encode(body["prompt"]),
                max_new_tokens=body.get("max_new_tokens",200),
                temperature=body.get("temperature",0.9),
                top_k=body.get("top_k",50),
                top_p=body.get("top_p",0.95),
                rep_penalty=body.get("rep_penalty",1.1))
    engine.submit(req)

    if body.get("stream",True):
        resp=web.StreamResponse(headers={"Content-Type":"text/plain; charset=utf-8"})
        await resp.prepare(request)
        toks,sent=[],""
        while True:
            t=await req.stream.get()
            if t is None:
                break
            toks.append(t)
            text=engine.tok.decode(toks)  #decode-and-diff so multi-token bytes merge correctly
            await resp.write(text[len(sent):].encode())
            sent=text
        await resp.write_eof()
        return resp

    while await req.stream.get() is not None:
        pass
    return web.json_response({"text":engine.tok.decode(req.out_ids()),
                              "prompt_tokens":req.prompt_len,"new_tokens":len(req.out_ids())})


async def health(request):
    s=request.app["engine"].sched
    return web.json_response({"active":len(s.active),"queued":len(s.queue),
                              "finished":len(s.finished),"free_blocks":s.free_blocks()})


async def main():
    torch.manual_seed(0)

    model=Transformer(vocab_size=cfg.VOCAB_SIZE,max_context=cfg.MAX_CONTEXT,max_freq=cfg.MAX_FREQ,
                      d_model=cfg.D_MODEL,n_heads=cfg.N_HEAD,num_layers=cfg.NUM_LAYERS,
                      ffn_hidden_dim=cfg.FFN_HIDDEN_DIM,group_size=cfg.NUM_GROUPS,mtp_heads=cfg.MTP_HEADS)
    opt_step=model.load_ckpt(CKPT_PATH,DEVICE)

    tokenizer=Tokenizer(vocab_size=48000)
    tokenizer.load_tokenizer(TOKENIZER_PATH)

    pool=block_table(cfg.NUM_LAYERS+cfg.MTP_HEADS,cfg.N_HEAD//cfg.NUM_GROUPS,cfg.D_MODEL//cfg.N_HEAD,
                     POOL_BYTES,TOKENS_PER_BLOCK,DEVICE)
    scheduler=Scheduler(model,pool,MAX_BATCH,cfg.MAX_CONTEXT,DEVICE,spec=True)
    engine=Engine(scheduler,tokenizer)

    app=web.Application()
    app["engine"]=engine
    app.router.add_post("/generate",generate)
    app.router.add_get("/health",health)

    runner=web.AppRunner(app)
    await runner.setup()
    await web.TCPSite(runner,"0.0.0.0",PORT).start()
    print(f"MODEL LOADED (opt_step={opt_step}), POOL {pool.num_blocks} blocks, serving on :{PORT}")

    await engine.loop()


if __name__=="__main__":
    asyncio.run(main())
