# llm_inference

An inference server for the ~1B model I trained in
[LLM_train](../LLM_train), written from scratch to understand how serving
actually works: paged KV cache, continuous batching, and speculative decoding
from the model's own MTP heads. Runs on a single RTX A6000.

I wrote the kernels and the scheduler myself instead of using vLLM so I could
see where the time really goes. Spoiler: at this model size it is almost all
Python and kernel-launch overhead, not the GPU.

## What's in here

- A paged KV cache. The cache is one big tensor of fixed-size pages
  (16 tokens each), and every request holds a list of page ids. A Triton
  flash-decode kernel reads through the page table, so a batch of requests
  with completely different context lengths runs in one launch.
- Continuous batching. Requests are admitted whenever there is a free slot,
  finished ones are evicted and their pages returned, and if the pool fills
  up the newest request is preempted and replayed later.
- Batched prefill through the intra-document attention kernel from the
  training repo. Prompts of different lengths are concatenated and run as
  one "packed sequence", which is exactly what that kernel was built for.
- Speculative decoding using the two MTP heads the model was trained with.
  The heads draft two tokens, one trunk pass verifies both, and if both are
  accepted a third token comes for free from the same pass.
- A small async HTTP server (`serve.py`) with token streaming.
- An eval harness (`evals.py`) for latency, throughput, acceptance rates,
  memory and a bandwidth roofline.

The whole thing is about 1800 lines including the kernels.

## How it works

Everything runs through one path. A "row" is `(request, position, token)`.
Each launch writes the KV for every row into its page first and then runs
attention where the row at position `t` sees `t+1` tokens. Because writes
happen before reads, a chunk of consecutive rows from one request is
causal automatically. Prefill, normal decode, MTP drafting and speculative
verification are all just different sets of rows.

Speculative decoding needs care with the KV cache: a rejected draft leaves
stale K/V behind in the trunk and in the head caches. Pages are addressed by
position, so "deleting" stale entries is just rewinding a per-plane length
and letting the next write overwrite the slot. The MTP heads keep their own
frontier and catch up over positions committed since they last ran, so their
caches never have holes.

The scheduler loop is:

```
admit()          fill free batch slots from the queue if pages allow
step()           preempt if the pool cannot fit this iteration
                 prefill new requests (intradoc kernel)
                 draft with the heads, one trunk pass to verify + bonus token
                 evict finished, free pages
```

## Model

Same model as the training repo, loaded from its checkpoint. 30 layers,
d_model 1536, 12 heads with 3 KV heads (GQA), 2 MTP heads, 48k custom BPE
vocab. 876M params, 1.75 GB in bf16.

## Setup

```bash
pip install torch triton aiohttp tiktoken regex
cp /path/to/checkpoint_step16800.pt models/
```

## Run

```bash
# batch a few prompts and print outputs + tok/s
python server.py

# http server on :8000, streams tokens
python serve.py
curl -N localhost:8000/generate -d '{"prompt":"The capital of France is","max_new_tokens":100}'

# evals: in-process perf sweep, or poisson load through the http server
python evals.py perf
python evals.py load
```

## Metrics

All numbers on one A6000, bf16, 100 new tokens per request, sampling with
temperature 0.9 / top-k 50 / top-p 0.95 / repetition penalty 1.1. Everything
is measured after warming up, because Triton autotunes per context-length
bucket and the first run in a process pays for the whole sweep (I got this
wrong the first time and reported a 2x speedup that was mostly warm-up).

Throughput vs batch size:

| batch | plain tok/s | spec tok/s | spec tok/iter | ms/iter (plain / spec) | roofline % |
|---|---|---|---|---|---|
| 1 | 29.0 | 49.9 | 1.98 | 34.6 / 39.4 | 6.2 / 5.8 |
| 4 | 120.9 | 185.8 | 1.97 | 32.5 / 39.1 | 6.6 / 5.9 |
| 16 | 483.1 | 673.7 | 1.99 | 33.1 / 41.1 | 6.6 / 5.7 |
| 32 | 848.9 | 1226.0 | 1.98 | 37.3 / 43.7 | 6.0 / 5.5 |
| 64 | 1863.8 | 2082.5 | 2.01 | 34.1 / 49.1 | 6.9 / 5.0 |

Roofline is the fraction of the memory-bandwidth floor: decode has to read
the 1.75 GB of weights once per iteration, which is 2.3 ms at 768 GB/s. We
are at 5-7% of that, and an iteration takes the same ~33 ms (plain) or
~40 ms (spec) whether the batch is 1 or 64. That fixed cost is kernel
launches and Python, roughly 1000 launches per iteration. Throughput
scales almost linearly with batch because extra rows are nearly free, so
per-token latency does not get worse as the batch grows.

The first version of this sampled each row separately in Python (its own
logits matvec reading the whole 147 MB embedding matrix, its own top-k /
top-p, and a GPU sync for every accept test). That cost ~1 ms per request
per iteration for plain decode and ~3.9 ms with speculation, enough that
speculation actually lost to plain decode at batch 16 (307 vs 325 tok/s).
Batching the logits, the sort and the accept test across all rows in an
iteration removed it: batch 16 went from 307 to 674 tok/s with speculation.

How much each piece buys, starting from the textbook loop that recomputes
the whole sequence for every token (`python evals.py naive`). 100 new tokens
per request, short prompts are ~10 tokens, long prompts are 1000 tokens:

| | short tok/s | long tok/s |
|---|---|---|
| no KV cache, one request at a time | 39.5 | 28.3 |
| KV cache, one request at a time | 30.0 | 32.3 |
| + continuous batching, batch 16 | 490.5 | 420.5 |
| + speculative decoding, batch 16 | 714.4 | 737.3 |
| + batch 64 | 2250.2 | 1365.8 |

The first two rows are the honest surprise. The KV cache does not help at
all on short prompts and is only 14% faster at 1000 tokens. At this model
size the recompute is cheap (a full forward over 1000 tokens is ~2 TFLOP,
20-30 ms on this GPU) and the decode path actually has *more* kernel
launches per layer than the prefill path (page write + flash decode +
reduce, versus one intra-doc attention launch), so at batch 1 the cache
saves compute that was never the bottleneck. It would start to matter at a
few thousand tokens of context, or with a bigger model.

Batching is where the real win is (about 15x), because an iteration costs
the same ~33 ms whether it carries 1 row or 16. Speculation adds another
1.5-1.75x on top, and going to batch 64 another 2-3x. End to end that is
57x over the naive loop on short prompts and 48x on long ones. The long
prompt is a repeated paragraph, which the model predicts well, so draft
acceptance is a bit higher there.

Speculative decoding stats (batch 8):

| | alpha1 | alpha2 given d1 accepted | tokens / iteration |
|---|---|---|---|
| temp 0.6 | 0.63 | 0.58 | 1.98 |
| temp 0.9 | 0.61 | 0.59 | 1.96 |
| temp 1.2 | 0.64 | 0.63 | 2.03 |
| code prompts | 0.63 | 0.59 | 1.99 |

The second head is nearly as accurate as the first, and acceptance barely
moves with temperature or prompt domain.

Latency at batch 8 (ms, p50 / p95 / p99):

| | plain | spec |
|---|---|---|
| inter-token, per iteration | 40.0 / 40.2 / 40.7 | 67.4 / 71.5 / 72.8 |
| inter-token, per token | 40.0 / 40.3 / 41.1 | 47.3 / 71.8 / 92.8 |
| prefill step (8 short prompts) | 33.4 / 33.6 / 33.6 | 28.9 / 34.5 / 36.3 |

With speculation tokens arrive in clumps, so the per-token p50 is below the
iteration time. The p99 is above it because a new request's prefill runs in
the same step as everyone else's decode and stalls them.

HTTP load test (Poisson arrivals, 24 requests per point, streaming):

| offered rate | tok/s | ttft p50 / p99 (ms) | e2e p50 / p99 (ms) |
|---|---|---|---|
| 0.5 req/s | 36 | 30 / 354 | 2079 / 2610 |
| 1.0 req/s | 89 | 211 / 393 | 2568 / 2945 |
| 2.0 req/s | 161 | 402 / 2937 | 4890 / 6195 |
| 4.0 req/s | 174 | 527 / 3424 | 4120 / 6318 |
| 8.0 req/s | 188 | 3143 / 7200 | 7398 / 10007 |

The server saturates around 2 requests/s for 100-token requests; past that
everything queues.

Memory: the 4 GB page pool is <1% used by these workloads. With the
16-token pages about 14% of allocated slots are empty at any time. A pool
starved down to 40 pages hits 100% utilisation, preempts 10 times and still
produces the same outputs.

Correctness (`test_server.py`): with `top_k=1` speculative decoding must
produce exactly the same tokens as plain decoding, including under forced
preemption; batched prefill and a decode step over prefill-written pages
are checked against a dense attention reference at every position; and
because greedy cannot see the residual-resampling path, 4096 sampled
3-token continuations are compared as histograms, speculative vs plain,
against a plain-vs-plain baseline.

## Status

Works end to end, but slow in absolute terms because it is overhead-bound.
In order of expected payoff:

1. CUDA graphs for the decode iteration, to get rid of the fixed ~32 ms.
   Everything else is now small next to it.
2. Run prefill for new arrivals in its own iteration so it does not stall
   in-flight decodes.
3. The head-cache rollback is written for 2 MTP heads. With 3 or more, a
   rejected draft can invalidate more than the tip entry; the fix is known
   but not done.

Done: batched sampling (was the per-request cost that made speculation lose
at large batch).

## Layout

```
Config/                       model config (shared with training)
Tokenizers/                   custom BPE tokenizer + vocab
main_models_new.py            the transformer, inference-only, with MTP heads
block_table.py                page pool, per-request allocator, kv write kernel
paged_attention_kernels.py    triton flash decode over a page table (GQA)
intradocatt_fwd_kernels.py    triton intra-doc attention, used for prefill
server.py                     scheduler: admit / prefill / spec decode / preempt
serve.py                      aiohttp server with streaming
evals.py                      perf + load evals
test_server.py                correctness tests (tiny random model)
test_gqa_paged.py             kernel test
```
