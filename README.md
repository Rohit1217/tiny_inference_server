# llm_inference

An inference server for the 0.9B model I trained in [LLM_train](../LLM_train),
written from scratch to understand how LLM serving actually works: a paged KV
cache, continuous batching, and speculative decoding using the model's own
multi-token-prediction (MTP) heads. It runs on a single RTX A6000.

I wrote the kernels and the scheduler myself instead of using vLLM so I could
see where the time really goes. The short answer: at this model size almost
all of it is kernel-launch and Python overhead, not GPU work. The GPU is busy
about 7% of the time.

About 1000 lines of Python for the model, cache, scheduler and HTTP server,
600 lines of Triton kernels, and 600 lines of evals and tests.

## Terms used below

- **KV cache**: the keys and values of every token a request has seen, kept
  on the GPU so each new token only needs one forward pass instead of
  recomputing the whole sequence.
- **Page**: the KV cache is split into fixed 16-token pages. A request holds a
  list of page ids, so its cache does not have to be contiguous.
- **Prefill / decode**: prefill runs the whole prompt through the model once;
  decode then produces tokens one iteration at a time.
- **Iteration**: one step of the server loop, in which every running request
  advances together in one batched forward pass.
- **Continuous batching**: requests join and leave the batch between
  iterations, instead of the whole batch starting and finishing together.
- **MTP heads / speculative decoding**: the model was trained with two extra
  heads that guess the next two tokens. Those guesses (drafts) are checked by
  the main model in one pass; accepted drafts are free extra tokens.
- **Acceptance rate (alpha)**: how often a draft is accepted. alpha2 is
  measured only when the first draft was accepted.
- **Preemption**: if the running requests grow past the memory available,
  the newest one is evicted, its pages freed, and it is re-run from its
  prompt plus the tokens it already produced when memory frees up.
- **Roofline**: the fastest an iteration could possibly be. Decode has to
  read all the weights once per iteration, so the floor is weight bytes /
  memory bandwidth, about 2.2 ms here.

## What's in here

- A paged KV cache. One big GPU tensor of 16-token pages, a free list, and a
  per-request page list. A Triton flash-decode kernel reads through the page
  table, so requests with very different context lengths run in one launch.
- Continuous batching with page-aware admission and preemption.
- Batched prefill through the intra-document attention kernel from the
  training repo: prompts of different lengths are concatenated and run as one
  packed sequence, which is exactly what that kernel was built for.
- Speculative decoding with the two MTP heads. The heads draft two tokens,
  one pass of the main model checks both, and if both are accepted a third
  token comes free from the same pass.
- Sampling with temperature, top-k, top-p and repetition penalty, done for
  every request in the batch at once.
- A small async HTTP server (`serve.py`) with token streaming.
- Evals (`evals.py`) and correctness tests (`test_server.py`).

## How it works

Everything is expressed as rows. A row is `(request, position, token)`. Each
launch writes the K/V of every row into its page first and then runs
attention, where the row at position `t` looks at `t+1` tokens. Because the
writes happen before the reads, a run of consecutive rows from one request is
causal automatically. Normal decode, MTP drafting and speculative
verification are all just different sets of rows through the same path.
Prefill uses the same rows but the intra-document kernel for attention,
which is exact because a request being prefilled always starts with an empty
cache.

Speculative decoding needs care with the KV cache: a rejected draft leaves
stale K/V behind. Pages are addressed by position, so throwing away stale
entries is just rewinding a length counter; the slot gets overwritten when
that position is written again. Each MTP head has its own cache and keeps a
"frontier", the first position it has not processed yet, and catches up over
every position committed since it last ran, so its cache never has holes.

The server loop is:

```
admit()   move requests from the queue into the batch while memory allows
step()    preempt the newest request if this iteration will not fit
          prefill requests that just joined
          heads draft 2 tokens, one main-model pass verifies them (+ bonus token)
          evict finished requests and free their pages
```

There are two separate memory policies:

- **Storage.** Paged (a request takes pages as it grows) or static (a request
  reserves every page it could ever need when it is admitted). Static exists
  only for comparison.
- **Admission.** `plan="final"` (default) only admits a request if everyone
  still fits once they have all generated their maximum length, so it never
  preempts. `plan="cur"` admits based on what requests hold right now, which
  fits more requests but preempts when they grow.

## Model

The same model as the training repo, loaded from its checkpoint: 0.9B
parameters (817M in the main model, 59M in the two MTP heads), 30 layers,
d_model 1536, 12 query heads sharing 3 KV heads (GQA), a 48k custom BPE
vocabulary, 1.75 GB in bf16.

## Setup

```bash
conda activate llm            # or: pip install torch triton aiohttp tiktoken regex
cp /path/to/checkpoint_step16800.pt models/
```

## Run

```bash
# run a few prompts through the scheduler, print outputs and tok/s
python server.py

# http server on :8000, streams tokens
python serve.py
curl -N localhost:8000/generate -d '{"prompt":"The capital of France is","max_new_tokens":100}'

# correctness tests (tiny random model, a few minutes)
python test_server.py

# evals
python evals.py perf     # latency, throughput vs batch, speculation stats, roofline
python evals.py naive    # what each technique buys, from a no-cache baseline
python evals.py mem      # paged vs static KV cache at equal memory
python evals.py load     # poisson load through the http server
```

The defaults are `MAX_BATCH=8` and a 4 GB page pool, set at the top of `server.py`.

## Results

### How these were measured

One A6000, bf16, 100 new tokens per request unless stated, sampling at
temperature 0.9, top-k 50, top-p 0.95, repetition penalty 1.1.

Two things about measuring on this machine turned out to matter a lot:

- **Warm-up.** Triton re-tunes the attention kernel for every new
  context-length bucket, and the first run in a process pays for all of it.
  My first speedup numbers compared a cold run against a warm one and
  claimed a 2x gain from speculation that was mostly warm-up. Every number
  here comes after a warm-up run.
- **Run-to-run noise.** Throughput on this shared machine jumps between two
  levels about 1.5x apart from one run to the next, with identical code and
  input. It is not which CPU socket the process runs on, not the process
  moving between cores, and not a memory leak (I checked all three), and I
  haven't found the cause. The noise only ever slows a run down, so each
  number is the fastest of 3 runs (5 for the paged vs static table, which
  also shows the median). Configurations being compared there run
  interleaved (A, B, C, A, B, C, ...) so a slow stretch hits all of them.
  The HTTP load test is a single run per point.

### Where the time goes

Throughput against batch size, plain decoding vs speculative decoding:

| batch | plain tok/s | spec tok/s | spec / plain | tokens per iteration (spec) | ms per iteration (plain / spec) | % of roofline (plain / spec) |
|---|---|---|---|---|---|---|
| 1 | 32.7 | 52.6 | 1.61x | 1.98 | 30.6 / 37.3 | 7.0 / 6.1 |
| 4 | 124.3 | 188.3 | 1.51x | 1.97 | 31.6 / 38.6 | 6.8 / 6.0 |
| 16 | 498.5 | 682.4 | 1.37x | 1.99 | 32.1 / 40.5 | 6.8 / 5.8 |
| 32 | 984.7 | 1289.1 | 1.31x | 1.97 | 32.4 / 43.6 | 6.9 / 5.5 |
| 64 | 1871.6 | 2089.8 | 1.12x | 2.01 | 33.9 / 49.0 | 6.9 / 5.0 |

An iteration takes about 31 ms (plain) or 37 ms (speculative) no matter how
many requests are in it, and the roofline floor is about 2.2 ms. So around
93% of every iteration is fixed overhead: roughly 1000 small kernel launches
plus Python. Two things follow from that:

- Batching is almost free. Going from 1 to 64 requests multiplies
  throughput by 57x while an iteration only gets 11% slower.
- Speculation's advantage shrinks as the batch grows (1.61x down to 1.12x).
  Each iteration it still commits about 2 tokens per request, but its own
  per-request cost (drafting, the accept test, more rows in every launch)
  grows with the batch while plain decoding's barely does.

The first version sampled every request separately in Python: its own
logits computation (a full read of the 147 MB embedding matrix per request),
its own top-k / top-p, and a GPU sync for every accept decision. That cost
about 1 ms per request per iteration for plain decoding and 3.9 ms with
speculation, enough that speculation was *slower* than plain decoding at
batch 16 (307 vs 325 tok/s). Doing all of it for the whole batch at once
fixed that; batch 16 with speculation went from 307 to 682 tok/s.

### What each technique buys

Starting from the textbook loop that recomputes the whole sequence for every
new token (`python evals.py naive`). Short prompts are about 10 tokens, long
prompts are 1000 tokens:

| | short prompts, tok/s | long prompts, tok/s |
|---|---|---|
| no KV cache, one request at a time | 38.5 | 28.1 |
| KV cache, one request at a time | 31.8 | 32.1 |
| + continuous batching, 16 requests | 488.3 | 431.2 |
| + speculative decoding | 677.4 | 750.3 |
| + 64 requests | 2076.3 | 1373.9 |

The KV cache is the surprise. With short prompts it makes things *slower*
(31.8 vs 38.5), and with 1000-token prompts it is only 14% faster.
Recomputing a 1000-token sequence adds only about 4 ms per token on top of the
~31 ms fixed cost, so there is little for the cache to save. On top of that,
the cached decode path launches more kernels per layer (page write, flash
decode, a reduce step, two buffer allocations) than the recompute path (page
write and one attention kernel). The cache would start to matter at a few
thousand tokens of context or with a bigger model.

The big win is batching (13-15x), because an iteration costs the same with
1 or 16 requests in it. Speculation adds another 1.4-1.7x, and going to 64
requests another 1.8-3.1x. End to end that is 54x over the naive loop on
short prompts and 49x on long ones. Speculation gains more on the long
prompts because they are a repeated paragraph, which the heads predict well.

### Speculative decoding

Batch 8:

| | alpha1 | alpha2 (given draft 1 accepted) | tokens per iteration |
|---|---|---|---|
| temperature 0.6 | 0.64 | 0.58 | 2.00 |
| temperature 0.9 | 0.61 | 0.58 | 1.95 |
| temperature 1.2 | 0.63 | 0.64 | 2.03 |
| code prompts | 0.69 | 0.62 | 2.11 |

Tokens per iteration is `1 + alpha1 + alpha1*alpha2`: the 1 is the token
that always comes out (the main model's own pick after a rejection, or the
bonus token when both drafts are accepted). The second head is nearly as
accurate as the first, acceptance hardly moves with temperature, and code is
a bit easier to draft than prose. A third head would add about 0.2 tokens
per iteration for one more launch, which is not worth it while the fixed
cost is 31 ms.

An earlier version ran the main model twice per iteration: once to produce
the next token and once to verify the drafts. Drafting from the hidden state
the verify pass had already computed removed one of the two passes. Measured
that way, speculation went from 1.03x to 1.33x over plain decoding at
batch 4.

### Paged vs static KV cache, at equal memory

The point of paging is that a request does not have to reserve its
worst-case length when it starts. To measure that, the scheduler has a
static mode that reserves every page a request could need at admission.
Everything else (kernels, scheduler, sampling) is the same.

How many requests fit at once is just arithmetic. The KV cache costs 48 KiB
per token (30 layers plus 2 MTP heads, 3 KV heads, head size 128, K and V,
2 bytes each), so a 16-token page is 768 KiB:

| pool | pages | request length | if the length is known (paged or static) | static, reserving the max context (8192) |
|---|---|---|---|---|
| 256 MB | 341 | 110 tokens | 48 | does not fit |
| 1 GB | 1365 | 110 tokens | 195 | 2 |
| 4 GB | 5461 | 110 tokens | 780 | 10 |
| 4 GB | 5461 | 1510 tokens | 57 | 10 |

A server that doesn't know how long a request will run has to reserve the
maximum context, which is the last column: 10 requests in 4 GB instead of
780, and not even one in 256 MB. That is the main argument for paging.

Measured with a 256 MB pool, so that memory is what limits every
configuration. Requests arrive a few per iteration so they are at different
lengths at any moment, and the batch cap is raised to 512. "Running" is the
most requests that ran together in one iteration. "Replayed" counts requests
that were preempted after doing work and had to be re-run.

| workload | storage, admission | running | tok/s best | tok/s median | replayed |
|---|---|---|---|---|---|
| 160 x 100 tokens | paged, admit on current | 120 | 2327 | 2223 | 84 |
| | paged, admit on final | 48 | 1744 | 1683 | 0 |
| | static, reserve prompt + max_new | 48 | 1711 | 1167 | 0 |
| | static, reserve max context | does not fit | | | |
| 160 x 30-800 tokens | paged, admit on current | 113 | 1104 | 1019 | 102 |
| | paged, admit on final | 37 | 1018 | 926 | 0 |
| | static, reserve prompt + max_new | 39 | 1035 | 915 | 0 |
| | static, reserve max context | does not fit | | | |
| 24 x 1200 tokens | paged, admit on current | 24 | 642 | 526 | 13 |
| | paged, admit on final | 4 | 253 | 233 | 0 |
| | static, reserve prompt + max_new | 4 | 246 | 235 | 0 |
| | static, reserve max context | does not fit | | | |

What this shows:

- **When the length is known, paging alone buys nothing.** Paged storage
  with conservative admission runs the same number of requests as static
  storage, at the same speed (1744 vs 1711, 1018 vs 1035, 253 vs 246).
- **What paging buys is the option to over-admit.** Since a request only
  holds the pages it has grown into so far, you can admit 2.5-6x more
  requests than would fit at full length and handle the overflow by
  preempting.
- **Over-admitting pays off here**: 1.33x faster on the 100-token workload
  and 2.5x on the long one, even though there were 84 replays for 160
  requests. That is because this server is overhead-bound: an
  iteration costs the same with 48 requests or 120, and a replay is one
  batched prefill. On a server that was actually limited by memory
  bandwidth the trade could go the other way.
- On the mixed-length workload the gain is only 8%, which is within the
  noise. I haven't dug into why. My guess is that the end of the run is
  dominated by a few 800-token requests that run at low concurrency under
  either policy.

An earlier version of this table was wrong. For paged storage, `admit()`
checked each waiting request against the number of free pages, but pages are
only taken when a request first runs, so that number never went down while
admitting. One request fitting let the whole queue in, and the next step
preempted most of them before they ran. That made "running" look like 158
instead of 120, and most of the "preemptions" were requests that never did
any work. Admission now keeps a running budget, and "running" is counted
after preemption.

The default is still `plan="final"`: it never preempts, so no request ever
pays the extra latency of being re-run. I haven't measured that latency cost
yet, so I haven't switched the default.

### Latency

Batch 8, 16 prompts all submitted at once, times in ms:

| | plain p50 / p99 | spec p50 / p99 |
|---|---|---|
| iteration time | 31.7 / 32.2 | 38.8 / 41.2 |
| time between tokens | 31.8 / 32.3 | 38.6 / 66.5 |
| prefill step for newly joined requests | 26.1 / 26.9 | 27.8 / 29.4 |
| time to first token | 1614 / 3202 | 591 / 2403 |

With speculation about 2 tokens arrive per iteration, so on average a token
arrives every ~20 ms even though an iteration takes 39 ms. The p99 of 66.5 ms
between tokens is an iteration plus a prefill (38.8 + 27.8): when a new
request joins, its prefill runs in the same step and every other request
waits for it. Time to first token is mostly waiting for one of the 8 batch
slots: 16 requests arrive at once and half of them queue.

### HTTP load test

Requests arrive at random (Poisson) at the given rate, 24 per point, one run
per point, through the streaming HTTP server with the default `MAX_BATCH=8`:

| offered rate | tok/s | time to first token p50 / p99 (ms) | full request p50 / p99 (ms) |
|---|---|---|---|
| 0.5 req/s | 36 | 31 / 326 | 2083 / 2616 |
| 1 req/s | 89 | 204 / 332 | 2150 / 2543 |
| 2 req/s | 165 | 220 / 397 | 2380 / 2732 |
| 4 req/s | 225 | 337 / 1361 | 2504 / 3378 |
| 8 req/s | 228 | 4257 / 5675 | 6530 / 7859 |
| 16 req/s | 321 | 1919 / 3668 | 4059 / 5847 |

The server keeps up with about 2 requests/s. Past that it finishes at most
about 2.3-3.3 requests/s, and anything more waits in the queue. That limit
comes from the 8 batch slots, not from the GPU; the batch
table above shows the same hardware doing about 2000 tok/s at batch 64. The
16 req/s row is better than the 8 req/s row, which is noise from using only
24 requests per point.

### Memory

With the default 4 GB pool these workloads use under 1% of it. With 16-token
pages about 14% of the allocated slots are empty at any moment (the unused
tail of each request's last page). Starving the pool down to 40 pages, the
default admission policy holds requests in the queue instead of preempting:
peak use is 85% of the pool, with no preemptions.

### Correctness

`test_server.py` runs on a tiny randomly initialised model:

- Batched prefill and a decode step that reads the prefill's pages are
  compared against a plain dense-attention forward pass, at every position.
- With `top_k=1` (greedy), speculative decoding must produce exactly the same
  tokens as plain decoding, including when preemption is forced with a
  5-page pool. Any stale-cache or rollback bug shows up as a mismatch.
- Greedy decoding can't test the rejection sampling path, so 4096 sampled
  3-token continuations are compared as histograms, speculative vs plain,
  against a plain-vs-plain baseline, with and without top-k/top-p and
  repetition penalty.

## Limitations and next steps

In order of expected payoff:

1. **CUDA graphs** to remove the ~31 ms fixed cost per iteration, which is
   93% of the time. Everything else is small next to it.
2. **Raise `MAX_BATCH`.** It defaults to 8, but batch 64 gives about 4x the
   throughput at nearly the same time per iteration. It is a one-line
   change, but the time to first token and memory behaviour at high load
   should be checked first.
3. **Run prefill in its own iteration**, so a new request doesn't stall
   everyone else (the 66 ms p99 above).
4. **Choose the admission policy with latency in mind.** Over-admitting
   gives more throughput, but replayed requests take longer, and I haven't
   measured by how much.
5. **More than 2 MTP heads.** The cache rollback for rejected drafts is
   written for 2 heads. With 3 or more, one rejection can invalidate more
   than the last entry of a head's cache. The fix is known but not done.
   Output would stay correct either way; only acceptance would drop.
6. **Find the source of the 1.5x run-to-run noise.** Ruled out so far: CPU
   socket, core migration, memory leaks.

## Layout

```
Config/                       model config (shared with training)
Tokenizers/                   custom BPE tokenizer + vocab
main_models_new.py            the transformer, inference only, with MTP heads
block_table.py                page pool, per-request page list, kv write kernel
paged_attention_kernels.py    triton flash decode over a page table (GQA)
intradocatt_fwd_kernels.py    triton intra-document attention, used for prefill
server.py                     scheduler: admission / prefill / speculative decode / preemption
serve.py                      aiohttp server with streaming
evals.py                      perf, naive, mem and load evals
test_server.py                correctness tests (tiny random model)
test_gqa_paged.py             kernel test for the paged attention kernel
generate_samples_old.py       the old no-cache generation loop, kept for reference
```
