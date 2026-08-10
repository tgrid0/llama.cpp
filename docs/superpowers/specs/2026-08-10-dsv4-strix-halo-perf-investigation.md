# DeepSeek-V4-Flash on Strix Halo: performance investigation brief

Status: investigation brief for a future agent session. No code has been changed yet.
Owner context: single Strix Halo (gfx1151, 128GB unified memory) box, ROCm 7.14 toolbox
(`amd-strix-halo-toolboxes`), llama.cpp branch `b10331_zgfs`.

## The problem

Model: `DeepSeek-V4-Flash-Layers37-42Q4KExperts-OtherExpertLayersIQ2XXSGateUp-Q2KDown-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-fixed-0731.gguf`
(antirez quant, 91GB, arch `deepseek4`, 43 layers, 256 experts/layer, 6 active). Full GGUF
tensor dump: `tmpwork/antirez_deepseek-v4-flash-0731_gguf_dump.txt`.

Fully loaded into unified memory (`--n-gpu-layers 999 --fit off`), ROCm/HIP backend.

| Metric | Expected (other users, different servers) | Observed |
|---|---|---|
| Prefill (t/s, ctx depth 0) | ~250 | ~80 |
| Decode (t/s, ctx depth 0) | ~11 (17-32 with MTP/DSpark, not in scope here) | ~7 |

**Hard observational data (btop/htop, not a guess)**, comparing against Qwen (a known-good
control model on this same box/branch):

| Model | Phase | GPU | CPU |
|---|---|---|---|
| Qwen | prefill | 100% | ~0% |
| Qwen | decode | 100% | ~0%, occasional bumps on random threads |
| DeepSeek-V4-Flash | prefill | 100% | ~0% |
| DeepSeek-V4-Flash | decode | **45%** | **100% on all 32 threads** |

This is a much stronger signal than "decode switches to CPU" — it's specifically **decode
only**, and it's **all threads pegged**, not one op stealing a slow generic-CPU-kernel path
on a single thread. Two things this rules in/out:

- **Rules out a permanently-CPU-resident tensor** (Branch B in the fix doc) as the sole
  explanation. If some MoE expert tensor (or any weight) were simply buffer-placed on CPU,
  prefill would touch that same tensor too — usually *more* expensively under batching, not
  less. Clean 0% CPU at prefill with 100%-on-all-threads at decode strongly suggests the
  differentiator is something in the **decode-specific graph path** (single query token,
  incremental KV-state update), not a static buffer-placement mistake that would apply
  equally to both phases.
- **The "all 32 threads at 100%" signature matches ggml-cpu's own multithreaded op
  execution**, not idle/blocked I/O threads (which wouldn't show sustained 100% compute
  across every thread) and not a single missing-kernel op running single-threaded on the
  scheduler thread (which would show one thread pegged, not all 32). This looks like the ggml
  scheduler assigned a real, non-trivial chunk of the decode graph to the CPU backend, and
  ggml-cpu is doing what it always does with `--threads 32`: splitting that work across all
  configured threads. Finding *which* graph nodes via `GGML_SCHED_DEBUG=2` (step 1 below) is
  now even more clearly the right first move — this isn't a vague "check for a slow kernel"
  hunt anymore, it's confirming a specific, testable hypothesis.

**New lead from this data**: since prefill (batched, N tokens per forward pass) is clean but
decode (single token, incremental) is not, look specifically for graph-building logic that
**differs structurally between the two paths**, not just an op that's used in both but happens
to be slow. `src/models/deepseek4.cpp` has exactly this kind of fork already read this
session: `dsv4_build_state_restore` / `dsv4_build_state_snapshot` (lines ~217-259) handle
incremental compressed-KV-cache state updates via `ggml_get_rows`/`cpy_kv`/`cpy_score` with
index tensors (`inp.state_restore_src_idxs` etc.) — this machinery exists specifically to
splice a *single new position* into the CSA/HCA compressed-attention state cache as tokens
arrive one at a time, which reads as a decode-specific concern (prefill likely builds the
compressed blocks directly from the full batch without needing incremental restore/snapshot
via indexed gather). Check whether any op in that chain, or in whatever consumes its output,
lacks HIP coverage or is being computed on CPU only in the `n_tokens == 1` case. This doesn't
replace the `GGML_SCHED_DEBUG=2` step — it's a specific place to look once that step names
the CPU-bound ops.

Full launch command for reference:
```
llama-server --port ${PORT} --threads 32 --flash-attn on --fit off --no-warmup \
  --batch-size 4096 --ubatch-size 1024 --cache-type-k q8_0 --cache-type-v q8_0 --jinja \
  --load-mode dio --cache-prompt --cache-ram 0 --parallel 1 --n-gpu-layers 999 \
  --ctx-checkpoints 4 --model .../DeepSeek-V4-Flash-...-chat-v2-imatrix-fixed-0731.gguf \
  --ctx-size 1048576 --temp 1.0 --reasoning on --reasoning-preserve \
  --cache-disk /home/walker/mnt/llm-cache/deepseek-v4-flash-0731-antirez --cache-disk-size 32768 \
  --alias antirez/deepseek-v4-flash-0731-q2q4
```
No `--cpu-moe` / `--n-cpu-moe` / `--override-tensor` flags are set, so any CPU placement is
either automatic (scheduler op-support fallback) or a side effect of `--load-mode dio` /
the disk-cache feature interacting with tensor buffer placement — not an explicit request.

## What's already confirmed in this codebase (read, not yet run)

The `deepseek4` architecture is **fully implemented** here, not a stub — see
`src/models/deepseek4.cpp` (~1550 lines): hyper-connections (`hc_attn`/`hc_ffn`/`hc_head`),
compressed-KV attention (CSA/HCA, ratios 4/128), the DeepSeek Sparse Attention lightning
indexer, and a hash-based MoE router override for the first `hash_layer_count` layers
(`ffn_gate_tid2eid`, a token-id -> expert-id lookup table, consumed via
`ggml_get_rows(ctx0, layer.ffn_gate_tid2eid, res->t_inp_tokens)` at
`src/models/deepseek4.cpp:1335` — this is a normal `GET_ROWS` op, should run on GPU fine,
probably not a CPU-fallback source but worth confirming with the profiling method below).

Custom fused ggml ops used by this arch and their backend coverage:
- `GGML_OP_DSV4_HC_PRE/COMB/POST` (`ggml/src/ggml-cuda/dsv4-hc.cu`) — **HIP-covered**. The
  file only excludes HIP for one internal fast-path block (lines ~450-511), falling back to
  a generic branch that HIP still compiles and runs. Not a suspected CPU-fallback source.
- `GGML_OP_LIGHTNING_INDEXER` (`ggml/src/ggml-cuda/lightning-indexer.cu`) — **partially
  degraded on HIP, but still GPU**. Lines 6-237 wrap an NVIDIA-only Turing-MMA/WMMA fast
  kernel in `#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)` with the comment
  `// TODO add support for AMD cards via rocWMMA`. HIP falls through to
  `lightning_indexer_kernel_vec` (line 244 on), a generic per-warp vector kernel — slower
  than WMMA, but it is a GPU kernel, not a CPU fallback. `ggml_cuda_lightning_indexer_supported`
  (line 536) does not appear to special-case HIP out entirely — confirm this holds for the
  actual K-cache type used at runtime (should be F16 per `llama-kv-cache-dsv4.h`, worth
  checking that file didn't get missed).
- `llama_mul_mat_hadamard` — used for RoPE application on q/kv in compressed attention.
  Location and backend coverage not yet checked. **Check this next** — it's called on nearly
  every attention op (`build_raw_attention`, `build_hca_attention`, `build_csa_lid_attention`),
  so if it's CPU-only or missing a HIP kernel, it would explain decode being consistently
  slow across every layer, every token.
- `ggml_top_k` / `ggml_set_rows` (`build_top_k_mask`, `build_lid_top_k`) — used to build the
  sparse top-k attention mask every decode step. Check HIP support for `GGML_OP_TOP_K` and
  `GGML_OP_SET_ROWS` specifically — these are less common ops that may not have been
  HIP-optimized industry-wide (they're newer additions to ggml). This is a strong lead: the
  indexer top-k selection runs on **every single decode token**, unlike prefill where it's
  amortized over a batch.

Sibling repos independently building Vulkan/HIP Lightning Indexer kernels for gfx1151
(see the sibling-repo survey doc) is circumstantial evidence that upstream's HIP coverage
for this op family is known-incomplete across projects, not just this branch.

**Caveat given the prefill-is-clean observation above**: the lightning indexer and its
top-k/set_rows mask-building path run on *every* forward pass, batched or not — if they were
simply CPU-bound, prefill should show CPU load too (arguably more, with more positions to
score). Either they're not the (sole) explanation, or something about which attention path
gets chosen differs between prefill and decode in a way that changes which ops actually run.
`deepseek4.cpp`'s attention dispatch picks between `build_raw_attention` (small sliding
window), `build_hca_attention`, and `build_csa_lid_attention` (the lightning-indexer path)
per layer based on `hparams.is_swa(il)` / compression ratio — check whether decode-time
single-token queries disproportionately hit a different one of these three than a batched
prefill does, before concluding the indexer itself is or isn't involved.

## Investigation steps for the next agent

1. **Get the exact op list with `GGML_SCHED_DEBUG=2`.** The btop/htop signal above already
   confirms *that* a real chunk of the decode graph runs on CPU across all threads — this
   step is about getting *which nodes*, not re-confirming the phenomenon. Run the server (or
   `llama-cli`/`llama-bench` with `-n 32` or so — 91GB model load is slow, plan for that) with
   `GGML_SCHED_DEBUG=2` (`ggml/src/ggml-backend.cpp:1793`,
   `ggml_backend_sched_print_assignments`) and diff the per-node backend assignment between a
   prefill graph and a decode (single-token, `n_tokens==1`, `n_past>0`) graph. Everything
   below is ranked by how well it explains "CPU-bound in decode, clean in prefill," not just
   "CPU-bound somewhere."
2. Also run `llama-bench` (or the server's own `/props` and timing logs) with
   `-ngl 999 -fa 1 -ctk q8_0 -ctv q8_0` and compare pp/tg at small context depth against the
   antirez quant, to get a clean number isolated from the user's full server flags
   (`--cache-disk`, `--ctx-checkpoints`, `--load-mode dio`, 1M ctx-size allocation) before
   assuming those flags are involved. Rule out or confirm the disk-cache/dio path's
   contribution to *prefill* separately from the decode CPU-fallback question — they may be
   two unrelated problems.
3. If `GGML_SCHED_DEBUG=2` shows specific fused ops (`DSV4_HC_*`, `LIGHTNING_INDEXER`,
   `TOP_K`, `SET_ROWS`, hadamard mul_mat) or specific tensor types (`IQ2_XXS`, `Q2_K`
   MoE experts) landing on CPU, grep each op's `ggml_backend_cuda_device_supports_op` /
   `ggml_backend_hip_*` path (HIP reuses ggml-cuda's `supports_op` via macro definitions —
   check whether any `GGML_CUDA_CC_IS_NVIDIA` or compute-capability gate silently
   disqualifies HIP for that specific op/type combo, the same pattern found in
   `lightning-indexer.cu`).
4. Check `ggml_backend_hip_*` / `ggml-cuda/mmq.cu` and `mmvq.cu` for `IQ2_XXS` and `Q2_K`
   `mul_mat`/`mul_mat_id` kernel coverage on gfx1151 specifically (RDNA3.5 uses a different
   code path than RDNA3/CDNA in some of ggml's AMD-specific kernel selection logic — check
   `ggml_cuda_info().devices[device].cc` handling for gfx1151's compute-capability constant
   and whether it's in any exclusion list). antirez's quant uses IQ2_XXS for most expert
   gate/up tensors and Q2_K for down — these are aggressive/exotic quant types; if HIP's
   `mul_mat_id` (the batched grouped-GEMM used for MoE) doesn't have a fast quantized-matmul
   path for them on this arch target, ggml may fall back to a slow dequant+CPU or
   dequant+generic-GPU path. This is a very plausible independent contributor to *both* slow
   prefill (large batched MoE matmuls) and slow decode.
5. Check `--load-mode dio` (direct I/O) interaction with weight buffer placement — confirm
   it doesn't force any tensor class to pinned CPU-only memory that then can't be scheduled
   on HIP for certain ops. Grep `load_mode` handling in `src/llama-model-loader.cpp` /
   `src/llama-mmap.cpp` (or wherever `--load-mode` was added — check blame/recent commits
   for `load_mode` if not obvious) alongside `--cache-disk`/disk-cache system (see
   `[[project_disk_cache_bug]]`-style memory notes from prior sessions — the disk-cache
   system in this branch is custom, not upstream, so it's a first-class suspect for anything
   unusual specific to *this* branch vs plain upstream llama.cpp).
6. Cross-check against a plain-upstream llama.cpp build (or the CachyLLama/qvac-fabric
   forks, which stay closer to upstream) with the same GGUF and same ROCm toolbox, if
   feasible, to separate "this is an upstream deepseek4/HIP gap" from "this is something
   specific to the custom disk-cache/MoE-streaming patches on this branch." The most recent
   two commits on this branch (`7bffc5d8c feat: pull 25294 - MoE disk streaming`,
   `09a88d8c5 feat: automatic disk cache system for text-only requests`) are exactly the kind
   of thing that could unexpectedly interact with MoE expert tensor placement — read those
   diffs and check whether they special-case MoE expert tensors' buffer type in a way that
   could force CPU residency for some of them. This subsystem has a history of subtle bugs
   (see `docs/superpowers/specs/2026-05-10-disk-kv-cache-design.md` and
   `2026-05-29-disk-cache-shutdown-prefix-design.md` for prior design context and known
   failure modes) so treat it as a first-class suspect, not an afterthought.

## Open questions to resolve, not assumptions to act on

- Is the CPU fallback (if confirmed) localized to a few ops per layer (cheap fix: HIP kernel
  for a missing op) or does it look like whole-tensor CPU residency (different fix: buffer
  placement / disk-cache interaction)?
- Is prefill slowness the same root cause as decode slowness, or two separate problems? The
  investigation must not assume they share a fix.
- Does the "hash layer" MoE router (first `hash_layer_count=3` layers) behave any differently
  performance-wise than the learned-router layers? Worth isolating in profiling since it's a
  genuinely unusual code path unique to this arch's early layers.

## Output of this investigation

A written finding (append to this doc or a follow-up doc) stating: which specific ops/paths
are CPU-bound during decode (with `GGML_SCHED_DEBUG` evidence), whether prefill has a
separate cause, and a ranked list of suspects for the fix-planning doc
(`2026-08-10-dsv4-strix-halo-perf-fix.md`) to act on. Do not jump to writing kernels before
this data exists — the fix doc currently has hypotheses, not a confirmed root cause.

## Findings (2026-08-10 follow-up session, on the actual Strix Halo box)

**Small-context `GGML_SCHED_DEBUG=2` test (`-n 4`, ~5-token prompt, `--load-mode dio`
required - see note below) showed only 1 CPU-backend node in both the prefill pass and the
first decode pass** - i.e. no reproduction of the original CPU-100%-all-threads symptom at
this tiny scale. This is expected, not a refutation, once you read the code below: the
mechanism only triggers once context depth crosses 1024 tokens, and this test never got
past ~5.

**Root cause candidate, now confirmed by reading the code (not just inferred from btop):**
- `ggml/src/ggml-cuda/ggml-cuda.cu:5211-5217` - `supports_op` for `GGML_OP_TOP_K` and
  `GGML_OP_ARGSORT`:
  ```cpp
  case GGML_OP_TOP_K:
  case GGML_OP_ARGSORT:
  #ifndef GGML_CUDA_USE_CUB
      return op->src[0]->ne[0] <= 1024;
  #else
      return true;
  #endif
  ```
- `ggml/src/ggml-cuda/common.cuh:110-112` - `GGML_CUDA_USE_CUB` is defined only
  `!defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA) && CUDART_VERSION >= 11070`. **HIP never
  gets this macro**, so on this branch's ROCm/HIP build, `supports_op` for TOP_K/ARGSORT is
  unconditionally `ne[0] <= 1024` - any row longer than 1024 elements is scheduled onto the
  **CPU backend entirely**, not a slow GPU kernel. This is a clean, mechanical explanation for
  "CPU 100% on all 32 threads specifically during decode": once a session's context depth
  (or whatever dimension feeds the lightning-indexer's `index_topk` selection - `ne[0]` here)
  passes 1024, every decode step's TOP_K call gets rerouted to `ggml-cpu`, which then
  multithreads it across all configured threads exactly as observed.
- `ggml/src/ggml-vulkan/ggml-vulkan.cpp:18270-18295` - Vulkan's `ARGSORT`/`TOP_K` have **no
  hardcoded 1024 cap**: `ARGSORT` uses a "large" pipeline gated on `vulkan_memory_model`
  (unbounded when available), and `TOP_K` is gated on a per-size pipeline table
  (`num_topk_pipelines`), not a fixed ceiling. This is a strong, mechanical explanation for
  why other Strix Halo users (who report ~250/~11 t/s and, per the sibling-repo survey doc,
  mostly run Vulkan not ROCm) don't hit this - their backend was never artificially capped at
  1024 in the first place.

**Independent corroboration from upstream GitHub** (found by the user searching issues, not
derived from this session's own reasoning):
- **PR #26592** ("CUDA: Enable CUB path on HIP via hipCUB") - directly targets this exact
  gap. Its own description: "For DeepSeek-V4-Flash models, any TOP_K operation exceeding 1024
  context length executed on the CPU... reportedly 6.4x token-generation loss." Same op,
  same model family, same root cause as derived above independently. **Status: open, blocked**
  - a reviewer (IMbackK) found segfaults in `test-backend-ops` on gfx908/gfx1100 with `k=1`
  after enabling the hipCUB path; not yet merged upstream as of 2026-08-10.
- **Issue #26746** ("ROCm gfx1151 RPC worker crashes in GGML_OP_TOP_K during DeepSeek V4
  prefill after 4096 tokens") - **same GPU family as this box** (Strix Halo/gfx1151), same
  model, a *crash* (not just a slow CPU fallback) in the TOP_K kernel path specifically during
  **prefill** past 4096 tokens. This is a second, separate TOP_K-related problem on the exact
  same hardware - open, unconfirmed, no fix yet. Relevant caveat for the "prefill is clean"
  observation in this doc: that observation was made with the user's real session at
  whatever context depth was active at the time, not verified across the 1024/4096-token
  boundaries - **do not assume prefill is unaffected until re-tested past those thresholds.**
- **Issue #26820** (RPC backend `[create_node] invalid data ptr` for DeepSeek-V4-Flash on a
  Windows CPU-only 9-node RPC cluster) - almost certainly **not relevant** here: different
  failure mode (RPC tensor-buffer deserialization), different hardware (CPU-only, distributed,
  not a single Strix Halo box), different transport (`rpc-server`, which the user isn't using).
  Noted so a future agent doesn't waste time chasing it as a lead.

**Load-mode note (new, unplanned finding):** the user could not get the model to load at all
*without* `--load-mode dio` - it stayed stuck at "Loading model..." with GPU pinned at 100%.
This is a separate, currently-unexplained problem (default/mmap load path apparently hangs for
this 91GB model on this box) - out of scope for the decode-CPU-fallback investigation, but
worth its own follow-up doc/issue before assuming `--load-mode dio` is merely a preference
rather than a workaround for a load-path bug.

**Revised next step (before writing any fix):** re-run the same `GGML_SCHED_DEBUG=2 -v`
methodology from the "Investigation steps" section above, but with a **prompt long enough to
push context depth past 1024 tokens** (and ideally past 4096, to also probe the #26746 crash
report), and check both the prefill pass(es) and decode passes for `TOP_K`/`ARGSORT` nodes
landing on the `CPU` backend. This directly confirms or refutes the mechanism above and
resolves whether prefill is actually clean or was just under-tested, before committing to
"port PR #26592" as the fix (see updated fix-plan doc).
