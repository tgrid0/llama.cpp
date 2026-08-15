# DeepSeek-V4 sparse-attention prefill: design

Status: approved design, ready for implementation planning. Branch `b10331_zgfs`.

## Problem

Prefill throughput on DeepSeek-V4-Flash degrades with context depth on this box (~110 t/s at
start, ~30 t/s at 128k, ~12 t/s at 470k), while `strix-halo-llamacpp` (Vulkan, not ROCm)
reports prefill flattening to ~110 t/s past 64k instead of continuing to decay. The cause is
architectural, not a missing/slow kernel in the usual sense: our `build_csa_lid_attention`
(`src/models/deepseek4.cpp`) already runs the DeepSeek Sparse Attention lightning indexer to
pick the top-k most relevant compressed-KV positions every layer (`build_lid_top_k`), but then
only uses that selection to build a dense `-inf`/`0` mask (`build_top_k_mask`) fed into a normal
flash-attention call over the *entire* compressed KV cache. This is mathematically correct
(masked-out positions contribute nothing to the softmax) but computationally is still
O(context length) per query - masking suppresses contributions after they're computed, it
doesn't skip the compute. That's why prefill keeps degrading instead of flattening.

## What the source project actually did

`strix-halo-llamacpp` points at `Nathanw1014/llama.cpp`'s `vulkan-dsv4-lightning-indexer`
branch (credited to Gaetan Puleo), read directly for this design (shallow-cloned into the
session scratchpad, not vendored into this repo). It adds a **true sparse** flash-attention
path: a fused kernel that only computes attention over the `n_kv_raw` dense/recent keys plus
the `n_top_k` gathered compressed-KV indices per query row - O(n_kv_raw + n_top_k) cost, flat
regardless of total context length once engaged.

The ggml-core plumbing for this is a small, purely additive change:

- `ggml_flash_attn_ext_add_top_k(tensor, top_k, n_kv_raw)` (new, `ggml.h`/`ggml.c`) sets the
  previously-unused `src[5]` slot on a `GGML_OP_FLASH_ATTN_EXT` node plus an op_param for
  `n_kv_raw`. Shape mirrors our existing `ggml_flash_attn_ext_add_sinks` (`src[4]`) almost
  exactly - our `GGML_MAX_SRC` is already 10, so this is a clean addition, not a struct resize.
- `build_attn_mha` (`src/llama-graph.h`/`.cpp`) gains two new optional trailing parameters
  (`top_k = nullptr`, `n_kv_raw = 0`); when set, it calls the hint setter right after building
  the flash-attn node.
- `build_csa_lid_attention` passes its already-computed `top_k` tensor and
  `n_kv_raw = raw_k->ne[2]` into `build_attn_mha`. The existing `build_top_k_mask` dense-mask
  path is untouched - it remains the correctness fallback for every backend that doesn't
  understand the hint.
- Backends that don't implement the hint (CPU, CUDA, SYCL today, and HIP) simply never look at
  `src[5]` and compute the identical dense-masked result. Only Vulkan was taught to check
  `src[5]` and, if a narrow shape gate matches (head_dim 512, 64-head MQA, F16 K/V, batched
  queries `ne[1] >= 64` - i.e. **prefill only**, decode's single-token queries never qualify),
  dispatch a fused shader (`flash_attn_top_k.comp`) instead of the general dense FA path.

This is exactly the kind of "reuse existing infrastructure, minimal invasive surface" change
AGENTS.md asks for: no new op type, no change to any other backend's behavior, fails closed to
today's (correct, just slow) dense path whenever the gate doesn't match.

## What needs porting/adapting for ROCm

The ggml-core plumbing (items 1-3 above) ports essentially verbatim - it's backend-agnostic.
The part that doesn't exist anywhere yet is a HIP/CUDA equivalent of the Vulkan fused kernel;
Vulkan is the only backend with an actual fast-path implementation.

One difference from the source's assumptions: the Vulkan kernel requires the CSA/LID K cache
to be F16. Our DSV4 KV cache code constructs the CSA/LID compressed caches with the same
`type_k`/`type_v` passed for the main cache (`src/llama-kv-cache-dsv4.cpp:1240-1257`), and this
setup's actual launch command uses `--cache-type-k q8_0 --cache-type-v q8_0`. Ported as-is, the
fast path would silently never engage for this box's real configuration. The kernel gets a
Q8_0 dequant-on-load path for K in addition to F16, so it engages under the memory-saving
config actually in use, rather than requiring a cache-type change.

## Design

1. **`ggml/include/ggml.h` + `ggml/src/ggml.c`** - add `ggml_flash_attn_ext_add_top_k`,
   mirroring `ggml_flash_attn_ext_add_sinks`'s structure (asserts, `src[5]` assignment,
   `ggml_set_op_params_i32` for `n_kv_raw`).
2. **`src/llama-graph.h` + `src/llama-graph.cpp`** - extend `build_attn_mha`'s signature with
   `ggml_tensor * top_k = nullptr, int64_t n_kv_raw = 0`; call
   `ggml_flash_attn_ext_add_top_k(cur, top_k, n_kv_raw)` when `top_k` is non-null, right after
   the existing `ggml_flash_attn_ext_add_sinks` call. No other call site changes.
3. **`src/models/deepseek4.cpp`** - `build_csa_lid_attention` passes `top_k, raw_k->ne[2]` as
   the new trailing args to its `build_attn_mha` call. `build_top_k_mask` and the rest of the
   function are unchanged.
4. **New kernel file `ggml/src/ggml-cuda/fattn-top-k.cu`** (+ header), compiled for both CUDA
   and HIP via the existing shared ggml-cuda source tree (this box only tests HIP, but nothing
   here is HIP-specific in principle - same pattern as other fattn kernels in this tree).
   Literal port of `flash_attn_top_k.comp`'s scalar online-softmax loop: per-query-row
   iteration over `n_kv_raw` dense keys plus `n_top_k` gathered keys (via the `src[5]` index
   tensor), warp-shuffle reduction in place of GLSL `subgroupAdd`, running max/sum exactly as
   the source shader does. Adds a Q8_0 dequant-on-load branch for K (decode one block of K
   into registers/shared memory per tile, same shape as existing HIP tile-dequant FA work)
   alongside the F16 path the source has.
5. **Dispatch wiring in `ggml/src/ggml-cuda/fattn.cu`** (or wherever the existing dispatcher
   picks a flash-attn kernel variant) - check `dst->src[5]` plus the same shape/dtype gate the
   Vulkan backend uses (adding the Q8_0 K allowance), and try the new kernel first; on any gate
   miss, fall through unchanged to the existing dense FA dispatch. This preserves current
   behavior for every other model/shape/backend combination.

## Data flow

- `build_lid_top_k` (unchanged) computes the top_k `I32` tensor
  `[n_top_k, n_tokens, 1, n_stream]` via `ggml_top_k`.
- `build_csa_lid_attention` still builds the dense `-inf`/`0` mask fallback exactly as today,
  and additionally passes `top_k` + `n_kv_raw` as a hint via the extended `build_attn_mha`.
- At dispatch time, HIP/CUDA checks the hint + shape gate. Match: run the sparse kernel,
  computing online softmax over only `n_kv_raw + n_top_k` keys per query row. No match (short
  context, decode's `n_tokens == 1` queries, non-DSV4 shapes, other archs): fall through to
  today's dense masked FA, byte-for-byte the same code path as before this change.

## Testing

1. **Correctness gate**: extend `test-backend-ops`'s `FLASH_ATTN_EXT` coverage with a
   top-k-hinted case at the DSV4 CSA shape, both F16 and Q8_0 K, compared against the existing
   dense CPU reference. Must pass before this touches a real model.
2. **Real-model sanity check** (not a blocking test, but expected before relying on this in a
   live session): antirez DSV4 GGUF, long-context prompt, fast path on vs. off (env- or
   build-flag-gated for the A/B), diff generated text/logits for identical output, then compare
   prefill t/s at a few context depths (matching the fix-plan doc's own validation pattern for
   the decode issue).
3. If the change touches any code path shared with non-DSV4 flash-attention (it shouldn't,
   given the narrow shape gate, but confirm), re-run relevant existing `test-backend-ops`
   FLASH_ATTN_EXT cases for other shapes to check for regressions.

## Non-goals

- Decode is out of scope. The shape gate excludes single-token queries by design (`ne[1] >= 64`
  matches the source); decode's separate slowness is the HIP TOP_K/ARGSORT hipCUB gap already
  covered by `2026-08-10-dsv4-strix-halo-perf-fix.md` (Branch A), not this feature.
- No change to the lightning-indexer scoring itself (`build_lid_top_k` is untouched) - only to
  what the FA kernel does with its output.
- Not attempting a rocWMMA/tiled kernel. This hardware is memory-bandwidth-bound per the
  sibling-repo docs (`docs/superpowers/specs/2026-08-10-strix-halo-sibling-repo-survey.md`), so
  a literal scalar port of the source algorithm is expected to capture most of the available
  win at much lower implementation/review risk; a fancier kernel is not ruled in unless the
  scalar port's measured performance says otherwise.
- Not vendoring or copying source files from `Nathanw1014/llama.cpp` wholesale - this design
  reimplements the mechanism understood from reading that branch, adapted to this branch's own
  ggml-cuda/HIP structure and this box's actual runtime config (Q8_0 KV cache).
