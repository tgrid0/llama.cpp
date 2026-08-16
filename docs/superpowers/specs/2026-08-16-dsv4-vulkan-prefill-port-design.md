# DeepSeek-V4 sparse-attention prefill: Vulkan port

Status: approved design, ready for implementation planning. Branch `b10331_zgfs`
(builds on top of the `dsv4-sparse-attn-prefill` worktree/branch).

## Problem

The prior effort (`docs/superpowers/specs/2026-08-15-dsv4-sparse-attn-prefill-design.md`,
plan `docs/superpowers/plans/2026-08-15-dsv4-sparse-attn-prefill.md`) ported a single sparse
flash-attention kernel to HIP/CUDA, based on research against `Nathanw1014/llama.cpp`'s
`vulkan-dsv4-lightning-indexer` branch. On real Strix Halo/ROCm hardware this yielded only a
small prefill improvement (125 t/s vs. 110 t/s baseline at depth 0) - nowhere near the ~290 t/s
the actual reference implementation achieves.

Two things were discovered testing this:

1. `vulkan-dsv4-lightning-indexer` was the wrong branch to research from. The branch
   `strix-halo-llamacpp` actually tracks is `Nathanw1014/llama.cpp`'s **`strix-halo-vulkan-beta3`**
   - which shares a common ancestor with `vulkan-dsv4-lightning-indexer` (diverging only 2
   commits after `ggml_flash_attn_ext_add_top_k` was introduced) but continues **154 commits**
   further. The ggml-core API from the prior plan matches beta3 exactly - same function name,
   same doc-comment semantics, verified byte-for-byte - so that work is not wasted, just
   incomplete: it only ported the smallest of beta3's three DSV4 kernel families.
2. Our fork's Vulkan backend has zero DSV4-specific shaders at all (`ggml-vulkan/vulkan-shaders/`
   has no `lightning_indexer*.comp`, `flash_attn_top_k*.comp`, or `dsv4_hc_*.comp`), so on
   Vulkan the lightning indexer (`GGML_OP_LIGHTNING_INDEXER`, upstream op, already merged into
   our fork with CPU/CUDA/Metal/SYCL implementations) and the DSV4 HC pre/comb/post fused ops
   fall back to CPU/unfused every layer, every token. This is a second, likely larger, prefill
   bottleneck than the flash-attention masking cost the prior plan addressed, and it is backend-
   agnostic infrastructure our fork already carries - only the Vulkan kernels are missing.

Separately, running our fork's Vulkan backend at large context (`--ctx 1048576`, `--ctx 65536`)
crashes: `Failed to allocate pinned memory (ErrorOutOfDeviceMemory)` and `ErrorDeviceLost`.
Confirmed via code inspection this is **not** caused by the prior plan's work - Vulkan's flash-
attention path never reads `src[5]`/`op_params[4]` (the hint the prior plan added), and the
crash reproduces on paths the prior plan never touched. Leading hypothesis: running HC pre/
comb/post and the indexer unfused (today's Vulkan fallback) allocates much larger intermediate
buffers at huge `n_kv` than beta3's fused kernels do - this plan's tasks test that hypothesis
before treating it as a separate investigation.

## What needs porting

Diffing beta3's shader inventory against our fork's, and grouping its DSV4-relevant commits
since the shared ancestor, splits fairly cleanly into two independent clusters:

- **Prefill cluster** (in scope for this plan, ~13 commits): lightning-indexer Vulkan kernels
  (`lightning_indexer.comp`, `lightning_indexer_cm.comp`), sparse top-k flash-attention
  (`flash_attn_top_k.comp`, `flash_attn_top_k_cm.comp`), DSV4 HC fusion (`dsv4_hc_pre/comb/
  post.comp`), and the tiling/splitting/multi-sequence-hardening work layered on top of those
  three kernel families.
- **Decode small-batch cluster** (out of scope, deferred to a later plan, ~9 commits):
  `lightning_indexer_decode_cm.comp` and quantized-K/V gather-to-compact optimizations for
  single-token decode batches. Decode is already close to parity with beta3 (22 vs. 27 t/s) so
  this is lower priority than closing the much larger prefill gap.

## Reuse from the prior plan

The `dsv4-sparse-attn-prefill` worktree already has, committed and reviewed:

- `ggml_flash_attn_ext_add_top_k(a, top_k, n_kv_raw)` in `ggml.h`/`ggml.c` - verified identical
  to beta3's own version of this function, including doc-comment wording.
- `build_attn_mha`/`build_csa_lid_attention` wiring in `src/llama-graph.*` and
  `src/models/deepseek4.cpp` that produces the `src[5]`/`op_params[4]` hint.
- `test-backend-ops` coverage in `tests/test-backend-ops.cpp` for the DSV4 MQA/top-k shape.

This plan branches from that worktree's tip, not from `b10331_zgfs` directly, so the new
Vulkan kernels consume the exact same hint the HIP kernel already does. None of the above is
touched by this plan except where a task needs to read a field the prior plan didn't wire up
(callouts below).

## Components

1. **Lightning indexer Vulkan kernel.** Port `lightning_indexer.comp` and
   `lightning_indexer_cm.comp` (coopmat2 variant) plus the `ggml-vulkan.cpp` dispatch/pipeline-
   selection wiring for `GGML_OP_LIGHTNING_INDEXER`. This is an existing upstream op our fork
   already has CPU/CUDA/Metal/SYCL kernels for; only the Vulkan kernel and its dispatch
   registration are new. Likely the single largest prefill win, since indexer scoring runs on
   CPU today for every layer of every token on Vulkan.
2. **Sparse top-k flash-attention kernel.** Port `flash_attn_top_k.comp` and
   `flash_attn_top_k_cm.comp`, wired to read the `src[5]`/`op_params[4]` hint the prior plan's
   Task 1/2 already produce. Structurally the Vulkan sibling of the HIP kernel from the prior
   plan, but this is the proven-fast reference implementation (beta3's own numbers: 290+ t/s).
3. **DSV4 HC pre/comb/post fusion kernels.** Port `dsv4_hc_pre.comp`, `dsv4_hc_comb.comp`,
   `dsv4_hc_post.comp` and their dispatch wiring, so `llama_context::resolve_fused_ops` stops
   disabling these three fusions on Vulkan (`src/llama-context.cpp:587-591`).
4. **Tiling, splitting, multi-sequence hardening.** The remaining prefill-cluster commits that
   refine components 1-3 for correctness/performance at scale: per-key-block mask caching,
   probability-fragment reuse, split sparse-prefill attention, query tiling coexisting with
   multiple sequences, and the resource-limit-recording commit. Depends on 1-3 landing first;
   ported as a single task since these commits modify the same few files repeatedly and don't
   decompose cleanly into independent units.
5. **Device-lost/OOM crash investigation.** Depends on 1-4 being in place. First tests whether
   the fused kernels resolve the crash (the buffer-size hypothesis above) at the context sizes
   that triggered it (`--ctx 65536`, `--ctx 1048576`). If the hypothesis doesn't hold, falls
   back to a from-scratch investigation of Vulkan pinned-memory/buffer-size limits for this
   workload.

Each task ports beta3's **current file state** at the tip of the prefill cluster, adapted to
our fork's structure - not a per-commit cherry-pick/replay. Our fork diverged from beta3's
ancestor with its own independent DSV4 implementation and unrelated work (disk-cache, DSpark
drafter), so replaying 13 commits of conflict resolution against unrelated history would be
slower and riskier than a direct port of the end state.

## Testing strategy

This round has real local tooling the prior (HIP) round lacked:

- **glslc.exe** (Windows Vulkan SDK, already installed) compiles each ported `.comp` shader to
  SPIR-V standalone - real syntax/type validation per task.
- **WSL build** - after installing `glslc`, `libvulkan-dev`, `spirv-headers`, `glslang-tools`,
  `vulkan-tools`, `ninja-build` (done this session), `cmake -DGGML_VULKAN=ON` configures and
  `make test-backend-ops` builds and links cleanly, including all existing Vulkan shaders and
  dispatch code. Every task's new shader and new `ggml-vulkan.cpp` code gets real compile+link
  validation, not a hand-trace.
- **No local execution.** WSL's only Vulkan device is Mesa's software `llvmpipe`, and
  `ggml-vulkan.cpp` deliberately filters to `eDiscreteGpu`/`eIntegratedGpu` device types
  (`ggml-vulkan.cpp:7444`), excluding CPU-type devices - confirmed via a real build (`ggml_vulkan:
  No devices found`). Functional correctness (`test-backend-ops -b Vulkan0 -o
  LIGHTNING_INDEXER`/`FLASH_ATTN_EXT`/`DSV4_HC_*`) and performance numbers still require a
  handoff to the user's Strix Halo box, same pattern as the prior plan's Task 5/6.

Each task's reviewer therefore gets a real compile+link result to check per task (a
meaningfully stronger gate than the fully-blind HIP round), with functional/perf verification
concentrated into one final hardware-validation task instead of spread across every task.

## Non-goals

- Decode small-batch quantized-gather cluster (deferred to a later plan).
- Any change to the HIP/CUDA kernel from the prior plan - it stays as committed.
- Any change to the ggml-core hint API - reused as-is.
- Matching beta3's exact perf numbers is a goal, not a hard requirement; the crash fix and the
  three kernel families landing correctly (verified functionally on real hardware) is the bar
  for calling this plan done.
