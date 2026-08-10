# Survey sibling Strix Halo projects for portable features

Status: plan for a future agent session. A shallow first pass has been done (summary below);
this doc scopes the deeper follow-up pass. No code has been ported yet.

Four sibling projects were cloned for comparison, ranked by popularity among other Strix Halo
users per the user:

1. `CachyLLama` — real llama.cpp source fork
2. `lucebox` — independent inference server, vendors llama.cpp/ggml
3. `qvac-fabric-llm.cpp` — real llama.cpp source fork
4. `strix-halo-llamacpp` — build/packaging tooling, not a source fork itself

## Shallow-pass findings (already done, this session)

**CachyLLama** (origin `fewtarius/CachyLLama`, tracks upstream master, merges regularly,
last merge `6ea215d17`). AMD APU/iGPU agentic-workload focus:
- Persistent SSD-backed KV cache, MoE expert SSD residency via `madvise`, `--cpu-moe` offload
  tuning — compare against this branch's own custom disk-cache/MoE-streaming system (see
  `docs/superpowers/specs/2026-05-10-disk-kv-cache-design.md`,
  `2026-05-29-disk-cache-shutdown-prefix-design.md`, and the two most recent commits on this
  branch) for overlapping or divergent design choices.
- Strix Halo/RDNA3.5 Vulkan flash-attention tuning — docs at `STRIX_HALO_NOTES.md`,
  `CEZANNE_NOTES.md`, `RDNA3_NOTES.md` in that repo.
- **DeepSeek-V4 Lightning Indexer Vulkan kernels** (`ggml-vulkan.cpp`,
  `lightning_indexer.comp`) — directly relevant to the perf-investigation doc's Branch A
  (this branch's HIP lightning-indexer lacks a fast path on AMD; CachyLLama apparently
  solved the analogous problem for Vulkan). Even if this branch stays on HIP rather than
  switching to Vulkan, the kernel's math/tiling approach is a useful reference for writing a
  rocWMMA HIP equivalent.
- DFlash/MTP-style draft support under `decoder_arch="laguna"` — worth understanding whether
  this is the same DFlash/DSpark family already in this branch or a third, incompatible
  design, before assuming it's portable.

**lucebox** (origin `Luce-Org/lucebox`, Python + custom C++/HIP, vendors llama.cpp as a
submodule plus a separate `Luce-Org/llama.cpp-dflash-ggml` fork for kernel patches). Heavy
DeepSeek-V4-Flash focus, most relevant single repo for this project:
- `server/docs/DS4.md` — architecture notes on 43-layer MoE/MLA and monolithic vs.
  layer-split vs. expert-parallel HIP backends — worth reading in full for any Strix
  Halo-specific MoE placement strategy this branch doesn't already do.
- `server/docs/HIP_PERF_PLAN.md` — ROCm MMQ tile overrides specifically for gfx1100/gfx1151/
  gfx1201 — **directly relevant** to the perf-investigation doc's quant-kernel-coverage
  hypothesis (IQ2_XXS/Q2_K mul_mat_id on gfx1151). Read this before writing any new HIP
  kernel — it may already document exactly which quant/op combos are slow on this hardware
  and why.
- Custom megakernel/pflash/spark/kvflash optimization directory, ROCm fp4/fpx GGML backends
  — survey what these do structurally; fp4/fpx backends in particular may be a cleaner
  alternative to the IQ2_XXS/Q2_K quant scheme antirez used, if the perf-fix doc's
  re-quantization branch turns out to be the right call.
- Recommended model pairing: `Lucebox/DeepSeek-V4-Flash-0731-ROCMFPX` +
  `Lucebox/DeepSeek-V4-Flash-DSpark-Drafter-GGUF` — the drafter is directly relevant to the
  MTP plan (Plan C there). The base model uses a different quant scheme (ROCMFPX) than
  antirez's IQ2_XXS/Q2_K — worth downloading and benchmarking as a comparison point even
  independent of any code porting, to separate "our HIP kernels are slow" from "this specific
  quant choice is slow on HIP" as explanations for the user's low numbers.

**qvac-fabric-llm.cpp** (origin `tetherto/qvac-fabric-llm.cpp`, based on llama.cpp b7248,
release-tagged v9341/v9840.x). Distinguishing feature: "TurboQuant" KV-cache quantization
(TBQ3/4_0, PQ3/4_0 formats, CPU+Vulkan kernels) and native multi-backend LoRA training.
README has Strix Halo pp/tg benchmark numbers but no DeepSeek-V4/MTP-specific code was found
in the shallow pass, and no ROCm-specific kernel work was evident. Lower priority for the
DeepSeek-V4 problem specifically; the KV-cache quantization work (TurboQuant) may be worth a
look independent of this project if the user cares about long-context memory footprint
(1M ctx-size is in the user's actual launch command), but that's a separate thread from the
DeepSeek-V4 perf problem.

**strix-halo-llamacpp** (packaging/toolbox repo, Dockerfiles + build/push/benchmark scripts,
assembles binaries from `Nathanw1014/llama.cpp`, tagged v0.4-v0.6.1). No local kernel source
to port from directly, but extremely detailed documentation:
- `README.md`, `BRANCHES.md`, `benchmarks/BENCHMARKS.md` — a curated list of per-concern
  "upstream-candidate" branches: Vulkan coopmat1 FA dequant-once, MUL_MAT_ID row-list/
  scale-epilogue fixes, tiled transposed CONCAT, silu*mul fusion, HIP tile-dequant KV decode,
  and **a dedicated DeepSeek-V4 Lightning Indexer branch** (independent confirmation, along
  with CachyLLama, that this is a known, real gap worth fixing — two projects solved it
  independently rather than one).
- The `Nathanw1014/llama.cpp` fork itself (not yet cloned locally) is worth pulling to see
  the actual diffs for these branches, since this repo only has orchestration, not source.

## Deeper-pass plan for the next agent

1. **Read `lucebox/server/docs/HIP_PERF_PLAN.md` and `DS4.md` in full first** — highest
   information density for this specific problem, written by people solving the exact same
   DeepSeek-V4-Flash-on-gfx1151 problem. Cross-reference every claim there against the
   perf-investigation doc's hypotheses before assuming anything needs independent discovery.
2. **Clone and diff `Nathanw1014/llama.cpp`'s DeepSeek-V4 Lightning Indexer branch** (via
   `strix-halo-llamacpp`'s `BRANCHES.md` pointer) against this branch's
   `ggml/src/ggml-cuda/lightning-indexer.cu` and `ggml/src/ggml-vulkan/` — determine whether
   it's a HIP fix (portable as-is or nearly) or a Vulkan-backend fix (would need translation
   to HIP, or would argue for adding Vulkan as a second backend option on this branch, which
   is a much bigger decision — flag this as a question for the user, not a unilateral choice).
3. **Diff CachyLLama's `lightning_indexer.comp` (Vulkan) and Lucebox's HIP MMQ tile overrides
   against this branch's equivalent files** — for each, produce a short compatibility note:
   same ggml op signature? same tensor shapes/types? any upstream ggml API drift between
   when they wrote it and this branch's current ggml version (this branch is close to current
   upstream llama.cpp, sibling forks may be pinned to older bases — qvac-fabric-llm.cpp
   explicitly is, at b7248 vs. this branch's much more recent HEAD)?
4. **For each portable feature identified, write it up as its own scoped mini-plan** (don't
   bundle unrelated ports into one PR/commit) with: what it fixes, which file(s) it touches,
   estimated risk (kernel correctness risk is high for anything touching numerics — flag
   clearly per AGENTS.md's contributor-understanding requirement, since none of this would be
   upstream-contributable without the porting agent fully understanding the borrowed code,
   not just copy-pasting it).
5. Explicitly separate "features relevant to the DeepSeek-V4-Flash perf problem" (lightning
   indexer kernels, MoE quant-kernel tuning) from "generally interesting but unrelated"
   (TurboQuant KV-cache quantization, LoRA training) — only the former should feed back into
   the perf-fix doc; the latter can be a separate low-priority backlog note.

## Non-goals

- Do not port entire subsystems (e.g. lucebox's whole HIP backend, or CachyLLama's SSD
  expert-residency system) wholesale — this branch already has its own disk-cache/MoE-
  streaming system; the goal is targeted kernel/technique borrowing, not a rewrite.
- Do not treat any of these repos' numbers as validated without independently reproducing
  them on this exact box — they may be running different ROCm versions, ubatch sizes, or
  quant schemes than the user's setup.
