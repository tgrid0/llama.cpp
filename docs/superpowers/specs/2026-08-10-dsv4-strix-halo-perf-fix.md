# DeepSeek-V4-Flash on Strix Halo: fix plan

Status: contingent plan for a future agent session. **Depends on the findings from
`2026-08-10-dsv4-strix-halo-perf-investigation.md`** — do not start implementing here until
that investigation has produced hard `GGML_SCHED_DEBUG=2` evidence of what's actually
CPU-bound. This doc lays out the remediation path for each plausible root cause found during
initial code reading, so the next agent doesn't have to re-derive them, but the actual choice
of which branch to pursue is gated on that data.

Target: close the gap between observed (~80 t/s prefill / ~7 t/s decode) and reported-elsewhere
(~250 t/s prefill / ~11 t/s decode) numbers for the antirez DeepSeek-V4-Flash-0731 Q2/Q4 quant
on this Strix Halo box, ROCm 7.14, branch `b10331_zgfs`.

## Branch A: a specific ggml op has no (fast) HIP kernel

If `GGML_SCHED_DEBUG=2` shows a specific op consistently on the CPU backend during decode:

- **If it's `GGML_OP_LIGHTNING_INDEXER`**: it already has a HIP-compiled generic vector
  kernel (`lightning_indexer_kernel_vec` in `ggml/src/ggml-cuda/lightning-indexer.cu:244`),
  so it should NOT be landing on CPU — if it is, that's a `supports_op` bug to fix directly
  (check `ggml_cuda_lightning_indexer_supported` at line 536 for an unintended HIP
  exclusion). If it's on GPU but just slow (not CPU), the fix is a rocWMMA fast-path kernel
  analogous to the existing NVIDIA WMMA one (lines 6-237) — this is real kernel-writing work;
  check whether CachyLLama or strix-halo-llamacpp (see sibling-repo survey doc) already wrote
  a Vulkan or HIP Lightning Indexer kernel for gfx1151 that can be ported/adapted instead of
  starting from scratch. Porting a working implementation is much lower risk than writing a
  new rocWMMA kernel from the NVIDIA WMMA reference.
- **If it's `GGML_OP_TOP_K` or `GGML_OP_SET_ROWS`** (used every decode step in
  `build_top_k_mask`/`build_lid_top_k`): check whether these ops have HIP kernels at all in
  `ggml/src/ggml-cuda/`. If missing, this is the highest-value narrow fix — these are small,
  well-defined ops (not fused megakernels), so a CPU->HIP port is tractable and low-risk. Grep
  `ggml/src/ggml-cuda/*.cu` for `top-k` / `set-rows` files before assuming a from-scratch
  write is needed — llama.cpp upstream may already carry a HIP kernel for these under a
  slightly different name.
- **If it's `mul_mat`/`mul_mat_id` on `IQ2_XXS`/`Q2_K` expert tensors**: check ggml's AMD
  kernel-selection logic (`mmq.cu`, `mmvq.cu`, and the `GGML_CUDA_CC_*` / architecture
  dispatch tables) for gfx1151 (RDNA3.5) coverage of these specific quant types. If HIP
  genuinely lacks a fast quantized-matmul kernel for `IQ2_XXS` on this architecture target,
  the practical fix is **not** writing a new kernel (out of scope, high risk, upstream
  maintainers are unlikely to accept a narrow gfx1151-only kernel per AGENTS.md's
  "simpler change that does 90% of the job" guidance) but instead **re-quantizing the model**
  with quant types that do have fast HIP coverage on this target (e.g. Q4_K/Q4_0 instead of
  IQ2_XXS for the expert gate/up tensors) — check what quant types the other Strix Halo users
  who report the good numbers are actually using (their inference servers differ, but if they
  all avoid IQ2_XXS/Q2_K for MoE experts, that's a strong, cheap signal). This is worth
  checking BEFORE any kernel work: re-quantizing is hours, not days.
- **If it's `llama_mul_mat_hadamard`**: find its implementation (not yet located in this
  session — search `ggml/src/ggml-cuda/` and `ggml/src/ggml-cpu/` for `hadamard`) and check
  HIP coverage the same way. It's called on nearly every attention op, so if this is CPU-only
  the fix is high-value but also higher-risk (correctness-sensitive, touches every layer's
  attention output).

## Branch B: whole-tensor CPU residency (buffer placement, not op support)

If profiling shows large blocks of tensors (not just specific ops) resident in CPU memory
during decode — e.g. some experts or some layer's weights never got a HIP buffer — the
suspects are the two most recent commits on this branch:

- `7bffc5d8c feat: pull 25294 - MoE disk streaming`
- `09a88d8c5 feat: automatic disk cache system for text-only requests`

Read both diffs in full and check whether they introduce any per-tensor buffer-type decision
that could special-case MoE expert tensors (e.g. streaming logic that assumes CPU-resident
staging buffers, applied more broadly than intended). Also check `--load-mode dio` handling
for the same reason — `dio` (direct I/O) loading may interact with how tensor buffers get
allocated/pinned in a way that differs from the default mmap-based load path. Compare
behavior with `--load-mode mmap` (or whatever the default/other mode is called — check
`common/arg.cpp` for the full enum) as a quick isolation test: if performance recovers with a
different `--load-mode`, that's a strong, cheap confirmation without needing to read the full
diff first.

If confirmed, the fix is scoped entirely within this branch's custom disk-cache/streaming
code (not upstream llama.cpp). This subsystem has a history of subtle bugs (a sleeping-state
crash after wake, prefix-entry accumulation on long sessions, recurrent/hybrid-model unload
hangs) — check prior design docs in this same `docs/superpowers/specs/` directory
(`2026-05-10-disk-kv-cache-design.md`, `2026-05-29-disk-cache-shutdown-prefix-design.md`) and
`git log` on `src/*disk-cache*`/`src/*dio*` files for related past fixes in the same
subsystem before re-diagnosing from scratch.

## Branch C: prefill and decode have different root causes

Don't assume a single fix addresses both numbers. If the investigation doc's step 2
(isolated `llama-bench` run without the user's full server flags) shows prefill recovering
close to expected while decode stays slow (or vice versa), split this into two independent
fixes and re-scope each against Branches A/B above separately — e.g. prefill slowness with
batched sequences may be dominated by MoE `mul_mat_id` quant-kernel throughput (Branch A,
quant-type angle) while decode slowness may be dominated by the per-token lightning-indexer
top-k path (Branch A, small-op angle), with Branch B being an unrelated multiplier on top of
either.

## Validation plan (once a fix is implemented)

1. `llama-bench` before/after on the exact antirez GGUF, `-ngl 999 -fa 1 -ctk q8_0 -ctv q8_0`,
   at a few context depths (0, and something representative of real usage, e.g. 8k-32k) to
   confirm the fix isn't a zero-context-only win.
2. Full server launch with the user's actual flags (disk cache, ctx-checkpoints, dio load
   mode, 1M ctx-size) to confirm the fix holds under real conditions, not just the clean
   benchmark harness.
3. Re-run `GGML_SCHED_DEBUG=2` after the fix to confirm the previously-CPU op/tensor now
   shows GPU placement, not just that the t/s number went up (numbers can be noisy; confirm
   the actual mechanism changed).
4. If the fix touches shared ggml code (not deepseek4-specific), run the existing test suite
   for affected backends to check for regressions on other models/architectures using the
   same op.

## Non-goals for this fix plan

- MTP/speculative decoding is out of scope here — see
  `2026-08-10-dsv4-mtp-plan.md`. Do not conflate "decode is slow" (this doc) with "decode
  could be 2-3x faster with a draft model" (the MTP doc) — they're additive, not the same
  problem.
- Do not attempt to upstream a gfx1151-only kernel hack without reading AGENTS.md's
  maintainer-burden guidance first if this work is ever intended for a PR — this branch is a
  private fork, so local hacks are fine here, but say so explicitly in any commit if the code
  is not something you'd defend to an upstream reviewer.
