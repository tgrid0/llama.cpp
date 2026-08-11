# Bringing DSpark speculative decoding for DeepSeek-V4-Flash to this branch

Status: **DONE, validated on hardware.** Plan B (repack antirez's ds4-quantized DSpark support
GGUF) shipped as `gguf-py/gguf/scripts/dsv4_dspark_repack.py`. Confirmed 2026-08-11 on Strix
Halo (gfx1151): loads via `--spec-draft-model <repacked>.gguf --spec-type draft-dspark`,
DeepSeek-V4-Flash-0731 decode speed went from 11 t/s to 20-22 t/s. Plan A (raw HF conversion)
was never attempted - Plan B worked first try (after one metadata bug fix, see below), no
reason to chase the larger download. One bug found and fixed during validation: several DSV4
hparams are per-layer arrays sized to the TARGET's block_count (43) -
`swiglu_clamp_exp`/`swiglu_clamp_shexp`, and generically `feed_forward_length`/
`attention.head_count`/`attention.head_count_kv` - copying them verbatim into the drafter's
GGUF fails to load ("wrong array length") because llama.cpp checks array length against the
*loading* model's own layer count (the drafter's 3 stages, not the target's 43). Fixed by
detecting any array whose length matches the target's block_count and collapsing it to the
drafter's stage count (verified all-equal first, per DeepSeek's own convention of repeating a
single value uniformly per layer).

Written by comparing this branch's own `src/models/dflash.cpp` against the sibling project
`W:\projects\llm\ds4` (dwarfstar4, "ds4" — a from-scratch C inference engine for DeepSeek-V4,
not based on llama.cpp, written by the same author who publishes the `antirez/deepseek-v4-gguf`
quants this branch already uses). ds4 has its own working DSpark support; this doc figures out
what it takes to get the same speedup here.

## Headline finding

**No llama.cpp C++ changes are needed.** `src/models/dflash.cpp` already implements a
`LLM_ARCH_DFLASH` ("dflash") architecture with a DSpark-specific graph path
(`llama_model_dflash::graph_dsv4`, active when GGUF key `dflash.hyper_connection.count` > 0)
that includes the Markov head + confidence head DSpark needs
(`build_dspark_markov_head`, `dflash.cpp:221-306`), selectable today via
`--spec-type draft-dspark` + `--spec-draft-model <path>` (`common/arg.cpp:4127-4144`,
`common/speculative.cpp:36-37, 941-945`). `conversion/deepseek.py::DeepseekV4DSparkModel`
(lines 938-1037) already has a converter for it. This is a **data/conversion problem, not a
missing-feature problem** — same shape as the native-MTP finding in
`2026-08-10-dsv4-mtp-plan.md`, and it reuses the same "repack, don't reimplement" playbook.

## What DSpark is (per ds4's own docs, `ds4/README.md:196-249`)

DSpark is DeepSeek's own auxiliary draft model for DeepSeek-V4-Flash: it reads hidden states
from `target_layer_ids` layers of the main model, then autoregressively proposes up to
`block_size` (default 5) future tokens through a small stack of full DSV4-style
(hyper-connection + MLA + MoE) transformer stages, each token conditioned on the previous
one via a low-rank "Markov head" bias, with a sigmoid "confidence head" pruning low-value
suffixes before target verification. It replaces the plain single-token MTP/NextN path when
enabled. ds4 exposes it as `--dspark --mtp <support.gguf>`.

## Side-by-side: tensor & metadata contract

Confirmed via direct source reads on both sides (llama.cpp: `src/llama-arch.cpp`,
`src/models/dflash.cpp`, `conversion/deepseek.py`; ds4: `gguf-tools/deepseek4-quantize.c`,
`ds4.c`). Per-layer/per-stage tensor **short names already match exactly** — both sides
independently arrived at `attn_q_a`, `attn_q_b`, `attn_kv`, `attn_kv_a_norm`,
`attn_output_a/b`, `attn_sinks`, `attn_norm`, `hc_attn_fn/base/scale`, `hc_ffn_fn/base/scale`,
`ffn_gate_inp`, `exp_probs_b`, `ffn_norm`, `ffn_{gate,up,down}_exps`,
`ffn_{gate,up,down}_shexp`. The only difference is the **prefix**:

| Concept | ds4 GGUF output | llama.cpp `dflash` expects |
|---|---|---|
| Per-stage/per-layer tensors | `mtp.{N}.<short-name>.weight` (0-based stage) | `blk.{N}.<short-name>.weight` |
| Stage-0-only "target feature" tensors | `mtp.0.main_proj.weight`, `mtp.0.main_norm.weight` | root `fc.weight`, `enc.output_norm.weight` |
| Final-stage-only head tensors | `mtp.{last}.norm.weight`, `mtp.{last}.hc_head_{fn,base,scale}.weight`, `mtp.{last}.markov_head.markov_w1/w2.weight`, `mtp.{last}.confidence_head.proj.weight` | root `output_norm.weight`, `output_hc_{fn,base,scale}.weight`, `markov_w1/w2.weight`, `conf_proj.weight` |
| `general.architecture` | `"deepseek4-dspark"` | `"dflash"` |
| Block-size key | `dspark.block_size` (UINT32) | `dflash.block_size` (raw literal read, `dflash.cpp:233`) |
| Target layer ids | `dspark.target_layer_ids` (ARRAY UINT32) | `dflash.target_layers` (`LLM_KV_TARGET_LAYERS`) |
| Markov rank | `dspark.markov_rank` (UINT32) | **not read from metadata** — inferred from `markov_w1.weight`'s shape (`dflash.cpp:86-90`), so this key can simply be dropped |
| Noise token id | `dspark.noise_token_id` (UINT32) | not seen consumed in `dflash.cpp`; open question, see below |
| Stage/layer count | `dspark.stage_count` / `dspark.n_layers` | standard `dflash.block_count` (not present in ds4's file at all — see gap below) |

This mapping is exhaustive enough to script mechanically — this is the same kind of rename
table the native-MTP plan already worked out for antirez's other file, and this one is a
cleaner 1:1 match (no `e_proj`/`h_proj` concat step needed).

**Confirms the "not just Lucebox" note in the MTP plan doc was slightly stale**: DSpark here is
DeepSeek's own official architecture (same one ds4 implements from scratch and antirez
distributes pre-quantized), not merely "what Lucebox distributes" — that framing undersold it.

## The real gap: ds4's DSpark support GGUF carries almost no hparams

ds4's `--dspark-support` output file has **exactly 9 GGUF KV pairs total**
(`deepseek4-quantize.c:2420-2434, 2562-2620`): the two `general.*` identity keys, alignment,
and the six `dspark.*` keys in the table above. It does **not** carry `hyper_connection.count`,
`attention.q_lora_rank`, `attention.sliding_window`, `expert_feed_forward_length`,
`expert_shared_count`, `expert_weights_scale/norm`, `expert_gating_func`,
`swiglu_clamp_exp/shexp`, `attention.output_group_count/lora_rank`,
`hyper_connection.sinkhorn_iterations/epsilon`, `attention.compress_ratios`, RMS eps, or
`block_count` — all of which `llama_model_dflash::load_arch_hparams`
(`dflash.cpp:7-63`) requires unconditionally when `dsv4_hc_mult > 0`, i.e. required, not
optional, throws if missing (`ml.get_key(..., hparams.X)` with no `false` default flag on most
of these).

This makes sense from ds4's side: it's a monolithic single-binary engine, so its DSpark stages
just reuse the trunk model's already-loaded config in memory (`ds4.c:5654` reads
`deepseek4.hyper_connection.count` from the **main model** `m`, not the support file).
llama.cpp's drafter, by contrast, loads as a fully separate `llama_model`/`llama_context` via
`--spec-draft-model` (`common/speculative.cpp`), so `load_arch_hparams` has no trunk model to
borrow from at load time — it needs every DSV4 hparam key present **in the draft file's own
GGUF metadata**, copied from the corresponding target model's GGUF.

This is the exact same category of gap the native-MTP plan (`2026-08-10-dsv4-mtp-plan.md`)
already identified and solved for a different file — same fix applies here: copy the missing
keys from the target model's own GGUF metadata into the repacked support file.

One thing DSpark does **not** need that native MTP did: `tok_embd`/`output` sharing.
`dflash.cpp`'s decoder already falls back to `cparams.ctx_other`'s target-model tensors when
its own `tok_embd`/`output` are null (lines 398-406, 481-488), so the repacked DSpark file does
not need its own copies of those — one less open question than the MTP plan had.

## Two paths to a working file

### Plan A (cleanest, contingent on a download that may not exist): convert from raw HF weights

`conversion/deepseek.py::DeepseekV4DSparkModel` reads directly from HF-checkpoint tensors
prefixed `mtp.<stage>.<rest>` (`filter_tensors`, line 993) — i.e. it expects the **same raw
source data** ds4's `--dspark-support` tool also reads directly from an HF checkpoint directory
(confirmed: ds4's tool input is HF safetensors, not a GGUF). If DeepSeek's official
`deepseek-ai/DeepSeek-V4-Flash` HF release includes these `mtp.*` sidecar tensors (plausible —
the native-MTP plan already established the release carries `layers.N.nextn.*`-style sidecar
tensors for the plain single-stage variant), running the existing converter against that raw
checkpoint is a **zero-repack, zero-guesswork** path: no rename table, no hparam backfill,
correct by construction. The catch is practical, not technical: this may require pulling a
large slice of the official multi-hundred-GB release just to reach the `mtp.*` tensors, which
may not be worth it next to Plan B's already-available 5.6 GB file. Worth a quick check (does
DeepSeek publish the MTP/DSpark sidecar as a separately-downloadable subset?) before committing
to Plan B.

### Plan B (practical today): repack antirez's ds4-quantized file

antirez already hosts a ready-to-download DSpark support GGUF today:
`https://huggingface.co/antirez/deepseek-v4-gguf` →
`DeepSeek-V4-Flash-DSpark-support-0731.gguf` (~5.6 GiB, per `ds4/README.md:220-221`), fetchable
with `ds4/download_model.sh ds4f-dspark`. This is already a `deepseek4-dspark`-arch GGUF
produced by ds4's own `deepseek4-quantize --dspark-support` tool (routed experts quantized to
`Q2_K`/`IQ2_XXS`, plain tensors `F16`, norms `F32` — see quant-policy table below), checkpoint-
matched to the `0731` Flash release this branch's own quants (per `antirez/deepseek-v4-gguf`)
already target.

Write a small `gguf-py`-based Python repack script that:
1. Reads `DeepSeek-V4-Flash-DSpark-support-0731.gguf`.
2. Renames every tensor per the table above: strip `mtp.{N}.` → `blk.{N}.` for per-stage
   tensors; for stage-0-only and final-stage-only tensors, strip the stage prefix entirely and
   rename to the root llama.cpp name (`main_proj`→`fc`, `main_norm`→`enc.output_norm`,
   `norm`→`output_norm`, `hc_head_fn/base/scale`→`output_hc_fn/base/scale`,
   `markov_head.markov_w1/w2`→`markov_w1/w2`, `confidence_head.proj`→`conf_proj`).
3. Sets `general.architecture = "dflash"`.
4. Sets `dflash.block_size` from `dspark.block_size`, `dflash.target_layers` from
   `dspark.target_layer_ids`. Drops `dspark.markov_rank`/`noise_token_id`/`stage_count`/
   `n_layers` (superseded by tensor-shape inference and standard `block_count`, except resolve
   the noise-token-id question below before assuming it's safe to drop).
5. Sets `dflash.block_count` = stage count (from `dspark.stage_count`).
6. Copies every DSV4 hparam key `load_arch_hparams` requires (see gap list above) from the
   **target model's own GGUF** — the local antirez `DeepSeek-V4-Flash-0731` quant already in
   use — reading each with the `deepseek4.*` prefix and rewriting as `dflash.*`. Also copy
   `attention.layernorm_rms_eps` and any rope params `dflash.cpp`'s SWA/rope block reads.
7. Validate: `llama-server -m <target.gguf> --spec-draft-model <repacked>.gguf
   --spec-type draft-dspark`, confirm it loads without shape errors, then benchmark decode
   t/s against the plain (no-speculation) baseline from
   `2026-08-10-dsv4-strix-halo-perf-fix.md`, on real hardware (Strix Halo, matching how the
   PR 26592 TOP_K fix was validated — this branch's dev loop needs the actual gfx1151 box, not
   this machine).

## Open questions to resolve during implementation, not before

- **Noise token id** (`dspark.noise_token_id`, default 128799): `dflash.cpp`'s DSpark path
  doesn't visibly consume a GGUF-level noise-token-id constant in the code read so far — the
  masked/noise positions in the draft ubatch are more likely constructed on the `common/`
  runtime side (ubatch construction in `common_speculative_impl_draft_dflash`,
  `common/speculative.cpp:941-945` and around). Grep `common/speculative.cpp` for how it builds
  the "noise block" token sequence before assuming this key is genuinely unused — if it's
  hardcoded or derived differently there, the repack may still need to communicate this value
  somehow (a vocab-adjacent special token, or a runtime flag).
- **Routed-expert double-quantization**: antirez's file already quantized MoE experts to
  `Q2_K`/`IQ2_XXS` via ds4's own policy. A draft model's own quality only affects acceptance
  rate, not target output correctness (the target model verifies), so this is very likely
  fine, but if acceptance rate benchmarks come back low, that's a candidate suspect before
  assuming a repack bug — compare against Plan A's unquantized-conversion result if available.
- **`dflash.block_count`**: confirm the standard GGUF `block_count` key name/format
  llama.cpp's model loader expects generically (check how other archs set it, e.g.
  `gguf_writer.py`'s `add_block_count`) rather than assuming `dflash.n_layers` verbatim.
- **MLA `o_groups`/`o_lora_rank` and any other DSV4-specific dims** not itemized above: do one
  more pass of `dflash.cpp:load_arch_hparams` against the copy list in step 6 above at
  implementation time — this doc's list was compiled from the two research passes and could
  have missed one required key; the load will throw a clear "missing key" error either way,
  so treat a failed load as a to-do list, not a stop sign.

## Suggested order of operations

1. Quick check: does `deepseek-ai/DeepSeek-V4-Flash`'s official HF repo expose the `mtp.*`
   sidecar tensors as a separately-downloadable subset (would make Plan A cheap)? If yes,
   prefer Plan A — it's strictly more correct and skips the whole rename/hparam-backfill script.
2. If not, or if the download is impractically large: download antirez's
   `DeepSeek-V4-Flash-DSpark-support-0731.gguf` today via `ds4/download_model.sh ds4f-dspark`
   and start Plan B — the file is small (5.6 GiB) and already sitting one command away.
3. Write and test the Plan B repack script (`gguf-py`-based, same tooling pattern as the
   native-MTP repack script in `2026-08-10-dsv4-mtp-plan.md`).
4. Validate load + benchmark on the actual Strix Halo hardware, not this machine — same
   constraint noted in the sibling perf docs.
5. Compare acceptance rate / decode t/s against both the no-speculation baseline and (if
   available) native MTP once that's also working, to see which speculative path wins on this
   hardware — they are not mutually exclusive, and DeepSeek's own DSpark design (multi-token
   Markov-chained blocks with confidence gating) is architecturally more ambitious than plain
   single-token NextN, so it's worth knowing which one actually pays off before investing
   further tuning time in either.
