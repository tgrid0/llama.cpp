# Bringing MTP speculative decoding to DeepSeek-V4-Flash on this branch

Status: plan for a future agent session. No code changes made yet. This doc is unusually
well-grounded (verified against actual source, not guessed) because the mechanism turned out
to already exist end-to-end in this codebase — the work is much smaller than "implement MTP,"
it's closer to "produce a correctly-formatted file and wire up flags that already exist."

## The headline finding

**Native MTP/NextN speculative decoding for `deepseek4` is already fully implemented in this
branch.** This is not a gap to fill with new C++ — it's an existing first-class speculative
decoding backend, one of four (`common/common.h:171-177`):
```
COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE   // standalone draft model
COMMON_SPECULATIVE_TYPE_DRAFT_EAGLE3
COMMON_SPECULATIVE_TYPE_DRAFT_MTP      // <-- this one
COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH   // separate arch, src/models/dflash.cpp — this is
                                        //     what Lucebox calls "DSpark Drafter"
COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK
```
Selected via `--spec-type draft-mtp` (see `common/speculative.cpp:35`) or auto-detected
(`common/arg.cpp:544-549`). Implementation: `common_speculative_impl_draft_mtp` in
`common/speculative.cpp:1285`. Graph building: `llama_model_deepseek4::graph_mtp` in
`src/models/deepseek4.cpp:1407` (and the `build_arch_graph` dispatch at line 180-185, which
picks `graph_mtp` when `params.gtype == LLM_GRAPH_TYPE_DECODER_MTP`).

antirez's statement that his separate MTP GGUF "requires a custom loader" is about **his file
not being in the tensor-naming format the existing loader expects**, not about MTP support
being absent from llama.cpp. Confirmed by direct comparison below.

## Why antirez's MTP GGUF doesn't load: a tensor-naming mismatch, not a missing feature

Dump: `tmpwork/antirez_deepseek-v4-flash_custom_mtp_gguf_dump.txt`
(`general.architecture = 'deepseek4_mtp_support'`, 32 tensors, all prefixed `mtp.0.*`, e.g.
`mtp.0.e_proj.weight`, `mtp.0.h_proj.weight`, `mtp.0.enorm.weight`, `mtp.0.hnorm.weight`,
`mtp.0.norm.weight`, `mtp.0.ffn_gate/up/down_exps.weight` (Q4_K, full 256 experts — same
MoE routing as the main trunk), `mtp.0.hc_attn_*`/`mtp.0.hc_ffn_*`/`mtp.0.hc_head_*`.

The native loader (`src/models/deepseek4.cpp:79-177`, `load_arch_tensors`) expects, for the
NextN block appended at layer index `n_layer_main` (i.e. `blk.43.*` for this 43-layer model):
```
blk.43.nextn.eh_proj.weight           (required)   {2*n_embd, n_embd}
blk.43.nextn.enorm.weight             (required)   {n_embd}
blk.43.nextn.hnorm.weight             (required)   {n_embd}
blk.43.nextn.embed_tokens.weight      (NOT required — falls back to root tok_embd)
blk.43.nextn.shared_head_head.weight  (NOT required — falls back to root output)
blk.43.nextn.shared_head_norm.weight  (NOT required)
```
plus the normal per-layer tensors (attn_norm, wq_a/wq_b, wkv, ffn_gate_inp, ffn_*_exps, etc.)
that any regular deepseek4 layer needs — the NextN block is architecturally just "one more
transformer layer" with a couple of extra projections, not a separate lightweight head.

**The rename/repack mapping already exists in this repo's own converter** —
`conversion/deepseek.py`. Two relevant, distinct code paths, do not conflate them:

1. `DeepseekV4Model.generate_extra_tensors` (lines 796-814): when converting from the
   *original HF safetensors* checkpoint, it finds `layers.{bid}.nextn.e_proj.weight` and
   `layers.{bid}.nextn.h_proj.weight` for `bid >= main_layers` and does
   `torch.cat((e_proj, h_proj), dim=1)` to produce the single `nextn.eh_proj.weight` the C++
   loader wants (line 813). This confirms the **e_proj/h_proj -> eh_proj split is exactly
   what antirez's file has**, just not yet concatenated/renamed.
2. `DeepseekV4Model._map_dsv4_tensor_name` (lines 825-893) is the full tensor rename table —
   `hc_attn_fn` -> `HC_ATTN_FN`, `hc_head_fn` -> `HC_HEAD_FN`, etc. — the same names visible
   in antirez's dump map 1:1 onto entries in this table.
3. A second, unrelated MTP path also exists in this same file: `DeepseekV4DSparkModel`
   (lines 938-1018), registered for the separate standalone "DSpark" drafter architecture
   (`gguf.MODEL_ARCH.DFLASH`, loaded by `src/models/dflash.cpp` — this is what Lucebox
   distributes as `DeepSeek-V4-Flash-DSpark-Drafter-GGUF`). **Do not confuse this with
   antirez's file** — antirez's tensors (full 256-expert MoE FFN, hyper-connections) match
   the native in-model NextN design, not the lightweight DSpark markov/confidence-head
   design (`_DSPARK_ROOT_MAP` in that class has only `main_proj`, `markov_head`,
   `confidence_head` — nothing like antirez's `ffn_gate_exps`/`ffn_up_exps`/`ffn_down_exps`).
   antirez's `general.architecture = 'deepseek4_mtp_support'` string is his own ad-hoc tag,
   not either of llama.cpp's two real MTP-family architectures.

## Plan A (preferred): repack antirez's GGUF, don't write a custom loader

Write a small Python script (using `gguf-py`, this repo already has it under `gguf-py/`) that:

1. Reads antirez's `DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf`.
2. Renames every `mtp.0.X` tensor using the same mapping `_map_dsv4_tensor_name` uses for
   `layers.43.X` (i.e. treat `mtp.0` as `layers.{n_layer_main}`, output as `blk.43.nextn.X`
   or `blk.43.X` depending on whether X is nextn-specific or a regular per-layer tensor —
   work through `layer_map` in `conversion/deepseek.py:845-877` entry by entry against the
   32 tensors in the dump; most map directly, `e_proj`+`h_proj` need the dim=1 concat from
   `generate_extra_tensors:813`).
3. Sets `general.architecture = 'deepseek4'` (matching the main model, not antirez's ad-hoc
   tag) and the hparams the loader reads in `mtp_only` mode — check
   `load_arch_hparams` (`deepseek4.cpp:18-77`) for which keys are read unconditionally vs.
   only for trunk layers, and copy those from the main model's GGUF metadata (dump:
   `tmpwork/antirez_deepseek-v4-flash-0731_gguf_dump.txt`) since the MTP-only
   file will not carry a trunk to derive them from.
4. Sets `deepseek4.nextn_predict_layers = 1` (confirms via the existing key
   `LLM_KV_NEXTN_PREDICT_LAYERS`) and `block_count` to cover the appended NextN block, per
   the python `mtp_only` path (`self.block_count += num_nextn_predict_layers`).
5. Names the output `mtp-<matching-basename>.gguf` — this matches the sidecar-file naming
   convention the download tooling already looks for (`common/download.cpp:641-644`,
   `find_best_sibling(files, model, "mtp-", tag)`), so it's discoverable the same way an
   official HF-hosted MTP sidecar would be.

**Open question to resolve before or during implementation, don't skip this**: the draft
context (`ctx_dft` in `common_speculative_impl_draft_mtp`, `common/speculative.cpp:1285-1296`)
is a genuinely separate `llama_context`/`llama_model`, loaded standalone via `--model-draft`.
`load_arch_tensors` creates `tok_embd`, `output_norm`, `output` **unconditionally** (flags=0,
not gated by `mtp_only`/`trunk_flags`) at `deepseek4.cpp:97-100`. antirez's 32-tensor dump has
**no** `token_embd.weight` or `output.weight`/`output_norm.weight`. Either:
   - the standalone draft file genuinely needs its own copy of these (likely: copy them out
     of the main model's GGUF into the repacked MTP file — token_embd is ~1GB in F16, output
     is the same size, so the sidecar file grows substantially but is still tiny next to the
     91GB main model), or
   - there's a tensor-sharing/tied-embedding mechanism for the draft context that reuses the
     target context's tok_embd/output without a separate copy (would need to be found in
     `common_speculative_impl_draft_mtp` or `llama_model_load_from_file`'s handling of a
     draft-role load — search for how `ctx_dft` gets constructed in
     `common/speculative.cpp` around where `common_speculative_impl_draft_mtp` is
     instantiated, and in `common/common.cpp`'s model-loading helpers for any "share weights
     with target" logic).
   Resolve this by reading the code path fully before writing the repack script's tensor
   list, OR just try the cheap experiment first: repack without tok_embd/output and attempt
   to load with `--model-draft <repacked>.gguf --spec-type draft-mtp` against the running
   target model — the error message (if any) will directly answer this question faster than
   further code reading.
6. Validate: `llama-server ... --model-draft <repacked-mtp>.gguf` (or whatever flag actually
   wires `params.draft.model`, confirm exact flag name in `common/arg.cpp` — `--model-draft`
   is a reasonable guess based on convention but wasn't directly confirmed this session)
   alongside the main `--model <91GB antirez quant>`, check it loads without tensor-shape
   errors, then benchmark decode t/s against the no-MTP baseline established in the perf-fix
   doc. Compare against the user's reported 17-32 t/s ceiling with MTP/DSpark.

## Plan B (fallback, more work, more correctness risk): re-convert from original HF checkpoint

If Plan A's tensor-by-tensor repack turns out to be unreliable (shape mismatches, missing
hparams that can't be reconstructed from the main model's GGUF, or the standalone-loadability
question above turns out to be a bigger structural problem than a rename), check whether the
original DeepSeek-V4-Flash-0731 HF safetensors release includes the `layers.N.nextn.*`
tensors directly (most DeepSeek MTP releases do, per `deepseek.py`'s handling of it as the
common case, lines 483-511 in the base `DeepseekV2Model`-family class, and 796-814 for the
V4-specific e_proj/h_proj variant). If so, running `conversion/deepseek.py` directly against
the original checkpoint in `--mtp`/`mtp_only` mode produces a guaranteed-correct sidecar file
with zero reverse-engineering, at the cost of a large download and needing to re-quantize
(match antirez's Q4_K/Q8_0 mix for consistency, or just use Q8_0 throughout since the MTP
block is a small fraction of total model size).

## Plan C (do instead of, not in addition to, if A/B stall): standalone DFlash/DSpark drafter

Lucebox hosts a separate, architecturally distinct drafter:
`https://huggingface.co/Lucebox/DeepSeek-V4-Flash-DSpark-Drafter-GGUF` (not yet downloaded —
no local dump available this session). This targets `COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK` /
`COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH` via `src/models/dflash.cpp`, a completely separate,
already-upstream-supported code path from native MTP — no antirez-format reverse-engineering
needed at all, just download and point `--model-draft` at it (flag name TBD, see above) with
the right `--spec-type`. This is the lowest-implementation-risk option but depends on
Lucebox's drafter being compatible with the antirez quant as the target model (same
vocabulary/tokenizer, same `n_embd_out` — check `llama_model_n_embd_out` compatibility per
the assertion at `common/speculative.cpp:1294`). Worth downloading and trying this in
parallel with Plan A, since it requires no code archaeology at all — cheapest experiment to
run first, even before finishing Plan A's repack script, to get a baseline "does speculative
decoding work at all for this target model on this branch" signal.

## Suggested order of operations for the implementing agent

1. Try Plan C first (download Lucebox's DSpark drafter, attempt to run it) — cheapest,
   answers "does the speculative decoding plumbing work at all here" independent of the
   antirez-format question.
2. In parallel or after, resolve the tok_embd/output open question for Plan A (the cheap
   experiment: attempt-and-read-the-error, per above).
3. Write and test the Plan A repack script.
4. Only fall back to Plan B if Plan A proves structurally broken, not just fiddly.
5. Benchmark whichever path works against the user's reported targets (17-32 t/s decode)
   using the same clean `llama-bench`-style methodology as the perf-fix doc, not just the
   full server flags, so the MTP win is measured in isolation from the base-inference fix.
