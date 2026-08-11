#!/usr/bin/env python3
"""Repack a ds4-format DeepSeek-V4 DSpark support GGUF into this branch's native `dflash` arch.

Background: docs/superpowers/specs/2026-08-11-dsv4-dspark-drafter-port-plan.md

antirez's dwarfstar4 ("ds4") engine ships a DSpark speculative-decoding support file
(general.architecture = "deepseek4-dspark", tensors named "mtp.{stage}.<short-name>",
9 GGUF metadata keys total) produced by ds4's own `deepseek4-quantize --dspark-support`
tool. This branch's own `dflash` architecture (src/models/dflash.cpp) already implements
the same DSpark drafter end-to-end - same per-stage tensor short names, same graph - but
expects llama.cpp's own tensor-naming convention ("blk.{N}.<short-name>" plus a handful of
root-level tensors) and expects every DSV4 hyperparameter to be present in the drafter
file's own GGUF metadata, because llama.cpp loads a speculative-decoding draft model as a
fully independent llama_model/llama_context (unlike ds4, which is monolithic and just
reuses the trunk model's in-memory config for its DSpark stages).

This script does not requantize or touch tensor data - only tensor/key names are rewritten,
and DSV4 hparams + the tokenizer are copied verbatim from the target model's own GGUF (the
same DeepSeek-V4-Flash-0731 quant the drafter will run alongside).

Usage:
    python dsv4_dspark_repack.py DSPARK.gguf TARGET.gguf OUTPUT.gguf [--dry-run]

Then in llama.cpp:
    llama-server -m TARGET.gguf --spec-draft-model OUTPUT.gguf --spec-type draft-dspark ...
"""
from __future__ import annotations

import argparse
import logging
import os
import re
import sys
from pathlib import Path

if "NO_LOCAL_GGUF" not in os.environ and (Path(__file__).parent.parent.parent.parent / 'gguf-py').exists():
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tqdm import tqdm

import gguf

logger = logging.getLogger("dsv4-dspark-repack")

# ---------------------------------------------------------------------------
# Tensor rename table (verified against src/llama-arch.cpp, src/models/dflash.cpp,
# conversion/deepseek.py::DeepseekV4DSparkModel, ds4/gguf-tools/deepseek4-quantize.c
# and ds4/ds4.c - see the plan doc above for file:line references)
# ---------------------------------------------------------------------------

# Per-stage tensors: identical short name on both projects, only the prefix changes
# ("mtp.{N}." on ds4's side -> "blk.{N}." on llama.cpp's side).
PER_STAGE_SUFFIXES = {
    "hc_attn_base.weight", "hc_attn_fn.weight", "hc_attn_scale.weight",
    "hc_ffn_base.weight", "hc_ffn_fn.weight", "hc_ffn_scale.weight",
    "attn_sinks.weight", "attn_norm.weight",
    "attn_q_a.weight", "attn_q_a_norm.weight", "attn_q_b.weight",
    "attn_kv.weight", "attn_kv_a_norm.weight",
    "attn_output_a.weight", "attn_output_b.weight",
    "ffn_gate_inp.weight", "exp_probs_b.bias", "ffn_norm.weight",
    "ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight",
    "ffn_gate_shexp.weight", "ffn_up_shexp.weight", "ffn_down_shexp.weight",
}

# Stage-0-only tensors ("target feature" injection): root-level in llama.cpp, stage dropped.
STAGE0_ROOT_RENAME = {
    "main_proj.weight": "fc.weight",
    "main_norm.weight": "enc.output_norm.weight",
}

# Final-stage-only tensors (LM-head-adjacent): root-level in llama.cpp, stage dropped.
FINAL_STAGE_ROOT_RENAME = {
    "norm.weight": "output_norm.weight",
    "hc_head_base.weight": "output_hc_base.weight",
    "hc_head_fn.weight": "output_hc_fn.weight",
    "hc_head_scale.weight": "output_hc_scale.weight",
    "markov_head.markov_w1.weight": "markov_w1.weight",
    "markov_head.markov_w2.weight": "markov_w2.weight",
    "confidence_head.proj.weight": "conf_proj.weight",
}

MTP_TENSOR_RE = re.compile(r"^mtp\.(\d+)\.(.+)$")

# DSV4 hparam keys that must NOT be blindly copied from the target model - DSpark's drafter
# defines these independently (see the plan doc's "the real gap" section).
TARGET_HPARAM_OVERRIDES = {
    "deepseek4.block_count",
    "deepseek4.attention.compress_ratios",
    "deepseek4.target_layers",
}


def plan_tensor_rename(name: str) -> str:
    m = MTP_TENSOR_RE.match(name)
    if not m:
        raise ValueError(f"unexpected tensor name (not 'mtp.N.*'): {name!r}")
    stage, rest = int(m.group(1)), m.group(2)

    if rest in PER_STAGE_SUFFIXES:
        return f"blk.{stage}.{rest}"
    if rest in STAGE0_ROOT_RENAME:
        if stage != 0:
            raise ValueError(f"expected {rest!r} only at stage 0, found at stage {stage}")
        return STAGE0_ROOT_RENAME[rest]
    if rest in FINAL_STAGE_ROOT_RENAME:
        return FINAL_STAGE_ROOT_RENAME[rest]

    raise ValueError(
        f"unrecognized DSpark tensor suffix {rest!r} (full name {name!r}) - the rename "
        f"table in this script is incomplete, check it against "
        f"docs/superpowers/specs/2026-08-11-dsv4-dspark-drafter-port-plan.md and ds4's "
        f"gguf-tools/deepseek4-quantize.c dspark_stage_rules[]"
    )


def get_field_value(reader: "gguf.GGUFReader", key: str):
    field = reader.get_field(key)
    return field.contents() if field else None


def backfill_target_hparams(reader_target: "gguf.GGUFReader", writer: "gguf.GGUFWriter",
                             n_stages: int, target_block_count: int) -> tuple[list[str], list[str]]:
    """Copy every deepseek4.* key from the target GGUF into dflash.* on the output.

    Some of these keys are per-layer arrays (e.g. swiglu_clamp_exp, and generically
    feed_forward_length / attention.head_count / attention.head_count_kv read via
    ml.get_key_or_arr(..., hparams.n_layer_all) in llama-model.cpp/dflash.cpp) sized to the
    TARGET's block_count (43 layers). Copied verbatim they fail to load against the
    drafter's own much smaller block_count (n_stages) with "wrong array length". DeepSeek's
    own conversion script writes these as a single value repeated block_count times (see
    conversion/deepseek.py:662-663, `[hparams["swiglu_limit"]] * self.block_count`) - so if
    every element is identical, it's safe to re-emit as n_stages copies of that one value.
    """
    copied = []
    resized = []
    for field in reader_target.fields.values():
        if not field.name.startswith("deepseek4."):
            continue
        if field.name in TARGET_HPARAM_OVERRIDES:
            continue
        new_key = "dflash." + field.name[len("deepseek4."):]
        val_type = field.types[0]
        sub_type = field.types[-1] if val_type == gguf.GGUFValueType.ARRAY else None
        value = field.contents()

        if (val_type == gguf.GGUFValueType.ARRAY and isinstance(value, list)
                and len(value) == target_block_count and target_block_count != n_stages):
            unique_vals = set(value)
            if len(unique_vals) == 1:
                value = list(value[:1]) * n_stages
            else:
                logger.warning(
                    f"{field.name} is a per-layer array of {target_block_count} DIFFERING "
                    f"values (first few: {value[:5]}) - can't safely collapse to a single "
                    f"value for {n_stages} drafter stages. Using its first {n_stages} values "
                    f"as a best-effort guess - verify {new_key} manually if the model behaves "
                    f"oddly, this key needs a real decision, not an automatic one."
                )
                value = list(value[:n_stages])
            resized.append(new_key)

        writer.add_key_value(new_key, value, val_type, sub_type=sub_type)
        copied.append(new_key)
    return copied, resized


def copy_tokenizer(reader_target: "gguf.GGUFReader", writer: "gguf.GGUFWriter") -> list[str]:
    """Copy every tokenizer.* key from the target GGUF verbatim (matches
    conversion/deepseek.py::DeepseekV4DSparkModel.set_vocab(), which reuses the target
    tokenizer rather than shipping a separate one for the drafter)."""
    copied = []
    for field in reader_target.fields.values():
        if not field.name.startswith("tokenizer."):
            continue
        val_type = field.types[0]
        sub_type = field.types[-1] if val_type == gguf.GGUFValueType.ARRAY else None
        writer.add_key_value(field.name, field.contents(), val_type, sub_type=sub_type)
        copied.append(field.name)
    return copied


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dspark", type=Path, help="ds4-format DSpark support GGUF (e.g. DeepSeek-V4-Flash-DSpark-support-0731.gguf)")
    parser.add_argument("target", type=Path, help="target DeepSeek-V4-Flash GGUF to borrow DSV4 hparams + tokenizer from")
    parser.add_argument("output", type=Path, help="output path for the repacked dflash-arch GGUF")
    parser.add_argument("--dry-run", action="store_true", help="print the rename/metadata plan and exit without writing tensor data")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(message)s")

    logger.info(f"* Reading DSpark support file: {args.dspark}")
    reader_dspark = gguf.GGUFReader(args.dspark, "r")

    src_arch = get_field_value(reader_dspark, gguf.Keys.General.ARCHITECTURE)
    if src_arch != "deepseek4-dspark":
        logger.warning(f"expected general.architecture='deepseek4-dspark' in {args.dspark}, found {src_arch!r} - continuing anyway")

    # --- plan tensor renames ---
    renames: dict[str, str] = {}
    max_stage = -1
    for tensor in reader_dspark.tensors:
        new_name = plan_tensor_rename(tensor.name)
        renames[tensor.name] = new_name
        stage = int(MTP_TENSOR_RE.match(tensor.name).group(1))
        max_stage = max(max_stage, stage)

    if max_stage < 0:
        raise SystemExit(f"found no 'mtp.N.*' tensors in {args.dspark} - is this really a ds4 --dspark-support file?")
    n_stages = max_stage + 1

    stage_count_field = get_field_value(reader_dspark, "dspark.stage_count")
    if stage_count_field is not None and int(stage_count_field) != n_stages:
        logger.warning(f"dspark.stage_count={stage_count_field} but highest tensor stage index implies {n_stages} stages - using {n_stages}")

    block_size = get_field_value(reader_dspark, "dspark.block_size")
    target_layer_ids = list(get_field_value(reader_dspark, "dspark.target_layer_ids") or [])
    noise_token_id = get_field_value(reader_dspark, "dspark.noise_token_id")

    if block_size is None or not target_layer_ids:
        raise SystemExit(f"{args.dspark} is missing dspark.block_size or dspark.target_layer_ids - is this really a ds4 --dspark-support file?")

    logger.info(f"* {len(renames)} tensors across {n_stages} stages")
    logger.info(f"* block_size={block_size} target_layer_ids={target_layer_ids} noise_token_id={noise_token_id}")

    logger.info(f"* Reading target model header: {args.target}")
    reader_target = gguf.GGUFReader(args.target, "r")
    target_arch = get_field_value(reader_target, gguf.Keys.General.ARCHITECTURE)
    if target_arch != "deepseek4":
        logger.warning(f"expected target general.architecture='deepseek4' in {args.target}, found {target_arch!r} - hparam backfill below will silently copy 0 keys if this is wrong")

    target_block_count = get_field_value(reader_target, "deepseek4.block_count")
    if target_block_count is None:
        raise SystemExit(f"{args.target} has no deepseek4.block_count - is this really the target model's GGUF?")
    target_block_count = int(target_block_count)

    dflash_target_layers = [layer + 1 for layer in target_layer_ids]  # matches conversion/deepseek.py's own +1 offset

    if args.dry_run:
        print("\n--- Tensor renames ---")
        shown = 0
        for old, new in renames.items():
            if shown >= 40:
                print(f"  ... and {len(renames) - 40} more")
                break
            print(f"  {old}  ->  {new}")
            shown += 1

        hparam_fields = [f for f in reader_target.fields.values()
                         if f.name.startswith("deepseek4.") and f.name not in TARGET_HPARAM_OVERRIDES]
        print(f"\n--- deepseek4.* keys to be copied from target as dflash.* ({len(hparam_fields)} keys) ---")
        for f in hparam_fields:
            val = f.contents()
            val_type = f.types[0]
            note = ""
            if val_type == gguf.GGUFValueType.ARRAY and isinstance(val, list) and len(val) == target_block_count and target_block_count != n_stages:
                unique_vals = set(val)
                if len(unique_vals) == 1:
                    note = f"  [per-layer array of {target_block_count}, all equal -> collapsed to {n_stages}x {next(iter(unique_vals))}]"
                else:
                    note = (f"  [WARNING: per-layer array of {target_block_count} DIFFERING values "
                             f"-> best-effort first {n_stages} taken, verify manually]")
            print(f"  {f.name} -> dflash.{f.name[len('deepseek4.'):]}{note}")

        print("\n--- overridden / synthesized keys ---")
        print("  general.architecture = dflash")
        print(f"  dflash.block_size = {block_size}")
        print(f"  dflash.target_layers = {dflash_target_layers}  (dspark.target_layer_ids + 1)")
        print(f"  dflash.block_count = {n_stages}")
        print(f"  dflash.attention.compress_ratios = {[0] * n_stages}  (DSpark requires uncompressed attention on all stages)")
        if noise_token_id is not None:
            print(f"  tokenizer.ggml.mask_token_id = {noise_token_id}")

        tok_keys = [f.name for f in reader_target.fields.values() if f.name.startswith("tokenizer.")]
        print(f"\n--- tokenizer.* keys to be copied verbatim from target ({len(tok_keys)} keys) ---")
        for k in tok_keys[:10]:
            print(f"  {k}")
        if len(tok_keys) > 10:
            print(f"  ... and {len(tok_keys) - 10} more")

        if len(hparam_fields) == 0:
            print("\n*** WARNING: 0 deepseek4.* keys found on target - the target's "
                  "general.architecture is probably not 'deepseek4'. Fix --target before "
                  "running for real, the output would fail to load. ***")

        print("\nDry run only - no output file written. Re-run without --dry-run to write it.")
        return

    if os.path.isfile(args.output):
        logger.warning(f"{args.output} already exists and will be overwritten")

    logger.info(f"* Writing: {args.output}")
    writer = gguf.GGUFWriter(args.output, arch="dflash", endianess=reader_dspark.endianess)

    alignment = get_field_value(reader_dspark, gguf.Keys.General.ALIGNMENT)
    if alignment is not None:
        writer.data_alignment = alignment

    writer.add_name(get_field_value(reader_dspark, gguf.Keys.General.NAME) or "DeepSeek-V4-Flash DSpark drafter (repacked)")

    copied, resized = backfill_target_hparams(reader_target, writer, n_stages, target_block_count)
    if len(copied) == 0:
        writer.close()
        raise SystemExit(
            f"copied 0 'deepseek4.*' hparam keys from {args.target} - refusing to write a "
            f"broken output file. Check --target points at the right GGUF (general.architecture "
            f"should be 'deepseek4', found {target_arch!r})."
        )
    logger.info(f"* Copied {len(copied)} DSV4 hparam keys from target as dflash.* ({len(resized)} per-layer arrays resized {target_block_count} -> {n_stages})")
    if resized:
        logger.info(f"* Resized keys: {', '.join(resized)}")

    writer.add_block_size(int(block_size))
    writer.add_target_layers(dflash_target_layers)
    writer.add_block_count(n_stages)
    writer.add_attention_compress_ratios([0] * n_stages)

    if noise_token_id is not None:
        writer.add_mask_token_id(int(noise_token_id))

    tok_copied = copy_tokenizer(reader_target, writer)
    logger.info(f"* Copied {len(tok_copied)} tokenizer.* keys from target")

    # --- tensors: renamed, byte-for-byte copy, no requantization ---
    total_bytes = sum(t.n_bytes for t in reader_dspark.tensors)
    for tensor in reader_dspark.tensors:
        writer.add_tensor_info(renames[tensor.name], tensor.data.shape, tensor.data.dtype, tensor.data.nbytes, tensor.tensor_type)

    bar = tqdm(desc="Writing", total=total_bytes, unit="byte", unit_scale=True)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()
    for tensor in reader_dspark.tensors:
        writer.write_tensor_data(tensor.data, tensor_endianess=reader_dspark.endianess)
        bar.update(tensor.n_bytes)
    writer.close()

    logger.info(f"* Done: {args.output}")


if __name__ == "__main__":
    main()
