# Disk Cache: Ignore Saved n_stream on Restore

## Problem

The automatic disk KV cache (`9f25e5cb2`) refuses to restore an entry that was
saved under a different KV-stream layout. A prompt cached with `--parallel 4`
(`kv-unified=false`, n_stream=4) is treated as a miss when the server is
restarted with `--parallel 1` (n_stream=1), even though the prompt and model
are identical:

```
disk cache: hash=00000000... was saved with n_stream=4, current n_stream=1, treating as miss
```

### Why it fails today

- The cache is keyed by SHA-256 of the prompt tokens only, one blob per
  distinct prompt. A second slot saving the same prompt is a no-op
  (timestamp refresh), so only the first-saved slot's stream data is persisted.
  This is fine: on restore, all non-empty stream data in the blob is routed
  into the destination seq's stream (`seq_to_stream[seq_id]`), and identical
  prompts produce identical KV content.
- The real blocker is the core state format: `llama_kv_cache::state_write`
  embeds the saving context's `n_stream` in the blob header
  (src/llama-kv-cache.cpp:1999), and `llama_kv_cache::state_read` throws
  "n_stream mismatch" when it differs from the current context
  (src/llama-kv-cache.cpp:2071-2075). `llama_state_seq_set_data_ext` then
  reports a partial restore, so the disk cache must pre-check `n_stream` and
  treat the entry as a miss (tools/server/server-disk-cache.cpp:510, 1210).

### Key insight

In `state_read`, for a single-seq restore (`seq_id != -1`) the read loop
already routes every non-empty saved stream into `seq_to_stream[seq_id]`
regardless of the saved stream index. The stored layout only matters for:
1. the header check, and
2. how many stream `cell_count` fields to read (the loop iterates the
   *current* `n_stream`, which can read past the end of a smaller blob).

So a single-seq state is layout-independent as long as the check is skipped
and the loop iterates the *saved* stream count. Full-state restore
(`seq_id == -1`, session files) must keep requiring an identical layout.

## Solution

### Part 1 - Lenient single-seq restore in the core

`llama_kv_cache::state_read` (src/llama-kv-cache.cpp):

- Read `n_stream_cur` from the blob.
- If `n_stream_cur != n_stream`:
  - `seq_id == -1`: throw "n_stream mismatch" as today.
  - `seq_id != -1`: continue; a single-seq state can be restored into any
    layout because all non-empty data lands in `seq_to_stream[seq_id]`.
- Loop `for (s = 0; s < n_stream_cur; ++s)` instead of `n_stream`.

Works in both directions:
- saved 4 -> current 1: three empty stream entries are skipped, the one
  non-empty stream is restored into stream 0.
- saved 1 -> current 4: only the one stored stream entry is read (no read
  past end of buffer), restored into `seq_to_stream[seq_id]`.

Recurrent/hybrid memory state (llama_memory_recurrent) has no stream
dimension and is unaffected.

### Part 2 - Disk cache stops treating n_stream as a compatibility check

tools/server/server-disk-cache.cpp:

- `load()`: delete the n_stream mismatch pre-check (lines 510-516).
- `load_by_hash()`: delete the n_stream mismatch pre-check (lines 1210-1214).
- `save()`: delete the incompatible-n_stream overwrite branch; an existing
  entry is always compatible, so a re-save only refreshes `last_used_us`.

`disk_cache_entry::n_stream` stays in index.json (v3, no version bump) as an
informational field; `init()` keeps the `n_stream` argument and logs it.
Header comments in server-disk-cache.h are updated to match.

### Part 3 - Tests

tools/server/tests/unit/test_disk_cache.py:

- `test_disk_cache_survives_parallel_count_change`: flip the expectation.
  A restart with a different `--parallel` on the same cache dir now yields a
  real disk-cache hit (`prompt_n < first_prompt_n`) instead of a full
  reprocess; the third-server step (same layout) still expects a hit.

## Out of scope

- Per-slot blob files. Not needed: the prompt hash already separates
  distinct prompts, and a shared prompt's KV content is identical across
  slots.
- Model identity in the hash. Sharing a cache dir across different models
  would still collide (pre-existing limitation, unchanged).
