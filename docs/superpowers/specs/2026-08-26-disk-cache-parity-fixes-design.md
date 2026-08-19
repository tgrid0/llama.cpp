# Disk Cache: Port Correctness Fixes from nathanw-llamacpp

## Problem

This branch's automatic disk KV cache (`tools/server/server-disk-cache.{h,cpp}`, hooked into
`tools/server/server-context.cpp`) was ported from an earlier point in the same feature's
lineage than two sibling forks (`nathanw-llamacpp` @ `strix-halo-vulkan-zgfs`, later ported again
to `ik_llama.cpp`). Since that port, three correctness bugs were found and fixed upstream in
nathanw's branch, and never ported back here:

1. **Broken embedded SHA-256** (nathanw `99c7e3ece`). The cache's self-contained SHA-256 (used
   as the sole content-addressing key for exact-match `load()`/`save()`) has three independent
   bugs that together make it near input-independent:
   - `K[47]` is `0x106a0070`; the correct FIPS-180-4 constant is `0x106aa070`.
   - `embed_sha256_transform`'s round macro is invoked 64 times against the same fixed `a..h`
     variable bindings, never rotating which physical variable plays which round role - not the
     real SHA-256 compression function.
   - `ROTL32`/`ROTR32` lack an enclosing set of parentheses around the macro body, so `^` (which
     binds tighter than `|` in C) silently reaches across the macro boundary and mis-groups every
     `S0`/`S1`/`s0`/`s1` computation.

   Confirmed present, byte-for-byte, in this branch's `tools/server/server-disk-cache.cpp`.
   Impact: unrelated token sequences can hash-collide, so the cache can silently serve KV state
   from a different, unrelated prompt/conversation.

2. **`n_stream` mismatch is treated as a permanent, fatal miss** (nathanw `567dcffe4`). This
   branch already has the *original* strict check added when disk cache first landed: an entry
   saved under one `--parallel` (n_stream) layout is refused and overwritten if the server
   restarts under a different one, even though the prompt and model are identical. The real
   restriction lives in `llama_kv_cache::state_read` (`src/llama-kv-cache.cpp`), which throws on
   any `n_stream` mismatch; the disk cache layer pre-checks and refuses. In fact, for a
   single-sequence restore (`seq_id != -1`, which is exactly what the disk cache always does),
   every non-empty saved stream lands in `seq_to_stream[seq_id]` regardless of the saved stream
   index - the strictness is unnecessary for this call shape and only needed for full-state
   (`seq_id == -1`) session-file restores.

3. **No drafter/MTP/DFLASH context in saved entries** (nathanw `ba8ed1a96`). `save()`/`load()`
   only persist `ctx_tgt`'s KV state. With speculative decoding active (MTP or DFLASH/DSPARK -
   this branch's most-used configuration per prior work), a disk-cache restore snaps `ctx_tgt` to
   a high saved position while `ctx_dft` is left wherever it last was - a real desync. nathanw's
   regression test reproduces this as "inconsistent sequence positions" / a failed-decode loop
   after a cache hit.

Multimodal-aware hashing/matching (chunk-identity-aware `hash_tokens`/`find_best_prefix`, the
`.chunks` sidecar) is **already at parity** with nathanw - confirmed via full-file diff, no
changes needed there.

## Approach

Port all three fixes from `nathanw-llamacpp`, adapted to this branch's current line numbers and
surrounding code (which already matches nathanw's structure closely - same `ctx_dft` member
resolved via `spec_init->context()`, same three disk-cache call sites in `server-context.cpp`).
Not a blind patch-apply: each fix is re-derived against this branch's actual current code, per
the diffs already collected during investigation.

Out of scope (confirmed with user): the diagnostic k_stream/v_stream null-data-pointer logging
nathanw's original commit added (no correctness fix, pure startup diagnostics) and the cosmetic
`has_media()` -> `has_mtmd` gate tweak in `save()`'s multimodal blob-build check (functionally
equivalent in every case that matters - `has_mtmd` true with no actual media chunks just builds
an empty, harmless blob map either way).

## Design

### 1. SHA-256 fix (`tools/server/server-disk-cache.cpp`)

Mechanical, self-contained, in `embed_sha256_transform` and its surrounding macros/table:

- Wrap `ROTL32`'s body in parentheses: `#define ROTL32(v, n) (((uint32_t)(v) << (n)) | ((uint32_t)(v) >> (32 - (n))))`.
- Fix `K[47]`: `0x106a0070` -> `0x106aa070`.
- Replace the 64x unrolled `R(i)` macro (fixed `a..h` bindings) with the standard loop form:
  explicit `T1`/`T2`, register rotation each round (`h=g; g=f; f=e; e=d+T1; d=c; c=b; b=a; a=T1+T2`).

No interface change, no callers affected. Verify against known SHA-256 test vectors (empty
string, `"abc"`, standard pangram) the same way nathanw's commit did.

### 2. `n_stream` leniency (`src/llama-kv-cache.cpp` + `tools/server/server-disk-cache.{h,cpp}`)

**Core** (`llama_kv_cache::state_read`, `src/llama-kv-cache.cpp`):
- Only throw `"n_stream mismatch"` when `seq_id == -1` (full-state restore).
- Loop `for (s = 0; s < n_stream_cur; ++s)` (the *saved* count) instead of the current context's
  `n_stream`, so a smaller saved layout doesn't read past the end of the blob and a larger one
  doesn't leave stored streams unread.
- Recurrent/hybrid memory (`llama_memory_recurrent`) has no stream dimension - unaffected.

**Disk cache** (`server-disk-cache.h/.cpp`):
- `load()`: delete the `n_stream` mismatch pre-check (treat-as-miss branch).
- `load_by_hash()`: delete the equivalent pre-check.
- `save()`: delete the incompatible-n_stream overwrite branch; an existing entry is always
  compatible now, so re-saving the same hash only refreshes `last_used_us`.
- `disk_cache_entry::n_stream` and the `init(..., n_stream)` param stay (informational logging
  only); update the header comments that currently describe the old strict behavior.

**Test** (`tools/server/tests/unit/test_disk_cache.py`):
- Flip `test_disk_cache_survives_parallel_count_change`'s expectations to assert a real
  cache hit across a layout change in both directions (1->4 and 4->1), matching nathanw's
  updated test, instead of asserting a forced full reprocess.

### 3. Drafter/MTP context save+restore (`server-disk-cache.h/.cpp` + `server-context.cpp`)

**`disk_cache_entry`**: add `uint64_t dft_size_bytes = 0` (size of a new `<hash>.dft.bin`
sidecar; 0 = no draft state saved, or speculative decoding wasn't active when the entry was
written). Persisted in `index.json` (no version bump needed - matches how `media_size_bytes` was
added).

**`server_disk_cache` API** gains an optional `llama_context * ctx_dft = nullptr` parameter on
`load()`, `save()`, and `load_by_hash()`, and `bool require_dft = false` on `find_best_prefix()`:

- `save(..., ctx_dft, ...)`: if `ctx_dft` is non-null and has non-empty per-sequence state for
  `slot_id`, serialize it via `llama_state_seq_get_size_ext`/`get_data_ext` and write it to
  `<hash>.dft.bin` (atomic temp-file-then-rename, same pattern as the existing `.bin`/`.ckpt`
  sidecars). Non-fatal if the draft context has no state yet for this sequence - entry is simply
  target-only.
- `load()`/`load_by_hash(..., ctx_dft, ...)`: if `ctx_dft` is non-null and the matched entry's
  `dft_size_bytes == 0`, treat it as a miss (log and return false) rather than restoring
  `ctx_tgt` alone and leaving `ctx_dft` desynced. Otherwise read `<hash>.dft.bin` and restore via
  `llama_state_seq_set_data_ext`.
- `find_best_prefix(..., require_dft)`: when true, skip any candidate entry with
  `dft_size_bytes == 0` - never let prefix search pick an entry that would desync `ctx_dft`.
- Eviction (`evict_oldest`), `validate_and_rebuild`, and total-size accounting all gain the
  `.dft.bin` file alongside the existing `.bin`/`.ckpt`/`.chunks` handling (remove-on-evict,
  size-account, prune-stale-sidecar-reference-on-rebuild).
- `save_index()`/`load_index()`: add `dft_size_bytes` to the JSON round-trip.

**`server-context.cpp`**: thread the existing per-slot `ctx_dft` member (already resolved via
`spec_init->context()` at slot-launch time, already used throughout speculative-decoding
bookkeeping in this file) through the three existing disk-cache call sites:
- the save-on-evict/idle-save helper (~`server-context.cpp:1013`)
- the exact-match load (~`:1803`)
- the prefix search + `load_by_hash` (~`:1823`/`:1825`), passing `ctx_dft != nullptr` as
  `require_dft`

No new state resolution needed - `ctx_dft` is already the right value at each of these call
sites (same member used by the checkpoint/rewind code elsewhere in this file).

**Test**: port `test_disk_cache_hit_with_speculative_decoding` (self-speculation setup - draft
model == target model, sidesteps needing a second vocab-compatible model) as a regression test:
fill a slot via speculative decoding, evict it with an unrelated short prompt (forces `ctx_dft`
to a small nonzero position), then re-request the original long prompt (exact disk-cache hit,
`ctx_tgt` restored far ahead of `ctx_dft`'s stale position) and assert generation succeeds with no
"inconsistent sequence positions" / "failed to decode" in the log.

## Changed files

| File | Change |
|------|--------|
| `tools/server/server-disk-cache.h` | `dft_size_bytes` field; `ctx_dft`/`require_dft` params on 4 methods; updated `n_stream` comments |
| `tools/server/server-disk-cache.cpp` | SHA-256 fix; `n_stream` check removal; drafter save/load/evict/rebuild/index plumbing |
| `src/llama-kv-cache.cpp` | `state_read`: lenient `n_stream` for single-seq restore |
| `tools/server/server-context.cpp` | pass `ctx_dft` through the 3 disk-cache call sites |
| `tools/server/tests/unit/test_disk_cache.py` | flip n_stream test; add speculative-decoding regression test |

## Testing

- Build (WSL, per project convention): `cmake -B build && cmake --build build --config Release -j 8`
  for `llama-server`, `test-chat`, `test-chat-peg-parser` (regression check - unrelated to this
  change, but part of standing verification convention for this branch).
- SHA-256: quick standalone check against known test vectors (empty string, `"abc"`, pangram)
  before relying on the C++ build alone.
- Python: `tools/server/tests/unit/test_disk_cache.py` - full suite, including the flipped
  n_stream test and the new speculative-decoding regression test. Prior sessions already
  bootstrapped a working pytest venv (`/home/t_grid0/.venv`) with the tiny test model cached, so
  this branch can actually run this instead of skipping it as prior ports had to.

## Out of scope

- Diagnostic k_stream/v_stream null-data-pointer logging (no correctness effect).
- `has_media()` -> `has_mtmd` cosmetic gate tweak in `save()`.
- Any further multimodal work - already at parity.
- ik_llama.cpp is not consulted further as a source here: it was ported *from* nathanw's same
  three commits with multimodal and n_stream explicitly dropped (older/simpler server structure),
  so it has no fixes beyond what nathanw already has.
