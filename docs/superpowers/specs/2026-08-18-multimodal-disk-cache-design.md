# Full multimodal disk-cache save/load/match: design

Status: approved design, ready for implementation planning. Branch `b10331_zgfs`.

## Problem

The automatic disk-cache system (`tools/server/server-disk-cache.{h,cpp}`) persists KV state to
disk keyed by a hash of the prompt's token sequence, so repeat/continued prompts skip
reprocessing. It works only for text-only prompts: every call site in
`tools/server/server-context.cpp` (lines 1021, 1062, 1794, 1800) gates disk-cache save/load
behind `!tokens.has_media()`. Any request containing an image or audio chunk bypasses the disk
cache entirely, even though this server is used heavily with multimodal + recurrent/hybrid
models where cold reprocessing cost is highest.

The gate exists because the matching logic underneath is genuinely text-only, not just
conservatively disabled:

- `hash_tokens()` (`server-disk-cache.cpp:214`) hashes `tokens.get_text_tokens()`, which silently
  **drops** every `LLAMA_TOKEN_NULL` media placeholder before hashing.
- `save()` stores `entry.tokens` as `full_tokens->get_text_tokens()` (`server-disk-cache.cpp:468`)
  - also stripped.
- `find_best_prefix()` does a flat index-by-index longest-common-prefix over these stripped
  arrays - no chunk-boundary awareness.

Removing the gate without fixing the matching logic would be a correctness bug: two prompts with
different images but the same surrounding text would hash identically, and the cache would
silently serve KV state computed from the wrong image.

## What already works (why this is low-risk)

1. **The KV blob itself is modality-agnostic.** Both `server_disk_cache::save/load` and the
   RAM-based `server_prompt_cache` (`tools/server/server-context.cpp:257-290`) call the same
   `llama_state_seq_get_data_ext`/`set_data_ext` API. The RAM cache is **not** gated by
   `has_media()` and already relies on this API for multimodal KV state, including M-RoPE
   position bookkeeping for VLMs. The hard part - serializing/restoring KV tensors for
   image/audio-derived positions - is already solved and exercised in production.
2. **Content-derived chunk identity already exists.** Every image/audio chunk carries an `id`
   string, an FNV hash of the raw bytes (`tools/mtmd/mtmd-helper.cpp:360-392`,
   `mtmd_bitmap_set_id`). `server_tokens::get_common_prefix()`
   (`tools/server/server-common.cpp:471-519`) already does chunk-aware prefix matching for the
   RAM slot-reuse path: at a `LLAMA_TOKEN_NULL` position it resolves both chunks via
   `find_chunk()`, compares `mtmd_input_chunk_get_id()`, and on match skips the whole chunk
   atomically via its token count.
3. **A chunk metadata (de)serializer already exists and is tested.** `mtmd_input_chunk_save()` /
   `mtmd_input_chunk_load()` (`tools/mtmd/mtmd.cpp:2258-2292`, public API in `mtmd.h:241,243`)
   round-trip a chunk's identity/shape (`type`, `id`, `nx`/`ny`/pos-type/`image_idx`/
   `n_temporal_merge` for images, `n_tokens` for audio) while **intentionally dropping the raw
   pixel/audio buffer** - the loaded chunk always comes back a placeholder. Covered by
   `tests/test-mtmd-c-api.c:67-113`. This is precisely "persist enough to reconstruct token/pos
   counts and identity, without re-storing the media bytes."
4. **Recurrent/hybrid checkpoint rewind does not need a second mechanism.** Checkpoint restore
   (`server-context.cpp:3495-3530`) rolls recurrent state back to a saved snapshot, then replays
   any tokens between the checkpoint and the target position by re-running them through
   `llama_decode()`. That replay reads from `slot.task->tokens` - the live, freshly-submitted
   request, which (per normal chat-completions semantics) carries the real image/audio bytes
   again this turn - never from the restored `slot.prompt.tokens`. The restored
   `slot.prompt.tokens` is used only for bookkeeping (`size_up_to_pos()`, `pos_next()`, prefix
   comparison against the new request), never fed back into the vision/audio encoder. Placeholder
   chunks (metadata only, no pixels) are therefore sufficient for that role. One fix (below)
   unlocks both the plain-KV-cache case and the recurrent-checkpoint case.

   The narrower, per-iteration `do_checkpoint = do_checkpoint && !has_mtmd` at
   `server-context.cpp:3755` is a different, local safeguard: it only skips creating a checkpoint
   on the exact batch that just enqueued an unprocessed media chunk (the chunk hasn't been
   decoded yet, so its effect isn't reflected in state). It is not a blanket "no checkpoints if
   media ever appeared in this conversation," and does not need to change.

## Approaches considered

- **Rejected - independent content-hash blob store.** Re-hash raw image/audio bytes ourselves and
  store them content-addressed separately. Redundant: `mtmd_bitmap`'s `id` is already a
  content hash of the raw bytes. Doubles storage/hashing logic for no matching-quality benefit.
- **Rejected - position/count-only matching.** Match if the same number of media chunks appear in
  the same token-index slots, without checking content identity. Reintroduces the exact
  false-positive-hit risk the `has_media()` gate currently avoids: silently serving KV state
  computed from a different image. Unacceptable for a cache whose entire purpose is silent reuse.
- **Recommended - chunk-identity-aware hashing/matching, reusing `mtmd_input_chunk_save/load`.**
  Treat each on-disk multimodal cache entry's token/chunk data as a reconstructable
  `server_tokens`, and drive matching through the *same* `server_tokens::get_common_prefix()` the
  RAM cache already uses, instead of hand-rolling a second LCP/equality implementation in the
  disk cache. Minimal new code, all of it composing existing, tested primitives - in line with
  AGENTS.md's "reuse existing infrastructure over introducing new components."

## Design

### 1. On-disk format (`server-disk-cache.h/.cpp`)

- `entry.tokens` stops stripping `LLAMA_TOKEN_NULL`. `save()`'s `new_tokens` computation
  (`server-disk-cache.cpp:468`) keeps the full sequence with placeholders intact. No JSON schema
  change needed for the flat token array itself - `LLAMA_TOKEN_NULL` is `-1`
  (`include/llama.h:39`), already round-trips as a signed int32 through the existing
  `index.json` `"tokens"` field.
- New sidecar file `<hash>.chunks`, parallel to the existing `<hash>.ckpt`: for each
  `(start_idx, chunk)` pair in the saved prompt's media map, call `mtmd_input_chunk_save()` and
  write `{start_idx, blob}` entries. Metadata-only (no pixel/audio bytes), so this stays tiny
  even for prompts with many images. Written/read the same way the `.ckpt` sidecar already is:
  atomic temp-file-then-rename on write, best-effort on read with the entry pruned from the
  index if the file is missing/corrupt.
- Bump `index.json`'s `"version"` field from 1 to 2. On load, a missing/mismatched version means
  the cache directory predates chunk-aware matching: clear it (delete all `.bin`/`.ckpt`/
  `.chunks` files and start with an empty index) rather than writing migration code - this is a
  local, cheaply-rebuilt cache.
- `disk_cache_entry` gains an in-memory (not separately persisted) reconstructed `server_tokens`
  for any entry with media, built once at index-load time by reading that entry's `.chunks`
  sidecar and replaying `push_back()` calls - not rebuilt per lookup.

### 2. Hashing (`hash_tokens`)

When the token sequence has media, walk it the same way `get_common_prefix()` already does: at a
`LLAMA_TOKEN_NULL` position, resolve the chunk via `find_chunk()` and fold
`mtmd_input_chunk_get_id()` into the running SHA-256 instead of the sentinel value, then advance
the loop by the chunk's token count (`mtmd_input_chunk_get_n_tokens()`). Plain text tokens hash
exactly as today.

### 3. Matching (`find_best_prefix`, prune-dedup)

Replace the current flat `std::equal`/index-by-index LCP loops with calls to the reconstructed
entry's `server_tokens::get_common_prefix(task.tokens)`:

- `find_best_prefix()` (`server-disk-cache.cpp:846-893`) uses the returned common-prefix length
  in place of its current byte-for-byte token loop.
- The prefix-supersession pruning in `save()` (`server-disk-cache.cpp:513-545`), which currently
  uses a raw `std::equal` that would incorrectly treat two entries with different images at the
  same null-slot position as identical/superseding, switches to the same `get_common_prefix()`
  based check (an old entry is superseded only if the common prefix with the new entry equals
  the old entry's full length).

### 4. Reconstruction on every restore path

Exact hit (`server-context.cpp:1812`), prefix hit (`server-context.cpp:1829`), and checkpoint
restore (depends on `slot.prompt.tokens.size_up_to_pos()` at `server-context.cpp:3521`, which
needs real `map_idx_to_media` entries to compute correctly across chunk boundaries) all currently
construct the restored prompt via `server_tokens(entry->tokens, has_mtmd)` - the raw-vector
constructor (`server-common.cpp:244`) that stores tokens with **no** backing chunk objects. A
later `find_chunk()` call on such a position throws (`server-common.cpp:345`,
`"Chunk not found"`), which would surface on the very next turn of a cached multimodal
conversation.

Fix: when the entry has media, build the restored `server_tokens` from the entry's
pre-reconstructed placeholder chunks (per section 1) instead of the raw-vector constructor - i.e.
replay the same chunk objects the index already holds in memory, positioned at their recorded
`start_idx`.

### 5. Lift the gates

Remove the four `!tokens.has_media()` / `!ret->prompt.tokens.has_media()` guards in
`server-context.cpp` (lines 1021, 1062, 1794, 1800) now that sections 1-4 make media-aware
matching and reconstruction safe.

Also fix the prefix-hit-ratio calculation at `server-context.cpp:1821`
(`task.tokens.get_text_tokens().size()`), which would badly undercount a prompt containing images
(each worth hundreds to thousands of tokens) when computing `min_prefix_len` - switch to
`tokens.size()`.

### 6. Recurrent/hybrid checkpoints

No code changes beyond sections 1-5 - per the "what already works" analysis above, checkpoint
save/restore rides on the same `server_tokens` bookkeeping and the same
`llama_state_seq_get_data_ext`/`set_data_ext` API, and the replay-from-live-request property means
placeholder chunks are sufficient. The per-iteration `!has_mtmd` suppression at
`server-context.cpp:3755` is unrelated and stays as-is.

This is the priority path for this server (recurrent/hybrid models are the most-used models
here), so it gets explicit test coverage rather than being treated as an incidental side effect:
a multi-turn conversation with an image early on, using a recurrent/hybrid model, cached to disk
and restored mid-conversation via both the plain KV path and a checkpoint rewind, must keep
`size_up_to_pos()` / checkpoint pos-range math correct across the chunk boundary.

### 7. Testing

- Extend `tools/server/tests/unit/test_disk_cache.py` with multimodal fixtures: exact hit,
  prefix hit, and a recurrent/hybrid-model checkpoint-rewind-through-an-image case.
- Manual server smoke test with an actual VLM and, separately, a recurrent/hybrid model with
  vision input, exercising: cold save, warm exact-hit restart, prefix-hit on an edited earlier
  turn, and multi-turn checkpoint rewind.

## Scope notes (from design discussion)

- Fork-only for now: changes stay server-internal (`server-disk-cache.{h,cpp}`,
  `server-context.cpp`); no new public `mtmd.h` API needed since `mtmd_input_chunk_save/load`
  already covers the requirement.
- Old-format disk caches are invalidated (not migrated) on first run after this change, per the
  `index.json` version bump in section 1.
