# Full multimodal disk-cache save/load/match Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the automatic disk-cache system (`tools/server/server-disk-cache.{h,cpp}`) so it saves, loads, and prefix-matches prompts containing images/audio, not just text - including recurrent/hybrid-model checkpoint-based rewind through a media chunk.

**Architecture:** Reuse existing, already-tested primitives instead of inventing new ones: `mtmd_input_chunk_get_id()` (content-hash identity, already used by the RAM prompt cache's `server_tokens::get_common_prefix()`) drives chunk-aware hashing and prefix matching; `mtmd_input_chunk_save()`/`mtmd_input_chunk_load()` (already round-trip chunk identity/shape without pixel data) persist enough per-chunk metadata to a new `<hash>.chunks` sidecar file to reconstruct placeholder chunks on a cache hit. The KV/recurrent-state blob itself needs no changes - `llama_state_seq_get_data_ext`/`set_data_ext` are already modality-agnostic (proven by the RAM cache, which is not gated on `has_media()`).

**Tech Stack:** C++17, nlohmann/json, the `mtmd` C API (`tools/mtmd/mtmd.h`), llama.cpp server (`tools/server/`), pytest server integration tests (`tools/server/tests/`).

**Spec:** `docs/superpowers/specs/2026-08-18-multimodal-disk-cache-design.md`

## Global Constraints

- ASCII-only source: no em-dash `-`, no unicode arrows/multiplication signs; use `-`, `->`, `x` (AGENTS.md).
- Code comments stay to 1-2 lines, explain the non-obvious "why," never restate the code (AGENTS.md).
- Prefer reusing existing infrastructure over new components; this plan must not add invasive new subsystems (AGENTS.md, spec "Approaches considered").
- Fork-only scope: no new public `mtmd.h` API surface. `mtmd_input_chunk_save`/`_load` already cover the requirement (spec "Scope notes").
- Old-format disk caches are invalidated (not migrated) on first run after this change - `index.json` version bump, wipe on mismatch (spec section 1, "Scope notes").
- Do not commit anything unless the user explicitly asks (session-level instruction).

---

## Task 1: `server_tokens::get_media_chunks()` helper

**Files:**
- Modify: `tools/server/server-common.h:187-196` (add declaration after `find_next_media_chunk`)
- Modify: `tools/server/server-common.cpp:348-354` (add definition after `find_next_media_chunk`)

**Interfaces:**
- Produces: `std::vector<std::pair<size_t, const mtmd_input_chunk *>> server_tokens::get_media_chunks() const` - all media chunks in ascending `start_idx` order, non-owning pointers valid for the lifetime of the `server_tokens` object. Used by Task 2's `save()` changes to enumerate what to serialize.

- [ ] **Step 1: Add the declaration**

In `tools/server/server-common.h`, right after this existing method (around line 191):

```cpp
    // find next media chunk after idx
    // returns a pair of pointer to the chunk (nullptr if not found) and its start index in tokens
    std::pair<const mtmd::input_chunk_ptr *, size_t> find_next_media_chunk(size_t idx) const;
```

add:

```cpp
    // list of (start_idx, chunk) for all media chunks, in ascending start_idx order.
    // used to serialize chunk identity/shape (not pixel/audio data) to external storage.
    std::vector<std::pair<size_t, const mtmd_input_chunk *>> get_media_chunks() const;
```

- [ ] **Step 2: Add the implementation**

In `tools/server/server-common.cpp`, right after `find_next_media_chunk`'s definition (around line 354):

```cpp
std::vector<std::pair<size_t, const mtmd_input_chunk *>> server_tokens::get_media_chunks() const {
    std::vector<std::pair<size_t, const mtmd_input_chunk *>> res;
    res.reserve(map_idx_to_media.size());
    for (const auto & it : map_idx_to_media) {
        res.emplace_back(it.first, it.second.get());
    }
    return res;
}
```

- [ ] **Step 3: Build**

Run: `cmake --build build --config Release -j 8 --target llama-server`
Expected: builds with no errors or new warnings.

- [ ] **Step 4: Regression-check the existing disk cache test suite**

Run: `cd tools/server/tests && python3 -m pytest unit/test_disk_cache.py -v`
Expected: all tests PASS (this task adds an unused-by-anyone-yet helper; behavior is unchanged).

- [ ] **Step 5: Commit**

```bash
git add tools/server/server-common.h tools/server/server-common.cpp
git commit -m "server: add server_tokens::get_media_chunks() helper"
```

---

## Task 2: Multimodal-aware disk-cache entry format (sidecar file, `disk_cache_entry` fields, index version bump)

This is the persistence-layer foundation every later task builds on. It adds a new `<hash>.chunks`
sidecar (parallel to the existing `<hash>.ckpt`), extends `disk_cache_entry`, stops stripping
`LLAMA_TOKEN_NULL` placeholders out of the persisted token sequence, and bumps the index format
version so pre-existing text-only caches are cleanly invalidated instead of silently misread.

**Files:**
- Modify: `tools/server/server-disk-cache.h`
- Modify: `tools/server/server-disk-cache.cpp`

**Interfaces:**
- Consumes: `server_tokens::get_media_chunks()` (Task 1).
- Produces:
  - `disk_cache_entry` gains `std::map<size_t, std::vector<uint8_t>> media_chunks` and
    `uint64_t media_size_bytes` fields.
  - `server_tokens server_disk_cache::rebuild_tokens(const std::string & hash) const` (public) -
    reconstructs a `server_tokens` for an index entry, including placeholder media chunks.
    Returns an empty `server_tokens` if the hash isn't found or reconstruction fails. Used by
    Task 4 (`find_best_prefix`) and Task 5 (server-context.cpp restore paths).
  - `server_tokens server_disk_cache::rebuild_tokens_impl(const disk_cache_entry & entry) const`
    (private) - same reconstruction, taking the entry directly (used internally by `save()`'s
    prune-dedup in Task 4, and by `rebuild_tokens`/`find_best_prefix`).

- [ ] **Step 1: Add `<map>` include and the two new `disk_cache_entry` fields**

In `tools/server/server-disk-cache.h`, add `#include <map>` next to the existing
`#include <unordered_map>` (line 7), then extend the struct:

```cpp
struct disk_cache_entry {
    uint64_t size_bytes = 0;
    uint32_t n_tokens = 0;
    int64_t last_used_us = 0;
    std::vector<llama_token> tokens;  // full token sequence; LLAMA_TOKEN_NULL marks a media position

    // size of the "<hash>.ckpt" sidecar file holding serialized context checkpoints
    // (0 if no checkpoints were saved alongside this entry)
    uint64_t ckpt_size_bytes = 0;

    // start_idx -> serialized mtmd_input_chunk blob (mtmd_input_chunk_save output: identity and
    // shape only, no pixel/audio data). Empty for text-only entries.
    std::map<size_t, std::vector<uint8_t>> media_chunks;

    // size of the "<hash>.chunks" sidecar file holding the blobs above (0 if none)
    uint64_t media_size_bytes = 0;
};
```

- [ ] **Step 2: Declare the new methods on `server_disk_cache`**

In `tools/server/server-disk-cache.h`, right after `get_entry()`'s declaration, add:

```cpp
    // Rebuild a server_tokens for the given entry hash, including placeholder media chunks
    // (identity/shape only, no pixel/audio data) reconstructed from persisted chunk blobs.
    // Returns an empty server_tokens if the hash is not found or reconstruction fails.
    server_tokens rebuild_tokens(const std::string & hash) const;
```

and in the `private:` section, next to `hash_tokens`, add:

```cpp
    // Same reconstruction as rebuild_tokens(), taking the entry directly.
    server_tokens rebuild_tokens_impl(const disk_cache_entry & entry) const;

    // Serialize/deserialize the "<hash>.chunks" sidecar file (media chunk blobs keyed by start_idx)
    static bool write_media_chunks_file(const std::string & filepath, const std::map<size_t, std::vector<uint8_t>> & blobs);
    static bool read_media_chunks_file(const std::string & filepath, std::map<size_t, std::vector<uint8_t>> & out_blobs);
```

- [ ] **Step 3: Add the chunk-blob builder, sidecar (de)serializers, and reconstruction, in `server-disk-cache.cpp`**

Add `#include <cstring>` to the top-of-file include block (needed for `strlen` in Task 3, add it
now to keep the include list settled). In the anonymous namespace, right after `append_hex`
(around line 156), add:

```cpp
// Serialize each media chunk's identity/shape (not pixel/audio data) via mtmd_input_chunk_save.
static bool build_media_chunk_blobs(
        const std::vector<std::pair<size_t, const mtmd_input_chunk *>> & chunks,
        std::map<size_t, std::vector<uint8_t>> & out_blobs) {
    out_blobs.clear();
    for (const auto & [start_idx, chunk] : chunks) {
        size_t expected_len = 0;
        if (mtmd_input_chunk_save(chunk, nullptr, 0, &expected_len) != 0) {
            return false;
        }
        std::vector<uint8_t> blob(expected_len);
        if (mtmd_input_chunk_save(chunk, reinterpret_cast<char *>(blob.data()), blob.size(), nullptr) != 0) {
            return false;
        }
        out_blobs[start_idx] = std::move(blob);
    }
    return true;
}
```

Then, right after `read_checkpoints_file`'s definition (the `.ckpt` sidecar reader), add the
`.chunks` sidecar format and its (de)serializers:

```cpp
// Binary format for "<hash>.chunks" sidecar files:
//   uint32_t magic   ("MMCH")
//   uint32_t version (1)
//   uint32_t count
//   repeated `count` times:
//     uint64_t start_idx
//     uint64_t blob_size; uint8_t blob[blob_size]   (mtmd_input_chunk_save output)
static const uint32_t kChunksMagic   = 0x48434d4du; // "MMCH"
static const uint32_t kChunksVersion = 1u;

bool server_disk_cache::write_media_chunks_file(const std::string & filepath, const std::map<size_t, std::vector<uint8_t>> & blobs) {
    std::ofstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        return false;
    }

    const uint32_t count = (uint32_t) blobs.size();
    file.write(reinterpret_cast<const char *>(&kChunksMagic),   sizeof(kChunksMagic));
    file.write(reinterpret_cast<const char *>(&kChunksVersion), sizeof(kChunksVersion));
    file.write(reinterpret_cast<const char *>(&count),          sizeof(count));

    for (const auto & [start_idx, blob] : blobs) {
        const uint64_t idx  = (uint64_t) start_idx;
        const uint64_t size = (uint64_t) blob.size();
        file.write(reinterpret_cast<const char *>(&idx),  sizeof(idx));
        file.write(reinterpret_cast<const char *>(&size), sizeof(size));
        if (size > 0) {
            file.write(reinterpret_cast<const char *>(blob.data()), (std::streamsize) size);
        }
    }

    file.flush();
    const bool ok = !file.fail();
    file.close();
    return ok;
}

bool server_disk_cache::read_media_chunks_file(const std::string & filepath, std::map<size_t, std::vector<uint8_t>> & out_blobs) {
    out_blobs.clear();

    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        return false;
    }

    uint32_t magic = 0, version = 0, count = 0;
    file.read(reinterpret_cast<char *>(&magic),   sizeof(magic));
    file.read(reinterpret_cast<char *>(&version), sizeof(version));
    file.read(reinterpret_cast<char *>(&count),    sizeof(count));
    if (!file || magic != kChunksMagic || version != kChunksVersion) {
        SRV_WRN("disk cache: invalid media chunks file '%s' (magic/version mismatch)\n", filepath.c_str());
        return false;
    }

    for (uint32_t i = 0; i < count; i++) {
        uint64_t idx = 0, size = 0;
        file.read(reinterpret_cast<char *>(&idx),  sizeof(idx));
        file.read(reinterpret_cast<char *>(&size), sizeof(size));
        if (!file) {
            SRV_WRN("disk cache: truncated media chunks file '%s'\n", filepath.c_str());
            out_blobs.clear();
            return false;
        }

        std::vector<uint8_t> blob(size);
        if (size > 0) {
            file.read(reinterpret_cast<char *>(blob.data()), (std::streamsize) size);
            if (!file) {
                SRV_WRN("disk cache: truncated media chunks file '%s'\n", filepath.c_str());
                out_blobs.clear();
                return false;
            }
        }

        out_blobs[(size_t) idx] = std::move(blob);
    }

    return true;
}
```

Then add the reconstruction methods, right after `get_entry()`'s definition:

```cpp
server_tokens server_disk_cache::rebuild_tokens_impl(const disk_cache_entry & entry) const {
    server_tokens st;
    st.has_mtmd = !entry.media_chunks.empty();

    for (size_t i = 0; i < entry.tokens.size(); ) {
        const auto it = entry.media_chunks.find(i);
        if (entry.tokens[i] == LLAMA_TOKEN_NULL && it != entry.media_chunks.end()) {
            mtmd_input_chunk * chunk = mtmd_input_chunk_load(
                    reinterpret_cast<const char *>(it->second.data()), it->second.size());
            if (!chunk) {
                SRV_WRN("%s", "disk cache: failed to reconstruct media chunk, treating entry as unmatched\n");
                return server_tokens();
            }
            st.push_back(chunk);
            const size_t n_tok = mtmd_input_chunk_get_n_tokens(chunk);
            mtmd_input_chunk_free(chunk);
            i += n_tok;
        } else if (entry.tokens[i] == LLAMA_TOKEN_NULL) {
            // malformed entry: null placeholder with no matching chunk blob
            SRV_WRN("%s", "disk cache: media placeholder with no chunk data, treating entry as unmatched\n");
            return server_tokens();
        } else {
            st.push_back(entry.tokens[i]);
            i += 1;
        }
    }

    return st;
}

server_tokens server_disk_cache::rebuild_tokens(const std::string & hash) const {
    auto it = m_index.find(hash);
    if (it == m_index.end()) {
        return server_tokens();
    }
    return rebuild_tokens_impl(it->second);
}
```

- [ ] **Step 4: Stop stripping media placeholders and persist chunk blobs in `save()`**

In `tools/server/server-disk-cache.cpp`, `save()` currently computes:

```cpp
    const llama_tokens new_tokens = (full_tokens && !full_tokens->empty()) ? full_tokens->get_text_tokens() : tokens.get_text_tokens();
```

Replace it with:

```cpp
    const server_tokens & src = (full_tokens && !full_tokens->empty()) ? *full_tokens : tokens;

    llama_tokens new_tokens;
    new_tokens.reserve(src.size());
    for (size_t i = 0; i < src.size(); i++) {
        new_tokens.push_back(src[i]);
    }

    std::map<size_t, std::vector<uint8_t>> media_blobs;
    if (src.has_media()) {
        if (!build_media_chunk_blobs(src.get_media_chunks(), media_blobs)) {
            SRV_WRN("%s", "disk cache: failed to serialize media chunks, saving without media metadata\n");
            media_blobs.clear();
        }
    }
```

Then, where `entry` is populated (right after `entry.last_used_us = ggml_time_us();`, before the
existing checkpoint-writing block), add the `.chunks` sidecar write, mirroring the existing
`.ckpt` block exactly in structure:

```cpp
    if (!media_blobs.empty()) {
        const std::string chunks_filepath     = cache_path(m_path, hash + ".chunks");
        const std::string chunks_tmp_filepath = chunks_filepath + ".tmp";

        if (write_media_chunks_file(chunks_tmp_filepath, media_blobs)) {
            std::error_code chunks_ec;
            fs::rename(chunks_tmp_filepath, chunks_filepath, chunks_ec);
            if (!chunks_ec) {
                std::error_code size_ec;
                const uint64_t chunks_bytes = (uint64_t) fs::file_size(chunks_filepath, size_ec);
                if (!size_ec) {
                    entry.media_size_bytes = chunks_bytes;
                    entry.media_chunks     = media_blobs;
                    m_total_size += chunks_bytes;
                }
            } else {
                SRV_WRN("disk cache: failed to rename media chunks file: %s\n", chunks_ec.message().c_str());
                fs::remove(chunks_tmp_filepath);
            }
        } else {
            SRV_WRN("disk cache: failed to write media chunks file '%s'\n", chunks_tmp_filepath.c_str());
            fs::remove(chunks_tmp_filepath);
        }
    }
```

(Leave the existing prune-dedup `std::equal` block and the rest of `save()` untouched here -
Task 4 fixes prune-dedup's chunk-awareness.)

- [ ] **Step 5: Persist `media_size_bytes` in `index.json`, read `.chunks` back on load**

In `save_index()`, next to `entry_json["ckpt_size_bytes"] = entry.ckpt_size_bytes;`, add:

```cpp
        entry_json["media_size_bytes"] = entry.media_size_bytes;
```

In `load_index()`'s entry-parsing loop, next to
`entry.ckpt_size_bytes = entry_json.value("ckpt_size_bytes", (uint64_t)0);`, add:

```cpp
            entry.media_size_bytes = entry_json.value("media_size_bytes", (uint64_t)0);
```

and, after the existing `tokens` array parsing block (still inside the per-entry loop, before
`m_index[hash] = entry;`), add:

```cpp
            if (entry.media_size_bytes > 0) {
                const std::string chunks_filepath = cache_path(m_path, hash + ".chunks");
                if (!read_media_chunks_file(chunks_filepath, entry.media_chunks)) {
                    SRV_WRN("disk cache: failed to read media chunks file '%s', entry will be treated as text-only\n", chunks_filepath.c_str());
                }
            }
```

and change the size accumulation right below it:

```cpp
            m_index[hash] = entry;
            m_total_size += entry.size_bytes + entry.ckpt_size_bytes + entry.media_size_bytes;
```

- [ ] **Step 6: Mirror `.ckpt` cleanup for `.chunks` in `evict_oldest()` and `validate_and_rebuild()`**

In `evict_oldest()`, next to the existing:

```cpp
    if (oldest_it->second.ckpt_size_bytes > 0) {
        fs::remove(cache_path(m_path, oldest_it->first + ".ckpt"));
    }
```

add:

```cpp
    if (oldest_it->second.media_size_bytes > 0) {
        fs::remove(cache_path(m_path, oldest_it->first + ".chunks"));
    }
```

and update the total-size line right below to include it:

```cpp
    const uint64_t entry_total = oldest_it->second.size_bytes + oldest_it->second.ckpt_size_bytes + oldest_it->second.media_size_bytes;
```

In `validate_and_rebuild()`:
- where it removes files for a missing index entry (next to the existing
  `if (it->second.ckpt_size_bytes > 0) { fs::remove(...".ckpt"); }`), add the same for `.chunks`:

```cpp
            if (it->second.media_size_bytes > 0) {
                fs::remove(cache_path(m_path, it->first + ".chunks"));
            }
```

- where it drops stale `ckpt_size_bytes` bookkeeping for a still-present entry (next to
  `if (it->second.ckpt_size_bytes > 0 && !fs::exists(cache_path(m_path, it->first + ".ckpt"))) { it->second.ckpt_size_bytes = 0; }`),
  add:

```cpp
            if (it->second.media_size_bytes > 0 && !fs::exists(cache_path(m_path, it->first + ".chunks"))) {
                it->second.media_size_bytes = 0;
                it->second.media_chunks.clear();
            }
```

- in the total-size rebuild loop, change:

```cpp
    for (const auto& [hash, entry] : m_index) {
        m_total_size += entry.size_bytes + entry.ckpt_size_bytes + entry.media_size_bytes;
    }
```

- [ ] **Step 7: Bump the index format version and wipe stale caches on mismatch**

In the anonymous namespace, next to `kIndexFilename`/`kStatsFilename`, add:

```cpp
static const uint32_t kIndexFormatVersion = 2; // bumped for multimodal-aware entries (chunks sidecar)
```

Declare a new private method in `server-disk-cache.h`, next to `load_index()`:

```cpp
    // Delete all files in the cache directory and reset in-memory state (used when the on-disk
    // index format is stale, e.g. predates multimodal-aware entries).
    void wipe_cache_dir();
```

Implement it in `server-disk-cache.cpp`, right before `load_index()`:

```cpp
void server_disk_cache::wipe_cache_dir() {
    std::error_code ec;
    for (const auto & dir_entry : fs::directory_iterator(m_path, ec)) {
        fs::remove(dir_entry.path(), ec);
    }
    m_index.clear();
    m_total_size = 0;
}
```

In `save_index()`, change `j["version"] = 1;` to `j["version"] = kIndexFormatVersion;`.

In `load_index()`, right after the JSON parse succeeds (after the `try { j = nlohmann::json::parse(file); } catch (...) { ... }` block, before `m_index.clear(); m_total_size = 0;`), add:

```cpp
    const uint32_t file_version = j.value("version", (uint32_t)0);
    if (file_version != kIndexFormatVersion) {
        SRV_WRN("disk cache: index format version %u != %u (multimodal-aware format), clearing stale cache\n",
                file_version, kIndexFormatVersion);
        wipe_cache_dir();
        return true;
    }
```

- [ ] **Step 8: Build**

Run: `cmake --build build --config Release -j 8 --target llama-server`
Expected: builds with no errors. Fix any signature mismatches against Task 1's
`get_media_chunks()` before moving on.

- [ ] **Step 9: Regression-check the existing disk cache test suite**

Run: `cd tools/server/tests && python3 -m pytest unit/test_disk_cache.py -v`
Expected: all tests PASS. Text-only saves never populate `media_blobs`, so `.chunks` is never
written and behavior for the existing test is unchanged; the version bump only affects
directories written by a pre-change binary, which the test suite creates fresh each run.

- [ ] **Step 10: Commit**

```bash
git add tools/server/server-disk-cache.h tools/server/server-disk-cache.cpp
git commit -m "server: persist media chunk metadata in disk cache entries (format v2)"
```

---

## Task 3: Chunk-identity-aware `hash_tokens()`

**Files:**
- Modify: `tools/server/server-disk-cache.cpp` (`hash_tokens`, around line 214)

**Interfaces:**
- Consumes: `server_tokens::find_chunk()`, `mtmd_input_chunk_get_id()`,
  `mtmd_input_chunk_get_n_tokens()` (all pre-existing).
- Produces: `hash_tokens()`'s behavior for media-containing token sequences changes from
  "silently ignore media" to "fold each chunk's content-hash id into the digest." Its signature
  is unchanged (`static std::string hash_tokens(const server_tokens& tokens)`), so `save()`/
  `load()`/`load_by_hash()` callers need no changes.

- [ ] **Step 1: Replace the token-hashing loop**

In `tools/server/server-disk-cache.cpp`, `hash_tokens()` currently does:

```cpp
    // Hash each token as 4-byte little-endian
    const auto tok_vec = tokens.get_text_tokens();
    for (auto tok : tok_vec) {
        uint8_t buf[4];
        buf[0] = (uint8_t)(tok & 0xFF);
        buf[1] = (uint8_t)((tok >> 8) & 0xFF);
        buf[2] = (uint8_t)((tok >> 16) & 0xFF);
        buf[3] = (uint8_t)((tok >> 24) & 0xFF);
        embed_sha256_update(&ctx, buf, 4);
    }
```

Replace it with:

```cpp
    // Hash each text token as 4-byte little-endian; hash a media chunk by its content-derived
    // id instead (mtmd_input_chunk_get_id, an FNV hash of the raw bytes) so two different images
    // never collide just because they sit at the same position with the same surrounding text.
    auto hash_u32 = [&ctx](uint32_t v) {
        uint8_t buf[4];
        buf[0] = (uint8_t)(v & 0xFF);
        buf[1] = (uint8_t)((v >> 8) & 0xFF);
        buf[2] = (uint8_t)((v >> 16) & 0xFF);
        buf[3] = (uint8_t)((v >> 24) & 0xFF);
        embed_sha256_update(&ctx, buf, 4);
    };

    for (size_t i = 0; i < tokens.size(); ) {
        const llama_token tok = tokens[i];
        if (tok == LLAMA_TOKEN_NULL) {
            const auto & chunk = tokens.find_chunk(i);
            const char * id = mtmd_input_chunk_get_id(chunk.get());
            if (id) {
                embed_sha256_update(&ctx, reinterpret_cast<const uint8_t *>(id), strlen(id));
            }
            i += mtmd_input_chunk_get_n_tokens(chunk.get());
        } else {
            hash_u32((uint32_t) tok);
            i += 1;
        }
    }
```

- [ ] **Step 2: Build**

Run: `cmake --build build --config Release -j 8 --target llama-server`
Expected: builds with no errors (needs `#include <cstring>`, already added in Task 2 step 3).

- [ ] **Step 3: Regression-check**

Run: `cd tools/server/tests && python3 -m pytest unit/test_disk_cache.py -v`
Expected: all tests PASS - text-only prompts never hit the `LLAMA_TOKEN_NULL` branch, so hashes
are unchanged for them.

- [ ] **Step 4: Commit**

```bash
git add tools/server/server-disk-cache.cpp
git commit -m "server: make disk-cache hashing chunk-identity-aware for media prompts"
```

---

## Task 4: Chunk-aware `find_best_prefix()` and prune-dedup

**Files:**
- Modify: `tools/server/server-disk-cache.cpp` (`find_best_prefix`, around line 846; `save()`'s
  prune loop, around line 513)

**Interfaces:**
- Consumes: `rebuild_tokens_impl()` (Task 2), `server_tokens::get_common_prefix()` (pre-existing,
  `server-common.cpp:471`).
- Produces: `find_best_prefix()`'s signature is unchanged. Prune-dedup in `save()` now correctly
  distinguishes two entries that share a token-index layout but differ in media content.

- [ ] **Step 1: Replace `find_best_prefix()`'s matching loop**

Replace the whole function body (from `if (!m_enabled || tokens.empty())` through the closing
`return best_hash;`) with:

```cpp
std::string server_disk_cache::find_best_prefix(const server_tokens & tokens, size_t min_prefix_len) const {
    if (!m_enabled || tokens.empty()) {
        return "";
    }

    size_t best_len = min_prefix_len > 0 ? min_prefix_len - 1 : 0;
    std::string best_hash;

    // Tracked purely for diagnostics: the closest match seen, even if it falls
    // short of min_prefix_len (so a miss can be explained: "closest was X%, needed Y%").
    size_t closest_len = 0;
    std::string closest_hash;

    for (const auto & [hash, entry] : m_index) {
        if (entry.tokens.empty()) {
            continue;  // old-format entry without token data
        }

        const server_tokens entry_tokens = rebuild_tokens_impl(entry);
        if (entry_tokens.empty()) {
            continue;  // malformed/unreadable media entry
        }

        const size_t lcp = entry_tokens.get_common_prefix(tokens);

        if (lcp > closest_len) {
            closest_len = lcp;
            closest_hash = hash;
        }
        if (lcp > best_len) {
            best_len = lcp;
            best_hash = hash;
        }
    }

    const size_t n_req = tokens.size();
    const double req_pct = 100.0 * min_prefix_len / std::max((size_t)1, n_req);
    if (!best_hash.empty()) {
        SRV_INF("disk cache: prefix match, hash=%.8s..., lcp=%zu/%zu tokens (%.1f%%, required >= %zu = %.1f%%)\n",
                best_hash.c_str(), best_len, n_req,
                100.0 * best_len / std::max((size_t)1, n_req), min_prefix_len, req_pct);
    } else {
        SRV_INF("disk cache: no prefix match for %zu tokens (required >= %zu = %.1f%%, %zu entries in index, closest lcp=%zu tokens = %.1f%% from hash=%.8s...)\n",
                n_req, min_prefix_len, req_pct, m_index.size(),
                closest_len, 100.0 * closest_len / std::max((size_t)1, n_req),
                closest_hash.empty() ? "none" : closest_hash.c_str());
    }

    return best_hash;
}
```

Note the perf tradeoff versus the old flat-array comparison: this now reconstructs a
`server_tokens` per candidate entry per search call (cheap for text-only entries - no
`mtmd_input_chunk_load` calls at all - and metadata-only, so still cheap for media entries, but
no longer a bare integer-array scan). Acceptable for a cache whose lookup happens once per
incoming request, not per token.

- [ ] **Step 2: Fix prune-dedup in `save()` to be chunk-aware**

Replace the existing block:

```cpp
    size_t n_pruned = 0;
    for (auto it2 = m_index.begin(); it2 != m_index.end();) {
        const disk_cache_entry & old_entry = it2->second;
        if (!old_entry.tokens.empty() &&
            old_entry.tokens.size() < new_tokens.size() &&
            std::equal(old_entry.tokens.begin(), old_entry.tokens.end(), new_tokens.begin())) {
            fs::remove(cache_path(m_path, it2->first + ".bin"));
            if (old_entry.ckpt_size_bytes > 0) {
                fs::remove(cache_path(m_path, it2->first + ".ckpt"));
            }
            const uint64_t old_total = old_entry.size_bytes + old_entry.ckpt_size_bytes;
            m_total_size = (old_total <= m_total_size) ? (m_total_size - old_total) : 0;
            it2 = m_index.erase(it2);
            n_pruned++;
        } else {
            ++it2;
        }
    }
```

with:

```cpp
    size_t n_pruned = 0;
    for (auto it2 = m_index.begin(); it2 != m_index.end();) {
        const disk_cache_entry & old_entry = it2->second;

        bool superseded = false;
        if (!old_entry.tokens.empty() && old_entry.tokens.size() < new_tokens.size()) {
            const server_tokens old_reconstructed = rebuild_tokens_impl(old_entry);
            if (!old_reconstructed.empty() &&
                old_reconstructed.get_common_prefix(src) == old_entry.tokens.size()) {
                superseded = true;
            }
        }

        if (superseded) {
            fs::remove(cache_path(m_path, it2->first + ".bin"));
            if (old_entry.ckpt_size_bytes > 0) {
                fs::remove(cache_path(m_path, it2->first + ".ckpt"));
            }
            if (old_entry.media_size_bytes > 0) {
                fs::remove(cache_path(m_path, it2->first + ".chunks"));
            }
            const uint64_t old_total = old_entry.size_bytes + old_entry.ckpt_size_bytes + old_entry.media_size_bytes;
            m_total_size = (old_total <= m_total_size) ? (m_total_size - old_total) : 0;
            it2 = m_index.erase(it2);
            n_pruned++;
        } else {
            ++it2;
        }
    }
```

(`src` is the `const server_tokens &` introduced in Task 2 step 4 - already in scope at this
point in `save()`.)

- [ ] **Step 3: Build**

Run: `cmake --build build --config Release -j 8 --target llama-server`
Expected: builds with no errors.

- [ ] **Step 4: Regression-check**

Run: `cd tools/server/tests && python3 -m pytest unit/test_disk_cache.py -v`
Expected: all tests PASS. For text-only entries, `rebuild_tokens_impl` degenerates to a plain
token-vector rebuild and `get_common_prefix` takes its non-mtmd fast path
(`server-common.cpp:474-484`), so prefix matching and pruning behave identically to before.

- [ ] **Step 5: Commit**

```bash
git add tools/server/server-disk-cache.cpp
git commit -m "server: make disk-cache prefix matching and prune-dedup chunk-aware"
```

---

## Task 5: Lift the multimodal gates in `server-context.cpp`

This is the integration task: it turns on everything built in Tasks 1-4 for real requests, fixes
the restored-prompt reconstruction so a cached multimodal prompt doesn't crash on its next turn,
and fixes the prefix-hit-ratio calculation. It also adds the first end-to-end multimodal tests,
since this is the first point at which the feature is reachable through the server.

**Files:**
- Modify: `tools/server/server-context.cpp` (lines ~1021, ~1062, ~1794-1839)
- Test: `tools/server/tests/unit/test_disk_cache.py` (new tests)

**Interfaces:**
- Consumes: `server_disk_cache::rebuild_tokens()` (Task 2), `server_tokens::clone()`
  (pre-existing, `server-common.h:236`).

- [ ] **Step 1: Lift the gate in `flush_disk_cache()`**

Replace:

```cpp
    void flush_disk_cache() {
        if (!disk_cache || !ctx_tgt) {
            return;
        }
        for (auto & slot : slots) {
            if (slot.prompt.n_tokens() > 0 && !slot.prompt.tokens.has_media()) {
                disk_cache_save_slot(slot);
            }
        }
        disk_cache_flushed = true;
    }
```

with:

```cpp
    void flush_disk_cache() {
        if (!disk_cache || !ctx_tgt) {
            return;
        }
        for (auto & slot : slots) {
            if (slot.prompt.n_tokens() > 0) {
                disk_cache_save_slot(slot);
            }
        }
        disk_cache_flushed = true;
    }
```

- [ ] **Step 2: Lift the gate in `slot_save_and_clear()`**

Replace:

```cpp
        // Save to disk cache before clearing (tokens are needed for hashing)
        // Skip disk cache when tokens contain media (LLAMA_TOKEN_NULL placeholders)
        if (disk_cache && !slot.prompt.tokens.has_media()) {
            saved = disk_cache_save_slot(slot);
        }
```

with:

```cpp
        // Save to disk cache before clearing (tokens are needed for hashing)
        if (disk_cache) {
            saved = disk_cache_save_slot(slot);
        }
```

- [ ] **Step 3: Lift the gates and fix reconstruction in `get_available_slot()`**

Replace the whole block:

```cpp
                // Also save to disk cache if enabled (skip when tokens contain media)
                if (tokens.size() > 0 && disk_cache && !ret->prompt.tokens.has_media()) {
                    disk_cache_save_slot(*ret);
                }

                // Try loading: disk cache first, then RAM prompt cache (skip when tokens contain media)
                bool loaded = false;
                if (disk_cache && !task.tokens.has_media()) {
                    std::string exact_hash;
                    std::list<common_prompt_checkpoint> loaded_checkpoints;
                    loaded = disk_cache->load(task.tokens, ctx_tgt, ret->id, &exact_hash, &loaded_checkpoints);
                    if (loaded) {
                        // Disk cache restores KV state but does not set prompt.tokens.
                        // We must set it here so that update_slots() can compute n_past correctly.
                        // Use the entry's full token sequence (prompt + any generated
                        // continuation) so prompt.tokens stays consistent with the
                        // restored memory's pos_max (entry.tokens is a superset of
                        // task.tokens for an exact hash hit).
                        const disk_cache_entry * entry = disk_cache->get_entry(exact_hash);
                        ret->prompt.tokens = server_tokens(
                                (entry && !entry->tokens.empty()) ? entry->tokens : task.tokens.get_tokens(),
                                task.tokens.has_mtmd);
                        ret->prompt.checkpoints = std::move(loaded_checkpoints);
                    }

                    // Exact miss - try prefix search. Require at least 50% of the
                    // request tokens to be covered by the cached prefix.
                    if (!loaded) {
                        const size_t n_req = task.tokens.get_text_tokens().size();
                        const size_t min_prefix_len = std::max((size_t)1, n_req / 2);
                        const std::string prefix_hash = disk_cache->find_best_prefix(task.tokens, min_prefix_len);
                        if (!prefix_hash.empty()) {
                            loaded = disk_cache->load_by_hash(prefix_hash, ctx_tgt, ret->id, &loaded_checkpoints);
                            if (loaded) {
                                const disk_cache_entry * entry = disk_cache->get_entry(prefix_hash);
                                if (entry) {
                                    ret->prompt.tokens = server_tokens(entry->tokens, task.tokens.has_mtmd);
                                }
                                ret->prompt.checkpoints = std::move(loaded_checkpoints);
                                SLT_TRC(*ret, "disk prefix-hit, restored state; prompt.tokens=%d (req=%zu), checkpoints=%zu, mem pos_min=%d pos_max=%d\n",
                                        ret->prompt.n_tokens(), task.tokens.get_text_tokens().size(), ret->prompt.checkpoints.size(),
                                        llama_memory_seq_pos_min(llama_get_memory(ctx_tgt), ret->id),
                                        llama_memory_seq_pos_max(llama_get_memory(ctx_tgt), ret->id));
                            }
                        }
                    }
                }
```

with:

```cpp
                // Also save to disk cache if enabled
                if (tokens.size() > 0 && disk_cache) {
                    disk_cache_save_slot(*ret);
                }

                // Try loading: disk cache first, then RAM prompt cache
                bool loaded = false;
                if (disk_cache) {
                    std::string exact_hash;
                    std::list<common_prompt_checkpoint> loaded_checkpoints;
                    loaded = disk_cache->load(task.tokens, ctx_tgt, ret->id, &exact_hash, &loaded_checkpoints);
                    if (loaded) {
                        // Disk cache restores KV state but does not set prompt.tokens.
                        // We must set it here so that update_slots() can compute n_past correctly.
                        // rebuild_tokens() reconstructs media chunks (identity/shape only) so a
                        // later get_common_prefix()/find_chunk() call on this restored prompt
                        // doesn't crash on the very next turn of a cached multimodal conversation.
                        const disk_cache_entry * entry = disk_cache->get_entry(exact_hash);
                        ret->prompt.tokens = (entry && !entry->tokens.empty())
                                ? disk_cache->rebuild_tokens(exact_hash)
                                : task.tokens.clone();
                        ret->prompt.checkpoints = std::move(loaded_checkpoints);
                    }

                    // Exact miss - try prefix search. Require at least 50% of the
                    // request tokens to be covered by the cached prefix.
                    if (!loaded) {
                        const size_t n_req = task.tokens.size();
                        const size_t min_prefix_len = std::max((size_t)1, n_req / 2);
                        const std::string prefix_hash = disk_cache->find_best_prefix(task.tokens, min_prefix_len);
                        if (!prefix_hash.empty()) {
                            loaded = disk_cache->load_by_hash(prefix_hash, ctx_tgt, ret->id, &loaded_checkpoints);
                            if (loaded) {
                                ret->prompt.tokens = disk_cache->rebuild_tokens(prefix_hash);
                                ret->prompt.checkpoints = std::move(loaded_checkpoints);
                                SLT_TRC(*ret, "disk prefix-hit, restored state; prompt.tokens=%d (req=%zu), checkpoints=%zu, mem pos_min=%d pos_max=%d\n",
                                        ret->prompt.n_tokens(), task.tokens.size(), ret->prompt.checkpoints.size(),
                                        llama_memory_seq_pos_min(llama_get_memory(ctx_tgt), ret->id),
                                        llama_memory_seq_pos_max(llama_get_memory(ctx_tgt), ret->id));
                            }
                        }
                    }
                }
```

(`n_req` switched from `task.tokens.get_text_tokens().size()` to `task.tokens.size()` so images -
worth hundreds/thousands of tokens each - aren't dropped from the prefix-hit-ratio requirement.)

- [ ] **Step 4: Build**

Run: `cmake --build build --config Release -j 8 --target llama-server`
Expected: builds with no errors.

- [ ] **Step 5: Regression-check the existing (text-only) disk cache tests**

Run: `cd tools/server/tests && python3 -m pytest unit/test_disk_cache.py -v`
Expected: all tests PASS.

- [ ] **Step 6: Write the new multimodal disk-cache tests**

Append to `tools/server/tests/unit/test_disk_cache.py`:

```python
IMG_URL_TRUCK = "https://huggingface.co/ggml-org/tinygemma3-GGUF/resolve/main/test/11_truck.png"
IMG_URL_CAT = "https://huggingface.co/ggml-org/tinygemma3-GGUF/resolve/main/test/91_cat.png"


def _vision_server(cache_dir):
    s = ServerPreset.tinygemma3()
    s.n_slots = 1
    s.n_predict = 4
    s.temperature = 0.0
    s.cache_disk = cache_dir
    s.cache_disk_size = -1
    return s


def test_disk_cache_hit_multimodal_after_restart():
    """Exact hit: same image + same text, after a server restart, should reprocess fewer tokens."""
    cache_dir = tempfile.mkdtemp(prefix="llama_disk_cache_mm_")
    try:
        server_mm = _vision_server(cache_dir)
        server_mm.start()

        req = {
            "prompt": {
                "prompt_string": "What is this: <__media__>\n",
                "multimodal_data": [IMG_URL_TRUCK],
            },
        }
        res = server_mm.make_request("POST", "/completions", data=req)
        assert res.status_code == 200
        first_prompt_n = res.body["timings"]["prompt_n"]
        assert first_prompt_n > 0

        server_mm.stop()

        server_mm2 = _vision_server(cache_dir)
        server_mm2.start()
        res2 = server_mm2.make_request("POST", "/completions", data=req)
        assert res2.status_code == 200
        assert res2.body["timings"]["prompt_n"] < first_prompt_n, (
            f"expected disk cache hit: prompt_n={res2.body['timings']['prompt_n']} should be < {first_prompt_n}"
        )
    finally:
        shutil.rmtree(cache_dir, ignore_errors=True)


def test_disk_cache_multimodal_distinguishes_different_images():
    """Same surrounding text, different images: must NOT be treated as the same cache entry."""
    cache_dir = tempfile.mkdtemp(prefix="llama_disk_cache_mm_")
    try:
        server_mm = _vision_server(cache_dir)
        server_mm.start()

        req_truck = {
            "prompt": {
                "prompt_string": "What is this: <__media__>\n",
                "multimodal_data": [IMG_URL_TRUCK],
            },
        }
        req_cat = {
            "prompt": {
                "prompt_string": "What is this: <__media__>\n",
                "multimodal_data": [IMG_URL_CAT],
            },
        }

        res_truck = server_mm.make_request("POST", "/completions", data=req_truck)
        assert res_truck.status_code == 200

        res_cat = server_mm.make_request("POST", "/completions", data=req_cat)
        assert res_cat.status_code == 200
        # a wrong cache hit would reprocess ~0 tokens for the second (different) image
        assert res_cat.body["timings"]["prompt_n"] > 0
    finally:
        shutil.rmtree(cache_dir, ignore_errors=True)


def test_disk_cache_prefix_hit_multimodal_edited_turn():
    """Prefix hit: same image, edited trailing text, should still reuse the image's cached prefix."""
    cache_dir = tempfile.mkdtemp(prefix="llama_disk_cache_mm_")
    try:
        server_mm = _vision_server(cache_dir)
        server_mm.start()

        base_req = {
            "prompt": {
                "prompt_string": "What is this: <__media__>\nDescribe it in one word.\n",
                "multimodal_data": [IMG_URL_TRUCK],
            },
        }
        res = server_mm.make_request("POST", "/completions", data=base_req)
        assert res.status_code == 200

        server_mm.stop()

        server_mm2 = _vision_server(cache_dir)
        server_mm2.start()

        edited_req = {
            "prompt": {
                "prompt_string": "What is this: <__media__>\nName the color.\n",
                "multimodal_data": [IMG_URL_TRUCK],
            },
        }
        res2 = server_mm2.make_request("POST", "/completions", data=edited_req)
        assert res2.status_code == 200
        # the image prefix should be reused; only the edited trailing text is new
        assert res2.body["timings"]["prompt_n"] < res.body["timings"]["prompt_n"] + 5
    finally:
        shutil.rmtree(cache_dir, ignore_errors=True)
```

- [ ] **Step 7: Run the new tests**

Run: `cd tools/server/tests && python3 -m pytest unit/test_disk_cache.py -v -k multimodal`
Expected: all three new tests PASS. `test_disk_cache_hit_multimodal_after_restart` proves the
exact-hit path; `test_disk_cache_multimodal_distinguishes_different_images` proves the
correctness property the whole design exists to guarantee; `test_disk_cache_prefix_hit_multimodal_edited_turn`
proves the chunk-aware LCP path. If any fails, do not move on - re-check Tasks 2-4 against the
failure before touching Task 6, since this is the first point the whole chain is exercised
together.

- [ ] **Step 8: Full disk cache regression pass**

Run: `cd tools/server/tests && python3 -m pytest unit/test_disk_cache.py -v`
Expected: all tests (existing text-only + new multimodal) PASS.

- [ ] **Step 9: Commit**

```bash
git add tools/server/server-context.cpp tools/server/tests/unit/test_disk_cache.py
git commit -m "server: enable disk cache for multimodal prompts"
```

---

## Task 6: Checkpoint test-harness support and recurrent/hybrid verification

Recurrent/hybrid models are this server's most-used models, so the checkpoint-rewind path is the
priority to validate - but no tiny recurrent/hybrid model with vision support exists in this
repo's test fixtures (`ServerPreset` has no recurrent/hybrid or combined vision+recurrent
preset), so it can't be covered by an automated pytest the way Task 5's transformer/attention
case was. This task adds the missing test-harness plumbing for context checkpoints (useful for
future automated coverage once/if such a fixture exists) and a concrete manual verification
procedure against the user's real hybrid VLM.

**Files:**
- Modify: `tools/server/tests/utils.py`

**Interfaces:**
- Produces: `ServerProcess.ctx_checkpoints: int | None` and
  `ServerProcess.checkpoint_min_step: int | None`, wired to `--ctx-checkpoints` /
  `--checkpoint-min-step`.

- [ ] **Step 1: Add the two new `ServerProcess` fields**

In `tools/server/tests/utils.py`, next to the existing `cache_disk_size: int | None = None`
field (line 113), add:

```python
    ctx_checkpoints: int | None = None
    checkpoint_min_step: int | None = None
```

- [ ] **Step 2: Wire them into `server_args`**

Right after the existing block:

```python
        if self.cache_disk_size is not None:
            server_args.extend(["--cache-disk-size", self.cache_disk_size])
```

add:

```python
        if self.ctx_checkpoints is not None:
            server_args.extend(["--ctx-checkpoints", self.ctx_checkpoints])
        if self.checkpoint_min_step is not None:
            server_args.extend(["--checkpoint-min-step", self.checkpoint_min_step])
```

- [ ] **Step 3: Build and run the full server test suite once to confirm no regressions from the harness change**

Run: `cmake --build build --config Release -j 8 --target llama-server`
Run: `cd tools/server/tests && python3 -m pytest unit/test_disk_cache.py unit/test_vision_api.py unit/test_slot_save.py -v`
Expected: all PASS (this step only adds two optional fields defaulting to `None`, so no existing
test's generated `server_args` changes).

- [ ] **Step 4: Commit the harness change**

```bash
git add tools/server/tests/utils.py
git commit -m "server tests: expose --ctx-checkpoints/--checkpoint-min-step in ServerProcess"
```

- [ ] **Step 5: Manual verification - recurrent/hybrid model, vision, checkpoint rewind**

Run this against the actual recurrent/hybrid VLM this server uses (substitute the real
`--model`/`--mmproj` paths and image path). The goal is to exercise exactly the path the spec's
section 6 flags as needing explicit coverage: a multi-turn conversation with an image early on,
restored from disk cache mid-conversation via checkpoint rewind, with `size_min`/`size_max`
bookkeeping intact across the chunk boundary.

1. Start the server with disk cache and checkpoints both enabled:

   ```bash
   ./build/bin/llama-server \
     --model /path/to/hybrid-vlm.gguf \
     --mmproj /path/to/hybrid-vlm-mmproj.gguf \
     --ctx-size 8192 \
     --ctx-checkpoints 8 \
     --checkpoint-min-step 256 \
     --cache-disk /tmp/mm-disk-cache-verify \
     --cache-disk-size -1 \
     --verbose
   ```

2. Send turn 1 (image + question) via `/chat/completions`:

   ```bash
   curl -s http://127.0.0.1:8080/v1/chat/completions -d '{
     "messages": [{"role": "user", "content": [
       {"type": "text", "text": "What is in this image?"},
       {"type": "image_url", "image_url": {"url": "data:image/png;base64,<BASE64_IMAGE>"}}
     ]}]
   }' | tee /tmp/turn1.json
   ```

   Confirm in the server log: a line starting `disk cache: saved` after the response, with
   `n_pruned` absent or 0 (first save, nothing to prune yet).

3. Send turn 2, re-sending the same image plus the assistant's turn-1 reply plus a new question
   (standard multi-turn chat-completions shape - the full history, including the image, goes in
   `messages` again):

   ```bash
   curl -s http://127.0.0.1:8080/v1/chat/completions -d '{
     "messages": [
       {"role": "user", "content": [
         {"type": "text", "text": "What is in this image?"},
         {"type": "image_url", "image_url": {"url": "data:image/png;base64,<BASE64_IMAGE>"}}
       ]},
       {"role": "assistant", "content": "<PASTE_TURN_1_ASSISTANT_REPLY_HERE>"},
       {"role": "user", "content": "Now describe its color in one word."}
     ]
   }' | tee /tmp/turn2.json
   ```

   Confirm in the log: `disk cache: exact hit` or `disk cache: prefix match` followed by a
   successful `llama_state_seq_set_data_ext` restore (no `partial restore` warning), and no
   `Chunk not found` exception / crash.

4. Restart the server (kill and re-run the same command from step 1) to force turn 3's request
   through a cold slot, so restoration must go through the disk cache's checkpoint path rather
   than an in-memory slot that never left. Send turn 3 with the full 3-message history (turns 1-2
   plus a new question), same shape as step 3.

   Confirm in the log:
   - `disk cache: prefix match` / `disk cache: exact hit` (not a full cold reprocess - check
     `timings.prompt_n` in the response is much smaller than the full prompt's token count).
   - `restored context checkpoint` (proves the checkpoint-rewind path specifically engaged, not
     just the plain KV-state path - this is the recurrent/hybrid-specific case).
   - No `GGML_ABORT`, no `pos_min == -1` error, no `Chunk not found` exception.
   - The final assistant reply is coherent with respect to the *original* image (i.e. still
     describes the truck/cat/whatever was actually sent - proves the reconstructed placeholder
     chunk's identity correctly matched, not some other cached image).

5. Record the outcome (pass/fail per checklist item above) - if any item fails, it points at
   which task (2-5) to re-examine: a `Chunk not found` failure means Task 2's reconstruction or
   Task 5's `rebuild_tokens()` wiring; wrong-image content in the reply means Task 3's hashing;
   `pos_min`/checkpoint restore failures mean the interaction described in the spec's section 6
   needs an actual code change after all (report back before proceeding further).

---

## Self-Review Notes

- **Spec coverage:** Section 1 (format) -> Task 2. Section 2 (hashing) -> Task 3. Section 3
  (matching/prune-dedup) -> Task 4. Section 4 (reconstruction) -> Task 2 (`rebuild_tokens`) +
  Task 5 (call sites). Section 5 (lift gates, min_prefix_len fix) -> Task 5. Section 6
  (recurrent/hybrid) -> Task 6 (no code change needed per the spec's own analysis; verified
  manually since no test fixture exists). Section 7 (testing) -> Task 5 steps 6-7 (automated) +
  Task 6 step 5 (manual, recurrent/hybrid case). All spec sections are covered.
- **Placeholder scan:** no TBD/TODO markers; every step has runnable code or a concrete,
  executable manual procedure (Task 6 step 5 is manual by necessity, not by omission - explained
  inline why no automated fixture exists).
- **Type consistency:** `rebuild_tokens(const std::string & hash) const` (Task 2) is called
  identically in Task 4 (`find_best_prefix`, via the private `rebuild_tokens_impl` overload taking
  an entry) and Task 5 (via the public hash-taking overload) - signatures match their call sites.
  `get_media_chunks()` (Task 1) returns `std::vector<std::pair<size_t, const mtmd_input_chunk *>>`,
  consumed as-is by `build_media_chunk_blobs()` (Task 2) with no intermediate conversion.
