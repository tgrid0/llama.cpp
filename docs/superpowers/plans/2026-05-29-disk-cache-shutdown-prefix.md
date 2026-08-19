# Disk Cache: Shutdown Save & Prefix Search Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Save all active slot KV states to disk when the server shuts down (or sleeps), and add prefix-match lookup so a new request can load the longest cached prefix from disk instead of only exact matches.

**Architecture:** Extend `disk_cache_entry` with the full token array so prefix matching is possible; add `find_best_prefix` / `load_by_hash` / `get_entry` to `server_disk_cache`; call the shutdown flush inside `destroy()` before the llama context is freed; and extend the `get_available_slot()` cache-miss path to try prefix lookup.

**Tech Stack:** C++17, nlohmann/json (already used for index.json), existing `llama_state_seq_*` API for KV state I/O, Python/pytest for integration tests.

**Spec:** `docs/superpowers/specs/2026-05-29-disk-cache-shutdown-prefix-design.md`

---

## File Map

| File | Change |
|------|--------|
| `tools/server/server-disk-cache.h` | Add `tokens` field to `disk_cache_entry`; declare `find_best_prefix`, `load_by_hash`, `get_entry` |
| `tools/server/server-disk-cache.cpp` | Implement new methods; update `save`, `load_index`, `save_index` to handle `tokens` field |
| `tools/server/server-context.cpp` | Add shutdown flush in `destroy()`; add prefix search in `get_available_slot()` |
| `tools/server/tests/utils.py` | Add `slot_prompt_similarity` field + CLI wiring |
| `tools/server/tests/unit/test_disk_cache.py` | Add two new tests: shutdown save, prefix search |

---

### Task 1: Add `tokens` field to `disk_cache_entry` and update index serialisation

**Files:**
- Modify: `tools/server/server-disk-cache.h`
- Modify: `tools/server/server-disk-cache.cpp`

This adds the token array to the struct and keeps index.json backward compatible (old entries without `"tokens"` load fine; prefix search just skips them).

- [ ] **Step 1: Add `tokens` field to `disk_cache_entry` in the header**

In `tools/server/server-disk-cache.h`, change the struct:

```cpp
struct disk_cache_entry {
    uint64_t size_bytes = 0;
    uint32_t n_tokens = 0;
    int64_t last_used_us = 0;
    std::vector<llama_token> tokens;  // full token sequence; empty for old-format entries
};
```

- [ ] **Step 2: Update `save_index()` in `server-disk-cache.cpp` to write tokens**

Find the loop that serialises entries (around line 621) and add the tokens field:

```cpp
for (const auto& [hash, entry] : m_index) {
    nlohmann::json entry_json;
    entry_json["size_bytes"]    = entry.size_bytes;
    entry_json["n_tokens"]      = entry.n_tokens;
    entry_json["last_used_us"]  = entry.last_used_us;
    // write tokens as flat int32 array
    nlohmann::json toks = nlohmann::json::array();
    for (auto t : entry.tokens) {
        toks.push_back((int32_t)t);
    }
    entry_json["tokens"] = std::move(toks);
    entries[hash] = entry_json;
}
```

- [ ] **Step 3: Update `load_index()` to read tokens (backward-compatible)**

In the entry-parsing loop inside `load_index()` (around line 593):

```cpp
entry.size_bytes   = entry_json.value("size_bytes",   (uint64_t)0);
entry.n_tokens     = entry_json.value("n_tokens",     (uint32_t)0);
entry.last_used_us = entry_json.value("last_used_us", (int64_t)0);
// read tokens array if present (absent in old-format entries)
if (entry_json.contains("tokens") && entry_json["tokens"].is_array()) {
    entry.tokens.clear();
    entry.tokens.reserve(entry_json["tokens"].size());
    for (const auto & t : entry_json["tokens"]) {
        entry.tokens.push_back((llama_token)t.get<int32_t>());
    }
}
```

- [ ] **Step 4: Update `save()` to populate `entry.tokens`**

In `save()`, where the entry is constructed (around line 376):

```cpp
disk_cache_entry entry;
entry.size_bytes   = obtained;
entry.n_tokens     = (uint32_t)tokens.size();
entry.last_used_us = ggml_time_us();
entry.tokens       = tokens.get_text_tokens();   // <-- add this line
m_index[hash]      = entry;
```

- [ ] **Step 5: Build and verify no compile errors**

```bash
cmake --build build --target llama-server -j4 2>&1 | tail -20
```

Expected: builds cleanly.

- [ ] **Step 6: Smoke-test index round-trip manually**

Start the server with a disk cache dir, run one completion, stop. Check that `index.json` in the cache dir now contains a `"tokens"` array in each entry:

```bash
cat /tmp/test_cache/index.json | python -m json.tool | grep -A5 tokens
```

Expected: something like `"tokens": [1, 15043, 29892, ...]`.

- [ ] **Step 7: Commit**

```bash
git add tools/server/server-disk-cache.h tools/server/server-disk-cache.cpp
git commit -m "feat: store token sequence in disk cache index for prefix matching"
```

---

### Task 2: Add `find_best_prefix`, `load_by_hash`, and `get_entry` methods

**Files:**
- Modify: `tools/server/server-disk-cache.h`
- Modify: `tools/server/server-disk-cache.cpp`

- [ ] **Step 1: Declare the three new methods in the header**

Add to the public section of `server_disk_cache` in `tools/server/server-disk-cache.h`:

```cpp
// Find the entry whose tokens form the longest common prefix with `tokens`.
// Only considers entries with token sequence length >= min_prefix_len.
// Returns the hash of the best match, or "" if none qualifies.
std::string find_best_prefix(const server_tokens & tokens, size_t min_prefix_len) const;

// Load KV state by hash directly (skips re-hashing the token sequence).
// Updates last_used_us on the entry. Returns true on success.
bool load_by_hash(const std::string & hash, llama_context * ctx, int32_t slot_id);

// Read-only access to an index entry. Returns nullptr if hash not found.
const disk_cache_entry * get_entry(const std::string & hash) const;
```

- [ ] **Step 2: Implement `get_entry` in `server-disk-cache.cpp`**

Add before the closing brace of the file:

```cpp
const disk_cache_entry * server_disk_cache::get_entry(const std::string & hash) const {
    auto it = m_index.find(hash);
    if (it == m_index.end()) {
        return nullptr;
    }
    return &it->second;
}
```

- [ ] **Step 3: Implement `find_best_prefix` in `server-disk-cache.cpp`**

```cpp
std::string server_disk_cache::find_best_prefix(const server_tokens & tokens, size_t min_prefix_len) const {
    if (!m_enabled || tokens.empty()) {
        return "";
    }

    const llama_tokens & task_toks = tokens.get_tokens();
    size_t best_len = min_prefix_len > 0 ? min_prefix_len - 1 : 0;
    std::string best_hash;

    for (const auto & [hash, entry] : m_index) {
        if (entry.tokens.empty()) {
            continue;  // old-format entry without token data
        }
        // compute LCP length
        size_t lcp = 0;
        const size_t limit = std::min(entry.tokens.size(), task_toks.size());
        while (lcp < limit && entry.tokens[lcp] == task_toks[lcp]) {
            ++lcp;
        }
        if (lcp > best_len) {
            best_len = lcp;
            best_hash = hash;
        }
    }

    return best_hash;
}
```

- [ ] **Step 4: Implement `load_by_hash` in `server-disk-cache.cpp`**

`load_by_hash` is essentially `load()` but receives a hash instead of tokens. Extract the shared loading logic:

```cpp
bool server_disk_cache::load_by_hash(const std::string & hash, llama_context * ctx, int32_t slot_id) {
    if (!m_enabled || hash.empty() || !ctx) {
        return false;
    }

    auto it = m_index.find(hash);
    if (it == m_index.end()) {
        return false;
    }

    std::string filepath = cache_path(m_path, hash + ".bin");

    std::ifstream file(filepath, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        SRV_WRN("disk cache: failed to open cache file '%s', removing from index\n", filepath.c_str());
        m_index.erase(it);
        save_index();
        return false;
    }

    size_t file_size = (size_t)file.tellg();
    if (file_size == 0) {
        SRV_WRN("disk cache: empty cache file '%s'\n", filepath.c_str());
        return false;
    }

    file.seekg(0, std::ios::beg);
    std::vector<uint8_t> data(file_size);
    if (!file.read(reinterpret_cast<char*>(data.data()), file_size)) {
        SRV_WRN("disk cache: failed to read cache file '%s'\n", filepath.c_str());
        return false;
    }
    file.close();

    size_t restored = llama_state_seq_set_data_ext(ctx, data.data(), data.size(), slot_id, 0);
    if (restored != data.size()) {
        SRV_WRN("disk cache: partial restore: expected %zu, got %zu\n", data.size(), restored);
        return false;
    }

    it = m_index.find(hash);
    if (it != m_index.end()) {
        it->second.last_used_us = ggml_time_us();
        save_index();
    }

    SRV_DBG("disk cache: prefix hit, loaded %zu bytes for hash %.8s...\n", data.size(), hash.c_str());
    return true;
}
```

- [ ] **Step 5: Build and verify no compile errors**

```bash
cmake --build build --target llama-server -j4 2>&1 | tail -20
```

Expected: builds cleanly.

- [ ] **Step 6: Commit**

```bash
git add tools/server/server-disk-cache.h tools/server/server-disk-cache.cpp
git commit -m "feat: add find_best_prefix, load_by_hash, get_entry to disk cache"
```

---

### Task 3: Shutdown flush — save active slots in `destroy()`

**Files:**
- Modify: `tools/server/server-context.cpp`

- [ ] **Step 1: Add shutdown flush before `llama_init.reset()` in `destroy()`**

In `tools/server/server-context.cpp`, find `destroy()` (around line 688):

```cpp
void destroy() {
```

Add the flush block at the very top of `destroy()`, before `llama_init.reset()`:

```cpp
void destroy() {
    // Flush active slot KV states to disk before freeing the llama context.
    // This covers both clean shutdown and sleeping-state entry (both call destroy()).
    if (disk_cache && ctx_tgt) {
        for (auto & slot : slots) {
            if (slot.prompt.n_tokens() > 0 && !slot.prompt.tokens.has_media()) {
                disk_cache->save(slot.prompt.tokens, ctx_tgt, slot.id);
            }
        }
    }

    llama_init.reset();
    // ... rest of existing destroy() body unchanged
```

The existing body continues with `ctx_tgt = nullptr;`, `mtmd_free(mctx);`, etc. — leave all of that intact.

- [ ] **Step 2: Build**

```bash
cmake --build build --target llama-server -j4 2>&1 | tail -20
```

Expected: builds cleanly.

- [ ] **Step 3: Commit**

```bash
git add tools/server/server-context.cpp
git commit -m "feat: flush slot KV states to disk cache on server shutdown/sleep"
```

---

### Task 4: Write the shutdown-flush integration test

**Files:**
- Modify: `tools/server/tests/unit/test_disk_cache.py`

The test starts a server, runs one completion (so the slot has a KV state), stops the server *without* first triggering an idle-slot save (no second completion on another slot), restarts with the same cache dir, and checks that the prompt is served from the disk cache (fewer tokens processed).

- [ ] **Step 1: Add `test_disk_cache_hit_after_clean_shutdown` to `test_disk_cache.py`**

Append to `tools/server/tests/unit/test_disk_cache.py`:

```python
def test_disk_cache_hit_after_clean_shutdown():
    """Active slot KV state is flushed to disk on shutdown; restarted server loads it."""
    cache_dir = tempfile.mkdtemp(prefix="llama_disk_cache_shutdown_")
    try:
        server.cache_disk = cache_dir
        server.start()

        # Send one completion — fills slot 0; do NOT trigger idle save via a second slot
        res = server.make_request("POST", "/completion", data={
            "prompt": LONG_PROMPT,
            "id_slot": 0,
            "cache_prompt": True,
        })
        assert res.status_code == 200
        first_prompt_n = res.body["timings"]["prompt_n"]
        assert first_prompt_n > 0

        # Stop server — shutdown flush should save slot 0 to disk
        server.stop()

        # Verify that at least one .bin file now exists in the cache dir
        bin_files = [f for f in os.listdir(cache_dir) if f.endswith(".bin")]
        assert len(bin_files) > 0, "expected at least one .bin cache file after shutdown"

        # Restart with same cache dir
        server2 = ServerPreset.tinyllama2()
        server2.n_slots = 2
        server2.n_predict = 4
        server2.temperature = 0.0
        server2.kv_unified = True
        server2.cache_disk = cache_dir
        server2.start()

        # Same prompt — should hit disk cache (fewer tokens processed)
        res2 = server2.make_request("POST", "/completion", data={
            "prompt": LONG_PROMPT,
            "cache_prompt": True,
        })
        assert res2.status_code == 200
        assert res2.body["timings"]["prompt_n"] < first_prompt_n, (
            f"expected disk cache hit after clean shutdown: "
            f"prompt_n={res2.body['timings']['prompt_n']} should be < {first_prompt_n}"
        )

        server2.stop()
    finally:
        shutil.rmtree(cache_dir, ignore_errors=True)
```

- [ ] **Step 2: Run the new test**

```bash
cd tools/server/tests
pytest unit/test_disk_cache.py::test_disk_cache_hit_after_clean_shutdown -v
```

Expected: PASS.

- [ ] **Step 3: Run the full disk cache test suite to check for regressions**

```bash
pytest unit/test_disk_cache.py -v
```

Expected: all tests PASS.

- [ ] **Step 4: Commit**

```bash
git add tools/server/tests/unit/test_disk_cache.py
git commit -m "test: disk cache hit after clean shutdown (shutdown flush)"
```

---

### Task 5: Wire prefix search into `get_available_slot()`

**Files:**
- Modify: `tools/server/server-context.cpp`

- [ ] **Step 1: Add prefix search after the exact-hash miss in `get_available_slot()`**

In `tools/server/server-context.cpp`, find the block (around line 1194):

```cpp
                // Try loading: disk cache first, then RAM prompt cache (skip when tokens contain media)
                bool loaded = false;
                if (disk_cache && !task.tokens.has_media()) {
                    loaded = disk_cache->load(task.tokens, ctx_tgt, ret->id);
                    if (loaded) {
                        // Disk cache restores KV state but does not set prompt.tokens.
                        // We must set it here so that update_slots() can compute n_past correctly.
                        ret->prompt.tokens = server_tokens(task.tokens.get_tokens(), task.tokens.has_mtmd);
                    }
                }
                if (!loaded && prompt_cache) {
                    loaded = ret->prompt_load(*prompt_cache, task.tokens);
                }
```

Replace it with:

```cpp
                // Try loading: disk cache exact match first, then prefix match, then RAM prompt cache
                bool loaded = false;
                if (disk_cache && !task.tokens.has_media()) {
                    loaded = disk_cache->load(task.tokens, ctx_tgt, ret->id);
                    if (loaded) {
                        // Exact hit: restore full task token sequence
                        ret->prompt.tokens = server_tokens(task.tokens.get_tokens(), task.tokens.has_mtmd);
                    }
                }
                // On exact miss, try disk prefix search
                if (!loaded && disk_cache && !task.tokens.has_media()) {
                    const float threshold = (slot_prompt_similarity > 0.0f) ? slot_prompt_similarity : 0.5f;
                    const size_t min_prefix = std::max((size_t)1,
                        (size_t)(task.tokens.size() * threshold));
                    const std::string prefix_hash = disk_cache->find_best_prefix(task.tokens, min_prefix);
                    if (!prefix_hash.empty()) {
                        loaded = disk_cache->load_by_hash(prefix_hash, ctx_tgt, ret->id);
                        if (loaded) {
                            // Prefix hit: set prompt.tokens to the matched prefix so update_slots()
                            // computes the correct n_past and processes only the remaining tokens.
                            const disk_cache_entry * entry = disk_cache->get_entry(prefix_hash);
                            if (entry) {
                                ret->prompt.tokens = server_tokens(
                                    llama_tokens(entry->tokens.begin(), entry->tokens.end()),
                                    task.tokens.has_mtmd);
                                SRV_INF("disk cache: prefix hit, loaded %zu/%zu tokens\n",
                                        entry->tokens.size(), task.tokens.size());
                            } else {
                                loaded = false; // entry disappeared from index, treat as miss
                            }
                        }
                    }
                }
                if (!loaded && prompt_cache) {
                    loaded = ret->prompt_load(*prompt_cache, task.tokens);
                }
```

- [ ] **Step 2: Build**

```bash
cmake --build build --target llama-server -j4 2>&1 | tail -20
```

Expected: builds cleanly.

- [ ] **Step 3: Commit**

```bash
git add tools/server/server-context.cpp
git commit -m "feat: disk cache prefix search on exact hash miss"
```

---

### Task 6: Add `slot_prompt_similarity` to test utils and write the prefix-search integration test

**Files:**
- Modify: `tools/server/tests/utils.py`
- Modify: `tools/server/tests/unit/test_disk_cache.py`

- [ ] **Step 1: Add `slot_prompt_similarity` field to `ServerProcess` in `utils.py`**

In `tools/server/tests/utils.py`, find the block of `cache_*` fields (around line 107):

```python
    cache_ram: int | None = None
    cache_disk: str | None = None
    cache_disk_size: int | None = None
```

Add below:

```python
    slot_prompt_similarity: float | None = None
```

Then in the `server_args` building block (find where `cache_disk_size` is handled, around line 256):

```python
        if self.cache_disk_size is not None:
            server_args.extend(["--cache-disk-size", self.cache_disk_size])
```

Add after it:

```python
        if self.slot_prompt_similarity is not None:
            server_args.extend(["--slot-prompt-similarity", self.slot_prompt_similarity])
```

- [ ] **Step 2: Add `test_disk_cache_prefix_search` to `test_disk_cache.py`**

Append to `tools/server/tests/unit/test_disk_cache.py`:

```python
def test_disk_cache_prefix_search():
    """New request matching a long prefix of a cached prompt loads from disk cache."""
    cache_dir = tempfile.mkdtemp(prefix="llama_disk_cache_prefix_")
    try:
        # Use slot_prompt_similarity=0.3 so prefix search kicks in for a 60%+ prefix match
        server.cache_disk = cache_dir
        server.slot_prompt_similarity = 0.3
        server.start()

        # First request: cache LONG_PROMPT
        res = server.make_request("POST", "/completion", data={
            "prompt": LONG_PROMPT,
            "id_slot": 0,
            "cache_prompt": True,
        })
        assert res.status_code == 200
        full_prompt_n = res.body["timings"]["prompt_n"]
        assert full_prompt_n > 0

        # Trigger idle save of slot 0 by using slot 1
        server.make_request("POST", "/completion", data={
            "prompt": "Hello",
            "id_slot": 1,
            "cache_prompt": True,
        })

        # Build a prompt that is LONG_PROMPT + a short suffix (shares the full prefix)
        extended_prompt = LONG_PROMPT + " The end."

        # Second request with extended prompt — disk prefix search should restore
        # LONG_PROMPT's KV state, so only " The end." is processed fresh
        res2 = server.make_request("POST", "/completion", data={
            "prompt": extended_prompt,
            "cache_prompt": True,
        })
        assert res2.status_code == 200
        # prompt_n should be much less than processing the full extended prompt from scratch
        # At minimum it should process fewer tokens than the original full_prompt_n
        assert res2.body["timings"]["prompt_n"] < full_prompt_n, (
            f"expected prefix cache hit: prompt_n={res2.body['timings']['prompt_n']} "
            f"should be < full {full_prompt_n}"
        )
    finally:
        shutil.rmtree(cache_dir, ignore_errors=True)
```

- [ ] **Step 3: Run the new prefix test**

```bash
cd tools/server/tests
pytest unit/test_disk_cache.py::test_disk_cache_prefix_search -v
```

Expected: PASS.

- [ ] **Step 4: Run full disk cache test suite**

```bash
pytest unit/test_disk_cache.py -v
```

Expected: all 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add tools/server/tests/utils.py tools/server/tests/unit/test_disk_cache.py
git commit -m "test: disk cache prefix search integration test"
```

---

## Self-Review

**Spec coverage:**
- ✅ "cache unloads by any reason → save to disk" — Task 3 (`destroy()` flush)
- ✅ "New prompt, exact disk hit" — already existed; Task 1/2 keep it working
- ✅ "Disk prefix search on exact miss" — Task 5
- ✅ `slot_prompt_similarity` threshold reuse with 0.5 fallback — Task 5
- ✅ Old-format index entries (no `tokens`) are backward compatible — Task 1 Step 3
- ✅ `load_by_hash` updates `last_used_us` — Task 2 Step 4
- ✅ `get_entry` returns `nullptr` on miss — Task 2 Step 2; guarded in Task 5

**Placeholder scan:** None found.

**Type consistency:**
- `disk_cache_entry::tokens` is `std::vector<llama_token>` everywhere ✅
- `find_best_prefix` returns `std::string` (hash or `""`); used with `.empty()` check ✅
- `load_by_hash(hash, ctx, slot_id)` signature matches declaration and call sites ✅
- `get_entry(hash)` returns `const disk_cache_entry *`; dereferenced via `entry->tokens` ✅
- `tokens.get_tokens()` used (not `get_text_tokens()`) when setting `ret->prompt.tokens` in Task 5 — this is correct; `get_text_tokens()` strips media tokens (LLAMA_TOKEN_NULL), but we've already guarded with `!has_media()` so both are equivalent here; using `get_tokens()` is consistent with the existing exact-match path ✅
