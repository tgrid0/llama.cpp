# Disk Cache n_stream-Agnostic Restore Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the server disk KV cache restore entries saved under a different `--parallel`/`--kv-unified` (n_stream) layout instead of treating them as misses.

**Architecture:** The blob already stores per-stream KV sections and routes all non-empty data into the destination seq's stream on single-seq restore. So the core `llama_kv_cache::state_read` only needs to skip the n_stream mismatch throw for single-seq states and loop over the *saved* stream count. The disk cache layer then drops its n_stream pre-checks and the save-time overwrite branch.

**Tech Stack:** C++ (llama.cpp core + tools/server), Python pytest integration tests, CMake.

## Global Constraints

- ASCII only in code and comments: no emdash, unicode arrow, `x`, or `...` ellipsis characters; use `-`, `->`, `x`, `...`
- Code comments: concise, 1-2 lines, plain wordings, no hard-wrap to a fixed column
- Do NOT commit without explicit user approval for each commit. If the user asks the agent to commit on their behalf, use `Assisted-by: opencode` in the message, never `Co-authored-by:`. Never write PR descriptions or reviewer responses
- No index format change: `kIndexFormatVersion` stays `3`; the `n_stream` field remains in index.json and becomes informational only
- Build (Windows, run from repo root): `cmake -B build` (first time only), then `cmake --build build --config Release --target llama-server`. Binary lands at `build/bin/Release/llama-server.exe` (the test harness resolves it automatically)
- Tests (PowerShell, run from `tools/server/tests/`): set `$env:LLAMA_CACHE = "tmp"` first so the model cache is shared between prep and test runs. Run with `python -m pytest unit/test_disk_cache.py -v`

---

### Task 1: Flip the n_stream test to expect cross-layout hits (RED)

**Files:**
- Modify: `tools/server/tests/unit/test_disk_cache.py:95-165` (replace `test_disk_cache_survives_parallel_count_change`)
- Build: repo root (no source change yet)

**Interfaces:**
- Consumes: existing server disk-cache behavior (`--cache-disk`, `--cache-disk-size`), `ServerPreset.tinyllama2()` from `tools/server/tests/utils.py`
- Produces: a failing test that Tasks 2+3 make green. The test covers both blob directions: n_stream 1 -> 4 (prompt A) and 4 -> 1 (prompt B, the user's scenario)

- [ ] **Step 1: Replace the test**

Replace the whole `test_disk_cache_survives_parallel_count_change` function (lines 95-165) with:

```python
LONG_PROMPT_B = (
    "In a quiet harbor town, an old lighthouse keeper kept a logbook "
    "of every ship that passed the cape. Each page recorded the wind, "
    "the swell, and the names of sailors who asked for water before "
    "they set sail into the open sea beyond the fog line."
)


def test_disk_cache_survives_parallel_count_change():
    """A cache entry saved under one --parallel (n_stream) layout is still
    usable by a server started with a different --parallel on the same
    --cache-disk dir, in both directions: the KV state is a single-sequence
    snapshot, so restore works across layouts (1 -> 4 for prompt A, 4 -> 1
    for prompt B)."""
    cache_dir = tempfile.mkdtemp(prefix="llama_disk_cache_nstream_")
    try:
        server.n_slots = 1
        server.kv_unified = False
        server.cache_disk = cache_dir
        server.cache_disk_size = -1
        server.start()

        res = server.make_request("POST", "/completion", data={
            "prompt": LONG_PROMPT,
            "cache_prompt": True,
        })
        assert res.status_code == 200
        first_prompt_n_a = res.body["timings"]["prompt_n"]
        assert first_prompt_n_a > 0

        # stop() hard-kills the process (no shutdown flush), so persist entry A
        # explicitly: a dissimilar request selects slot 0 via LRU and saves its
        # state to the disk cache before the slot is reused.
        server.make_request("POST", "/completion", data={
            "prompt": "Hello",
            "cache_prompt": True,
        })

        server.stop()

        # Different --parallel: entry A (saved with n_stream=1) must restore
        # (1 -> 4). Prompt B is new here; the next request's idle-slot pass
        # saves it, so blob B has n_stream=4.
        server2 = ServerPreset.tinyllama2()
        server2.n_predict = 4
        server2.temperature = 0.0
        server2.n_slots = 4
        server2.kv_unified = False
        server2.cache_disk = cache_dir
        server2.cache_disk_size = -1
        server2.start()

        res2a = server2.make_request("POST", "/completion", data={
            "prompt": LONG_PROMPT,
            "cache_prompt": True,
        })
        assert res2a.status_code == 200
        assert res2a.body["timings"]["prompt_n"] < first_prompt_n_a, (
            f"expected disk cache hit across n_stream layouts (1 -> 4): "
            f"prompt_n={res2a.body['timings']['prompt_n']} should be < {first_prompt_n_a}"
        )

        res2b = server2.make_request("POST", "/completion", data={
            "prompt": LONG_PROMPT_B,
            "cache_prompt": True,
        })
        assert res2b.status_code == 200
        first_prompt_n_b = res2b.body["timings"]["prompt_n"]
        assert first_prompt_n_b > 0

        # Persist entry B: launching any task saves all idle slots holding
        # state (cache_idle_slots is on by default).
        server2.make_request("POST", "/completion", data={
            "prompt": "Hello",
            "cache_prompt": True,
        })

        server2.stop()

        # Prompt B was saved by the 4-slot server; a 1-slot server must get a
        # real hit (4 -> 1).
        server3 = ServerPreset.tinyllama2()
        server3.n_predict = 4
        server3.temperature = 0.0
        server3.n_slots = 1
        server3.kv_unified = False
        server3.cache_disk = cache_dir
        server3.cache_disk_size = -1
        server3.start()

        res3 = server3.make_request("POST", "/completion", data={
            "prompt": LONG_PROMPT_B,
            "cache_prompt": True,
        })
        assert res3.status_code == 200
        assert res3.body["timings"]["prompt_n"] < first_prompt_n_b, (
            f"expected disk cache hit across n_stream layouts (4 -> 1): "
            f"prompt_n={res3.body['timings']['prompt_n']} should be < {first_prompt_n_b}"
        )

        server3.stop()
    finally:
        shutil.rmtree(cache_dir, ignore_errors=True)
```

- [ ] **Step 2: Build the server**

Run (from repo root):

```powershell
cmake -B build
cmake --build build --config Release --target llama-server
```

Expected: build succeeds (first configure/build takes several minutes).

- [ ] **Step 3: One-time model prep**

The disk-cache tests run the server with `--offline`, so the model must already be in `LLAMA_CACHE`. From `tools/server/tests/`:

```powershell
$env:LLAMA_CACHE = "tmp"
python -c "import utils; s = utils.ServerPreset.tinyllama2(); s.offline = False; s.start(); s.stop()"
```

Expected: downloads `ggml-org/test-model-stories260K` into `tmp/`, server starts and stops cleanly. Skip if the model is already cached.

- [ ] **Step 4: Run the flipped test, verify it FAILS**

From `tools/server/tests/`:

```powershell
$env:LLAMA_CACHE = "tmp"
python -m pytest unit/test_disk_cache.py::test_disk_cache_survives_parallel_count_change -v
```

Expected: FAIL. `res2a` (the 1 -> 4 step) asserts `prompt_n < first_prompt_n_a` but the current code treats the n_stream mismatch as a miss, so `prompt_n == first_prompt_n_a`. Server log shows `was saved with n_stream=1, current n_stream=4, treating as miss`.

- [ ] **Step 5: Commit (only with user approval)**

Ask the user for the commit message (or to commit themselves). If the user asks the agent to commit on their behalf, stage only `tools/server/tests/unit/test_disk_cache.py` and use a concise message with an `Assisted-by: opencode` trailer.

---

### Task 2: Drop the n_stream checks in the disk cache layer (still RED)

**Files:**
- Modify: `tools/server/server-disk-cache.cpp:510-516` (load pre-check), `:591-617` (save overwrite branch), `:1210-1214` (load_by_hash pre-check)
- Modify: `tools/server/server-disk-cache.h:19-23, 52-55` (comments)

**Interfaces:**
- Consumes: `disk_cache_entry::n_stream` (stays, informational only), `m_n_stream` (stays, logged at init and written per entry)
- Produces: `load()`/`load_by_hash()` attempt the restore for any n_stream; `save()` on an existing entry only refreshes `last_used_us`. The core still throws on n_stream mismatch, so the Task 1 test stays RED with a different failure mode (partial restore instead of the disk-cache pre-check warning)

- [ ] **Step 1: Remove the load() pre-check**

In `tools/server/server-disk-cache.cpp`, delete this block from `load()` (lines 510-516, between the index lookup and the file path build):

```cpp
    if (it->second.n_stream != 0 && it->second.n_stream != m_n_stream) {
        m_misses++;
        save_stats();
        SRV_WRN("disk cache: hash=%.8s... was saved with n_stream=%u, current n_stream=%u, treating as miss\n",
                hash.c_str(), it->second.n_stream, m_n_stream);
        return false;
    }
```

- [ ] **Step 2: Simplify the save() already-cached branch**

In `save()`, replace the whole `// Check if already in cache` block (lines 591-617) with:

```cpp
    // Check if already in cache
    auto it = m_index.find(hash);
    if (it != m_index.end()) {
        it->second.last_used_us = ggml_time_us();
        save_index();
        SRV_INF("disk cache: already cached, hash=%.8s..., refreshed last_used\n", hash.c_str());
        return true;
    }
```

(the old branch also handled incompatible-n_stream entries by deleting the blob and re-saving; entries are compatible across layouts now, so that path goes away)

- [ ] **Step 3: Remove the load_by_hash() pre-check**

In `load_by_hash()`, delete this block (lines 1210-1214, after the index lookup):

```cpp
    if (it->second.n_stream != 0 && it->second.n_stream != m_n_stream) {
        SRV_WRN("disk cache: hash=%.8s... was saved with n_stream=%u, current n_stream=%u, treating as miss\n",
                hash.c_str(), it->second.n_stream, m_n_stream);
        return false;
    }
```

- [ ] **Step 4: Update the header comments**

In `tools/server/server-disk-cache.h`, replace lines 19-23:

```cpp
    // number of KV-cache streams (--parallel, unless --kv-unified) the entry was saved
    // under; a restore into a context with a different n_stream always fails, so this
    // lets load()/save() detect an incompatible entry without a doomed restore attempt.
    // 0 means unknown (pre-n_stream-aware index entry).
    uint32_t n_stream = 0;
```

with:

```cpp
    // number of KV-cache streams (--parallel, unless --kv-unified) the entry was
    // saved under. Informational only: single-seq restores work across layouts,
    // so this is not used for compatibility checks.
    // 0 means unknown (pre-n_stream-aware index entry).
    uint32_t n_stream = 0;
```

and replace the `init()` doc lines 52-55:

```cpp
    // n_stream: number of KV-cache streams this server's context uses (see
    //           disk_cache_entry::n_stream); entries saved under a different
    //           n_stream are treated as incompatible and replaced rather than restored
```

with:

```cpp
    // n_stream: number of KV-cache streams this server's context uses (see
    //           disk_cache_entry::n_stream); informational only
```

- [ ] **Step 5: Rebuild**

Run (from repo root):

```powershell
cmake --build build --config Release --target llama-server
```

Expected: build succeeds.

- [ ] **Step 6: Re-run the flipped test, verify it still FAILS (different reason)**

From `tools/server/tests/`:

```powershell
$env:LLAMA_CACHE = "tmp"
python -m pytest unit/test_disk_cache.py::test_disk_cache_survives_parallel_count_change -v
```

Expected: FAIL on the same `res2a` assertion. The difference: the `treating as miss` pre-check warning is gone; the server log now shows the core failing the restore instead (`error loading state: n_stream mismatch` followed by `disk cache: partial restore`), which confirms the disk layer no longer blocks and the core is the remaining blocker.

---

### Task 3: Lenient single-seq restore in the core (GREEN)

**Files:**
- Modify: `src/llama-kv-cache.cpp:2071-2077` (n_stream check + loop bound in `llama_kv_cache::state_read`)

**Interfaces:**
- Consumes: nothing new; same state blob format (`n_stream` header + per-stream `cell_count` sections, written by `state_write` at line 1999)
- Produces: `llama_kv_cache::state_read(io, seq_id, flags)` succeeds when `seq_id != -1` regardless of the saved n_stream; `seq_id == -1` (full session restore) still throws on mismatch. Any caller of `llama_state_seq_set_data_ext` (disk cache, `llama_state_seq_load_file`) benefits

- [ ] **Step 1: Make state_read lenient for single-seq states**

In `src/llama-kv-cache.cpp`, in `llama_kv_cache::state_read`, replace lines 2071-2077:

```cpp
    uint32_t n_stream_cur;
    io.read(&n_stream_cur, sizeof(n_stream_cur));
    if (n_stream_cur != n_stream) {
        throw std::runtime_error("n_stream mismatch");
    }

    for (uint32_t s = 0; s < n_stream; ++s) {
```

with:

```cpp
    uint32_t n_stream_cur;
    io.read(&n_stream_cur, sizeof(n_stream_cur));
    if (n_stream_cur != n_stream && seq_id == -1) {
        throw std::runtime_error("n_stream mismatch");
    }

    // the stored layout sets how many stream sections to read; a single seq's
    // data lands in seq_to_stream[seq_id] regardless of the saved layout
    for (uint32_t s = 0; s < n_stream_cur; ++s) {
```

Why this is correct: for `seq_id != -1`, every non-empty stream section in the buffer is restored into `seq_to_stream[seq_id]` (line 2085 already does this), so the saved layout is irrelevant; the loop bound must be the saved count, otherwise a smaller blob would be read past its end. For `seq_id == -1` the layouts must match, so the throw stays and the loop bound is identical either way.

- [ ] **Step 2: Rebuild**

Run (from repo root):

```powershell
cmake --build build --config Release --target llama-server
```

Expected: build succeeds.

- [ ] **Step 3: Re-run the flipped test, verify it PASSES**

From `tools/server/tests/`:

```powershell
$env:LLAMA_CACHE = "tmp"
python -m pytest unit/test_disk_cache.py::test_disk_cache_survives_parallel_count_change -v
```

Expected: PASS (1 -> 4 hit for prompt A, 4 -> 1 hit for prompt B).

- [ ] **Step 4: Commit (only with user approval)**

Ask the user for the commit message (or to commit themselves). If the user asks the agent to commit on their behalf, stage `src/llama-kv-cache.cpp`, `tools/server/server-disk-cache.cpp`, `tools/server/server-disk-cache.h` and use a concise message with an `Assisted-by: opencode` trailer.

---

### Task 4: Full disk-cache regression + final commit

**Files:**
- Test: `tools/server/tests/unit/test_disk_cache.py` (all 11 tests: 7 text + 4 multimodal)

**Interfaces:**
- Consumes: Tasks 1-3. The multimodal tests additionally need the `tinygemma3` model in `LLAMA_CACHE` and network access to Hugging Face (image URLs)

- [ ] **Step 1: One-time multimodal model prep (skip if already cached)**

From `tools/server/tests/`:

```powershell
$env:LLAMA_CACHE = "tmp"
python -c "import utils; s = utils.ServerPreset.tinygemma3(); s.offline = False; s.start(); s.stop()"
```

Expected: downloads the tinygemma3 model into `tmp/`, server starts and stops cleanly.

- [ ] **Step 2: Run the whole disk-cache test file**

From `tools/server/tests/`:

```powershell
$env:LLAMA_CACHE = "tmp"
python -m pytest unit/test_disk_cache.py -v
```

Expected: all 11 tests PASS. Watch for:
- `test_disk_cache_hit_after_restart` (basic hit regression)
- `test_disk_cache_prefix_search`, `test_disk_cache_prefix_hit_token_count_matches_restored_state`, `test_disk_cache_prunes_superseded_prefix_entries` (load_by_hash regression)
- `test_disk_cache_hit_after_clean_shutdown` (shutdown flush regression)
- the 4 multimodal tests (core state_read on a hybrid SWA model)

If a multimodal test fails on model download or image fetch (network), report it to the user and re-run just that test once before treating it as a real failure.

- [ ] **Step 3: Final commit (only with user approval)**

If the test file was modified since Task 1's commit (it was not, unless debugging required it), include it. Ask the user for the commit message or to commit. If the user asks the agent to commit on their behalf, use a concise message with an `Assisted-by: opencode` trailer.
