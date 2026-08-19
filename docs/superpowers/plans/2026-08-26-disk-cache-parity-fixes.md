# Disk Cache Parity Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port three correctness fixes from `nathanw-llamacpp`'s disk-cache lineage into this
branch's automatic disk KV cache: a broken embedded SHA-256, a fatal (should-be-lenient)
`n_stream` mismatch check, and missing MTP/DFLASH drafter-context save/restore.

**Architecture:** Each fix is applied directly to the existing `server_disk_cache` class
(`tools/server/server-disk-cache.{h,cpp}`), its three call sites in
`tools/server/server-context.cpp`, and (for the `n_stream` fix) `llama_kv_cache::state_read` in
`src/llama-kv-cache.cpp`. No new files, no architecture change - this widens/corrects existing
entry points.

**Tech Stack:** C++17 (llama.cpp server), pytest (`tools/server/tests/unit/test_disk_cache.py`).
Build via WSL Ubuntu 24.04 (project convention): the repo is mounted at
`/mnt/w/projects/llm/llama.cpp` inside WSL, with an already-configured `build/` directory and a
bootstrapped pytest venv at `/home/t_grid0/.venv`.

**Spec:** `docs/superpowers/specs/2026-08-26-disk-cache-parity-fixes-design.md`

## Global Constraints

- ASCII-only in new/edited comments and log strings (project convention; a stray em-dash was
  flagged in a prior port).
- No `DCDBG`-style temporary debug logging - any new log lines use real `SRV_INF`/`SRV_WRN` at
  their natural verbosity, not scaffolding.
- Every edit must match this branch's *current* code exactly (given below per task) - do not
  blind-apply nathanw's diffs; this branch has already diverged in surrounding code.
- Build command (WSL): `cd /mnt/w/projects/llm/llama.cpp && cmake --build build --config Release -j 8 --target llama-server`
- Python test command (WSL, run from `tools/server/tests`):
  `cd /mnt/w/projects/llm/llama.cpp/tools/server/tests && /home/t_grid0/.venv/bin/python -m pytest unit/test_disk_cache.py -k "<test_name>" -v`

---

### Task 1: Fix broken embedded SHA-256

**Files:**
- Modify: `tools/server/server-disk-cache.cpp:30` (ROTL32 macro), `:70` (K[47] constant),
  `:77-100` (`embed_sha256_transform`)

**Interfaces:**
- Consumes: nothing new
- Produces: nothing new - `hash_tokens()` and all callers are unaffected signature-wise; only the
  hash *values* it produces change (this is the point of the fix)

- [ ] **Step 1: Fix the `ROTL32` macro (missing parentheses around the macro body)**

In `tools/server/server-disk-cache.cpp`, change:

```cpp
#define ROTL32(v, n) U32V((uint32_t)(v) << (n)) | ((uint32_t)(v) >> (32 - (n)))
```

to:

```cpp
#define ROTL32(v, n) (((uint32_t)(v) << (n)) | ((uint32_t)(v) >> (32 - (n))))
```

(`ROTR32` is defined in terms of `ROTL32` right below it and needs no separate edit.)

- [ ] **Step 2: Fix the wrong `K[47]` round constant**

In the same file's `K[64]` table, change:

```cpp
    0xd192e819, 0xd6990624, 0xf40e3585, 0x106a0070,
```

to:

```cpp
    0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
```

- [ ] **Step 3: Replace the non-rotating round macro in `embed_sha256_transform` with the standard loop form**

Change:

```cpp
static void embed_sha256_transform(uint32_t *state, const uint32_t *data) {
    uint32_t W[64];
    uint32_t a,b,c,d,e,f,g,h;
    unsigned j;
    a = state[0]; b = state[1]; c = state[2]; d = state[3];
    e = state[4]; f = state[5]; g = state[6]; h = state[7];

    for (j = 0; j < 16; j++) W[j] = data[j];
    for (j = 16; j < 64; j++) W[j] = s1(W[j-2]) + W[j-7] + s0(W[j-15]) + W[j-16];

    #define R(i) h+=S1(e)+Ch(e,f,g)+K[i]+W[i]; d+=h; h+=S0(a)+Maj(a,b,c)
    R(0); R(1); R(2); R(3); R(4); R(5); R(6); R(7);
    R(8); R(9); R(10); R(11); R(12); R(13); R(14); R(15);
    R(16); R(17); R(18); R(19); R(20); R(21); R(22); R(23);
    R(24); R(25); R(26); R(27); R(28); R(29); R(30); R(31);
    R(32); R(33); R(34); R(35); R(36); R(37); R(38); R(39);
    R(40); R(41); R(42); R(43); R(44); R(45); R(46); R(47);
    R(48); R(49); R(50); R(51); R(52); R(53); R(54); R(55);
    R(56); R(57); R(58); R(59); R(60); R(61); R(62); R(63);
    #undef R

    state[0] += a; state[1] += b; state[2] += c; state[3] += d;
    state[4] += e; state[5] += f; state[6] += g; state[7] += h;
}
```

to:

```cpp
static void embed_sha256_transform(uint32_t *state, const uint32_t *data) {
    uint32_t W[64];
    uint32_t a, b, c, d, e, f, g, h, T1, T2;
    unsigned j;

    for (j = 0; j < 16; j++) W[j] = data[j];
    for (j = 16; j < 64; j++) W[j] = s1(W[j-2]) + W[j-7] + s0(W[j-15]) + W[j-16];

    a = state[0]; b = state[1]; c = state[2]; d = state[3];
    e = state[4]; f = state[5]; g = state[6]; h = state[7];

    for (j = 0; j < 64; j++) {
        T1 = h + S1(e) + Ch(e, f, g) + K[j] + W[j];
        T2 = S0(a) + Maj(a, b, c);
        h = g; g = f; f = e; e = d + T1;
        d = c; c = b; b = a; a = T1 + T2;
    }

    state[0] += a; state[1] += b; state[2] += c; state[3] += d;
    state[4] += e; state[5] += f; state[6] += g; state[7] += h;
}
```

- [ ] **Step 4: Verify the fixed algorithm against known SHA-256 test vectors in a standalone harness**

Create `scratchpad/sha256_verify.cpp` (path: use this session's scratchpad directory) containing
the exact fixed block from Steps 1-3 (macros, `K[64]`, `embed_sha256_init`,
`embed_sha256_transform`, `embed_sha256_update`, `embed_sha256_final` - copy
`embed_sha256_update`/`embed_sha256_final` verbatim from the current file, lines 102-153, they are
unchanged) plus a `main()`:

```cpp
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>

// --- paste the fixed macros / K[64] / embed_sha256_init / embed_sha256_transform /
//     embed_sha256_update / embed_sha256_final here, unchanged from the edited file ---

int main(int argc, char** argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s <string>\n", argv[0]);
        return 1;
    }
    embed_sha256_t ctx;
    embed_sha256_init(&ctx);
    embed_sha256_update(&ctx, reinterpret_cast<const unsigned char*>(argv[1]), strlen(argv[1]));
    unsigned char digest[32];
    embed_sha256_final(&ctx, digest);
    for (int i = 0; i < 32; i++) printf("%02x", digest[i]);
    printf("\n");
    return 0;
}
```

Compile and run (native Bash tool, no WSL needed - this is a standalone file with no project
dependencies):

```bash
g++ -O2 -o /tmp/sha256_verify scratchpad/sha256_verify.cpp
/tmp/sha256_verify "" 
/tmp/sha256_verify "abc"
/tmp/sha256_verify "The quick brown fox jumps over the lazy dog"
```

Compare each line against the real hash:

```bash
printf '' | sha256sum
printf 'abc' | sha256sum
printf 'The quick brown fox jumps over the lazy dog' | sha256sum
```

Expected: all three of `sha256_verify`'s outputs exactly match the corresponding `sha256sum`
output (ignoring the trailing ` *-` / ` -` that `sha256sum` appends). If any mismatch, re-check
Steps 1-3 against this task's code blocks before proceeding - do not continue with a still-broken
hash.

- [ ] **Step 5: Build `llama-server` and confirm it compiles clean**

```bash
wsl.exe -e bash -lc "cd /mnt/w/projects/llm/llama.cpp && cmake --build build --config Release -j 8 --target llama-server"
```

Expected: build succeeds, no new warnings from `server-disk-cache.cpp`.

- [ ] **Step 6: Commit**

```bash
git add tools/server/server-disk-cache.cpp
git commit -m "fix: correct broken embedded SHA-256 in server-disk-cache.cpp

Three independent bugs made the disk cache's content-addressing hash
near input-independent: wrong K[47] round constant, a round macro
that never rotated its register bindings, and a missing paren in
ROTL32/ROTR32 that let ^ misgroup across the macro boundary. Verified
against known SHA-256 test vectors via a standalone extracted build.

Ported from nathanw-llamacpp 99c7e3ece."
```

(Do not commit the throwaway `scratchpad/sha256_verify.cpp` file - it lives outside the repo's
scratchpad convention for this session and is not part of the deliverable.)

---

### Task 2: Make `n_stream` restore lenient in the core KV cache

**Files:**
- Modify: `src/llama-kv-cache.cpp:2043-2049` (`llama_kv_cache::state_read`)

**Interfaces:**
- Consumes: nothing new
- Produces: `state_read` now accepts a single-sequence (`seq_id != -1`) restore whose saved
  `n_stream` differs from the context's current `n_stream`; full-state (`seq_id == -1`) restores
  still require an exact match. Later tasks (3, 4) depend on this relaxed behavior.

- [ ] **Step 1: Relax the `n_stream` check and read loop bound**

In `src/llama-kv-cache.cpp`, inside `llama_kv_cache::state_read`, change:

```cpp
    uint32_t n_stream_cur;
    io.read(&n_stream_cur, sizeof(n_stream_cur));
    if (n_stream_cur != n_stream) {
        throw std::runtime_error("n_stream mismatch");
    }

    for (uint32_t s = 0; s < n_stream; ++s) {
```

to:

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

Do not touch anything else in this function - the rest of the loop body (which routes each
stream's data via `seq_id == -1 ? s : seq_to_stream[seq_id]`) already does the right thing once
the loop bound and guard are fixed; this is confirmed by re-reading the full function body before
editing.

- [ ] **Step 2: Build and confirm no compile errors**

```bash
wsl.exe -e bash -lc "cd /mnt/w/projects/llm/llama.cpp && cmake --build build --config Release -j 8 --target llama-server"
```

Expected: clean build. (This function has no dedicated C++ unit test in this codebase; the
behavior is exercised end-to-end by the Python test in Task 4, after Task 3 also lands.)

- [ ] **Step 3: Commit**

```bash
git add src/llama-kv-cache.cpp
git commit -m "fix: tolerate n_stream mismatch on single-sequence KV state restore

llama_kv_cache::state_read only needs an exact n_stream match for
full-state (seq_id == -1) restores. A single-sequence restore always
routes into seq_to_stream[seq_id] regardless of the saved layout, so
enforcing the match there needlessly rejects a valid restore across a
--parallel/--kv-unified layout change. Read the saved stream count
(n_stream_cur) rather than the current context's, so a smaller or
larger saved layout is handled correctly either direction.

Ported from nathanw-llamacpp 567dcffe4."
```

---

### Task 3: Stop treating `n_stream` as a disk-cache compatibility gate

**Files:**
- Modify: `tools/server/server-disk-cache.h:19-23,52-54` (comments only)
- Modify: `tools/server/server-disk-cache.cpp:510-516` (`load()`), `:591-617` (`save()`),
  `:1210-1214` (`load_by_hash()`)

**Interfaces:**
- Consumes: Task 2's relaxed `state_read` (a restore across a different `n_stream` must now
  actually succeed at the core level for this task's removed checks to be safe)
- Produces: `load()`/`save()`/`load_by_hash()` no longer reject or replace an entry based on
  `n_stream`; `disk_cache_entry::n_stream` remains present and populated (informational only) -
  no signature changes in this task (those come in Task 5)

- [ ] **Step 1: Update the header comments describing the old strict behavior**

In `tools/server/server-disk-cache.h`, change:

```cpp
    // number of KV-cache streams (--parallel, unless --kv-unified) the entry was saved
    // under; a restore into a context with a different n_stream always fails, so this
    // lets load()/save() detect an incompatible entry without a doomed restore attempt.
    // 0 means unknown (pre-n_stream-aware index entry).
    uint32_t n_stream = 0;
```

to:

```cpp
    // number of KV-cache streams (--parallel, unless --kv-unified) the entry was
    // saved under. Informational only: single-seq restores work across layouts,
    // so this is not used for compatibility checks.
    // 0 means unknown (pre-n_stream-aware index entry).
    uint32_t n_stream = 0;
```

And change:

```cpp
    // n_stream: number of KV-cache streams this server's context uses (see
    //           disk_cache_entry::n_stream); entries saved under a different
    //           n_stream are treated as incompatible and replaced rather than restored
    // Returns true on success
    bool init(const std::string& path, int32_t max_size_mib, uint32_t n_stream);
```

to:

```cpp
    // n_stream: number of KV-cache streams this server's context uses (see
    //           disk_cache_entry::n_stream); informational only
    // Returns true on success
    bool init(const std::string& path, int32_t max_size_mib, uint32_t n_stream);
```

- [ ] **Step 2: Delete the mismatch-as-miss check in `load()`**

In `tools/server/server-disk-cache.cpp`, inside `server_disk_cache::load`, delete this block
entirely (it sits between the "exact miss" early-return and the `// Build file path` comment):

```cpp
    if (it->second.n_stream != 0 && it->second.n_stream != m_n_stream) {
        m_misses++;
        save_stats();
        SRV_WRN("disk cache: hash=%.8s... was saved with n_stream=%u, current n_stream=%u, treating as miss\n",
                hash.c_str(), it->second.n_stream, m_n_stream);
        return false;
    }

```

(Leave one blank line between the closing `}` of the "exact miss" block above it and the
`// Build file path` comment below, matching normal spacing.)

- [ ] **Step 3: Simplify the "already cached" branch in `save()`**

In the same file, inside `server_disk_cache::save`, change:

```cpp
    // Check if already in cache
    auto it = m_index.find(hash);
    if (it != m_index.end()) {
        if (it->second.n_stream == 0 || it->second.n_stream == m_n_stream) {
            // Already cached under a compatible KV-stream layout, just update timestamp
            it->second.last_used_us = ggml_time_us();
            save_index();
            SRV_INF("disk cache: already cached, hash=%.8s..., refreshed last_used\n", hash.c_str());
            return true;
        }

        // Entry was saved under a different --parallel/--kv-unified layout and can
        // never be restored here: replace it instead of leaving this hash stuck
        // pointing at a blob this server can never load.
        SRV_INF("disk cache: hash=%.8s... has incompatible n_stream (entry=%u, current=%u), overwriting\n",
                hash.c_str(), it->second.n_stream, m_n_stream);
        fs::remove(cache_path(m_path, hash + ".bin"));
        if (it->second.ckpt_size_bytes > 0) {
            fs::remove(cache_path(m_path, hash + ".ckpt"));
        }
        if (it->second.media_size_bytes > 0) {
            fs::remove(cache_path(m_path, hash + ".chunks"));
        }
        const uint64_t old_total = it->second.size_bytes + it->second.ckpt_size_bytes + it->second.media_size_bytes;
        m_total_size = (old_total <= m_total_size) ? (m_total_size - old_total) : 0;
        m_index.erase(it);
    }
```

to:

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

- [ ] **Step 4: Delete the mismatch-as-miss check in `load_by_hash()`**

In the same file, inside `server_disk_cache::load_by_hash`, delete this block (it sits between the
"hash not found" early-return and the `std::string filepath = cache_path(...)` line):

```cpp
    if (it->second.n_stream != 0 && it->second.n_stream != m_n_stream) {
        SRV_WRN("disk cache: hash=%.8s... was saved with n_stream=%u, current n_stream=%u, treating as miss\n",
                hash.c_str(), it->second.n_stream, m_n_stream);
        return false;
    }

```

- [ ] **Step 5: Build**

```bash
wsl.exe -e bash -lc "cd /mnt/w/projects/llm/llama.cpp && cmake --build build --config Release -j 8 --target llama-server"
```

Expected: clean build.

- [ ] **Step 6: Commit**

```bash
git add tools/server/server-disk-cache.h tools/server/server-disk-cache.cpp
git commit -m "fix: disk cache no longer treats n_stream as a compatibility gate

Now that llama_kv_cache::state_read tolerates a single-sequence
restore across n_stream layouts (previous commit), the disk cache's
own pre-checks in load()/save()/load_by_hash() - which refused or
overwrote an entry saved under a different --parallel/--kv-unified
layout - are unnecessary and actively wrong: they discarded a
perfectly restorable entry. n_stream stays in disk_cache_entry as an
informational field only.

Ported from nathanw-llamacpp 567dcffe4."
```

---

### Task 4: Flip the `n_stream` pytest test to assert real cross-layout hits

**Files:**
- Modify: `tools/server/tests/unit/test_disk_cache.py:95-165`
  (`test_disk_cache_survives_parallel_count_change`)

**Interfaces:**
- Consumes: Task 2 + Task 3's lenient restore behavior
- Produces: nothing consumed by later tasks

- [ ] **Step 1: Replace the test function (and add its new `LONG_PROMPT_B` fixture) with the flipped version**

In `tools/server/tests/unit/test_disk_cache.py`, the existing `LONG_PROMPT` constant (lines
13-19) stays as-is. Immediately before `def test_disk_cache_survives_parallel_count_change():`
(currently line 95), insert:

```python
LONG_PROMPT_B = (
    "In a quiet harbor town, an old lighthouse keeper kept a logbook "
    "of every ship that passed the cape. Each page recorded the wind, "
    "the swell, and the names of sailors who asked for water before "
    "they set sail into the open sea beyond the fog line."
)


```

Then replace the entire existing function body (from `def test_disk_cache_survives_parallel_count_change():`
through its matching `finally: shutil.rmtree(cache_dir, ignore_errors=True)`, currently lines
95-165) with:

```python
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

- [ ] **Step 2: Run the flipped test**

```bash
wsl.exe -e bash -lc "cd /mnt/w/projects/llm/llama.cpp/tools/server/tests && /home/t_grid0/.venv/bin/python -m pytest unit/test_disk_cache.py -k test_disk_cache_survives_parallel_count_change -v"
```

Expected: PASS. (Requires Task 1-3's build to already be in place; rebuild first if Task 3's
build step wasn't the most recent build.)

- [ ] **Step 3: Commit**

```bash
git add tools/server/tests/unit/test_disk_cache.py
git commit -m "test: flip n_stream disk-cache test to expect cross-layout hits

Matches the behavior fixed in the previous two commits: a disk-cache
entry saved under one --parallel layout is now a real hit (not a
forced full reprocess) when restored under a different one, in both
directions.

Ported from nathanw-llamacpp 567dcffe4."
```

---

### Task 5: Add drafter (MTP/DFLASH) context save/restore to `server_disk_cache`

**Files:**
- Modify: `tools/server/server-disk-cache.h` (struct field + 4 method signatures)
- Modify: `tools/server/server-disk-cache.cpp` (`load`, `save`, `evict_oldest`,
  `validate_and_rebuild`, `load_index`, `save_index`, `find_best_prefix`, `load_by_hash`)

**Interfaces:**
- Consumes: nothing new
- Produces: `disk_cache_entry::dft_size_bytes` (`uint64_t`); `load()`, `save()`, `load_by_hash()`
  gain a `llama_context* ctx_dft = nullptr` parameter (inserted right after `slot_id`, before any
  existing trailing params); `find_best_prefix()` gains `bool require_dft = false` (appended
  after `min_prefix_len`). Task 6 (`server-context.cpp`) consumes these new parameters by name.

- [ ] **Step 1: Add `dft_size_bytes` to `disk_cache_entry` (header)**

In `tools/server/server-disk-cache.h`, change:

```cpp
    // size of the "<hash>.ckpt" sidecar file holding serialized context checkpoints
    // (0 if no checkpoints were saved alongside this entry)
    uint64_t ckpt_size_bytes = 0;

    // start_idx -> serialized mtmd_input_chunk blob (mtmd_input_chunk_save output: identity and
```

to:

```cpp
    // size of the "<hash>.ckpt" sidecar file holding serialized context checkpoints
    // (0 if no checkpoints were saved alongside this entry)
    uint64_t ckpt_size_bytes = 0;

    // size of the "<hash>.dft.bin" sidecar file holding the draft model's serialized
    // per-sequence KV state (0 if no speculative decoding was active, or the draft
    // context had no state to save, when this entry was written). A speculative-
    // decoding server must not restore an entry lacking this: doing so would leave
    // ctx_dft's position desynced from the freshly-restored ctx_tgt.
    uint64_t dft_size_bytes = 0;

    // start_idx -> serialized mtmd_input_chunk blob (mtmd_input_chunk_save output: identity and
```

- [ ] **Step 2: Add `ctx_dft` to `load()`'s signature (header)**

Change:

```cpp
    // Load KV state from disk cache for the given tokens
    // tokens: the token sequence to look up
    // ctx: llama context to restore state into
    // slot_id: slot ID to restore into
    // out_hash: if non-null and the load succeeds, receives the matched entry's hash
    //           (so the caller can look up the entry's full token sequence via get_entry())
    // out_checkpoints: if non-null and the load succeeds, receives the entry's saved
    //                  context checkpoints (cleared first; empty if none were saved)
    // Returns true if cache hit and state was loaded successfully
    bool load(const server_tokens& tokens, llama_context* ctx, int32_t slot_id,
              std::string * out_hash = nullptr,
              std::list<common_prompt_checkpoint> * out_checkpoints = nullptr);
```

to:

```cpp
    // Load KV state from disk cache for the given tokens
    // tokens: the token sequence to look up
    // ctx: llama context to restore state into
    // slot_id: slot ID to restore into
    // ctx_dft: if non-null (speculative decoding is active), the draft context to also
    //          restore. An entry that was saved without draft state is treated as a miss
    //          in this case, rather than leaving ctx_dft desynced from the restored ctx.
    // out_hash: if non-null and the load succeeds, receives the matched entry's hash
    //           (so the caller can look up the entry's full token sequence via get_entry())
    // out_checkpoints: if non-null and the load succeeds, receives the entry's saved
    //                  context checkpoints (cleared first; empty if none were saved)
    // Returns true if cache hit and state was loaded successfully
    bool load(const server_tokens& tokens, llama_context* ctx, int32_t slot_id,
              llama_context* ctx_dft = nullptr,
              std::string * out_hash = nullptr,
              std::list<common_prompt_checkpoint> * out_checkpoints = nullptr);
```

- [ ] **Step 3: Add `ctx_dft` to `save()`'s signature (header)**

Change:

```cpp
    // Returns true if saved successfully
    bool save(const server_tokens& tokens, llama_context* ctx, int32_t slot_id,
              const server_tokens* full_tokens = nullptr,
              const std::list<common_prompt_checkpoint> * checkpoints = nullptr);
```

to:

```cpp
    // ctx_dft: if non-null (speculative decoding is active), the draft context's
    //          per-sequence KV state is saved alongside the target's, so a later
    //          load()/load_by_hash() can restore both in sync. Skipped (non-fatal)
    //          if the draft context has no state for this sequence yet.
    // Returns true if saved successfully
    bool save(const server_tokens& tokens, llama_context* ctx, int32_t slot_id,
              llama_context* ctx_dft = nullptr,
              const server_tokens* full_tokens = nullptr,
              const std::list<common_prompt_checkpoint> * checkpoints = nullptr);
```

- [ ] **Step 4: Add `require_dft` to `find_best_prefix()` and `ctx_dft` to `load_by_hash()` (header)**

Change:

```cpp
    // Find the entry whose tokens form the longest common prefix with `tokens`.
    // Only considers entries with token sequence length >= min_prefix_len.
    // Returns the hash of the best match, or "" if none qualifies.
    std::string find_best_prefix(const server_tokens & tokens, size_t min_prefix_len) const;

    // Load KV state by hash directly (skips re-hashing the token sequence).
    // Updates last_used_us on the entry. Returns true on success.
    // out_checkpoints: if non-null, receives the entry's saved context checkpoints
    //                  (cleared first; empty if none were saved).
    bool load_by_hash(const std::string & hash, llama_context * ctx, int32_t slot_id,
                       std::list<common_prompt_checkpoint> * out_checkpoints = nullptr);
```

to:

```cpp
    // Find the entry whose tokens form the longest common prefix with `tokens`.
    // Only considers entries with token sequence length >= min_prefix_len.
    // require_dft: if true, only consider entries that also carry draft-model KV
    //              state (see disk_cache_entry::dft_size_bytes) - pass true when
    //              speculative decoding is active, so the caller never picks an
    //              entry that would leave ctx_dft desynced after restore.
    // Returns the hash of the best match, or "" if none qualifies.
    std::string find_best_prefix(const server_tokens & tokens, size_t min_prefix_len, bool require_dft = false) const;

    // Load KV state by hash directly (skips re-hashing the token sequence).
    // Updates last_used_us on the entry. Returns true on success.
    // ctx_dft: if non-null (speculative decoding is active), the draft context to also
    //          restore. An entry saved without draft state is treated as a miss.
    // out_checkpoints: if non-null, receives the entry's saved context checkpoints
    //                  (cleared first; empty if none were saved).
    bool load_by_hash(const std::string & hash, llama_context * ctx, int32_t slot_id,
                       llama_context * ctx_dft = nullptr,
                       std::list<common_prompt_checkpoint> * out_checkpoints = nullptr);
```

- [ ] **Step 5: `load()` - signature, miss-on-no-draft-state check, draft restore (cpp)**

In `tools/server/server-disk-cache.cpp`, change the function signature:

```cpp
bool server_disk_cache::load(const server_tokens& tokens, llama_context* ctx, int32_t slot_id,
                              std::string * out_hash,
                              std::list<common_prompt_checkpoint> * out_checkpoints) {
```

to:

```cpp
bool server_disk_cache::load(const server_tokens& tokens, llama_context* ctx, int32_t slot_id,
                              llama_context* ctx_dft,
                              std::string * out_hash,
                              std::list<common_prompt_checkpoint> * out_checkpoints) {
```

Then, right after the "exact miss" early-return block (`if (it == m_index.end()) { ... return
false; }`) and before `// Build file path`, insert:

```cpp
    // A speculative-decoding server can only use entries that also carry draft-model
    // KV state; otherwise ctx_dft would be left desynced from the freshly-restored
    // ctx position, corrupting the next draft decode. Treat this like a miss.
    if (ctx_dft && it->second.dft_size_bytes == 0) {
        m_misses++;
        save_stats();
        SRV_INF("disk cache: exact match for %zu tokens has no draft-model state, treating as miss, hash=%.8s... (hits=%" PRIu64 ", misses=%" PRIu64 ")\n",
                tokens.size(), hash.c_str(), m_hits, m_misses);
        return false;
    }

```

Then, in the same function, right after the target-state restore block and before `// Update
index: update last_used timestamp`:

```cpp
    // Restore KV state into context
    size_t restored = llama_state_seq_set_data_ext(ctx, data.data(), data.size(), slot_id, 0);
    if (restored != data.size()) {
        SRV_WRN("disk cache: partial restore: expected %zu, got %zu\n", data.size(), restored);
        return false;
    }

    // Update index: update last_used timestamp
```

insert the draft restore block between them, i.e. change to:

```cpp
    // Restore KV state into context
    size_t restored = llama_state_seq_set_data_ext(ctx, data.data(), data.size(), slot_id, 0);
    if (restored != data.size()) {
        SRV_WRN("disk cache: partial restore: expected %zu, got %zu\n", data.size(), restored);
        return false;
    }

    if (ctx_dft) {
        std::string dft_filepath = cache_path(m_path, hash + ".dft.bin");
        std::ifstream dft_file(dft_filepath, std::ios::binary | std::ios::ate);
        if (!dft_file.is_open()) {
            SRV_WRN("disk cache: failed to open draft cache file '%s'\n", dft_filepath.c_str());
            return false;
        }
        size_t dft_file_size = (size_t) dft_file.tellg();
        dft_file.seekg(0, std::ios::beg);
        std::vector<uint8_t> dft_data(dft_file_size);
        if (dft_file_size > 0 && !dft_file.read(reinterpret_cast<char*>(dft_data.data()), dft_file_size)) {
            SRV_WRN("disk cache: failed to read draft cache file '%s'\n", dft_filepath.c_str());
            return false;
        }
        dft_file.close();

        size_t dft_restored = llama_state_seq_set_data_ext(ctx_dft, dft_data.data(), dft_data.size(), slot_id, 0);
        if (dft_restored != dft_data.size()) {
            SRV_WRN("disk cache: partial draft restore: expected %zu, got %zu\n", dft_data.size(), dft_restored);
            return false;
        }
    }

    // Update index: update last_used timestamp
```

- [ ] **Step 6: `save()` - signature, draft state persist (cpp)**

Change the function signature:

```cpp
bool server_disk_cache::save(const server_tokens& tokens, llama_context* ctx, int32_t slot_id,
                              const server_tokens* full_tokens,
                              const std::list<common_prompt_checkpoint> * checkpoints) {
```

to:

```cpp
bool server_disk_cache::save(const server_tokens& tokens, llama_context* ctx, int32_t slot_id,
                              llama_context* ctx_dft,
                              const server_tokens* full_tokens,
                              const std::list<common_prompt_checkpoint> * checkpoints) {
```

Then update the prune-superseded-entries block's size accounting - change:

```cpp
            fs::remove(cache_path(m_path, it2->first + ".bin"));
            if (old_entry.ckpt_size_bytes > 0) {
                fs::remove(cache_path(m_path, it2->first + ".ckpt"));
            }
            if (old_entry.media_size_bytes > 0) {
                fs::remove(cache_path(m_path, it2->first + ".chunks"));
            }
            const uint64_t old_total = old_entry.size_bytes + old_entry.ckpt_size_bytes + old_entry.media_size_bytes;
```

to:

```cpp
            fs::remove(cache_path(m_path, it2->first + ".bin"));
            if (old_entry.ckpt_size_bytes > 0) {
                fs::remove(cache_path(m_path, it2->first + ".ckpt"));
            }
            if (old_entry.media_size_bytes > 0) {
                fs::remove(cache_path(m_path, it2->first + ".chunks"));
            }
            if (old_entry.dft_size_bytes > 0) {
                fs::remove(cache_path(m_path, it2->first + ".dft.bin"));
            }
            const uint64_t old_total = old_entry.size_bytes + old_entry.ckpt_size_bytes + old_entry.media_size_bytes + old_entry.dft_size_bytes;
```

Then, between the end of the media-chunks (`.chunks`) save block and the start of the checkpoints
(`.ckpt`) save block, change:

```cpp
        if (!chunks_ok) {
            // same "abort the whole save" policy as a failed build_media_chunk_blobs() above:
            // committing entry.tokens with placeholders but no chunk blobs would make this
            // entry permanently unmatchable (rebuild_tokens_impl rejects it forever)
            SRV_WRN("%s", "disk cache: failed to persist media chunks sidecar, aborting save\n");
            fs::remove(filepath);
            fs::remove(chunks_filepath);
            return false;
        }
    }

    // Persist context checkpoints alongside the main KV state, if any. These let a
```

to:

```cpp
        if (!chunks_ok) {
            // same "abort the whole save" policy as a failed build_media_chunk_blobs() above:
            // committing entry.tokens with placeholders but no chunk blobs would make this
            // entry permanently unmatchable (rebuild_tokens_impl rejects it forever)
            SRV_WRN("%s", "disk cache: failed to persist media chunks sidecar, aborting save\n");
            fs::remove(filepath);
            fs::remove(chunks_filepath);
            return false;
        }
    }

    // Persist the draft model's KV state alongside the target's, if speculative
    // decoding is active. Skipped (non-fatal) if the draft context has no state
    // for this sequence yet - the entry is simply treated as tgt-only later
    // (see load()/load_by_hash()/find_best_prefix()'s require_dft handling).
    if (ctx_dft) {
        size_t dft_state_size = llama_state_seq_get_size_ext(ctx_dft, slot_id, 0);
        if (dft_state_size > 0) {
            std::vector<uint8_t> dft_data(dft_state_size);
            size_t dft_obtained = llama_state_seq_get_data_ext(ctx_dft, dft_data.data(), dft_state_size, slot_id, 0);
            if (dft_obtained != dft_state_size) {
                SRV_WRN("disk cache: failed to get draft KV state: expected %zu, got %zu\n", dft_state_size, dft_obtained);
            } else {
                const std::string dft_filepath     = cache_path(m_path, hash + ".dft.bin");
                const std::string dft_tmp_filepath = dft_filepath + ".tmp";

                bool dft_ok = false;
                std::ofstream dft_file(dft_tmp_filepath, std::ios::binary);
                if (dft_file.is_open()) {
                    dft_file.write(reinterpret_cast<const char*>(dft_data.data()), dft_obtained);
                    dft_file.flush();
                    if (!dft_file.fail()) {
                        dft_file.close();
                        std::error_code dft_ec;
                        fs::rename(dft_tmp_filepath, dft_filepath, dft_ec);
                        if (!dft_ec) {
                            entry.dft_size_bytes = dft_obtained;
                            m_total_size += dft_obtained;
                            dft_ok = true;
                        } else {
                            SRV_WRN("disk cache: failed to rename draft cache file: %s\n", dft_ec.message().c_str());
                        }
                    } else {
                        SRV_WRN("disk cache: failed to write draft cache file '%s'\n", dft_tmp_filepath.c_str());
                        dft_file.close();
                    }
                } else {
                    SRV_WRN("disk cache: failed to create draft cache file '%s'\n", dft_tmp_filepath.c_str());
                }
                if (!dft_ok) {
                    fs::remove(dft_tmp_filepath);
                }
            }
        }
    }

    // Persist context checkpoints alongside the main KV state, if any. These let a
```

(`entry` is already declared earlier in `save()`, before the media-chunks block, so it is in
scope here.)

- [ ] **Step 7: `evict_oldest()` - remove/account for the `.dft.bin` sidecar (cpp)**

Change:

```cpp
    // Remove files
    std::string filepath = cache_path(m_path, oldest_it->first + ".bin");
    fs::remove(filepath); // Ignore errors (file might already be gone)
    if (oldest_it->second.ckpt_size_bytes > 0) {
        fs::remove(cache_path(m_path, oldest_it->first + ".ckpt"));
    }
    if (oldest_it->second.media_size_bytes > 0) {
        fs::remove(cache_path(m_path, oldest_it->first + ".chunks"));
    }

    // Update index
    const uint64_t entry_total = oldest_it->second.size_bytes + oldest_it->second.ckpt_size_bytes + oldest_it->second.media_size_bytes;
```

to:

```cpp
    // Remove files
    std::string filepath = cache_path(m_path, oldest_it->first + ".bin");
    fs::remove(filepath); // Ignore errors (file might already be gone)
    if (oldest_it->second.ckpt_size_bytes > 0) {
        fs::remove(cache_path(m_path, oldest_it->first + ".ckpt"));
    }
    if (oldest_it->second.media_size_bytes > 0) {
        fs::remove(cache_path(m_path, oldest_it->first + ".chunks"));
    }
    if (oldest_it->second.dft_size_bytes > 0) {
        fs::remove(cache_path(m_path, oldest_it->first + ".dft.bin"));
    }

    // Update index
    const uint64_t entry_total = oldest_it->second.size_bytes + oldest_it->second.ckpt_size_bytes + oldest_it->second.media_size_bytes + oldest_it->second.dft_size_bytes;
```

- [ ] **Step 8: `validate_and_rebuild()` - remove/account for the `.dft.bin` sidecar (cpp)**

Change:

```cpp
            SRV_WRN("disk cache: removing index entry for missing file '%s'\n", it->first.c_str());
            if (it->second.ckpt_size_bytes > 0) {
                fs::remove(cache_path(m_path, it->first + ".ckpt"));
            }
            if (it->second.media_size_bytes > 0) {
                fs::remove(cache_path(m_path, it->first + ".chunks"));
            }
            it = m_index.erase(it);
            removed++;
        } else {
            // Update size from actual file (in case index is stale)
            it->second.size_bytes = found_files[it->first];
            // Drop stale checkpoint-size bookkeeping if the sidecar file is gone
            if (it->second.ckpt_size_bytes > 0 && !fs::exists(cache_path(m_path, it->first + ".ckpt"))) {
                it->second.ckpt_size_bytes = 0;
            }
            if (it->second.media_size_bytes > 0 && !fs::exists(cache_path(m_path, it->first + ".chunks"))) {
                it->second.media_size_bytes = 0;
                it->second.media_chunks.clear();
            }
            ++it;
        }
    }

    // Rebuild total size
    m_total_size = 0;
    for (const auto& [hash, entry] : m_index) {
        m_total_size += entry.size_bytes + entry.ckpt_size_bytes + entry.media_size_bytes;
    }
```

to:

```cpp
            SRV_WRN("disk cache: removing index entry for missing file '%s'\n", it->first.c_str());
            if (it->second.ckpt_size_bytes > 0) {
                fs::remove(cache_path(m_path, it->first + ".ckpt"));
            }
            if (it->second.media_size_bytes > 0) {
                fs::remove(cache_path(m_path, it->first + ".chunks"));
            }
            if (it->second.dft_size_bytes > 0) {
                fs::remove(cache_path(m_path, it->first + ".dft.bin"));
            }
            it = m_index.erase(it);
            removed++;
        } else {
            // Update size from actual file (in case index is stale)
            it->second.size_bytes = found_files[it->first];
            // Drop stale checkpoint-size bookkeeping if the sidecar file is gone
            if (it->second.ckpt_size_bytes > 0 && !fs::exists(cache_path(m_path, it->first + ".ckpt"))) {
                it->second.ckpt_size_bytes = 0;
            }
            if (it->second.media_size_bytes > 0 && !fs::exists(cache_path(m_path, it->first + ".chunks"))) {
                it->second.media_size_bytes = 0;
                it->second.media_chunks.clear();
            }
            if (it->second.dft_size_bytes > 0 && !fs::exists(cache_path(m_path, it->first + ".dft.bin"))) {
                it->second.dft_size_bytes = 0;
            }
            ++it;
        }
    }

    // Rebuild total size
    m_total_size = 0;
    for (const auto& [hash, entry] : m_index) {
        m_total_size += entry.size_bytes + entry.ckpt_size_bytes + entry.media_size_bytes + entry.dft_size_bytes;
    }
```

- [ ] **Step 9: `load_index()` - read `dft_size_bytes` from JSON (cpp)**

Change:

```cpp
            entry.ckpt_size_bytes = entry_json.value("ckpt_size_bytes", (uint64_t)0);
            entry.media_size_bytes = entry_json.value("media_size_bytes", (uint64_t)0);
            entry.n_stream        = entry_json.value("n_stream",        (uint32_t)0);
```

to:

```cpp
            entry.ckpt_size_bytes = entry_json.value("ckpt_size_bytes", (uint64_t)0);
            entry.media_size_bytes = entry_json.value("media_size_bytes", (uint64_t)0);
            entry.dft_size_bytes  = entry_json.value("dft_size_bytes",  (uint64_t)0);
            entry.n_stream        = entry_json.value("n_stream",        (uint32_t)0);
```

and change:

```cpp
            m_index[hash] = entry;
            m_total_size += entry.size_bytes + entry.ckpt_size_bytes + entry.media_size_bytes;
        }
    }
```

to:

```cpp
            m_index[hash] = entry;
            m_total_size += entry.size_bytes + entry.ckpt_size_bytes + entry.media_size_bytes + entry.dft_size_bytes;
        }
    }
```

- [ ] **Step 10: `find_best_prefix()` - signature and `require_dft` skip (cpp)**

Change:

```cpp
std::string server_disk_cache::find_best_prefix(const server_tokens & tokens, size_t min_prefix_len) const {
```

to:

```cpp
std::string server_disk_cache::find_best_prefix(const server_tokens & tokens, size_t min_prefix_len, bool require_dft) const {
```

Then, inside the loop over `m_index`, change:

```cpp
        if (entry.tokens.empty()) {
            continue;  // old-format entry without token data
        }

        const server_tokens entry_tokens = rebuild_tokens_impl(entry);
```

to:

```cpp
        if (entry.tokens.empty()) {
            continue;  // old-format entry without token data
        }

        if (require_dft && entry.dft_size_bytes == 0) {
            continue;  // would leave ctx_dft desynced from the restored ctx_tgt
        }

        const server_tokens entry_tokens = rebuild_tokens_impl(entry);
```

- [ ] **Step 11: `load_by_hash()` - signature, miss-on-no-draft-state check, draft restore (cpp)**

Change the function signature:

```cpp
bool server_disk_cache::load_by_hash(const std::string & hash, llama_context * ctx, int32_t slot_id,
                                      std::list<common_prompt_checkpoint> * out_checkpoints) {
```

to:

```cpp
bool server_disk_cache::load_by_hash(const std::string & hash, llama_context * ctx, int32_t slot_id,
                                      llama_context * ctx_dft,
                                      std::list<common_prompt_checkpoint> * out_checkpoints) {
```

Then, right after the "hash not found" early-return and before `std::string filepath =
cache_path(...)`, insert:

```cpp
    // See load(): an entry without draft-model state must not be used to restore
    // a speculative-decoding server's ctx, or ctx_dft would end up desynced.
    if (ctx_dft && it->second.dft_size_bytes == 0) {
        return false;
    }

```

Then, after the target-state restore block (the `restored != data.size()` check inside this
function - note this is a *different* occurrence from Step 5's, in `load_by_hash` not `load`) and
before `it = m_index.find(hash);`, change:

```cpp
    size_t restored = llama_state_seq_set_data_ext(ctx, data.data(), data.size(), slot_id, 0);
    if (restored != data.size()) {
        SRV_WRN("disk cache: partial restore: expected %zu, got %zu\n", data.size(), restored);
        return false;
    }

    it = m_index.find(hash);
```

to:

```cpp
    size_t restored = llama_state_seq_set_data_ext(ctx, data.data(), data.size(), slot_id, 0);
    if (restored != data.size()) {
        SRV_WRN("disk cache: partial restore: expected %zu, got %zu\n", data.size(), restored);
        return false;
    }

    if (ctx_dft) {
        std::string dft_filepath = cache_path(m_path, hash + ".dft.bin");
        std::ifstream dft_file(dft_filepath, std::ios::binary | std::ios::ate);
        if (!dft_file.is_open()) {
            SRV_WRN("disk cache: failed to open draft cache file '%s'\n", dft_filepath.c_str());
            return false;
        }
        size_t dft_file_size = (size_t) dft_file.tellg();
        dft_file.seekg(0, std::ios::beg);
        std::vector<uint8_t> dft_data(dft_file_size);
        if (dft_file_size > 0 && !dft_file.read(reinterpret_cast<char*>(dft_data.data()), dft_file_size)) {
            SRV_WRN("disk cache: failed to read draft cache file '%s'\n", dft_filepath.c_str());
            return false;
        }
        dft_file.close();

        size_t dft_restored = llama_state_seq_set_data_ext(ctx_dft, dft_data.data(), dft_data.size(), slot_id, 0);
        if (dft_restored != dft_data.size()) {
            SRV_WRN("disk cache: partial draft restore: expected %zu, got %zu\n", dft_data.size(), dft_restored);
            return false;
        }
    }

    it = m_index.find(hash);
```

- [ ] **Step 12: `save_index()` - write `dft_size_bytes` to JSON (cpp)**

Change:

```cpp
        entry_json["ckpt_size_bytes"] = entry.ckpt_size_bytes;
        entry_json["media_size_bytes"] = entry.media_size_bytes;
        entry_json["n_stream"]        = entry.n_stream;
```

to:

```cpp
        entry_json["ckpt_size_bytes"] = entry.ckpt_size_bytes;
        entry_json["media_size_bytes"] = entry.media_size_bytes;
        entry_json["dft_size_bytes"]  = entry.dft_size_bytes;
        entry_json["n_stream"]        = entry.n_stream;
```

- [ ] **Step 13: Build**

```bash
wsl.exe -e bash -lc "cd /mnt/w/projects/llm/llama.cpp && cmake --build build --config Release -j 8 --target llama-server"
```

Expected: clean build. At this point no caller passes a non-null `ctx_dft` yet (Task 6 wires
that), so this build alone should not change any existing test's behavior - all `ctx_dft`
parameters default to `nullptr` at every current call site, which is a no-op for every new code
path added in this task.

- [ ] **Step 14: Run the full existing disk-cache test suite to confirm no regression from the plumbing change**

```bash
wsl.exe -e bash -lc "cd /mnt/w/projects/llm/llama.cpp/tools/server/tests && /home/t_grid0/.venv/bin/python -m pytest unit/test_disk_cache.py -v"
```

Expected: all tests PASS (same set as before this task - no new test yet, that's Task 7).

- [ ] **Step 15: Commit**

```bash
git add tools/server/server-disk-cache.h tools/server/server-disk-cache.cpp
git commit -m "feat: support saving/restoring drafter (MTP/DFLASH) context in disk cache

disk_cache_entry gains dft_size_bytes and a '<hash>.dft.bin' sidecar.
load()/save()/load_by_hash() gain an optional ctx_dft parameter to
save/restore the draft model's per-sequence KV state alongside the
target's; find_best_prefix() gains require_dft to avoid picking an
entry that would leave ctx_dft desynced from a restored ctx_tgt. All
new parameters default to nullptr/false, so this alone does not
change behavior until server-context.cpp passes them (next commit).

Ported from nathanw-llamacpp ba8ed1a96."
```

---

### Task 6: Wire `ctx_dft` through the disk-cache call sites in `server-context.cpp`

**Files:**
- Modify: `tools/server/server-context.cpp:1013,1803,1823,1825`

**Interfaces:**
- Consumes: Task 5's new `ctx_dft`/`require_dft` parameters
- Produces: nothing new - this task activates the drafter-context feature end-to-end for real
  speculative-decoding server configurations

- [ ] **Step 1: Pass `ctx_dft` in `disk_cache_save_slot()`**

In `tools/server/server-context.cpp`, inside `disk_cache_save_slot`, change:

```cpp
        return disk_cache->save(prompt_tokens, ctx_tgt, slot.id, &slot.prompt.tokens, &slot.prompt.checkpoints);
```

to:

```cpp
        return disk_cache->save(prompt_tokens, ctx_tgt, slot.id, ctx_dft, &slot.prompt.tokens, &slot.prompt.checkpoints);
```

(`ctx_dft` here is the `server_context_impl` member already resolved via `spec_init->context()`
at slot-launch time and already used throughout this file's speculative-decoding/checkpoint
bookkeeping - no new resolution logic needed.)

- [ ] **Step 2: Pass `ctx_dft` in the exact-match `disk_cache->load(...)` call**

In the same file, change:

```cpp
                    loaded = disk_cache->load(task.tokens, ctx_tgt, ret->id, &exact_hash, &loaded_checkpoints);
```

to:

```cpp
                    loaded = disk_cache->load(task.tokens, ctx_tgt, ret->id, ctx_dft, &exact_hash, &loaded_checkpoints);
```

- [ ] **Step 3: Pass `require_dft`/`ctx_dft` in the prefix-search fallback**

In the same file, change:

```cpp
                        const std::string prefix_hash = disk_cache->find_best_prefix(task.tokens, min_prefix_len);
                        if (!prefix_hash.empty()) {
                            loaded = disk_cache->load_by_hash(prefix_hash, ctx_tgt, ret->id, &loaded_checkpoints);
```

to:

```cpp
                        const std::string prefix_hash = disk_cache->find_best_prefix(task.tokens, min_prefix_len, ctx_dft != nullptr);
                        if (!prefix_hash.empty()) {
                            loaded = disk_cache->load_by_hash(prefix_hash, ctx_tgt, ret->id, ctx_dft, &loaded_checkpoints);
```

- [ ] **Step 4: Build**

```bash
wsl.exe -e bash -lc "cd /mnt/w/projects/llm/llama.cpp && cmake --build build --config Release -j 8 --target llama-server"
```

Expected: clean build.

- [ ] **Step 5: Run the full existing disk-cache test suite (non-speculative configs must be unaffected)**

```bash
wsl.exe -e bash -lc "cd /mnt/w/projects/llm/llama.cpp/tools/server/tests && /home/t_grid0/.venv/bin/python -m pytest unit/test_disk_cache.py -v"
```

Expected: all tests PASS.

- [ ] **Step 6: Commit**

```bash
git add tools/server/server-context.cpp
git commit -m "feat: thread ctx_dft through disk-cache save/load call sites

Activates the drafter-context save/restore support added in the
previous commit: the per-slot ctx_dft already resolved for
speculative decoding is now passed to every disk_cache save/load/
find_best_prefix/load_by_hash call, so an MTP/DFLASH draft context
stays in sync across a disk-cache restore instead of desyncing.

Ported from nathanw-llamacpp ba8ed1a96."
```

---

### Task 7: Add the speculative-decoding disk-cache regression test

**Files:**
- Modify: `tools/server/tests/unit/test_disk_cache.py` (insert new test + helpers before
  `test_disk_cache_hit_after_clean_shutdown`)

**Interfaces:**
- Consumes: Task 6's end-to-end wiring
- Produces: nothing consumed by later tasks

- [ ] **Step 1: Insert the new fixtures and test**

In `tools/server/tests/unit/test_disk_cache.py`, find:

```python
    finally:
        shutil.rmtree(cache_dir, ignore_errors=True)


def test_disk_cache_hit_after_clean_shutdown():
```

(this is the end of `test_disk_cache_prefix_search` immediately followed by the start of
`test_disk_cache_hit_after_clean_shutdown` - the exact `finally:` block shown here is unique to
this location in the file). Replace it with:

```python
    finally:
        shutil.rmtree(cache_dir, ignore_errors=True)


SPEC_DRAFT_MODEL_URL = "https://huggingface.co/ggml-org/tiny-llamas/resolve/main/stories15M-q4_0.gguf"


SHORT_UNRELATED_PROMPT = "Meanwhile, across the ocean, a merchant counted his coins."


def _spec_server(cache_dir):
    # Self-speculation (draft model == target model): the disk-cache index only
    # needs a real ctx_dft to exercise, and this sidesteps needing a second,
    # vocab-compatible offline-cached model just for the draft role.
    model_path = download_file(SPEC_DRAFT_MODEL_URL)
    s = ServerProcess()
    s.model_file = model_path
    s.model_alias = "stories15m-q4_0"
    s.n_ctx = 1024
    s.n_batch = 256
    s.n_slots = 1
    s.n_predict = 8
    s.temperature = 0.0
    s.seed = 42
    s.model_draft = model_path
    s.spec_type = "draft-simple"
    s.spec_draft_n_min = 1
    s.spec_draft_n_max = 4
    s.fa = "off"
    s.cache_disk = cache_dir
    s.cache_disk_size = -1
    return s


def test_disk_cache_hit_with_speculative_decoding():
    """Regression: a disk-cache restore must bring the draft model's KV cache back
    in sync with the target's. Before the fix, only ctx_tgt was saved/restored, so
    after a restore ctx_dft was left at whatever position it last held, desynced
    from the newly-restored ctx_tgt. llama-batch.cpp only rejects a mismatch when
    the sequence already has *some* stored position (seq_pos_max >= 0) - an empty
    sequence is always accepted as a fresh start - so reproducing the crash needs
    ctx_dft to hold real, nonzero, stale state, not just be cleared to empty.

    With a single slot, an unrelated prompt reusing that slot forces the target
    and draft contexts to reprocess from scratch and settle at a small, real
    position. Requesting the original long prompt again then hits the entry saved
    by request A: ctx_tgt gets restored straight to its high saved position, while
    (pre-fix) ctx_dft is left at the short unrelated prompt's small position - a
    real, nonzero mismatch, which is exactly what triggered the production bug's
    "inconsistent sequence positions" / failed-decode loop.
    """
    cache_dir = tempfile.mkdtemp(prefix="llama_disk_cache_spec_")
    fd, log_path = tempfile.mkstemp(suffix=".log", dir=cache_dir)
    os.close(fd)
    try:
        server_spec = _spec_server(cache_dir)
        server_spec.debug = True
        server_spec.log_path = log_path
        server_spec.start()

        # Request A: fills the only slot, generating with the draft model's help.
        # Evicted (and thus saved to disk) by request B below.
        res = server_spec.make_request("POST", "/completion", data={
            "prompt": LONG_PROMPT,
            "cache_prompt": True,
        })
        assert res.status_code == 200
        assert res.body["timings"]["draft_n"] > 0

        # Request B: unrelated short prompt reuses the same (only) slot, forcing
        # ctx_tgt and ctx_dft to reprocess from scratch and settle at a small,
        # real, nonzero position - the "stale state" ingredient the crash needs.
        res_b = server_spec.make_request("POST", "/completion", data={
            "prompt": SHORT_UNRELATED_PROMPT,
            "cache_prompt": True,
        })
        assert res_b.status_code == 200

        # Request C: the original long prompt again -> exact-hash disk cache hit,
        # restoring ctx_tgt to a position far beyond ctx_dft's stale one (still
        # sitting at request B's short prompt length). Generation must continue
        # normally, not desync and fail to decode.
        res_c = server_spec.make_request("POST", "/completion", data={
            "prompt": LONG_PROMPT,
            "cache_prompt": True,
        })
        assert res_c.status_code == 200
        assert len(res_c.body["content"]) > 0
        assert res_c.body["timings"]["cache_n"] > 0, (
            "expected request C to reuse cached tokens (a disk-cache restore), "
            f"got timings: {res_c.body['timings']}"
        )

        # Stop the server so its log is fully flushed before the final check
        # (the requests above only prove no crash happened; reading the full log
        # after a clean stop reliably confirms the restore path taken, without
        # racing the server's own log-flush timing).
        server_spec.stop()
        with open(log_path) as f:
            logs = f.read()
        assert "disk cache: exact hit" in logs, f"expected an exact disk-cache hit, got:\n{logs}"
        assert "inconsistent sequence positions" not in logs, (
            f"draft context desynced from target after disk-cache restore:\n{logs}"
        )
        assert "failed to decode" not in logs, f"decode failed after disk-cache restore:\n{logs}"
    finally:
        shutil.rmtree(cache_dir, ignore_errors=True)


def test_disk_cache_hit_after_clean_shutdown():
```

- [ ] **Step 2: Run the new test**

```bash
wsl.exe -e bash -lc "cd /mnt/w/projects/llm/llama.cpp/tools/server/tests && /home/t_grid0/.venv/bin/python -m pytest unit/test_disk_cache.py -k test_disk_cache_hit_with_speculative_decoding -v"
```

Expected: PASS. If it fails on the "exact hit" log assertion, double check the exact hit log line
text in `server-disk-cache.cpp`'s `load()` (currently: `"disk cache: exact hit for %zu tokens,
hash=%.8s..."`) matches the substring `"disk cache: exact hit"` asserted here - update the
assertion string (not the production log line) if this branch's wording differs from nathanw's.

- [ ] **Step 3: Commit**

```bash
git add tools/server/tests/unit/test_disk_cache.py
git commit -m "test: add speculative-decoding disk-cache regression test

Self-speculation setup (draft model == target model) exercises the
real production bug this branch now fixes: reuse the only slot with
an unrelated prompt to leave ctx_dft at a small, real, nonzero
position, then re-request the original long prompt to force an
exact-hash disk-cache hit that restores ctx_tgt far ahead of it.
Asserts generation succeeds with no 'inconsistent sequence positions'
or 'failed to decode' in the log.

Ported from nathanw-llamacpp ba8ed1a96."
```

---

### Task 8: Full regression build and test suite run

**Files:** none (verification only)

**Interfaces:**
- Consumes: all prior tasks
- Produces: final confirmation before this work is considered done

- [ ] **Step 1: Full reconfigure + build of the standing regression targets**

Per this project's established build convention (`llama-server`, `test-chat`,
`test-chat-peg-parser`):

```bash
wsl.exe -e bash -lc "cd /mnt/w/projects/llm/llama.cpp && cmake --build build --config Release -j 8 --target llama-server test-chat test-chat-peg-parser"
```

Expected: all three build clean.

- [ ] **Step 2: Run the C++ regression suites**

```bash
wsl.exe -e bash -lc "cd /mnt/w/projects/llm/llama.cpp && ./build/bin/test-chat"
wsl.exe -e bash -lc "cd /mnt/w/projects/llm/llama.cpp && ./build/bin/test-chat-peg-parser"
```

Expected: both pass fully (no failures), matching the baseline from prior ports on this branch
lineage (these are unrelated to disk cache but are this project's standing "did I break anything
else" check).

- [ ] **Step 3: Run the full `test_disk_cache.py` suite one final time**

```bash
wsl.exe -e bash -lc "cd /mnt/w/projects/llm/llama.cpp/tools/server/tests && /home/t_grid0/.venv/bin/python -m pytest unit/test_disk_cache.py -v"
```

Expected: all tests PASS, including the flipped n_stream test (Task 4) and the new speculative-
decoding test (Task 7).

- [ ] **Step 4: No commit needed for this task** (verification only; if any step above surfaces a
  failure, fix it as part of the task that introduced it and re-run from Step 1 rather than
  committing a fix here).
