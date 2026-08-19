# Disk Cache: Shutdown Save & Prefix Search

## Problem

Two gaps in the current disk KV cache system (`216485e`):

1. **Shutdown loss**: When the server stops (or enters sleeping state), active slot KV states in RAM/VRAM are freed without being saved to disk. Cache built up during a session is lost.

2. **Exact-only lookup**: Disk cache lookup is SHA256 of the full token sequence. If a new request shares a long prefix with a cached entry but is not identical, the cache is bypassed entirely and the prefix must be reprocessed from scratch.

## Solution

### Part 1 — Shutdown Save

Before `destroy()` frees the llama context, flush all non-empty, non-media slots to disk:

```cpp
void destroy() {
    if (disk_cache && ctx_tgt) {
        for (auto & slot : slots) {
            if (slot.prompt.n_tokens() > 0 && !slot.prompt.tokens.has_media()) {
                disk_cache->save(slot.prompt.tokens, ctx_tgt, slot.id);
            }
        }
    }
    llama_init.reset();  // frees ctx_tgt — must come after disk save
    ...
}
```

This also fires when the server enters sleeping state (which calls `destroy()`), covering "cache unloads by any reason → save to disk first."

### Part 2 — Disk Prefix Search

#### Data model

`disk_cache_entry` gains the full token sequence:

```cpp
struct disk_cache_entry {
    uint64_t size_bytes = 0;
    uint32_t n_tokens = 0;
    int64_t last_used_us = 0;
    std::vector<llama_token> tokens;  // new: stored in index.json as "tokens": [...]
};
```

Index format is backward compatible: old entries without a `tokens` field are kept, used for exact lookup, skipped during prefix search.

#### New methods on `server_disk_cache`

```cpp
// Find entry whose tokens are the longest common prefix of `tokens`
// Returns hash of best match, or "" if no entry reaches min_prefix_len
std::string find_best_prefix(const server_tokens & tokens, size_t min_prefix_len) const;

// Load KV state by hash directly (skips re-hashing, used after find_best_prefix)
// Updates last_used_us on the entry, same as load()
bool load_by_hash(const std::string & hash, llama_context * ctx, int32_t slot_id);

// Read-only access to an index entry (for retrieving tokens after find_best_prefix)
const disk_cache_entry * get_entry(const std::string & hash) const;
```

`find_best_prefix` iterates all entries with non-empty `tokens`, computes LCP with task tokens, returns the hash with the longest common prefix >= `min_prefix_len`.

#### Integration in `get_available_slot()`

After exact hash miss:

```cpp
// Exact match
bool loaded = false;
if (disk_cache && !task.tokens.has_media()) {
    loaded = disk_cache->load(task.tokens, ctx_tgt, ret->id);
    if (loaded) {
        ret->prompt.tokens = server_tokens(task.tokens.get_tokens(), task.tokens.has_mtmd);
    }
}

// Prefix match on exact miss
if (!loaded && disk_cache && !task.tokens.has_media()) {
    const float threshold = (slot_prompt_similarity > 0.0f) ? slot_prompt_similarity : 0.5f;
    const size_t min_prefix = std::max((size_t)1, (size_t)(task.tokens.size() * threshold));
    std::string prefix_hash = disk_cache->find_best_prefix(task.tokens, min_prefix);
    if (!prefix_hash.empty()) {
        loaded = disk_cache->load_by_hash(prefix_hash, ctx_tgt, ret->id);
        if (loaded) {
            // Set prompt.tokens to the matched prefix so update_slots() processes only the remainder
            const auto & entry = disk_cache->get_entry(prefix_hash);
            ret->prompt.tokens = server_tokens(
                llama_tokens(entry.tokens.begin(), entry.tokens.end()),
                task.tokens.has_mtmd);
        }
    }
}
```

#### Prefix threshold

| Condition | Threshold used |
|-----------|---------------|
| `slot_prompt_similarity` not set (0.0) | 0.5 (hardcoded default) |
| `slot_prompt_similarity` set (e.g. 0.8) | 0.8 |
| Disk cache disabled | N/A |

RAM slot selection behavior is unchanged; the fallback only affects disk prefix search.

## Changed Files

| File | Change |
|------|--------|
| `tools/server/server-disk-cache.h` | Add `tokens` to `disk_cache_entry`; add `find_best_prefix`, `load_by_hash`, `get_entry` |
| `tools/server/server-disk-cache.cpp` | Implement new methods; update `save`/`load_index`/`save_index` for `tokens` field |
| `tools/server/server-context.cpp` | Shutdown flush in `destroy()`; prefix search in `get_available_slot()` |

## Edge Cases

| Scenario | Behavior |
|----------|----------|
| Slot is processing at shutdown | Saved if non-empty and non-media; `prompt.tokens` at that point includes generated tokens, which is a valid cacheable state for that conversation |
| Entry has no `tokens` (old format) | Skipped in prefix search, used normally for exact lookup |
| Prefix match loads partial context | `prompt.tokens` set to matched prefix; `update_slots()` processes remainder normally |
| `find_best_prefix` scans large index | Linear scan O(n * min(prefix_len, entry_len)); acceptable at typical cache sizes |
| Sleeping state entry | Same as shutdown: `destroy()` is called, flush fires automatically |
