# Disk KV Cache Design

> Persistent disk-based KV cache for llama.cpp server, replacing in-memory prompt cache when enabled.

## Problem

The current in-memory prompt cache (`--cache-ram`) evicts slot KV states when RAM is full, losing them permanently. When context is large or many slots are active, RAM cache fills quickly and slots must reprocess their full prompts on reuse, wasting compute.

## Solution

A disk-based cache that stores slot KV states as files on disk, indexed by content hash of the token sequence. When a slot is evicted or goes idle, its KV state is saved to disk. When a new request arrives for a slot with matching tokens, the KV state is loaded from disk before falling back to full prompt reprocessing.

## Requirements

1. **Automatic**: Configured via CLI args, monitored continuously, oldest entries evicted when size exceeds threshold
2. **Reliable**: Opt-in feature, failures are non-disruptive (log warning, continue)
3. **Transparent**: Works the same way as current RAM cache slot matching flow
4. **Replace RAM cache**: When disk cache is enabled, `--cache-ram` is ignored

## Architecture

### Files

| File | Responsibility |
|------|---------------|
| `tools/server/server-disk-cache.h` | `server_disk_cache` class declaration |
| `tools/server/server-disk-cache.cpp` | Implementation: init, load, save, evict, index management |
| `common/common.h` | Add `cache_disk_path`, `cache_disk_size_mib` to `common_params` |
| `common/arg.cpp` | Add `--cache-disk` and `--cache-disk-size` CLI arguments |
| `tools/server/server-context.cpp` | Integrate disk cache into slot lifecycle |

### Cache Storage Layout

```
<cache_dir>/
  index.json          # Metadata index (JSON)
  a1b2c3d4...e5f6.bin # KV state file (SHA256 hash of tokens)
  7g8h9i0j...k1l2.bin # ...
```

### Cache Entry Format

**Filename**: SHA256 hash of the token sequence (hex string, 64 characters)

**File content**: Binary KV state in existing `llama_state_seq_save_file()` format (no format change needed)

**Index entry** (per entry in `index.json`):
```json
{
  "<sha256_hash>": {
    "size_bytes": 12345678,
    "n_tokens": 512,
    "last_used_us": 1700000000000000
  }
}
```

### `server_disk_cache` Class

```cpp
struct server_disk_cache {
    // Initialize cache: create directory, load index
    bool init(const std::string& path, int32_t max_size_mib);

    // Load KV state from disk if cache entry exists for these tokens
    // Returns true if loaded successfully, false otherwise
    bool load(const server_tokens& tokens, llama_context* ctx, int32_t slot_id);

    // Save KV state to disk for these tokens
    // Returns true if saved successfully
    bool save(const server_tokens& tokens, llama_context* ctx, int32_t slot_id);

    // Evict the oldest entry (by last_used_us)
    // Returns true if an entry was evicted
    bool evict_oldest();

    // Get total cache size in bytes
    size_t get_total_size() const;

    // Check if adding a new entry of given size would exceed limit
    // If so, evict oldest entries until under limit
    void check_and_evict(size_t new_entry_size);

    // Validate and rebuild index from directory
    void validate_and_rebuild();

    // Check if cache is enabled
    bool is_enabled() const;
};
```

### Token Hashing

SHA256 of token IDs serialized as 4-byte little-endian integers, concatenated.

```
hash_input = token_0 (4 bytes LE) || token_1 (4 bytes LE) || ... || token_n (4 bytes LE)
hash = SHA256(hash_input)
```

### Integration Points

#### 1. Slot Selection (`get_available_slot`)

In `server-context.cpp`, before the existing `prompt_cache` operations:

```
// Before prompt_save (slot eviction):
disk_cache.save(slot.prompt.tokens, ctx, slot.id)

// Before prompt_load (slot reuse):
if (!disk_cache.load(task.tokens, ctx, slot.id)) {
    // Fall back to RAM prompt_cache if disk cache miss
    if (!slot.prompt_load(*prompt_cache, task.tokens)) {
        slot.prompt_clear(false);
    }
}
```

#### 2. Idle Slot Saving (`cache_idle_slots`)

In `server-context.cpp` where idle slots are saved:

```
// After saving to prompt_cache (or instead of it when disk cache is active):
disk_cache.save(slot.prompt.tokens, ctx, slot.id)
```

#### 3. Size Enforcement

After each successful `save()`, call `check_and_evict(new_entry_size)` to ensure total cache size stays under the limit.

### CLI Arguments

```
--cache-disk PATH        Path to disk cache directory. When non-empty, disk cache is enabled and RAM cache is bypassed.
--cache-disk-size N      Maximum cache size in MiB (default: 0 = no limit, -1 = no limit)
```

Environment variables: `LLAMA_ARG_CACHE_DISK`, `LLAMA_ARG_CACHE_DISK_SIZE`

### Error Handling

| Scenario | Behavior |
|----------|----------|
| Cache dir doesn't exist | Create on init |
| Index file missing/corrupted | Log warning, rebuild from directory |
| Cache entry file missing | Remove from index, continue |
| Load fails (I/O error, corruption) | Log warning, slot processes prompt fresh |
| Save fails (disk full, permission) | Log warning, slot continues normally |
| Eviction fails | Log warning, continue (size may temporarily exceed limit) |

All disk operations wrapped in try/catch with `SRV_WRN` logging. No exceptions propagate to inference flow.

### Startup Sequence

1. If `--cache-disk PATH` is set:
   a. Create directory if it doesn't exist
   b. Load `index.json` if present
   c. Validate index: remove entries pointing to missing `.bin` files
   d. Log: `disk cache: path=PATH, entries=N, total_size=X MiB`
2. If `--cache-disk-size N` is set: log the limit
3. If both `--cache-disk` and `--cache-ram` are set: log that RAM cache is bypassed

### Index File Format

```json
{
  "version": 1,
  "entries": {
    "<sha256_hex>": {
      "size_bytes": <uint64>,
      "n_tokens": <uint32>,
      "last_used_us": <uint64>
    }
  },
  "total_size_bytes": <uint64>
}
```

- `version`: for future schema compatibility
- `entries`: map of hash → metadata
- `total_size_bytes`: sum of all entry sizes (redundant but avoids recomputation)

### Thread Safety

The disk cache operates within the server's single-threaded task processing flow (via `update_slots()`). No additional locking needed. Index file writes are atomic (write to temp file, then rename).

### Edge Cases

1. **Concurrent access to same cache entry**: Not possible - only one thread processes slots
2. **Partial write during save**: Write to temp file, rename on success
3. **Index corruption**: Detected on load, full rebuild from directory
4. **Very large single entry**: No per-entry limit, but eviction handles total size
5. **Empty token sequence**: Skip save/load (nothing to cache)
