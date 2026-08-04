#pragma once

#include "server-common.h"
#include "llama.h"

#include <string>
#include <unordered_map>
#include <list>
#include <cstdint>

// Metadata for a single cache entry stored in the index
struct disk_cache_entry {
    uint64_t size_bytes = 0;
    uint32_t n_tokens = 0;
    int64_t last_used_us = 0;
    std::vector<llama_token> tokens;  // full token sequence; empty for old-format entries

    // size of the "<hash>.ckpt" sidecar file holding serialized context checkpoints
    // (0 if no checkpoints were saved alongside this entry)
    uint64_t ckpt_size_bytes = 0;
};

class server_disk_cache {
public:
    server_disk_cache() = default;
    ~server_disk_cache() = default;

    // Disable copy (manages resources)
    server_disk_cache(const server_disk_cache&) = delete;
    server_disk_cache& operator=(const server_disk_cache&) = delete;

    // Thread safety: This class is not thread-safe.
    // It is only used from the single-threaded server task loop.

    // Initialize cache: create directory if needed, load index
    // path: directory path for cache files
    // max_size_mib: max total size in MiB (0 = disabled, >0 = limit, -1 = unlimited)
    // Returns true on success
    bool init(const std::string& path, int32_t max_size_mib);

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

    // Save KV state to disk cache for the given tokens
    // tokens: the token sequence to use as the cache key (hashed)
    // ctx: llama context containing the KV state to save
    // slot_id: slot ID (used for state extraction)
    // full_tokens: the full token sequence corresponding to the saved KV state's
    //               position range (prompt + any generated continuation). If null
    //               or empty, defaults to `tokens`. Stored as entry.tokens so that
    //               restoring this entry yields prompt.tokens consistent with the
    //               restored memory's pos_max.
    // checkpoints: if non-null and non-empty, the slot's context checkpoints are
    //              serialized alongside the main KV state and restored on a later
    //              load()/load_by_hash() of this entry.
    // Returns true if saved successfully
    bool save(const server_tokens& tokens, llama_context* ctx, int32_t slot_id,
              const server_tokens* full_tokens = nullptr,
              const std::list<common_prompt_checkpoint> * checkpoints = nullptr);

    // Evict the oldest entry (by last_used_us timestamp)
    // Returns true if an entry was evicted, false if cache was empty
    bool evict_oldest();

    // Get total cache size in bytes
    size_t get_total_size() const;

    // Check if adding a new entry of given size would exceed limit
    // If so, evict oldest entries until under limit (or cache is empty)
    void check_and_evict(size_t new_entry_size);

    // Validate index: remove entries with missing .bin files
    // Called on startup to repair corrupted indexes
    void validate_and_rebuild();

    // Check if cache is initialized and enabled
    bool is_enabled() const;

    // Get the number of entries in the cache
    size_t n_entries() const;

    // Get the cache path
    const std::string& get_path() const;

    // Get cumulative hit/miss counts (loaded from stats.json on init, persisted on each access)
    uint64_t get_hits()   const { return m_hits;   }
    uint64_t get_misses() const { return m_misses; }

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

    // Read-only access to an index entry. Returns nullptr if hash not found.
    const disk_cache_entry * get_entry(const std::string & hash) const;

private:
    // Compute SHA256 hash of token sequence
    static std::string hash_tokens(const server_tokens& tokens);

    // Serialize/deserialize context checkpoints to/from the "<hash>.ckpt" sidecar file
    static bool write_checkpoints_file(const std::string & filepath, const std::list<common_prompt_checkpoint> & checkpoints);
    static bool read_checkpoints_file(const std::string & filepath, std::list<common_prompt_checkpoint> & out_checkpoints);

    // Load index from index.json file
    bool load_index();

    // Save index to index.json file (atomic: write to temp, then rename)
    bool save_index();

    // Load hit/miss stats from stats.json (called on init; missing file is not an error)
    void load_stats();

    // Save hit/miss stats to stats.json (atomic write; called after each hit or miss)
    void save_stats();

    // Path to cache directory
    std::string m_path;

    // Max size in bytes (0 = no limit)
    uint64_t m_max_size_bytes = 0;

    // Current total size in bytes
    size_t m_total_size = 0;

    // Index: hash -> entry metadata
    std::unordered_map<std::string, disk_cache_entry> m_index;

    // Whether cache is initialized
    bool m_enabled = false;

    // Cumulative hit/miss counters (persisted in stats.json)
    uint64_t m_hits   = 0;
    uint64_t m_misses = 0;
};
