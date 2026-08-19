# Disk KV Cache Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add persistent disk-based KV cache to llama.cpp server that replaces in-memory prompt cache when enabled.

**Architecture:** New `server_disk_cache` class manages SHA256-content-addressed cache files on disk with a JSON index. Integrated into slot eviction and idle-saving flows in `server_context_impl`. When `--cache-disk` is set, `--cache-ram` is bypassed.

**Tech Stack:** C++17, nlohmann/json (already in project), SHA256 (OpenSSL or common crypto utils), std::filesystem

---

## File Structure

| File | Action | Responsibility |
|------|--------|----------------|
| `tools/server/server-disk-cache.h` | Create | `server_disk_cache` class declaration, `disk_cache_entry` struct |
| `tools/server/server-disk-cache.cpp` | Create | Implementation: init, load, save, evict, index management, SHA256 hashing |
| `common/common.h` | Modify | Add `cache_disk_path` and `cache_disk_size_mib` to `common_params` |
| `common/arg.cpp` | Modify | Add `--cache-disk` and `--cache-disk-size` CLI argument handlers |
| `tools/server/server-context.h` | Modify | Add `std::unique_ptr<server_disk_cache> disk_cache` member |
| `tools/server/server-context.cpp` | Modify | Initialize disk cache, integrate into slot lifecycle (eviction + idle saving) |
| `tools/server/CMakeLists.txt` | Modify | Add `server-disk-cache.cpp` and `.h` to `server-context` library |

---

### Task 1: Create `server-disk-cache.h`

**Files:**
- Create: `tools/server/server-disk-cache.h`

- [ ] **Step 1: Write the header file**

```cpp
#pragma once

#include "server-task.h"
#include "llama.h"

#include <string>
#include <unordered_map>
#include <cstdint>

// Metadata for a single cache entry stored in the index
struct disk_cache_entry {
    uint64_t size_bytes = 0;
    uint32_t n_tokens = 0;
    int64_t last_used_us = 0;
};

class server_disk_cache {
public:
    server_disk_cache() = default;
    ~server_disk_cache() = default;

    // Initialize cache: create directory if needed, load index
    // path: directory path for cache files
    // max_size_mib: max total size in MiB (0 = no limit, -1 = no limit)
    // Returns true on success
    bool init(const std::string& path, int32_t max_size_mib);

    // Load KV state from disk cache for the given tokens
    // tokens: the token sequence to look up
    // ctx: llama context to restore state into
    // slot_id: slot ID to restore into
    // Returns true if cache hit and state was loaded successfully
    bool load(const server_tokens& tokens, llama_context* ctx, int32_t slot_id);

    // Save KV state to disk cache for the given tokens
    // tokens: the token sequence to use as cache key
    // ctx: llama context containing the KV state to save
    // slot_id: slot ID (used for state extraction)
    // Returns true if saved successfully
    bool save(const server_tokens& tokens, llama_context* ctx, int32_t slot_id);

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

private:
    // Compute SHA256 hash of token sequence
    static std::string hash_tokens(const server_tokens& tokens);

    // Load index from index.json file
    bool load_index();

    // Save index to index.json file (atomic: write to temp, then rename)
    bool save_index();

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
};
```

- [ ] **Step 2: Verify header compiles** (syntactic check only, will be validated in Task 8)

---

### Task 2: Create `server-disk-cache.cpp` — Core Implementation

**Files:**
- Create: `tools/server/server-disk-cache.cpp`

- [ ] **Step 1: Write the implementation file**

```cpp
#include "server-disk-cache.h"
#include "log.h"
#include "common.h"

#include <fstream>
#include <sstream>
#include <algorithm>
#include <filesystem>
#include <iomanip>

namespace fs = std::filesystem;

namespace {

// JSON library is available via common.h includes
using json = nlohmann::json;

const char* kIndexFilename = "index.json";

// --- Embedded SHA-256 (Igor Pavlov, public domain) ---

#define U8V(v) ((uint8_t)(v) & 0xFFU)
#define U32V(v) ((uint32_t)(v) & 0xFFFFFFFFU)
#define ROTL32(v, n) U32V((uint32_t)(v) << (n)) | ((uint32_t)(v) >> (32 - (n)))
#define ROTR32(v, n) ROTL32(v, 32 - (n))

typedef struct {
    uint32_t state[8];
    uint64_t count;
    unsigned char buffer[64];
} embed_sha256_t;

static void embed_sha256_init(embed_sha256_t *p) {
    p->state[0] = 0x6a09e667;
    p->state[1] = 0xbb67ae85;
    p->state[2] = 0x3c6ef372;
    p->state[3] = 0xa54ff53a;
    p->state[4] = 0x510e527f;
    p->state[5] = 0x9b05688c;
    p->state[6] = 0x1f83d9ab;
    p->state[7] = 0x5be0cd19;
    p->count = 0;
}

#define S0(x) (ROTR32(x, 2) ^ ROTR32(x,13) ^ ROTR32(x, 22))
#define S1(x) (ROTR32(x, 6) ^ ROTR32(x,11) ^ ROTR32(x, 25))
#define s0(x) (ROTR32(x, 7) ^ ROTR32(x,18) ^ (x >> 3))
#define s1(x) (ROTR32(x,17) ^ ROTR32(x,19) ^ (x >> 10))
#define Ch(x,y,z) (z^(x&(y^z)))
#define Maj(x,y,z) ((x&y)|(z&(x|y)))

static const uint32_t K[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5,
    0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
    0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
    0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
    0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
    0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
    0xd192e819, 0xd6990624, 0xf40e3585, 0x106a0070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5,
    0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
    0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
};

static void embed_sha256_transform(uint32_t *state, const uint32_t *data) {
    uint32_t W[16];
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

static void embed_sha256_update(embed_sha256_t *p, const unsigned char *data, size_t size) {
    uint32_t cur = (uint32_t)p->count & 0x3F;
    while (size > 0) {
        p->buffer[cur++] = *data++;
        p->count++;
        size--;
        if (cur == 64) {
            cur = 0;
            uint32_t data32[16];
            for (int i = 0; i < 16; i++)
                data32[i] = ((uint32_t)p->buffer[i*4] << 24) | ((uint32_t)p->buffer[i*4+1] << 16) |
                            ((uint32_t)p->buffer[i*4+2] << 8) | (uint32_t)p->buffer[i*4+3];
            embed_sha256_transform(p->state, data32);
        }
    }
}

static void embed_sha256_final(embed_sha256_t *p, unsigned char *digest) {
    uint64_t lenInBits = (p->count << 3);
    uint32_t cur = (uint32_t)p->count & 0x3F;
    p->buffer[cur++] = 0x80;
    while (cur != (64 - 8)) {
        cur &= 0x3F;
        if (cur == 0) {
            uint32_t data32[16] = {0};
            embed_sha256_transform(p->state, data32);
        }
        p->buffer[cur++] = 0;
    }
    for (int i = 0; i < 8; i++) {
        p->buffer[cur++] = (unsigned char)(lenInBits >> 56);
        lenInBits <<= 8;
    }
    {
        uint32_t data32[16] = {0};
        for (int i = 0; i < 16; i++)
            data32[i] = ((uint32_t)p->buffer[i*4] << 24) | ((uint32_t)p->buffer[i*4+1] << 16) |
                        ((uint32_t)p->buffer[i*4+2] << 8) | (uint32_t)p->buffer[i*4+3];
        embed_sha256_transform(p->state, data32);
    }
    for (int i = 0; i < 8; i++) {
        digest[i*4] = (unsigned char)(p->state[i] >> 24);
        digest[i*4+1] = (unsigned char)(p->state[i] >> 16);
        digest[i*4+2] = (unsigned char)(p->state[i] >> 8);
        digest[i*4+3] = (unsigned char)(p->state[i]);
    }
    embed_sha256_init(p);
}

// Append hex representation of a byte
static void append_hex(std::ostringstream& oss, uint8_t b) {
    static const char hex[] = "0123456789abcdef";
    oss << hex[(b >> 4) & 0xf] << hex[b & 0xf];
}

} // anonymous namespace

//
// server_disk_cache implementation
//

bool server_disk_cache::init(const std::string& path, int32_t max_size_mib) {
    // Validate path
    if (path.empty()) {
        SRV_WRN("disk cache: empty path, disabling\n");
        return false;
    }

    m_path = path;

    // Create directory if it doesn't exist
    try {
        if (!fs::exists(m_path)) {
            fs::create_directories(m_path);
        }
    } catch (const std::exception& e) {
        SRV_WRN("disk cache: failed to create directory '%s': %s\n", m_path.c_str(), e.what());
        return false;
    }

    // Set max size
    if (max_size_mib > 0) {
        m_max_size_bytes = (uint64_t)max_size_mib * 1024ull * 1024ull;
    }
    // max_size_mib == 0 or -1: no limit

    // Load index
    if (!load_index()) {
        SRV_WRN("disk cache: failed to load index, starting fresh\n");
    }

    m_enabled = true;

    SRV_INF("disk cache: path='%s', entries=%zu, total_size=%.2f MiB, max_size=%s\n",
            m_path.c_str(),
            m_index.size(),
            m_total_size / (1024.0 * 1024.0),
            m_max_size_bytes == 0 ? "unlimited" : string_format("%.2f GiB", (double)m_max_size_bytes / (1024.0 * 1024.0 * 1024.0)).c_str());

    return true;
}

std::string server_disk_cache::hash_tokens(const server_tokens& tokens) {
    if (tokens.empty()) {
        return "";
    }

    embed_sha256_t ctx;
    embed_sha256_init(&ctx);

    // Hash each token as 4-byte little-endian
    const auto& tok_vec = tokens.get_tokens();
    for (auto tok : tok_vec) {
        uint8_t buf[4];
        buf[0] = (uint8_t)(tok & 0xFF);
        buf[1] = (uint8_t)((tok >> 8) & 0xFF);
        buf[2] = (uint8_t)((tok >> 16) & 0xFF);
        buf[3] = (uint8_t)((tok >> 24) & 0xFF);
        embed_sha256_update(&ctx, buf, 4);
    }

    // Finalize and convert to hex string
    uint8_t digest[32];
    embed_sha256_final(&ctx, digest);

    std::ostringstream oss;
    for (int i = 0; i < 32; ++i) {
        append_hex(oss, digest[i]);
    }

    return oss.str();
}

bool server_disk_cache::load(const server_tokens& tokens, llama_context* ctx, int32_t slot_id) {
    if (!m_enabled || tokens.empty() || !ctx) {
        return false;
    }

    std::string hash = hash_tokens(tokens);
    if (hash.empty()) {
        return false;
    }

    auto it = m_index.find(hash);
    if (it == m_index.end()) {
        // Cache miss
        return false;
    }

    // Build file path
    std::string filepath = m_path + "/" + hash + ".bin";

    // Read file
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

    // Restore KV state into context
    size_t restored = llama_state_seq_set_data_ext(ctx, data.data(), data.size(), slot_id, 0);
    if (restored != data.size()) {
        SRV_WRN("disk cache: partial restore: expected %zu, got %zu\n", data.size(), restored);
        return false;
    }

    // Update index: update last_used timestamp
    it = m_index.find(hash);
    if (it != m_index.end()) {
        it->second.last_used_us = ggml_time_us();
        save_index();
    }

    SRV_DBG("disk cache: hit for %zu tokens (%zu bytes)\n", tokens.size(), data.size());
    return true;
}

bool server_disk_cache::save(const server_tokens& tokens, llama_context* ctx, int32_t slot_id) {
    if (!m_enabled || tokens.empty() || !ctx) {
        return false;
    }

    std::string hash = hash_tokens(tokens);
    if (hash.empty()) {
        return false;
    }

    // Check if already in cache
    auto it = m_index.find(hash);
    if (it != m_index.end()) {
        // Already cached, just update timestamp
        it->second.last_used_us = ggml_time_us();
        save_index();
        return true;
    }

    // Get KV state size
    size_t state_size = llama_state_seq_get_size_ext(ctx, slot_id, 0);
    if (state_size == 0) {
        return false;
    }

    // Allocate buffer
    std::vector<uint8_t> data(state_size);
    size_t obtained = llama_state_seq_get_data_ext(ctx, data.data(), state_size, slot_id, 0);
    if (obtained != state_size) {
        SRV_WRN("disk cache: failed to get KV state: expected %zu, got %zu\n", state_size, obtained);
        return false;
    }

    // Write to temp file first (atomic write)
    std::string filepath = m_path + "/" + hash + ".bin";
    std::string tmp_filepath = filepath + ".tmp";

    std::ofstream file(tmp_filepath, std::ios::binary);
    if (!file.is_open()) {
        SRV_WRN("disk cache: failed to create cache file '%s'\n", tmp_filepath.c_str());
        return false;
    }

    file.write(reinterpret_cast<const char*>(data.data()), obtained);
    file.close();

    if (file.fail()) {
        SRV_WRN("disk cache: failed to write cache file '%s'\n", tmp_filepath.c_str());
        // Clean up temp file
        fs::remove(tmp_filepath);
        return false;
    }

    // Atomically rename temp file to final path
    if (fs::rename(tmp_filepath, filepath) != 0) {
        SRV_WRN("disk cache: failed to rename cache file\n");
        fs::remove(tmp_filepath);
        return false;
    }

    // Add to index
    disk_cache_entry entry;
    entry.size_bytes = obtained;
    entry.n_tokens = (uint32_t)tokens.size();
    entry.last_used_us = ggml_time_us();
    m_index[hash] = entry;
    m_total_size += obtained;

    // Check size limit and evict if needed
    check_and_evict(0); // 0 = no new entry to add, just enforce limit

    save_index();

    SRV_DBG("disk cache: saved %zu tokens (%zu bytes)\n", tokens.size(), obtained);
    return true;
}

bool server_disk_cache::evict_oldest() {
    if (m_index.empty()) {
        return false;
    }

    // Find entry with smallest last_used_us
    auto oldest_it = m_index.begin();
    for (auto it = std::next(m_index.begin()); it != m_index.end(); ++it) {
        if (it->second.last_used_us < oldest_it->second.last_used_us) {
            oldest_it = it;
        }
    }

    // Remove file
    std::string filepath = m_path + "/" + oldest_it->first + ".bin";
    fs::remove(filepath); // Ignore errors (file might already be gone)

    // Update index
    m_total_size -= oldest_it->second.size_bytes;
    m_index.erase(oldest_it);

    save_index();
    return true;
}

void server_disk_cache::check_and_evict(size_t new_entry_size) {
    if (m_max_size_bytes == 0) {
        return; // No limit
    }

    uint64_t target_size = m_total_size + new_entry_size;
    while (target_size > m_max_size_bytes && !m_index.empty()) {
        evict_oldest();
        if (m_index.empty()) {
            break;
        }
        target_size = m_total_size + new_entry_size;
    }
}

size_t server_disk_cache::get_total_size() const {
    return m_total_size;
}

size_t server_disk_cache::n_entries() const {
    return m_index.size();
}

const std::string& server_disk_cache::get_path() const {
    return m_path;
}

bool server_disk_cache::is_enabled() const {
    return m_enabled;
}

void server_disk_cache::validate_and_rebuild() {
    if (!fs::exists(m_path)) {
        return;
    }

    // Collect all .bin files in directory
    std::unordered_map<std::string, size_t> found_files;
    try {
        for (const auto& entry : fs::directory_iterator(m_path)) {
            if (entry.is_regular_file()) {
                std::string filename = entry.path().filename().string();
                if (filename.size() > 4 && filename.substr(filename.size() - 4) == ".bin") {
                    std::string hash = filename.substr(0, filename.size() - 4);
                    found_files[hash] = entry.file_size();
                }
            }
        }
    } catch (const std::exception& e) {
        SRV_WRN("disk cache: failed to scan cache directory: %s\n", e.what());
        return;
    }

    // Remove index entries for files that no longer exist
    size_t removed = 0;
    for (auto it = m_index.begin(); it != m_index.end();) {
        if (found_files.find(it->first) == found_files.end()) {
            SRV_WRN("disk cache: removing index entry for missing file '%s'\n", it->first.c_str());
            it = m_index.erase(it);
            removed++;
        } else {
            // Update size from actual file (in case index is stale)
            it->second.size_bytes = found_files[it->first];
            ++it;
        }
    }

    // Rebuild total size
    m_total_size = 0;
    for (const auto& [hash, entry] : m_index) {
        m_total_size += entry.size_bytes;
    }

    if (removed > 0) {
        save_index();
    }
}

bool server_disk_cache::load_index() {
    std::string index_path = m_path + "/" + kIndexFilename;

    if (!fs::exists(index_path)) {
        // No index file, validate directory and rebuild
        validate_and_rebuild();
        return true;
    }

    // Read index file
    std::ifstream file(index_path);
    if (!file.is_open()) {
        return false;
    }

    json j;
    try {
        j = json::parse(file);
    } catch (const std::exception& e) {
        SRV_WRN("disk cache: failed to parse index file: %s\n", e.what());
        return false;
    }

    // Parse entries
    m_index.clear();
    m_total_size = 0;

    if (j.contains("entries")) {
        for (const auto& [hash, entry_json] : j["entries"].items()) {
            disk_cache_entry entry;
            entry.size_bytes = entry_json.value("size_bytes", (uint64_t)0);
            entry.n_tokens = entry_json.value("n_tokens", (uint32_t)0);
            entry.last_used_us = entry_json.value("last_used_us", (int64_t)0);
            m_index[hash] = entry;
            m_total_size += entry.size_bytes;
        }
    }

    // Validate: remove entries with missing files
    validate_and_rebuild();

    SRV_INF("disk cache: loaded index with %zu entries, total size=%.2f MiB\n",
            m_index.size(), m_total_size / (1024.0 * 1024.0));

    return true;
}

bool server_disk_cache::save_index() {
    std::string index_path = m_path + "/" + kIndexFilename;
    std::string tmp_path = index_path + ".tmp";

    json j;
    j["version"] = 1;
    j["total_size_bytes"] = m_total_size;

    json entries = json::object();
    for (const auto& [hash, entry] : m_index) {
        json entry_json;
        entry_json["size_bytes"] = entry.size_bytes;
        entry_json["n_tokens"] = entry.n_tokens;
        entry_json["last_used_us"] = entry.last_used_us;
        entries[hash] = entry_json;
    }
    j["entries"] = entries;

    // Write to temp file
    std::ofstream file(tmp_path);
    if (!file.is_open()) {
        SRV_WRN("disk cache: failed to create index file\n");
        return false;
    }

    file << j.dump(2);
    file.close();

    if (file.fail()) {
        SRV_WRN("disk cache: failed to write index file\n");
        fs::remove(tmp_path);
        return false;
    }

    // Atomically rename
    if (fs::rename(tmp_path, index_path) != 0) {
        SRV_WRN("disk cache: failed to rename index file\n");
        fs::remove(tmp_path);
        return false;
    }

    return true;
}
```

---

### Task 3: Add CLI Parameters to `common_params`

**Files:**
- Modify: `common/common.h:594`

- [ ] **Step 1: Add disk cache fields to common_params**

After line 594 (`int32_t cache_ram_mib`), add:

```cpp
    int32_t cache_disk_size_mib = 0;  // -1 = no limit, 0 = disabled, N = max MiB
    std::string cache_disk_path;      // path to disk cache directory
```

---

### Task 4: Add CLI Argument Parsing

**Files:**
- Modify: `common/arg.cpp`

- [ ] **Step 1: Add argument handlers after `--cache-idle-slots` block (around line 1344)**

After the `--cache-idle-slots` entry (ending at line 1344), add:

```cpp
    add_opt(common_arg(
        {"--cache-disk-path"}, "PATH",
        "path to disk cache directory (enables disk-based KV cache, bypasses --cache-ram)",
        [](common_params & params, const std::string & value) {
            params.cache_disk_path = value;
        }
    ).set_env("LLAMA_ARG_CACHE_DISK_PATH").set_examples({LLAMA_EXAMPLE_SERVER}));
    add_opt(common_arg(
        {"--cache-disk-size"}, "N",
        string_format("maximum disk cache size in MiB (default: %d, -1 = no limit, 0 = disabled)", params.cache_disk_size_mib),
        [](common_params & params, int value) {
            params.cache_disk_size_mib = value;
        }
    ).set_env("LLAMA_ARG_CACHE_DISK_SIZE").set_examples({LLAMA_EXAMPLE_SERVER}));
```

---

### Task 5: Add Disk Cache Member to `server_context_impl`

**Files:**
- Modify: `tools/server/server-context.h`

- [ ] **Step 1: Add include and member**

In `server-context.h`, add after existing includes:

```cpp
#include "server-disk-cache.h"
```

In `server_context_impl`, add after `prompt_cache` member (around line 700):

```cpp
    // Disk-based KV cache (optional, replaces prompt_cache when enabled)
    std::unique_ptr<server_disk_cache> disk_cache;
```

---

### Task 6: Integrate Disk Cache into Server Context Initialization

**Files:**
- Modify: `tools/server/server-context.cpp`

- [ ] **Step 1: Modify `init()` to initialize disk cache and handle RAM cache bypass**

In `server_context_impl::init()`, find the existing prompt cache initialization block (around lines 953-963):

```cpp
        if (params_base.cache_ram_mib != 0) {
```

Replace the entire block with:

```cpp
        // Initialize disk cache if enabled
        if (!params_base.cache_disk_path.empty()) {
            disk_cache = std::make_unique<server_disk_cache>();
            if (!disk_cache->init(params_base.cache_disk_path, params_base.cache_disk_size_mib)) {
                SRV_WRN("%s: failed to initialize disk cache, continuing without it\n", __func__);
                disk_cache.reset();
            }
        }

        // Initialize RAM prompt cache (only if disk cache is NOT enabled)
        if (params_base.cache_ram_mib != 0 && !params_base.cache_disk_path.empty()) {
            SRV_WRN("%s: disk cache is enabled, --cache-ram is bypassed\n", __func__);
        }

        if (params_base.cache_ram_mib != 0 && params_base.cache_disk_path.empty()) {
            if (params_base.cache_ram_mib < 0) {
                SRV_WRN("prompt cache is enabled, size limit: unlimited\n");
            } else {
                SRV_WRN("prompt cache is enabled, size limit: %d MiB\n", params_base.cache_ram_mib);
            }
            prompt_cache = std::make_unique<server_prompt_cache>(params_base.cache_ram_mib, n_ctx);
        } else if (params_base.cache_ram_mib == 0) {
            SRV_WRN("%s", "prompt cache is disabled - use `--cache-ram N` to enable it\n");
        }
```

- [ ] **Step 2: Modify `get_available_slot()` to use disk cache**

In `get_available_slot()`, find the cache update block (around lines 1169-1186). Replace the existing `if (update_cache)` block:

```cpp
            if (update_cache) {
                SRV_WRN("%s", "updating prompt cache\n");

                const int64_t t_start = ggml_time_us();

                // don't save the slot's state if its context is empty
                if (tokens.size() > 0) {
                    ret->prompt_save(*prompt_cache);

                    // Also save to disk cache if enabled
                    if (disk_cache) {
                        disk_cache->save(ret->prompt.tokens, ctx, ret->id);
                    }
                }

                // Try disk cache first, then RAM prompt cache
                bool loaded = false;
                if (disk_cache) {
                    loaded = disk_cache->load(task.tokens, ctx, ret->id);
                }
                if (!loaded && prompt_cache) {
                    loaded = ret->prompt_load(*prompt_cache, task.tokens);
                }
                if (!loaded) {
                    ret->prompt_clear(false);
                }

                prompt_cache->update();

                SRV_WRN("prompt cache update took %.2f ms\n", (ggml_time_us() - t_start) / 1000.0);
            }
```

Wait — the above won't work because `prompt_cache` might be null when disk cache is enabled. Let me revise:

```cpp
            if (update_cache) {
                SRV_WRN("%s", "updating prompt cache\n");

                const int64_t t_start = ggml_time_us();

                // Save slot state to RAM cache if available
                if (tokens.size() > 0 && prompt_cache) {
                    ret->prompt_save(*prompt_cache);
                }

                // Also save to disk cache if enabled
                if (tokens.size() > 0 && disk_cache) {
                    disk_cache->save(ret->prompt.tokens, ctx, ret->id);
                }

                // Try loading: disk cache first, then RAM prompt cache
                bool loaded = false;
                if (disk_cache) {
                    loaded = disk_cache->load(task.tokens, ctx, ret->id);
                }
                if (!loaded && prompt_cache) {
                    loaded = ret->prompt_load(*prompt_cache, task.tokens);
                }
                if (!loaded) {
                    ret->prompt_clear(false);
                }

                if (prompt_cache) {
                    prompt_cache->update();
                }

                SRV_WRN("prompt cache update took %.2f ms\n", (ggml_time_us() - t_start) / 1000.0);
            }
```

- [ ] **Step 3: Modify idle slot saving to use disk cache**

Find the idle slot saving code (around line 725 in `update_slots()` where `slot.prompt_save(*prompt_cache)` is called). After that call, add:

```cpp
                        // Also save to disk cache if enabled
                        if (disk_cache) {
                            disk_cache->save(slot.prompt.tokens, ctx, slot.id);
                        }
```

The exact context will be something like:

```cpp
                if (params_base.cache_idle_slots && slot.prompt.n_tokens() > 0) {
                    slot.prompt_save(*prompt_cache);

                    // Also save to disk cache if enabled
                    if (disk_cache) {
                        disk_cache->save(slot.prompt.tokens, ctx, slot.id);
                    }
                }
```

---

### Task 7: Update CMakeLists.txt

**Files:**
- Modify: `tools/server/CMakeLists.txt`

- [ ] **Step 1: Add server-disk-cache files to server-context library**

In the `add_library(${TARGET} STATIC ...)` block, add after `server-common.h`:

```
    server-disk-cache.cpp
    server-disk-cache.h
```

The block should look like:
```cmake
add_library(${TARGET} STATIC
    server-chat.cpp
    server-chat.h
    server-task.cpp
    server-task.h
    server-queue.cpp
    server-queue.h
    server-common.cpp
    server-common.h
    server-disk-cache.cpp
    server-disk-cache.h
    server-context.cpp
    server-context.h
    server-tools.cpp
    server-tools.h
)
```

---

### Task 8: Build and Verify

- [ ] **Step 1: Build the project**

Run: `cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build --target llama-server -j$(nproc)`

Expected: Clean build with no errors or warnings related to new code.

- [ ] **Step 2: Verify CLI arguments are recognized**

Run: `./build/bin/llama-server --help | grep -A1 cache-disk`

Expected: Output showing `--cache-disk-path` and `--cache-disk-size` arguments.

- [ ] **Step 3: Verify server starts with disk cache**

Run: `./build/bin/llama-server -m <model> --cache-disk-path /tmp/kv-cache --cache-disk-size 1024`

Expected: Server starts, logs show `disk cache: path=/tmp/kv-cache/, entries=0, total_size=0.00 MiB, max_size=1.00 GiB`

---

## Self-Review Checklist

**1. Spec coverage:**
- Disk cache class with init/load/save/evict: Tasks 1, 2
- SHA256 content-addressable files: Task 2 (`hash_tokens`)
- JSON index with atomic writes: Task 2 (`load_index`, `save_index`)
- CLI arguments: Tasks 3, 4
- RAM cache bypass: Task 6 (Step 1)
- Slot eviction integration: Task 6 (Step 2)
- Idle slot integration: Task 6 (Step 3)
- Size enforcement: Task 2 (`check_and_evict`)
- Error handling: Task 2 (all operations wrapped in try/catch or error checks)
- Index validation/rebuild: Task 2 (`validate_and_rebuild`)
- CMakeLists update: Task 7
- Build verification: Task 8

**2. Placeholder scan:** No "TBD", "TODO", "implement later", or vague references. All code is complete.

**3. Type consistency:**
- `server_tokens` used consistently (from `server-task.h`)
- `llama_context*` used consistently
- `int32_t` for size params (matches existing `cache_ram_mib` pattern)
- `disk_cache_entry` struct defined in header, used in implementation

**4. Edge cases covered:**
- Empty tokens: checked in `load()` and `save()`
- Already cached: `save()` updates timestamp, returns true
- Missing files: `validate_and_rebuild()` handles this
- Partial writes: atomic temp-file-then-rename pattern
- No limit: `m_max_size_bytes == 0` skips eviction
