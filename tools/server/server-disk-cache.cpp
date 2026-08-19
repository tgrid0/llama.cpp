#include "server-disk-cache.h"
#include "log.h"
#include "common.h"

#include <cinttypes>
#include <cstring>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <filesystem>
#include <nlohmann/json.hpp>

namespace fs = std::filesystem;

namespace {

const char* kIndexFilename = "index.json";
const char* kStatsFilename = "stats.json";
static const uint32_t kIndexFormatVersion = 3; // bumped for n_stream-aware entries

// Build a path within the cache directory using std::filesystem for cross-platform compatibility
static std::string cache_path(const std::string& dir, const std::string& filename) {
    return (fs::path(dir) / filename).string();
}

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
        if (cur > 56) {
            memset(p->buffer + cur, 0, 64 - cur);
            uint32_t data32[16];
            for (int i = 0; i < 16; i++)
                data32[i] = ((uint32_t)p->buffer[i*4] << 24) | ((uint32_t)p->buffer[i*4+1] << 16) |
                            ((uint32_t)p->buffer[i*4+2] << 8) | (uint32_t)p->buffer[i*4+3];
            embed_sha256_transform(p->state, data32);
            cur = 0;
        } else {
            p->buffer[cur++] = 0;
        }
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
}

// Append hex representation of a byte
static void append_hex(std::ostringstream& oss, uint8_t b) {
    static const char hex[] = "0123456789abcdef";
    oss << hex[(b >> 4) & 0xf] << hex[b & 0xf];
}

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

} // anonymous namespace

//
// server_disk_cache implementation
//

bool server_disk_cache::init(const std::string& path, int32_t max_size_mib, uint32_t n_stream) {
    m_n_stream = n_stream;

    // Validate path
    if (path.empty()) {
        SRV_WRN("%s", "disk cache: empty path, disabling\n");
        return false;
    }

    // 0 = disabled (as documented in --cache-disk-size help text)
    if (max_size_mib == 0) {
        SRV_WRN("%s", "disk cache: size is 0, disabling\n");
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
    // max_size_mib < 0 (e.g. -1): no limit

    // Load index
    if (!load_index()) {
        SRV_WRN("%s", "disk cache: failed to load index, starting fresh\n");
    }

    load_stats();

    m_enabled = true;

    SRV_INF("disk cache: path='%s', entries=%zu, total_size=%.2f MiB, max_size=%s, n_stream=%u\n",
            m_path.c_str(),
            m_index.size(),
            m_total_size / (1024.0 * 1024.0),
            m_max_size_bytes == 0 ? "unlimited" : string_format("%zu MiB", (size_t)(m_max_size_bytes / (1024ull * 1024ull))).c_str(),
            m_n_stream);

    return true;
}

std::string server_disk_cache::hash_tokens(const server_tokens& tokens) {
    if (tokens.empty()) {
        return "";
    }

    embed_sha256_t ctx;
    embed_sha256_init(&ctx);

    // Hash text tokens as 4-byte little-endian; for media chunks, hash their content-derived id
    // instead so different images at the same position with identical surrounding text don't collide.
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
            const size_t n_tok = mtmd_input_chunk_get_n_tokens(chunk.get());
            if (n_tok == 0) {
                SRV_WRN("%s", "disk cache: media chunk reports 0 tokens, skipping\n");
                i += 1;  // Safely advance by 1 to avoid infinite loop
            } else {
                i += n_tok;
            }
        } else {
            hash_u32((uint32_t) tok);
            i += 1;
        }
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

// Binary format for "<hash>.ckpt" sidecar files:
//   uint32_t magic   ("CKPT")
//   uint32_t version (1)
//   uint32_t count
//   repeated `count` times:
//     int64_t  n_tokens
//     int32_t  pos_min
//     int32_t  pos_max
//     uint64_t data_tgt_size; uint8_t data_tgt[data_tgt_size]
//     uint64_t data_dft_size; uint8_t data_dft[data_dft_size]
static const uint32_t kCkptMagic   = 0x54504b43u; // "CKPT"
static const uint32_t kCkptVersion = 1u;

bool server_disk_cache::write_checkpoints_file(const std::string & filepath, const std::list<common_prompt_checkpoint> & checkpoints) {
    std::ofstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        return false;
    }

    const uint32_t count = (uint32_t) checkpoints.size();
    file.write(reinterpret_cast<const char *>(&kCkptMagic),   sizeof(kCkptMagic));
    file.write(reinterpret_cast<const char *>(&kCkptVersion), sizeof(kCkptVersion));
    file.write(reinterpret_cast<const char *>(&count),        sizeof(count));

    for (const auto & ckpt : checkpoints) {
        const int64_t  n_tokens   = ckpt.n_tokens;
        const int32_t  pos_min    = ckpt.pos_min;
        const int32_t  pos_max    = ckpt.pos_max;
        const uint64_t tgt_size   = ckpt.data_tgt.size();
        const uint64_t dft_size   = ckpt.data_dft.size();

        file.write(reinterpret_cast<const char *>(&n_tokens), sizeof(n_tokens));
        file.write(reinterpret_cast<const char *>(&pos_min),  sizeof(pos_min));
        file.write(reinterpret_cast<const char *>(&pos_max),  sizeof(pos_max));
        file.write(reinterpret_cast<const char *>(&tgt_size), sizeof(tgt_size));
        if (tgt_size > 0) {
            file.write(reinterpret_cast<const char *>(ckpt.data_tgt.data()), tgt_size);
        }
        file.write(reinterpret_cast<const char *>(&dft_size), sizeof(dft_size));
        if (dft_size > 0) {
            file.write(reinterpret_cast<const char *>(ckpt.data_dft.data()), dft_size);
        }
    }

    file.flush();
    const bool ok = !file.fail();
    file.close();
    return ok;
}

bool server_disk_cache::read_checkpoints_file(const std::string & filepath, std::list<common_prompt_checkpoint> & out_checkpoints) {
    out_checkpoints.clear();

    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        return false;
    }

    uint32_t magic = 0, version = 0, count = 0;
    file.read(reinterpret_cast<char *>(&magic),   sizeof(magic));
    file.read(reinterpret_cast<char *>(&version), sizeof(version));
    file.read(reinterpret_cast<char *>(&count),    sizeof(count));
    if (!file || magic != kCkptMagic || version != kCkptVersion) {
        SRV_WRN("disk cache: invalid checkpoint file '%s' (magic/version mismatch)\n", filepath.c_str());
        return false;
    }

    for (uint32_t i = 0; i < count; i++) {
        common_prompt_checkpoint ckpt;

        int64_t  n_tokens = 0;
        int32_t  pos_min  = 0;
        int32_t  pos_max  = 0;
        uint64_t tgt_size = 0;
        uint64_t dft_size = 0;

        file.read(reinterpret_cast<char *>(&n_tokens), sizeof(n_tokens));
        file.read(reinterpret_cast<char *>(&pos_min),  sizeof(pos_min));
        file.read(reinterpret_cast<char *>(&pos_max),  sizeof(pos_max));
        file.read(reinterpret_cast<char *>(&tgt_size), sizeof(tgt_size));
        if (!file) {
            SRV_WRN("disk cache: truncated checkpoint file '%s'\n", filepath.c_str());
            out_checkpoints.clear();
            return false;
        }

        ckpt.n_tokens = n_tokens;
        ckpt.pos_min  = pos_min;
        ckpt.pos_max  = pos_max;

        if (tgt_size > 0) {
            ckpt.data_tgt.resize(tgt_size);
            file.read(reinterpret_cast<char *>(ckpt.data_tgt.data()), tgt_size);
        }

        file.read(reinterpret_cast<char *>(&dft_size), sizeof(dft_size));
        if (!file) {
            SRV_WRN("disk cache: truncated checkpoint file '%s'\n", filepath.c_str());
            out_checkpoints.clear();
            return false;
        }

        if (dft_size > 0) {
            ckpt.data_dft.resize(dft_size);
            file.read(reinterpret_cast<char *>(ckpt.data_dft.data()), dft_size);
        }

        if (!file) {
            SRV_WRN("disk cache: truncated checkpoint file '%s'\n", filepath.c_str());
            out_checkpoints.clear();
            return false;
        }

        out_checkpoints.push_back(std::move(ckpt));
    }

    return true;
}

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

bool server_disk_cache::load(const server_tokens& tokens, llama_context* ctx, int32_t slot_id,
                              std::string * out_hash,
                              std::list<common_prompt_checkpoint> * out_checkpoints) {
    if (!m_enabled || tokens.empty() || !ctx) {
        return false;
    }

    std::string hash = hash_tokens(tokens);
    if (hash.empty()) {
        return false;
    }

    auto it = m_index.find(hash);
    if (it == m_index.end()) {
        m_misses++;
        save_stats();
        SRV_INF("disk cache: exact miss for %zu tokens, hash=%.8s... (hits=%" PRIu64 ", misses=%" PRIu64 ")\n",
                tokens.size(), hash.c_str(), m_hits, m_misses);
        return false;
    }

    if (it->second.n_stream != 0 && it->second.n_stream != m_n_stream) {
        m_misses++;
        save_stats();
        SRV_WRN("disk cache: hash=%.8s... was saved with n_stream=%u, current n_stream=%u, treating as miss\n",
                hash.c_str(), it->second.n_stream, m_n_stream);
        return false;
    }

    // Build file path
    std::string filepath = cache_path(m_path, hash + ".bin");

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

    m_hits++;
    save_stats();
    SRV_INF("disk cache: exact hit for %zu tokens, hash=%.8s... (%zu bytes, hits=%" PRIu64 ", misses=%" PRIu64 ")\n",
            tokens.size(), hash.c_str(), data.size(), m_hits, m_misses);
    if (out_hash) {
        *out_hash = hash;
    }

    if (out_checkpoints) {
        out_checkpoints->clear();
        if (it != m_index.end() && it->second.ckpt_size_bytes > 0) {
            const std::string ckpt_filepath = cache_path(m_path, hash + ".ckpt");
            if (!read_checkpoints_file(ckpt_filepath, *out_checkpoints)) {
                SRV_WRN("disk cache: failed to read checkpoint file '%s'\n", ckpt_filepath.c_str());
            }
        }
    }

    return true;
}

bool server_disk_cache::save(const server_tokens& tokens, llama_context* ctx, int32_t slot_id,
                              const server_tokens* full_tokens,
                              const std::list<common_prompt_checkpoint> * checkpoints) {
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

    const server_tokens & src = (full_tokens && !full_tokens->empty()) ? *full_tokens : tokens;

    llama_tokens new_tokens;
    new_tokens.reserve(src.size());
    for (size_t i = 0; i < src.size(); i++) {
        new_tokens.push_back(src[i]);
    }

    std::map<size_t, std::vector<uint8_t>> media_blobs;
    if (src.has_mtmd) {
        if (!build_media_chunk_blobs(src.get_media_chunks(), media_blobs)) {
            // new_tokens already has LLAMA_TOKEN_NULL placeholders reflecting src's actual
            // layout; saving them without matching chunk blobs would make this entry
            // permanently unmatchable later (rebuild_tokens_impl rejects unbacked
            // placeholders). Abort the whole save rather than persist a broken entry.
            SRV_WRN("%s", "disk cache: failed to serialize media chunks, aborting save\n");
            return false;
        }
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
    std::string filepath = cache_path(m_path, hash + ".bin");
    std::string tmp_filepath = filepath + ".tmp";

    std::ofstream file(tmp_filepath, std::ios::binary);
    if (!file.is_open()) {
        SRV_WRN("disk cache: failed to create cache file '%s'\n", tmp_filepath.c_str());
        return false;
    }

    file.write(reinterpret_cast<const char*>(data.data()), obtained);
    file.flush();
    if (file.fail()) {
        SRV_WRN("disk cache: failed to write cache file '%s'\n", tmp_filepath.c_str());
        file.close();
        // Clean up temp file
        fs::remove(tmp_filepath);
        return false;
    }
    file.close();

    // Atomically rename temp file to final path
    std::error_code ec;
    fs::rename(tmp_filepath, filepath, ec);
    if (ec) {
        SRV_WRN("disk cache: failed to rename cache file: %s\n", ec.message().c_str());
        fs::remove(tmp_filepath);
        return false;
    }

    // Prune older entries whose full token sequence is a strict prefix of this
    // entry's tokens: they are an earlier turn of the same conversation and are
    // fully superseded (find_best_prefix always prefers the longest match), so
    // keeping them around only wastes disk budget and accelerates LRU eviction
    // of genuinely distinct entries (other sessions' cached state). Without this,
    // a single long-running conversation accumulates one entry per turn forever.
    // Done only now (new file already safely on disk) so a failed save can never
    // destroy the old entry without replacing it; done before check_and_evict so
    // the freed space is available and we don't evict an unrelated, still-live
    // session's entry just to make room that this conversation's own stale
    // entries were already holding.
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
    if (n_pruned > 0) {
        SRV_INF("disk cache: pruned %zu superseded prefix entr%s from earlier turns of this conversation\n",
                n_pruned, n_pruned == 1 ? "y" : "ies");
    }

    // Evict before committing to avoid writing then immediately deleting
    check_and_evict(obtained);

    // Add to index
    disk_cache_entry entry;
    entry.tokens       = new_tokens;
    entry.n_tokens     = (uint32_t)entry.tokens.size();
    entry.size_bytes   = obtained;
    entry.last_used_us = ggml_time_us();
    entry.n_stream     = m_n_stream;

    if (!media_blobs.empty()) {
        const std::string chunks_filepath     = cache_path(m_path, hash + ".chunks");
        const std::string chunks_tmp_filepath = chunks_filepath + ".tmp";

        bool chunks_ok = false;
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
                    chunks_ok = true;
                }
            } else {
                SRV_WRN("disk cache: failed to rename media chunks file: %s\n", chunks_ec.message().c_str());
                fs::remove(chunks_tmp_filepath);
            }
        } else {
            SRV_WRN("disk cache: failed to write media chunks file '%s'\n", chunks_tmp_filepath.c_str());
            fs::remove(chunks_tmp_filepath);
        }

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
    // later restore roll back to an earlier position without a full reprocess
    // (needed for recurrent/hybrid memory, which has no other way to "rewind").
    if (checkpoints && !checkpoints->empty()) {
        const std::string ckpt_filepath     = cache_path(m_path, hash + ".ckpt");
        const std::string ckpt_tmp_filepath = ckpt_filepath + ".tmp";

        if (write_checkpoints_file(ckpt_tmp_filepath, *checkpoints)) {
            std::error_code ckpt_ec;
            fs::rename(ckpt_tmp_filepath, ckpt_filepath, ckpt_ec);
            if (!ckpt_ec) {
                std::error_code size_ec;
                const uint64_t ckpt_bytes = (uint64_t) fs::file_size(ckpt_filepath, size_ec);
                if (!size_ec) {
                    entry.ckpt_size_bytes = ckpt_bytes;
                    m_total_size += ckpt_bytes;
                }
            } else {
                SRV_WRN("disk cache: failed to rename checkpoint file: %s\n", ckpt_ec.message().c_str());
                fs::remove(ckpt_tmp_filepath);
            }
        } else {
            SRV_WRN("disk cache: failed to write checkpoint file '%s'\n", ckpt_tmp_filepath.c_str());
            fs::remove(ckpt_tmp_filepath);
        }
    }

    m_index[hash]      = entry;
    m_total_size += obtained;

    save_index();

    SRV_INF("disk cache: saved %zu tokens (%zu bytes), hash=%.8s...\n", tokens.size(), obtained, hash.c_str());
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
    if (entry_total <= m_total_size) {
        m_total_size -= entry_total;
    } else {
        SRV_WRN("disk cache: entry size (%" PRIu64 ") exceeds total size (%zu), resetting total\n",
                entry_total, m_total_size);
        m_total_size = 0;
    }
    m_index.erase(oldest_it);

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

static bool is_valid_sha256_hex(const std::string& s) {
    if (s.size() != 64) {
        return false;
    }
    for (char c : s) {
        if (!isxdigit(c)) {
            return false;
        }
    }
    return true;
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
                if (filename.size() == 68 && filename.substr(filename.size() - 4) == ".bin") {
                    std::string hash = filename.substr(0, filename.size() - 4);
                    if (is_valid_sha256_hex(hash)) {
                        found_files[hash] = entry.file_size();
                    } else {
                        SRV_WRN("disk cache: skipping file with invalid hash name '%s'\n", filename.c_str());
                    }
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

    if (removed > 0) {
        save_index();
    }
}

void server_disk_cache::load_stats() {
    std::string stats_path = cache_path(m_path, kStatsFilename);

    if (!fs::exists(stats_path)) {
        return; // no stats yet; counters stay at 0
    }

    std::ifstream file(stats_path);
    if (!file.is_open()) {
        return;
    }

    nlohmann::json j;
    try {
        j = nlohmann::json::parse(file);
    } catch (const std::exception &) {
        return; // corrupted stats file; start fresh
    }

    m_hits   = j.value("hits",   (uint64_t)0);
    m_misses = j.value("misses", (uint64_t)0);
}

void server_disk_cache::save_stats() {
    std::string stats_path = cache_path(m_path, kStatsFilename);
    std::string tmp_path   = stats_path + ".tmp";

    nlohmann::json j;
    j["hits"]   = m_hits;
    j["misses"] = m_misses;

    std::ofstream file(tmp_path);
    if (!file.is_open()) {
        return;
    }

    file << j.dump(2);
    file.flush();
    if (file.fail()) {
        file.close();
        fs::remove(tmp_path);
        return;
    }
    file.close();

    std::error_code ec;
    fs::rename(tmp_path, stats_path, ec);
    if (ec) {
        fs::remove(tmp_path);
    }
}

void server_disk_cache::wipe_cache_dir() {
    std::error_code ec;
    for (const auto & dir_entry : fs::directory_iterator(m_path, ec)) {
        fs::remove(dir_entry.path(), ec);
    }
    m_index.clear();
    m_total_size = 0;
}

bool server_disk_cache::load_index() {
    std::string index_path = cache_path(m_path, kIndexFilename);

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

    nlohmann::json j;
    try {
        j = nlohmann::json::parse(file);
    } catch (const std::exception& e) {
        SRV_WRN("disk cache: failed to parse index file: %s\n", e.what());
        return false;
    }

    const uint32_t file_version = j.value("version", (uint32_t)0);
    if (file_version != kIndexFormatVersion) {
        SRV_WRN("disk cache: index format version %u != %u (multimodal-aware format), clearing stale cache\n",
                file_version, kIndexFormatVersion);
        wipe_cache_dir();
        return true;
    }

    // Parse entries
    m_index.clear();
    m_total_size = 0;

    if (j.contains("entries")) {
        for (const auto& [hash, entry_json] : j["entries"].items()) {
            disk_cache_entry entry;
            entry.size_bytes      = entry_json.value("size_bytes",      (uint64_t)0);
            entry.n_tokens        = entry_json.value("n_tokens",        (uint32_t)0);
            entry.last_used_us    = entry_json.value("last_used_us",    (int64_t)0);
            entry.ckpt_size_bytes = entry_json.value("ckpt_size_bytes", (uint64_t)0);
            entry.media_size_bytes = entry_json.value("media_size_bytes", (uint64_t)0);
            entry.n_stream        = entry_json.value("n_stream",        (uint32_t)0);
            // read tokens array if present (absent in old-format entries)
            if (entry_json.contains("tokens") && entry_json["tokens"].is_array()) {
                entry.tokens.clear();
                entry.tokens.reserve(entry_json["tokens"].size());
                for (const auto & t : entry_json["tokens"]) {
                    if (t.is_number_integer()) {
                        entry.tokens.push_back((llama_token)t.get<int32_t>());
                    }
                }
            }
            if (entry.media_size_bytes > 0) {
                const std::string chunks_filepath = cache_path(m_path, hash + ".chunks");
                if (!read_media_chunks_file(chunks_filepath, entry.media_chunks)) {
                    SRV_WRN("disk cache: failed to read media chunks file '%s', entry will be treated as text-only\n", chunks_filepath.c_str());
                }
            }
            m_index[hash] = entry;
            m_total_size += entry.size_bytes + entry.ckpt_size_bytes + entry.media_size_bytes;
        }
    }

    // Validate: remove entries with missing files
    validate_and_rebuild();

    SRV_INF("disk cache: loaded index with %zu entries, total size=%.2f MiB\n",
            m_index.size(), m_total_size / (1024.0 * 1024.0));

    return true;
}

const disk_cache_entry * server_disk_cache::get_entry(const std::string & hash) const {
    auto it = m_index.find(hash);
    if (it == m_index.end()) {
        return nullptr;
    }
    return &it->second;
}

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
            if (n_tok == 0) {
                // malformed chunk blob: would never advance past this position
                SRV_WRN("%s", "disk cache: media chunk reports 0 tokens, treating entry as unmatched\n");
                return server_tokens();
            }
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

server_tokens server_disk_cache::rebuild_tokens(const std::string & hash, bool has_mtmd) const {
    auto it = m_index.find(hash);
    if (it == m_index.end()) {
        return server_tokens();
    }
    server_tokens st = rebuild_tokens_impl(it->second);
    if (!st.empty()) {
        // the caller (not the entry) knows whether the restoring slot is multimodal-capable;
        // a media-bearing entry still forces has_mtmd true regardless of what was passed in
        st.has_mtmd = has_mtmd || st.has_mtmd;
    }
    return st;
}

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

bool server_disk_cache::load_by_hash(const std::string & hash, llama_context * ctx, int32_t slot_id,
                                      std::list<common_prompt_checkpoint> * out_checkpoints) {
    if (!m_enabled || hash.empty() || !ctx) {
        return false;
    }

    auto it = m_index.find(hash);
    if (it == m_index.end()) {
        return false;
    }

    if (it->second.n_stream != 0 && it->second.n_stream != m_n_stream) {
        SRV_WRN("disk cache: hash=%.8s... was saved with n_stream=%u, current n_stream=%u, treating as miss\n",
                hash.c_str(), it->second.n_stream, m_n_stream);
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

    if (out_checkpoints) {
        out_checkpoints->clear();
        if (it != m_index.end() && it->second.ckpt_size_bytes > 0) {
            const std::string ckpt_filepath = cache_path(m_path, hash + ".ckpt");
            if (!read_checkpoints_file(ckpt_filepath, *out_checkpoints)) {
                SRV_WRN("disk cache: failed to read checkpoint file '%s'\n", ckpt_filepath.c_str());
            }
        }
    }

    m_hits++;
    save_stats();
    SRV_INF("disk cache: prefix hit, loaded %zu bytes for hash %.8s... (hits=%" PRIu64 ", misses=%" PRIu64 ")\n",
            data.size(), hash.c_str(), m_hits, m_misses);
    return true;
}

bool server_disk_cache::save_index() {
    std::string index_path = cache_path(m_path, kIndexFilename);
    std::string tmp_path = index_path + ".tmp";

    nlohmann::json j;
    j["version"] = kIndexFormatVersion;
    j["total_size_bytes"] = m_total_size;

    nlohmann::json entries = nlohmann::json::object();
    for (const auto& [hash, entry] : m_index) {
        nlohmann::json entry_json;
        entry_json["size_bytes"]      = entry.size_bytes;
        entry_json["n_tokens"]        = entry.n_tokens;
        entry_json["last_used_us"]    = entry.last_used_us;
        entry_json["ckpt_size_bytes"] = entry.ckpt_size_bytes;
        entry_json["media_size_bytes"] = entry.media_size_bytes;
        entry_json["n_stream"]        = entry.n_stream;
        // write tokens as flat int32 array
        nlohmann::json toks = nlohmann::json::array();
        for (auto t : entry.tokens) {
            toks.push_back((int32_t)t);
        }
        entry_json["tokens"] = std::move(toks);
        entries[hash] = entry_json;
    }
    j["entries"] = entries;

    // Write to temp file
    std::ofstream file(tmp_path);
    if (!file.is_open()) {
        SRV_WRN("%s", "disk cache: failed to create index file\n");
        return false;
    }

    file << j.dump(2);
    file.flush();
    if (file.fail()) {
        SRV_WRN("%s", "disk cache: failed to write index file\n");
        file.close();
        fs::remove(tmp_path);
        return false;
    }
    file.close();

    // Atomically rename
    std::error_code ec;
    fs::rename(tmp_path, index_path, ec);
    if (ec) {
        SRV_WRN("disk cache: failed to rename index file: %s\n", ec.message().c_str());
        fs::remove(tmp_path);
        return false;
    }

    return true;
}
