#include "server-disk-cache.h"
#include "server-cache-io.h"
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
static const uint32_t kIndexFormatVersion = 5; // includes model and context identity

// Build a path within the cache directory using std::filesystem for cross-platform compatibility
static std::string cache_path(const std::string& dir, const std::string& filename) {
    return (fs::path(dir) / filename).string();
}

static bool read_state_file(const std::string & path, llama_context * ctx, llama_seq_id seq, size_t size) {
    try {
        server_cache_file file(path, false);
        const auto read = [](void * user, void * dst, size_t n) {
            static_cast<server_cache_file *>(user)->read(dst, n);
            return true;
        };
        return size > 0 && llama_state_seq_read_stream(ctx, seq, size, read, &file) == size;
    } catch (const std::exception & err) {
        SRV_WRN("disk cache: %s\n", err.what());
        return false;
    }
}

static size_t write_state_file(const std::string & path, llama_context * ctx, llama_seq_id seq) {
    try {
        server_cache_file file(path, true);
        const auto write = [](void * user, const void * src, size_t n) {
            static_cast<server_cache_file *>(user)->write(src, n);
            return true;
        };
        const size_t size = llama_state_seq_write_stream(ctx, seq, write, &file);
        if (size == 0) return 0;
        file.finish();
        return size;
    } catch (const std::exception & err) {
        SRV_WRN("disk cache: %s\n", err.what());
        return 0;
    }
}

// --- Embedded SHA-256 (Igor Pavlov, public domain) ---

#define U8V(v) ((uint8_t)(v) & 0xFFU)
#define U32V(v) ((uint32_t)(v) & 0xFFFFFFFFU)
#define ROTL32(v, n) (((uint32_t)(v) << (n)) | ((uint32_t)(v) >> (32 - (n))))
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
    0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5,
    0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
    0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
};

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

std::string server_disk_cache::identity(const common_params & p, llama_context * ctx, llama_context * ctx_dft) {
    if (!p.lora_adapters.empty() || !p.control_vectors.empty() || !p.kv_overrides.empty()) {
        throw std::runtime_error("disk cache does not support adapters, control vectors or metadata overrides");
    }
    nlohmann::json files = nlohmann::json::array();
    auto add_file = [&](const std::string & path) {
        if (path.empty()) return;
        files.push_back({fs::canonical(path).string(), fs::file_size(path), fs::last_write_time(path).time_since_epoch().count()});
    };
    auto add_model = [&](const std::string & path, llama_context * context) {
        if (!context || path.empty()) return;
        char count_buf[32] = {};
        llama_model_meta_val_str(llama_get_model(context), "split.count", count_buf, sizeof(count_buf));
        const int count = std::max(1, std::atoi(count_buf));
        if (count == 1) {
            add_file(path);
        } else {
            std::vector<char> prefix(path.size() + 64), part(path.size() + 64);
            if (llama_split_prefix(prefix.data(), prefix.size(), path.c_str(), 0, count) <= 0) {
                throw std::runtime_error("cannot identify split model files");
            }
            for (int i = 0; i < count; ++i) {
                llama_split_path(part.data(), part.size(), prefix.data(), i, count);
                add_file(part.data());
            }
        }
    };
    add_model(p.model.path, ctx);
    add_model(p.speculative.draft.mparams.path, ctx_dft);
    add_file(p.model.ple);
    add_file(p.speculative.draft.mparams.ple);
    add_file(p.mmproj.path);
    if (files.empty()) throw std::runtime_error("cannot identify model files");
    nlohmann::json settings = {
        p.rope_scaling_type, p.rope_freq_base, p.rope_freq_scale,
        p.yarn_ext_factor, p.yarn_attn_factor, p.yarn_beta_fast, p.yarn_beta_slow, p.yarn_orig_ctx,
        p.cache_type_k, p.cache_type_v, p.flash_attn_type, p.swa_full,
        p.speculative.draft.cache_type_k, p.speculative.draft.cache_type_v,
        p.speculative.types, llama_n_rs_seq(ctx), ctx_dft ? llama_n_rs_seq(ctx_dft) : 0};
    return nlohmann::json({{"files", files}, {"settings", settings}}).dump();
}

bool server_disk_cache::init(const std::string& path, int32_t max_size_mib, uint32_t n_stream, const std::string & identity) {
    m_identity = identity;
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
        m_index.clear();
    }

    validate_and_rebuild();
    check_and_evict(0);
    if (!save_index()) return false;

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
static const uint32_t kCkptVersion = 2u;

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
        const uint64_t spec_size  = ckpt.data_spec.size();

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
        file.write(reinterpret_cast<const char *>(&spec_size), sizeof(spec_size));
        file.write(reinterpret_cast<const char *>(ckpt.data_spec.data()), spec_size);
    }

    file.flush();
    const bool ok = !file.fail();
    file.close();
    return ok;
}

static bool read_blob(std::ifstream & file, std::vector<uint8_t> & out, uint64_t size) {
    const auto pos = file.tellg();
    if (!file || pos < 0) return false;
    file.seekg(0, std::ios::end);
    const auto end = file.tellg();
    file.seekg(pos);
    if (end < pos || size > uint64_t(end - pos)) return false;
    out.resize(size);
    return size == 0 || bool(file.read(reinterpret_cast<char *>(out.data()), size));
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

        if (!read_blob(file, ckpt.data_tgt, tgt_size)) {
            out_checkpoints.clear();
            return false;
        }

        file.read(reinterpret_cast<char *>(&dft_size), sizeof(dft_size));
        if (!file) {
            SRV_WRN("disk cache: truncated checkpoint file '%s'\n", filepath.c_str());
            out_checkpoints.clear();
            return false;
        }

        uint64_t spec_size = 0;
        if (!read_blob(file, ckpt.data_dft, dft_size) ||
            !file.read(reinterpret_cast<char *>(&spec_size), sizeof(spec_size)) ||
            !read_blob(file, ckpt.data_spec, spec_size)) {
            out_checkpoints.clear();
            return false;
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

        std::vector<uint8_t> blob;
        if (!read_blob(file, blob, size)) {
            out_blobs.clear();
            return false;
        }

        out_blobs[(size_t) idx] = std::move(blob);
    }

    return true;
}

bool server_disk_cache::load(const server_tokens & tokens, llama_context * ctx, int32_t slot_id,
                              llama_context * ctx_dft, std::string * out_hash,
                              std::list<common_prompt_checkpoint> * out_checkpoints) {
    m_restore_failed = false;
    if (!m_enabled || tokens.empty() || !ctx) return false;
    const std::string hash = hash_tokens(tokens);
    if (!load_by_hash(hash, ctx, slot_id, ctx_dft, out_checkpoints)) {
        ++m_misses;
        save_stats();
        return false;
    }
    if (out_hash) *out_hash = hash;
    SRV_INF("disk cache: exact hit for %zu tokens, hash=%.8s...\n", tokens.size(), hash.c_str());
    return true;
}

void server_disk_cache::erase(const std::string & hash) {
    auto it = m_index.find(hash);
    if (it == m_index.end()) return;
    const auto & e = it->second;
    const uint64_t size = e.size_bytes + e.ckpt_size_bytes + e.media_size_bytes + e.dft_size_bytes;
    std::error_code ec;
    for (const auto * suffix : {".bin", ".dft.bin", ".ckpt", ".chunks"}) {
        fs::remove(cache_path(m_path, hash + suffix), ec);
    }
    m_total_size -= std::min<uint64_t>(m_total_size, size);
    m_index.erase(it);
}

bool server_disk_cache::save(const server_tokens & tokens, llama_context * ctx, int32_t slot_id,
                            llama_context * ctx_dft, const server_tokens * full_tokens,
                            const std::list<common_prompt_checkpoint> * checkpoints) {
    if (!m_enabled || tokens.empty() || !ctx) return false;
    const std::string hash = hash_tokens(tokens);
    if (hash.empty()) return false;
    auto found = m_index.find(hash);
    if (found != m_index.end()) {
        if (!ctx_dft || found->second.dft_size_bytes > 0) {
            found->second.last_used_us = ggml_time_us();
            return save_index();
        }
        erase(hash);
    }

    const auto & src = full_tokens && !full_tokens->empty() ? *full_tokens : tokens;
    disk_cache_entry entry;
    for (size_t i = 0; i < src.size(); ++i) entry.tokens.push_back(src[i]);
    entry.n_tokens = entry.tokens.size();
    entry.n_stream = m_n_stream;
    entry.last_used_us = ggml_time_us();
    entry.size_bytes = llama_state_seq_get_size_ext(ctx, slot_id, 0);
    if (!entry.size_bytes) return false;
    if (ctx_dft) entry.dft_size_bytes = llama_state_seq_get_size_ext(ctx_dft, slot_id, 0);
    if (src.has_mtmd && !build_media_chunk_blobs(src.get_media_chunks(), entry.media_chunks)) return false;
    if (!entry.media_chunks.empty()) {
        entry.media_size_bytes = 12;
        for (const auto & chunk : entry.media_chunks) entry.media_size_bytes += 16 + chunk.second.size();
    }
    if (checkpoints && !checkpoints->empty()) {
        entry.ckpt_size_bytes = 12;
        for (const auto & cp : *checkpoints) entry.ckpt_size_bytes += 40 + cp.data_tgt.size() + cp.data_dft.size() + cp.data_spec.size();
    }
    const uint64_t total = entry.size_bytes + entry.dft_size_bytes + entry.media_size_bytes + entry.ckpt_size_bytes;
    if (m_max_size_bytes && total > m_max_size_bytes) {
        SRV_WRN("disk cache: entry of %" PRIu64 " bytes exceeds the size limit\n", total);
        return false;
    }
    check_and_evict(total);
    const auto path = [&](const char * suffix) { return cache_path(m_path, hash + suffix); };
    try {
        if (write_state_file(path(".bin.tmp"), ctx, slot_id) != entry.size_bytes ||
            (entry.dft_size_bytes && write_state_file(path(".dft.bin.tmp"), ctx_dft, slot_id) != entry.dft_size_bytes) ||
            (entry.ckpt_size_bytes && !write_checkpoints_file(path(".ckpt.tmp"), *checkpoints)) ||
            (entry.media_size_bytes && !write_media_chunks_file(path(".chunks.tmp"), entry.media_chunks))) {
            throw std::runtime_error("failed to write cache entry");
        }
        fs::rename(path(".bin.tmp"), path(".bin"));
        if (entry.dft_size_bytes) fs::rename(path(".dft.bin.tmp"), path(".dft.bin"));
        if (entry.ckpt_size_bytes) fs::rename(path(".ckpt.tmp"), path(".ckpt"));
        if (entry.media_size_bytes) fs::rename(path(".chunks.tmp"), path(".chunks"));
    } catch (const std::exception & err) {
        SRV_WRN("disk cache: %s\n", err.what());
        std::error_code ec;
        for (const auto * suffix : {".bin", ".dft.bin", ".ckpt", ".chunks"}) {
            fs::remove(path(suffix), ec);
            fs::remove(path(suffix) + ".tmp", ec);
        }
        save_index();
        return false;
    }

    // Drop superseded turns only after their replacement is on disk.
    std::vector<std::string> superseded;
    for (const auto & old : m_index) {
        const auto rebuilt = rebuild_tokens_impl(old.second);
        if (!rebuilt.empty() && rebuilt.size() < src.size() && rebuilt.get_common_prefix(src) == rebuilt.size()) {
            superseded.push_back(old.first);
        }
    }
    for (const auto & old : superseded) erase(old);
    m_index[hash] = std::move(entry);
    m_total_size += total;
    if (!save_index()) {
        erase(hash);
        return false;
    }
    SRV_INF("disk cache: saved %zu tokens (%" PRIu64 " bytes), hash=%.8s...\n", tokens.size(), total, hash.c_str());
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

    erase(oldest_it->first);

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

static bool is_cache_payload(const std::string & name) {
    if (name.size() <= 64 || !is_valid_sha256_hex(name.substr(0, 64))) return false;
    const auto suffix = name.substr(64);
    return suffix == ".bin" || suffix == ".dft.bin" || suffix == ".ckpt" || suffix == ".chunks" ||
           suffix == ".bin.tmp" || suffix == ".dft.bin.tmp" || suffix == ".ckpt.tmp" || suffix == ".chunks.tmp";
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

    // Remove abandoned writes and sidecars that are not owned by the index.
    for (const auto & file : fs::directory_iterator(m_path)) {
        const auto name = file.path().filename().string();
        if (!file.is_regular_file() || !is_cache_payload(name)) continue;
        const auto it = m_index.find(name.substr(0, 64));
        const auto suffix = name.substr(64);
        const bool owned = it != m_index.end() &&
            (suffix == ".bin" || (suffix == ".ckpt" && it->second.ckpt_size_bytes) ||
             (suffix == ".chunks" && it->second.media_size_bytes) || (suffix == ".dft.bin" && it->second.dft_size_bytes));
        if (!owned) fs::remove(file.path());
        else if (suffix == ".ckpt") it->second.ckpt_size_bytes = file.file_size();
        else if (suffix == ".chunks") it->second.media_size_bytes = file.file_size();
        else if (suffix == ".dft.bin") it->second.dft_size_bytes = file.file_size();
    }

    // Rebuild total size
    m_total_size = 0;
    for (const auto& [hash, entry] : m_index) {
        m_total_size += entry.size_bytes + entry.ckpt_size_bytes + entry.media_size_bytes + entry.dft_size_bytes;
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
        const auto name = dir_entry.path().filename().string();
        if (is_cache_payload(name) || name == kIndexFilename || name == kStatsFilename || name == "index.json.tmp" || name == "stats.json.tmp") {
            fs::remove(dir_entry.path(), ec);
        }
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
    if (file_version != kIndexFormatVersion || j.value("identity", std::string()) != m_identity) {
        SRV_WRN("disk cache: index format version %u != %u (or model/context identity changed), clearing stale cache\n",
                file_version, kIndexFormatVersion);
        wipe_cache_dir();
        return true;
    }

    // Parse entries
    m_index.clear();
    m_total_size = 0;

    if (j.contains("entries")) {
        for (const auto& [hash, entry_json] : j["entries"].items()) {
            if (!is_valid_sha256_hex(hash)) continue;
            disk_cache_entry entry;
            entry.size_bytes      = entry_json.value("size_bytes",      (uint64_t)0);
            entry.n_tokens        = entry_json.value("n_tokens",        (uint32_t)0);
            entry.last_used_us    = entry_json.value("last_used_us",    (int64_t)0);
            entry.ckpt_size_bytes = entry_json.value("ckpt_size_bytes", (uint64_t)0);
            entry.media_size_bytes = entry_json.value("media_size_bytes", (uint64_t)0);
            entry.dft_size_bytes  = entry_json.value("dft_size_bytes",  (uint64_t)0);
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
            m_total_size += entry.size_bytes + entry.ckpt_size_bytes + entry.media_size_bytes + entry.dft_size_bytes;
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

std::string server_disk_cache::find_best_prefix(const server_tokens & tokens, size_t min_prefix_len, bool require_dft) const {
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

        if (require_dft && entry.dft_size_bytes == 0) {
            continue;  // would leave ctx_dft desynced from the restored ctx_tgt
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
                                      llama_context * ctx_dft,
                                      std::list<common_prompt_checkpoint> * out_checkpoints) try {
    m_restore_failed = false;
    if (!m_enabled || hash.empty() || !ctx) {
        return false;
    }

    auto it = m_index.find(hash);
    if (it == m_index.end()) {
        return false;
    }

    // See load(): an entry without draft-model state must not be used to restore
    // a speculative-decoding server's ctx, or ctx_dft would end up desynced.
    if (ctx_dft && it->second.dft_size_bytes == 0) {
        return false;
    }

    if (rebuild_tokens_impl(it->second).empty()) return false;
    if (out_checkpoints) {
        out_checkpoints->clear();
        if (it->second.ckpt_size_bytes &&
            !read_checkpoints_file(cache_path(m_path, hash + ".ckpt"), *out_checkpoints)) return false;
    }

    const std::string filepath = cache_path(m_path, hash + ".bin");
    const size_t file_size = it->second.size_bytes;
    // A failed stream restore can mutate both contexts. The caller must clear its prompt too.
    if (!read_state_file(filepath, ctx, slot_id, file_size) ||
        (ctx_dft && !read_state_file(cache_path(m_path, hash + ".dft.bin"), ctx_dft, slot_id, it->second.dft_size_bytes))) {
        llama_memory_seq_rm(llama_get_memory(ctx), slot_id, -1, -1);
        if (ctx_dft) llama_memory_seq_rm(llama_get_memory(ctx_dft), slot_id, -1, -1);
        m_restore_failed = true;
        return false;
    }

    it = m_index.find(hash);
    if (it != m_index.end()) {
        it->second.last_used_us = ggml_time_us();
        save_index();
    }

    m_hits++;
    save_stats();
    SRV_INF("disk cache: prefix hit, loaded %zu bytes for hash %.8s... (hits=%" PRIu64 ", misses=%" PRIu64 ")\n",
            file_size, hash.c_str(), m_hits, m_misses);
    return true;
} catch (const std::exception & err) {
    SRV_WRN("disk cache: restore failed: %s\n", err.what());
    m_restore_failed = true;
    return false;
}

bool server_disk_cache::save_index() {
    std::string index_path = cache_path(m_path, kIndexFilename);
    std::string tmp_path = index_path + ".tmp";

    nlohmann::json j;
    j["version"] = kIndexFormatVersion;
    j["identity"] = m_identity;
    j["total_size_bytes"] = m_total_size;

    nlohmann::json entries = nlohmann::json::object();
    for (const auto& [hash, entry] : m_index) {
        nlohmann::json entry_json;
        entry_json["size_bytes"]      = entry.size_bytes;
        entry_json["n_tokens"]        = entry.n_tokens;
        entry_json["last_used_us"]    = entry.last_used_us;
        entry_json["ckpt_size_bytes"] = entry.ckpt_size_bytes;
        entry_json["media_size_bytes"] = entry.media_size_bytes;
        entry_json["dft_size_bytes"]  = entry.dft_size_bytes;
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
