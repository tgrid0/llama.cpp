#include "llama-ple-stream.h"

#include "llama-impl.h"
#include "llama-ple-sidecar.h"

#include <algorithm>
#include <cerrno>
#include <cinttypes>
#include <cstring>
#include <stdexcept>

#ifdef _WIN32
#include <malloc.h>
#else
#include <fcntl.h>
#include <unistd.h>
#endif

static const size_t PLE_STREAM_DIRECT_ALIGN = 4096;

static void * ple_aligned_alloc(size_t n) {
#ifdef _WIN32
    return _aligned_malloc(n, PLE_STREAM_DIRECT_ALIGN);
#else
    void * p = nullptr;
    if (posix_memalign(&p, PLE_STREAM_DIRECT_ALIGN, n) != 0) {
        p = nullptr;
    }
    return p;
#endif
}

static void ple_aligned_free(void * p) {
#ifdef _WIN32
    _aligned_free(p);
#else
    free(p);
#endif
}

// positional read; returns a pointer to the len bytes at offs inside staging
static const uint8_t * ple_pread(llama_file & file, uint8_t * staging, size_t len, size_t offs, bool direct) {
#ifdef _WIN32
    GGML_UNUSED(direct);
    // no positional read primitive; serialize the seek+read pairs
    static std::mutex io_mtx;
    std::lock_guard<std::mutex> lock(io_mtx);
    try {
        file.seek(offs, SEEK_SET);
        file.read_raw(staging, len);
        return staging;
    } catch (...) {
        return nullptr;
    }
#else
    const int fd = file.file_id();

    if (direct) {
        // O_DIRECT requires the offset, length, and buffer all block-aligned
        const size_t a     = PLE_STREAM_DIRECT_ALIGN;
        const size_t aoffs = offs & ~(a - 1);
        const size_t head  = offs - aoffs;
        const size_t total = ((head + len + a - 1) / a) * a;
        ssize_t r;
        do {
            r = pread(fd, staging, total, aoffs);
        } while (r < 0 && errno == EINTR);
        if (r < 0 || (size_t) r < head + len) {
            // mid-stream direct I/O failure: fall back to buffered I/O for this read,
            // as llama_file does
            try {
                file.seek(offs, SEEK_SET);
                file.read_raw(staging, len);
                return staging;
            } catch (...) {
                return nullptr;
            }
        }
        return staging + head;
    }

    try {
        file.seek(offs, SEEK_SET);
        file.read_raw(staging, len);
        return staging;
    } catch (...) {
        return nullptr;
    }
#endif
}

// clamps n_cache_rows to the table size (warning if it had to), and sizes the
// cache buffers; does not touch row_buf_ (see alloc_row_buf() for why)
void llama_ple_stream::init_cache(uint32_t n_cache_rows) {
    // a cache with more slots than the table has rows cannot improve the hit rate
    // (every row already has its own slot at n_cache_rows_ == n_rows_) and only
    // wastes host memory, which matters here since the whole point of streaming
    // is a bounded, user-controlled footprint
    n_cache_rows_ = (int64_t) std::min<uint64_t>(n_cache_rows, (uint64_t) n_rows_);
    if (n_cache_rows_ < (int64_t) n_cache_rows) {
        LLAMA_LOG_WARN("%s: --ple-cache-rows %u exceeds the table's %" PRId64 " rows, clamping\n",
                __func__, n_cache_rows, n_rows_);
    }

    if (n_cache_rows_ > 0) {
        cache_.resize((size_t) n_cache_rows_ * row_size_);
        cache_tags_.assign(n_cache_rows_, -1);
    }
}

// allocates row_buf_; callers must call this last, after every other throwing
// step in the constructor has already succeeded - row_buf_ is a raw pointer
// (no RAII), so allocating it earlier would leak it if a later step threw
void llama_ple_stream::alloc_row_buf() {
    row_buf_ = (uint8_t *) ple_aligned_alloc(row_size_ + 2 * PLE_STREAM_DIRECT_ALIGN);
    if (row_buf_ == nullptr) {
        throw std::runtime_error("PLE table streaming allocation failed");
    }
}

void llama_ple_stream::open_files(const std::vector<std::string> & paths, bool use_direct_io) {
    for (const auto & path : paths) {
        files_.push_back(std::make_unique<llama_file>(path.c_str(), "rb", use_direct_io));
    }

    if (use_direct_io) {
        bool ok = true;
        for (auto & file : files_) {
            ok = ok && file->has_direct_io();
        }
        if (ok) {
            uint8_t * probe = (uint8_t *) ple_aligned_alloc(PLE_STREAM_DIRECT_ALIGN);
            ok = probe != nullptr;
            for (auto & file : files_) {
                ok = ok && ple_pread(*file, probe, PLE_STREAM_DIRECT_ALIGN, 0, true) != nullptr;
            }
            ple_aligned_free(probe);
        }
        if (!ok) {
            LLAMA_LOG_WARN("%s: O_DIRECT not usable for the PLE table, falling back to buffered reads\n", __func__);
            files_.clear();
            for (const auto & path : paths) {
                files_.push_back(std::make_unique<llama_file>(path.c_str(), "rb", false));
            }
            use_direct_io = false;
        }
    }
    use_direct_io_ = use_direct_io;
}

llama_ple_stream::llama_ple_stream(
        const char * path, size_t offs, int64_t n_rows, int64_t head_dim,
        enum ggml_type type, uint32_t n_cache_rows, bool use_direct_io) {
    if (path == nullptr || path[0] == '\0') {
        throw std::runtime_error("PLE table streaming requires a file-based model (not a stream/file descriptor)");
    }
    if (n_rows <= 0 || head_dim <= 0) {
        throw std::runtime_error("PLE table has invalid dimensions");
    }

    row_size_ = ggml_row_size(type, head_dim);
    n_rows_   = n_rows;
    head_dim_ = head_dim;
    type_     = type;

    // clamp/warn on --ple-cache-rows and size the cache buffers before opening the
    // file, so the clamp warning (if any) always prints before any O_DIRECT log
    init_cache(n_cache_rows);

    // opens (and, on a failed O_DIRECT probe, reopens buffered) before the bounds
    // check below runs, so a bounds-check throw here never emits a misleading
    // "streaming enabled" log for a stream that was never actually built
    open_files({ path }, use_direct_io);

    if (offs + (size_t) n_rows_ * row_size_ > files_[0]->size()) {
        throw std::runtime_error("PLE table data is not within the file bounds, model is corrupted or incomplete");
    }
    segments_.push_back({ 0, n_rows_, 0, offs });

    // last throwing step, once nothing above it can still fail
    alloc_row_buf();

    if (use_direct_io_) {
        LLAMA_LOG_INFO("%s: PLE table streaming uses O_DIRECT (page cache bypassed)\n", __func__);
    }
    if (n_cache_rows_ > 0) {
        LLAMA_LOG_INFO("%s: PLE table streaming enabled, %" PRId64 " rows, %" PRId64 " row cache slots (%.2f MiB)\n",
                __func__, n_rows_, (int64_t) n_cache_rows_, (double) cache_.size() / (1024.0 * 1024.0));
    } else {
        LLAMA_LOG_INFO("%s: PLE table streaming enabled, %" PRId64 " rows, no row cache\n", __func__, n_rows_);
    }
}

llama_ple_stream::llama_ple_stream(
        const llama_ple_sidecar_manifest & manifest, int64_t head_dim,
        enum ggml_type type, uint32_t n_cache_rows, bool use_direct_io) {
    if (head_dim != manifest.row_dim) {
        throw std::runtime_error("PLE sidecar embedding_row_dimension does not match the model's PLE head dim");
    }

    row_size_ = ggml_row_size(type, head_dim);
    if (row_size_ != manifest.row_stride) {
        throw std::runtime_error("PLE sidecar row_stride_bytes does not match the model's PLE dtype/head dim");
    }
    n_rows_   = manifest.n_rows;
    head_dim_ = head_dim;
    type_     = type;

    // same ordering rationale as the single-file constructor: clamp/warn on the
    // cache size first, open (and validate) every file next, and only declare
    // success - the O_DIRECT and "enabled" logs - once nothing left can throw
    init_cache(n_cache_rows);

    std::vector<std::string> paths;
    for (const auto & file : manifest.files) {
        paths.push_back(file.path);
    }
    open_files(paths, use_direct_io);

    for (const auto & seg : manifest.segments) {
        const uint64_t need = seg.file_offset + (uint64_t) seg.rows * row_size_;
        if (need > files_[seg.file_index]->size()) {
            throw std::runtime_error("PLE sidecar segment is not within its physical file's bounds");
        }
        segments_.push_back({ seg.global_row_start, seg.rows, seg.file_index, seg.file_offset });
    }

    alloc_row_buf();

    if (use_direct_io_) {
        LLAMA_LOG_INFO("%s: PLE table streaming uses O_DIRECT (page cache bypassed)\n", __func__);
    }
    if (n_cache_rows_ > 0) {
        LLAMA_LOG_INFO("%s: PLE sidecar streaming enabled, %" PRId64 " rows across %zu files, "
                "%" PRId64 " row cache slots (%.2f MiB)\n",
                __func__, n_rows_, files_.size(), (int64_t) n_cache_rows_, (double) cache_.size() / (1024.0 * 1024.0));
    } else {
        LLAMA_LOG_INFO("%s: PLE sidecar streaming enabled, %" PRId64 " rows across %zu files, no row cache\n",
                __func__, n_rows_, files_.size());
    }
}

llama_ple_stream::~llama_ple_stream() {
    if (stats_.n_gather_calls > 0) {
        LLAMA_LOG_INFO("%s: PLE table stream stats: %" PRId64 " gathers, %" PRId64 " row hits, %" PRId64 " row misses, %" PRId64 " file reads (%.2f MiB)\n",
                __func__, stats_.n_gather_calls, stats_.n_row_hits, stats_.n_row_misses,
                stats_.n_file_reads, (double) stats_.n_file_bytes / (1024.0 * 1024.0));
    }
    ple_aligned_free(row_buf_);
}

// segments_ is sorted by global_row_start and covers [0, n_rows_) with no gaps, so a
// binary search on the segment start finds the (single) owning segment
void llama_ple_stream::resolve(int32_t row, llama_file *& file, size_t & offs) const {
    auto it = std::upper_bound(segments_.begin(), segments_.end(), (int64_t) row,
            [](int64_t r, const segment & s) { return r < s.global_row_start; });
    GGML_ASSERT(it != segments_.begin());
    --it;
    file = files_[it->file_index].get();
    offs = it->file_offset + (size_t) (row - it->global_row_start) * row_size_;
}

// read row into the cache slot (slot >= 0) or into row_buf_ (slot < 0); sets src
// to the row bytes
bool llama_ple_stream::read_row(int32_t row, int64_t slot, const uint8_t * & src) {
    // files_[0] approximates every file's I/O mode: they are opened together, so
    // this is normally accurate. A per-file mid-stream O_DIRECT failure can desync
    // files_[0] from another file's actual state, but that only costs one wasted
    // O_DIRECT-style read attempt on the desynced file - its fd stays valid and the
    // read still returns correct data either way, so no correctness risk follows.
    if (use_direct_io_ && !files_[0]->has_direct_io()) {
        use_direct_io_ = false;
    }

    llama_file * file = nullptr;
    size_t offs = 0;
    resolve(row, file, offs);

    uint8_t * dst_buf = row_buf_;
    if (slot >= 0) {
        dst_buf = cache_.data() + slot * row_size_;
    }

    const uint8_t * p = nullptr;
    if (use_direct_io_) {
        p = ple_pread(*file, row_buf_, row_size_, offs, true);
        if (p == nullptr) {
            return false;
        }
        if (slot >= 0) {
            memcpy(dst_buf, p, row_size_);
            cache_tags_[slot] = row;
            src = dst_buf;
        } else {
            src = p;
        }
    } else {
        p = ple_pread(*file, dst_buf, row_size_, offs, false);
        if (p == nullptr) {
            return false;
        }
        if (slot >= 0) {
            cache_tags_[slot] = row;
        }
        src = p;
    }

    stats_.n_file_reads++;
    stats_.n_file_bytes += row_size_;
    return true;
}

bool llama_ple_stream::gather(const int32_t * idx, size_t n_idx, float * dst) {
    std::lock_guard<std::mutex> lock(mtx_);

    stats_.n_gather_calls++;

    const ggml_type_traits * traits = ggml_get_type_traits(type_);
    GGML_ASSERT(traits->to_float && "PLE table type has no to_float");

    // (row, position) pairs kept in idx order so that a resident row is served
    // before a later row in the same gather can evict it; consecutive equal rows
    // are still grouped below and read once
    pairs_.clear();
    pairs_.reserve(n_idx);
    for (size_t k = 0; k < n_idx; ++k) {
        const int32_t row = idx[k];
        GGML_ASSERT(row >= 0 && row < n_rows_);
        pairs_.emplace_back(row, k);
    }

    const bool cache_on = n_cache_rows_ > 0;

    for (size_t i = 0; i < pairs_.size();) {
        const int32_t row = pairs_[i].first;
        size_t j = i + 1;
        while (j < pairs_.size() && pairs_[j].first == row) {
            j++;
        }

        const uint8_t * src = nullptr;
        if (cache_on) {
            const int64_t slot = row % n_cache_rows_;
            if (cache_tags_[slot] == row) {
                stats_.n_row_hits++;
                src = cache_.data() + slot * row_size_;
            } else {
                stats_.n_row_misses++;
                if (!read_row(row, slot, src)) {
                    return false;
                }
            }
        } else {
            stats_.n_row_misses++;
            if (!read_row(row, -1, src)) {
                return false;
            }
        }

        for (size_t m = i; m < j; ++m) {
            traits->to_float(src, dst + pairs_[m].second * head_dim_, head_dim_);
        }

        i = j;
    }

    return true;
}
