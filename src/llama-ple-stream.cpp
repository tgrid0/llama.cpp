#include "llama-ple-stream.h"

#include "llama-impl.h"

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

llama_ple_stream::llama_ple_stream(
        const char * path, size_t offs, int64_t n_rows, int64_t head_dim,
        enum ggml_type type, uint32_t n_cache_rows, bool use_direct_io) {
    if (path == nullptr || path[0] == '\0') {
        throw std::runtime_error("PLE table streaming requires a file-based model (not a stream/file descriptor)");
    }
    if (n_rows <= 0 || head_dim <= 0) {
        throw std::runtime_error("PLE table has invalid dimensions");
    }

    offs_     = offs;
    row_size_ = ggml_row_size(type, head_dim);
    n_rows_   = n_rows;
    head_dim_ = head_dim;
    type_     = type;
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

    // opened and validated before row_buf_ is allocated: both can throw, and row_buf_
    // is a raw pointer (no RAII), so allocating it first would leak on those paths
    auto open = [&](bool direct) {
        file_ = std::make_unique<llama_file>(path, "rb", direct);
    };

    open(use_direct_io);
    if (use_direct_io) {
        bool ok = file_->has_direct_io();
        if (ok) {
            uint8_t * probe = (uint8_t *) ple_aligned_alloc(PLE_STREAM_DIRECT_ALIGN);
            ok = probe != nullptr && ple_pread(*file_, probe, PLE_STREAM_DIRECT_ALIGN, 0, true) != nullptr;
            ple_aligned_free(probe);
        }
        if (!ok) {
            LLAMA_LOG_WARN("%s: O_DIRECT not usable for the PLE table, falling back to buffered reads\n", __func__);
            use_direct_io = false;
            open(false);
        }
    }
    use_direct_io_ = use_direct_io;

    if (offs_ + (size_t) n_rows_ * row_size_ > file_->size()) {
        throw std::runtime_error("PLE table data is not within the file bounds, model is corrupted or incomplete");
    }

    row_buf_ = (uint8_t *) ple_aligned_alloc(row_size_ + 2 * PLE_STREAM_DIRECT_ALIGN);
    if (row_buf_ == nullptr) {
        throw std::runtime_error("PLE table streaming allocation failed");
    }

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

llama_ple_stream::~llama_ple_stream() {
    if (stats_.n_gather_calls > 0) {
        LLAMA_LOG_INFO("%s: PLE table stream stats: %" PRId64 " gathers, %" PRId64 " row hits, %" PRId64 " row misses, %" PRId64 " file reads (%.2f MiB)\n",
                __func__, stats_.n_gather_calls, stats_.n_row_hits, stats_.n_row_misses,
                stats_.n_file_reads, (double) stats_.n_file_bytes / (1024.0 * 1024.0));
    }
    ple_aligned_free(row_buf_);
}

// read row into the cache slot (slot >= 0) or into row_buf_ (slot < 0); sets src
// to the row bytes
bool llama_ple_stream::read_row(int32_t row, int64_t slot, const uint8_t * & src) {
    // a direct read failure may have switched the file to buffered I/O
    if (use_direct_io_ && !file_->has_direct_io()) {
        use_direct_io_ = false;
    }

    const size_t offs = offs_ + (size_t) row * row_size_;

    uint8_t * dst_buf = row_buf_;
    if (slot >= 0) {
        dst_buf = cache_.data() + slot * row_size_;
    }

    const uint8_t * p = nullptr;
    if (use_direct_io_) {
        p = ple_pread(*file_, row_buf_, row_size_, offs, true);
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
        p = ple_pread(*file_, dst_buf, row_size_, offs, false);
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
