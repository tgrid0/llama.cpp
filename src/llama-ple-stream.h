#pragma once

#include "ggml-cpp.h"
#include "llama-mmap.h"

#include <cstdint>
#include <memory>
#include <mutex>
#include <utility>
#include <vector>

// SSD streaming of the qwen4exp PLE n-gram table.
//
// The table (per_layer_token_embd.weight) can be tens of GiB and is read by a few
// random row gathers per token. Instead of loading it into RAM, rows are pread from
// the GGUF on demand into a bounded direct-mapped row cache and dequantized to F32
// exactly like the resident-table gather. This is the no-mmap counterpart of
// TENSOR_READ_LAZY: the table never leaves the SSD and the cache size is an
// explicit user-controlled bound. With O_DIRECT the page cache is bypassed too.
//
// note: like llama_moe_stream, concurrent decodes of the same model are not
// supported - gather() takes an internal mutex, but two contexts evict each
// other's cached rows.

class llama_ple_stream {
public:
    // opens `path` and prepares row reads at data offset `offs`; throws on bad
    // arguments. n_rows * row_size bytes must fit in the file, row_size =
    // ggml_row_size(type, head_dim). O_DIRECT falls back to buffered reads when
    // the OS or filesystem does not support it.
    llama_ple_stream(
            const char * path,
            size_t offs,
            int64_t n_rows,
            int64_t head_dim,
            enum ggml_type type,
            uint32_t n_cache_rows, // direct-mapped row cache slots; 0 disables the cache
            bool use_direct_io);
    ~llama_ple_stream();

    // dequantize rows idx[0..n_idx) to F32 into dst (head_dim * n_idx floats),
    // head element varies fastest within a row and rows are laid out consecutively:
    // dst[k*head_dim + e] = to_float(row idx[k])[e], the same layout the resident
    // gather produces. Returns false on I/O failure. Asserts idx[k] in [0, n_rows).
    bool gather(const int32_t * idx, size_t n_idx, float * dst);

    size_t  row_size() const { return row_size_; }
    int64_t n_rows()   const { return n_rows_; }

    struct stats {
        int64_t n_gather_calls = 0;
        int64_t n_row_hits     = 0;
        int64_t n_row_misses   = 0;
        int64_t n_file_reads   = 0;
        int64_t n_file_bytes   = 0;
    };
    const stats & get_stats() const { return stats_; }

private:
    bool read_row(int32_t row, int64_t slot, const uint8_t * & src);

    std::unique_ptr<llama_file> file_;
    size_t   offs_     = 0;
    size_t   row_size_ = 0;
    int64_t  n_rows_   = 0;
    int64_t  head_dim_ = 0;
    enum ggml_type type_ = GGML_TYPE_F32;
    bool use_direct_io_ = false;

    // direct-mapped row cache: slot = row % n_cache_rows_
    int64_t n_cache_rows_ = 0;
    std::vector<uint8_t> cache_;      // n_cache_rows_ * row_size_
    std::vector<int64_t> cache_tags_; // [slot] row id, -1 = empty

    // aligned staging for O_DIRECT reads and cache=0 reads (row_size_ + 2*align)
    uint8_t * row_buf_ = nullptr;

    // scratch: (row, position) pairs, sorted by row so equal rows are grouped and
    // reads go in file order
    std::vector<std::pair<int32_t, size_t>> pairs_;

    std::mutex mtx_;
    stats stats_;
};
