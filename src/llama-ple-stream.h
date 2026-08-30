#pragma once

#include "ggml-cpp.h"
#include "llama-mmap.h"

#include <cstdint>
#include <memory>
#include <mutex>
#include <utility>
#include <vector>

struct llama_ple_sidecar_manifest;

// SSD streaming of the qwen4exp PLE n-gram table.
//
// The table (per_layer_token_embd.weight, or an external ds4-dfm-style SSD-PLE
// sidecar, see --ple) can be tens of GiB and is read by a few random row gathers
// per token. Instead of loading it into RAM, rows are pread on demand into a
// bounded direct-mapped row cache and dequantized to F32 exactly like the
// resident-table gather. Rows may live in one file (the embedded-tensor case) or
// be split across several (the sidecar case); either way they are addressed
// through the same sorted list of contiguous row segments.
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

    // opens every physical file the manifest references and prepares row reads
    // across its segments; throws on bad arguments, including a head_dim that
    // does not match manifest.row_dim or a row size that does not match
    // manifest.row_stride.
    llama_ple_stream(
            const llama_ple_sidecar_manifest & manifest,
            int64_t head_dim,
            enum ggml_type type,
            uint32_t n_cache_rows,
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
    // returns a copy; gather() updates stats_ under mtx_
    stats get_stats() const {
        std::lock_guard<std::mutex> lock(mtx_);
        return stats_;
    }

private:
    // one contiguous run of global rows stored in one open file
    struct segment {
        int64_t  global_row_start;
        int64_t  rows;
        uint32_t file_index;
        uint64_t file_offset;
    };

    void init_cache(uint32_t n_cache_rows);
    void open_files(const std::vector<std::string> & paths, bool use_direct_io);
    void resolve(int32_t row, llama_file *& file, size_t & offs) const;
    bool read_row(int32_t row, int64_t slot, const uint8_t * & src);

    std::vector<std::unique_ptr<llama_file>> files_;
    std::vector<segment> segments_; // sorted by global_row_start, contiguous, covers [0, n_rows_)

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

    // scratch: (row, position) pairs, kept in idx order; consecutive equal rows grouped
    std::vector<std::pair<int32_t, size_t>> pairs_;

    mutable std::mutex mtx_; // get_stats() is const
    stats stats_;
};
