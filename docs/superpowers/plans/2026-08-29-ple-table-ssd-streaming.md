# PLE Table SSD Streaming (--ple-stream) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Keep the qwen4exp PLE n-gram table (`per_layer_token_embd.weight`, ~48-100 GiB) on SSD instead of loading it into RAM, reading only the hashed rows per step into a bounded host row cache, so the model runs with `--load-mode none`/`dio` on a 128 GB UMA Strix Halo box with fully controlled memory.

**Architecture:** Add a small `llama_ple_stream` module (new files `src/llama-ple-stream.h/.cpp`, mirroring `llama-moe-stream`): it reopens the GGUF file, preads distinct rows on demand (optional O_DIRECT with buffered fallback), keeps a direct-mapped row cache sized by a user flag, and dequantizes rows to F32 exactly like the current host gather. The qwen4exp arch registers the table as `TENSOR_STREAMED` (loader skips it, no buffer, no data), and `llm_graph_input_ple::set_input` gathers through the stream instead of `per_layer_tok_embd->data`. New flags `--ple-stream`, `--ple-cache-rows`, `--ple-direct-io` flow common-params -> llama_model_params -> arch.

**Tech Stack:** C++ (llama.cpp fork), ggml, llama_file (O_DIRECT-capable file wrapper), ggml type traits for row dequantization.

## Global Constraints

- Per user: do NOT build the full Vulkan/RADV stack on this PC (builds happen on the separate Strix Halo machine). Only the CPU-only unit-test target may be built locally to run the new test.
- No unicode characters in code or comments: use ASCII only (`->`, `x`, `...`). No em-dashes, no arrows.
- Comments: concise (1-2 lines), plain wording, no restating the code. Match surrounding style.
- Follow existing patterns, do not restructure: mirror `llama-moe-stream.h/.cpp`, the `--moe-stream` arg blocks, and the loader's `TENSOR_STREAMED` skip path.
- Do not split a line mid-sentence to fit a column width.
- The joined single-tensor GGUF (`per_layer_token_embd.weight`) is required; head-split files (gguf_split_ple_heads.py) are out of scope for this fork.
- `--ple-stream` works with any `--load-mode`; it replaces the mmap `--tensor-read-lazy` path for the table. Recommended with `--load-mode none` or `dio`.
- Every task's requirements implicitly include this section.

---

### Task 1: `llama_ple_stream` module + unit test

**Files:**
- Create: `src/llama-ple-stream.h`
- Create: `src/llama-ple-stream.cpp`
- Modify: `src/CMakeLists.txt` (add the .cpp to the llama library sources, after `llama-moe-stream.cpp`)
- Create: `tests/test-ple-stream.cpp`
- Modify: `tests/CMakeLists.txt` (register the test with `llama_build_and_test`)
- Test: `tests/test-ple-stream.cpp`

**Interfaces:**
- Consumes: `llama_file` from `src/llama-mmap.h` (constructor `llama_file(const char * fname, const char * mode, bool use_direct_io = false)`, `file_id()`, `seek`, `read_raw`, `has_direct_io()`, `size()`); `ggml_row_size`, `ggml_get_type_traits`, `ggml_fp32_to_fp16` from `ggml/include/ggml.h`; `LLAMA_LOG_*` from `src/llama-impl.h`.
- Produces: class `llama_ple_stream` with constructor
  `llama_ple_stream(const char * path, size_t offs, int64_t n_rows, int64_t head_dim, enum ggml_type type, uint32_t n_cache_rows, bool use_direct_io)`,
  `bool gather(const int32_t * idx, size_t n_idx, float * dst)`,
  `size_t row_size() const`, `int64_t n_rows() const`,
  `struct llama_ple_stream::stats { int64_t n_gather_calls, n_row_hits, n_row_misses, n_file_reads, n_file_bytes; }` and `const stats & get_stats() const`.
  Task 2 consumes `gather`, `row_size`, and the class type. Task 4 consumes `get_stats()`.

- [ ] **Step 1: Write the failing test**

`tests/test-ple-stream.cpp` (assert/printf style, like `tests/test-llama-archs.cpp`; includes mirror that file: `"ggml.h"`, `"ggml-cpp.h"`, and `../src/llama-ple-stream.h`):

```cpp
#include "ggml.h"
#include "ggml-cpp.h"

#include "../src/llama-ple-stream.h"

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#undef NDEBUG
#include <cassert>

// row r holds value (float) r in every element
static void write_f16_rows(const std::string & path, int64_t n_rows, int64_t head_dim) {
    FILE * f = fopen(path.c_str(), "wb");
    assert(f != nullptr);
    std::vector<ggml_fp16_t> row(head_dim);
    for (int64_t r = 0; r < n_rows; ++r) {
        for (int64_t e = 0; e < head_dim; ++e) {
            row[e] = ggml_fp32_to_fp16((float) r);
        }
        fwrite(row.data(), sizeof(ggml_fp16_t), (size_t) head_dim, f);
    }
    fclose(f);
}

// one Q8_0 row with value r in every element: scale = r, all quantized bytes = 1,
// so to_float(row) = r
static void write_q8_rows(const std::string & path, int64_t n_rows, int64_t head_dim) {
    FILE * f = fopen(path.c_str(), "wb");
    assert(f != nullptr);
    assert(head_dim % 32 == 0);
    const int64_t n_blocks = head_dim / 32;
    for (int64_t r = 0; r < n_rows; ++r) {
        for (int64_t b = 0; b < n_blocks; ++b) {
            const ggml_fp16_t scale = ggml_fp32_to_fp16((float) r);
            fwrite(&scale, sizeof(ggml_fp16_t), 1, f);
            std::vector<int8_t> q(32, 1);
            fwrite(q.data(), sizeof(int8_t), 32, f);
        }
    }
    fclose(f);
}

static void check_dst(const std::vector<float> & dst, const std::vector<int32_t> & idx, int64_t head_dim) {
    for (size_t k = 0; k < idx.size(); ++k) {
        for (int64_t e = 0; e < head_dim; ++e) {
            assert(dst[k*head_dim + e] == (float) idx[k]);
        }
    }
}

static void test_f16() {
    const std::string path = "ple-test-f16.bin";
    const int64_t head_dim = 4, n_rows = 8;
    write_f16_rows(path, n_rows, head_dim);

    // no cache: values and layout
    {
        llama_ple_stream s(path.c_str(), 0, n_rows, head_dim, GGML_TYPE_F16, 0, false);
        const std::vector<int32_t> idx = {1, 3, 3, 2};
        std::vector<float> dst(head_dim * idx.size());
        assert(s.gather(idx.data(), idx.size(), dst.data()));
        check_dst(dst, idx, head_dim);
        assert(s.get_stats().n_row_misses == 3); // {1,3,2} distinct
    }

    // cache: second gather of the same rows is served without file reads
    {
        llama_ple_stream s(path.c_str(), 0, n_rows, head_dim, GGML_TYPE_F16, 4, false);
        const std::vector<int32_t> idx1 = {0, 1, 2, 3};
        std::vector<float> dst1(head_dim * idx1.size());
        assert(s.gather(idx1.data(), idx1.size(), dst1.data()));
        check_dst(dst1, idx1, head_dim);

        const std::vector<int32_t> idx2 = {3, 2, 1, 0};
        std::vector<float> dst2(head_dim * idx2.size());
        assert(s.gather(idx2.data(), idx2.size(), dst2.data()));
        check_dst(dst2, idx2, head_dim);
        assert(s.get_stats().n_file_reads == 4); // no re-reads on the second call
    }

    // eviction: 2 slots, rows 0 and 2 collide on slot 0, row 0 is re-read
    {
        llama_ple_stream s(path.c_str(), 0, n_rows, head_dim, GGML_TYPE_F16, 2, false);
        const std::vector<int32_t> idx = {0};
        std::vector<float> dst(head_dim * idx.size());
        assert(s.gather(idx.data(), idx.size(), dst.data()));
        const std::vector<int32_t> idx2 = {2};
        std::vector<float> dst2(head_dim * idx2.size());
        assert(s.gather(idx2.data(), idx2.size(), dst2.data()));
        const std::vector<int32_t> idx3 = {2, 0};
        std::vector<float> dst3(head_dim * idx3.size());
        assert(s.gather(idx3.data(), idx3.size(), dst3.data()));
        check_dst(dst3, idx3, head_dim);
        assert(s.get_stats().n_file_reads == 3); // row 0 re-read after eviction
    }

    // direct IO init must not fail: falls back to buffered where unsupported
    {
        llama_ple_stream s(path.c_str(), 0, n_rows, head_dim, GGML_TYPE_F16, 0, true);
        const std::vector<int32_t> idx = {1, 5, 3};
        std::vector<float> dst(head_dim * idx.size());
        assert(s.gather(idx.data(), idx.size(), dst.data()));
        check_dst(dst, idx, head_dim);
    }

    std::remove(path.c_str());
}

static void test_q8() {
    const std::string path = "ple-test-q8.bin";
    const int64_t head_dim = 32, n_rows = 8;
    write_q8_rows(path, n_rows, head_dim);

    llama_ple_stream s(path.c_str(), 0, n_rows, head_dim, GGML_TYPE_Q8_0, 0, false);
    const std::vector<int32_t> idx = {2, 5, 5};
    std::vector<float> dst(head_dim * idx.size());
    assert(s.gather(idx.data(), idx.size(), dst.data()));
    check_dst(dst, idx, head_dim);
    assert(s.get_stats().n_row_misses == 2);

    std::remove(path.c_str());
}

int main() {
    test_f16();
    test_q8();
    printf("test-ple-stream: all tests passed\n");
    return 0;
}
```

- [ ] **Step 2: Run test to verify it fails**

Configure a CPU-only test build (per the user's constraint, this is the only thing built on this PC):

```bash
cmake -S . -B build-tests -DGGML_VULKAN=OFF -DLLAMA_BUILD_TESTS=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-tests --target test-ple-stream --config Release
ctest --test-dir build-tests -R test-ple-stream --output-on-failure
```

Expected: build fails with `llama-ple-stream.h: No such file or directory` (or the test fails to link `llama_ple_stream`). That is the expected failure.

- [ ] **Step 3: Write the module header**

`src/llama-ple-stream.h`:

```cpp
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
```

- [ ] **Step 4: Write the module implementation**

`src/llama-ple-stream.cpp`:

```cpp
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
            return nullptr;
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
    n_cache_rows_ = n_cache_rows;

    if (n_cache_rows_ > 0) {
        cache_.resize((size_t) n_cache_rows_ * row_size_);
        cache_tags_.assign(n_cache_rows_, -1);
    }
    row_buf_ = (uint8_t *) ple_aligned_alloc(row_size_ + 2 * PLE_STREAM_DIRECT_ALIGN);
    if (row_buf_ == nullptr) {
        throw std::runtime_error("PLE table streaming allocation failed");
    }

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

    // (row, position) pairs, sorted by row: equal rows are grouped and read once,
    // and the reads go in file order
    pairs_.clear();
    pairs_.reserve(n_idx);
    for (size_t k = 0; k < n_idx; ++k) {
        const int32_t row = idx[k];
        GGML_ASSERT(row >= 0 && row < n_rows_);
        pairs_.emplace_back(row, k);
    }
    std::sort(pairs_.begin(), pairs_.end());

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
```

- [ ] **Step 5: Register the source and the test**

In `src/CMakeLists.txt`, add `llama-ple-stream.cpp` to the llama library sources, directly after the `llama-moe-stream.cpp` line (currently line 39).

In `tests/CMakeLists.txt`, add after the `test-model-load-cancel.cpp` registration block (currently around line 281):

```cmake
llama_build_and_test(test-ple-stream.cpp)
```

- [ ] **Step 6: Run test to verify it passes**

```bash
cmake --build build-tests --target test-ple-stream --config Release
ctest --test-dir build-tests -R test-ple-stream --output-on-failure
```

Expected: `test-ple-stream: all tests passed`, ctest reports PASS. If the module's MSVC path (seek+read under `_WIN32`) misbehaves locally, that is a real bug to fix in Step 4, not a test change.

- [ ] **Step 7: Commit**

```bash
git add src/llama-ple-stream.h src/llama-ple-stream.cpp src/CMakeLists.txt tests/test-ple-stream.cpp tests/CMakeLists.txt
git commit -m "ple: SSD-stream the PLE n-gram table rows into a bounded host cache"
```

---

### Task 2: Wire the stream into the qwen4exp arch

**Files:**
- Modify: `src/models/models.h` (qwen4exp class: member + destructor declaration)
- Modify: `src/models/qwen4exp.cpp` (load_arch_tensors skip + stream init; build_ple host-gather gating; set_input gather source)
- Modify: `src/models/qwen4exp.cpp` (out-of-line destructor definition)
- Test: existing `tests/test-llama-archs.cpp` must still pass (regression; ple_stream defaults off). End-to-end behavior is verified manually in Task 4 (the real model file is not on this PC).

**Interfaces:**
- Consumes: Task 1's `llama_ple_stream` class; `llama_model_loader::TENSOR_STREAMED` and `ml.create_tensor(hparams, buft_list_cpu, buft_list_input, buft_list_output, buft_list_layer, tn, ne, flags)` from `src/llama-model-loader.h`; `ml.get_weight(name)` returning `llama_tensor_weight { uint16_t idx; size_t offs; ggml_tensor * tensor; }`; `ml.file_paths` (`std::vector<std::string>`, same order as files, empty string for FILE*-based loading); `params.ple_stream/ple_cache_rows/ple_direct_io` from `llama_model_params` (Task 3 adds them to `include/llama.h`; implement Task 3 first, or compile Task 2 after Task 3 - the fields are referenced here).
- Produces: `llama_model_qwen4exp::ple_stream` member (`std::unique_ptr<llama_ple_stream>`, null when streaming is off); `per_layer_tok_embd == nullptr` when streaming is on. Task 4 uses these.

- [ ] **Step 1: Add the member and destructor declaration**

In `src/models/models.h`, inside `struct llama_model_qwen4exp` (currently around line 2309), add near the `gather_tables()` override:

```cpp
        // SSD streaming of the PLE n-gram table; null unless --ple-stream.
        // per_layer_tok_embd stays null then: the graph gathers through this stream.
        std::unique_ptr<llama_ple_stream> ple_stream;

        ~llama_model_qwen4exp() override;
```

Add the forward declaration just above the class (the header must not include llama-ple-stream.h):

```cpp
struct llama_ple_stream;
```

- [ ] **Step 2: Add the out-of-line destructor**

In `src/models/qwen4exp.cpp`, add after the class definitions (e.g. right after `llama_model_qwen4exp::load_arch_tensors`):

```cpp
llama_model_qwen4exp::~llama_model_qwen4exp() = default;
```

Add the include at the top of `src/models/qwen4exp.cpp`, after the existing `#include "llama-memory-recurrent.h"`:

```cpp
#include "llama-ple-stream.h"
```

The out-of-line destructor is required: `std::unique_ptr<llama_ple_stream>` needs the complete type where the destructor is instantiated, and other TUs that include `models.h` only see the forward declaration.

- [ ] **Step 3: Skip loading the table and create the stream**

In `src/models/qwen4exp.cpp`, `llama_model_qwen4exp::load_arch_tensors`, replace the PLE-table creation block (currently around lines 133-143):

```cpp
    // flat [ple_head_dim, n_rows] gather target; n_rows is padded, so read it back
    if (hparams.ple_n_heads > 0) {
        const std::string ple_name = tn(LLM_TENSOR_PER_LAYER_TOKEN_EMBD, "weight").str();
        const auto * ple_w = ml.get_weight(ple_name.c_str());
        GGML_ASSERT(ple_w != nullptr && "qwen4exp is missing the PLE n-gram table");
        const int64_t ple_rows = ple_w->tensor->ne[1];

        if (params.ple_stream) {
            // stream the table from the GGUF on demand: register the skip so the
            // loader accounting stays consistent; no buffer is allocated, no data
            // is loaded. the buft lists are unused for a TENSOR_STREAMED skip.
            ml.create_tensor(hparams, nullptr, nullptr, nullptr, nullptr,
                    tn(LLM_TENSOR_PER_LAYER_TOKEN_EMBD, "weight"),
                    { hparams.ple_head_dim, ple_rows }, TENSOR_STREAMED);

            ple_stream = std::make_unique<llama_ple_stream>(
                    ml.file_paths[ple_w->idx].c_str(), ple_w->offs, ple_rows,
                    hparams.ple_head_dim, ple_w->tensor->type,
                    params.ple_cache_rows, params.ple_direct_io);

            // per_layer_tok_embd stays nullptr: build_ple gathers through
            // ple_stream, which forces the host-gather path
        } else {
            // the PLE/engram table is gathered 16 random rows per token and never
            // read densely, so it wants MADV_RANDOM and no eager pull-in. See
            // --tensor-read-lazy.
            per_layer_tok_embd = create_tensor(tn(LLM_TENSOR_PER_LAYER_TOKEN_EMBD, "weight"),
                                               { hparams.ple_head_dim, ple_rows }, TENSOR_READ_LAZY);
        }
    }
```

Note: with `--ple-stream`, `gather_tables()` returns `{}` automatically (it returns `{ per_layer_tok_embd }` only when non-null), so the mmap gather-range machinery and `prefetch_rows` stay inert. No change needed there.

- [ ] **Step 4: Force host gather in build_ple**

In `src/models/qwen4exp.cpp`, `llama_model_qwen4exp::graph::build_ple`, change the branch that selects the gather path (currently around line 1029-1040):

```cpp
    // heads lie slowest within a token either way, as the reference does
    ggml_tensor * emb = nullptr;
    if (ple_host_gather() || pmodel.ple_stream != nullptr) {
        ple_inp->emb = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32,
                hparams.ple_head_dim * n_heads, n_tokens);
        ggml_set_input(ple_inp->emb);
        emb = ple_inp->emb;
        res->add_input(std::move(ple_inp));
    } else {
        ple_inp->rows = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_heads * n_tokens);
        ggml_set_input(ple_inp->rows);
        ggml_tensor * rows = ple_inp->rows;
        res->add_input(std::move(ple_inp));

        emb = ggml_get_rows(ctx0, model.per_layer_tok_embd, rows);
        emb = ggml_reshape_2d(ctx0, emb, hparams.ple_head_dim * n_heads, n_tokens);
    }
```

Add the local at the top of `build_ple` (after `GGML_UNUSED(inp);`):

```cpp
    const llama_model_qwen4exp & pmodel = static_cast<const llama_model_qwen4exp &>(model);
```

The in-graph `ggml_get_rows` fallback must never run while streaming: `per_layer_tok_embd` is null and the stream reads rows host-side. The `LLAMA_PLE_HOST_GATHER=0` env var is ignored in that case.

- [ ] **Step 5: Gather through the stream in set_input**

In `src/models/qwen4exp.cpp`, `llm_graph_input_ple::set_input`, replace the host-gather body (currently around lines 1134-1160):

```cpp
    // the table is far too big to offload, so it is gathered straight out of the mapping: one
    // fault per row, 16 per token, no two of them on the same page. left to the get_rows those
    // faults happen one at a time; queued here they are in flight before the graph even runs.
    pmodel.prefetch_rows(pmodel.per_layer_tok_embd, idx.data(), idx.size()); // no-op with ple_stream (per_layer_tok_embd == nullptr)

    if (rows) {
        ggml_backend_tensor_set(rows, idx.data(), 0, idx.size()*ggml_element_size(rows));
        return;
    }

    // Gather host-side. Head varies fastest within a token, the layout ggml_get_rows produced for
    // the same index vector, so the flattened [head_dim * n_heads] row per token is unchanged.
    const int64_t head_dim = pmodel.ple_stream != nullptr
        ? (int64_t) pmodel.hparams.ple_head_dim
        : pmodel.per_layer_tok_embd->ne[0];

    // get_rows dequantised to F32; keep that so the downstream matmuls are bit-identical
    std::vector<float> vals((size_t) head_dim * idx.size());
    if (pmodel.ple_stream != nullptr) {
        // the SSD stream reads the rows on demand and dequantizes them exactly like
        // the resident gather below
        if (!pmodel.ple_stream->gather(idx.data(), idx.size(), vals.data())) {
            throw std::runtime_error("PLE table streaming read failed");
        }
    } else {
        const ggml_tensor * tbl      = pmodel.per_layer_tok_embd;
        const size_t        row_sz   = ggml_row_size(tbl->type, head_dim);
        const char *        base     = (const char *) tbl->data;

        if (tbl->type == GGML_TYPE_F32) {
            for (size_t k = 0; k < idx.size(); ++k) {
                memcpy(vals.data() + k*head_dim, base + (size_t) idx[k]*row_sz, head_dim*sizeof(float));
            }
        } else {
            const ggml_type_traits * traits = ggml_get_type_traits(tbl->type);
            GGML_ASSERT(traits->to_float && "PLE table type has no to_float");
            for (size_t k = 0; k < idx.size(); ++k) {
                traits->to_float(base + (size_t) idx[k]*row_sz, vals.data() + k*head_dim, head_dim);
            }
        }
    }

    ggml_backend_tensor_set(emb, vals.data(), 0, vals.size()*sizeof(float));
```

- [ ] **Step 6: Compile and run the regression tests**

```bash
cmake --build build-tests --target test-ple-stream test-llama-archs --config Release
ctest --test-dir build-tests -R "test-ple-stream|test-llama-archs" --output-on-failure
```

Expected: both pass. `test-llama-archs` exercises the synthetic qwen4exp arch with `ple_stream` defaulting off, so this is the regression gate for the loader/arch changes.

- [ ] **Step 7: Commit**

```bash
git add src/models/models.h src/models/qwen4exp.cpp
git commit -m "qwen4exp: stream the PLE table through the SSD row cache (--ple-stream)"
```

---

### Task 3: Expose `--ple-stream`, `--ple-cache-rows`, `--ple-direct-io`

**Files:**
- Modify: `include/llama.h` (`llama_model_params` fields)
- Modify: `src/llama-model.cpp` (`llama_model_default_params` defaults)
- Modify: `common/common.h` (`common_params` fields)
- Modify: `common/common.cpp` (map common_params -> llama_model_params)
- Modify: `common/arg.cpp` (three `add_opt` blocks)
- Modify: `tests/test-arg-parser.cpp` (registration assertion)
- Test: `tests/test-arg-parser.cpp`

**Interfaces:**
- Consumes: nothing from Tasks 1-2 at runtime, but Task 2 references these `llama_model_params` fields, so implement this task before compiling Task 2.
- Produces: `common_params::{ple_stream(bool, default false), ple_cache_rows(uint32_t, default 1<<20), ple_direct_io(bool, default false)}`; `llama_model_params::{ple_stream, ple_cache_rows, ple_direct_io}` with the same defaults. Task 2 and Task 4 consume these.

- [ ] **Step 1: Add the fields to `llama_model_params`**

In `include/llama.h`, inside `struct llama_model_params`, after the `bool moe_stream;` field (currently around line 350):

```cpp
        // SSD streaming of the PLE n-gram table (qwen4exp): the table is not loaded;
        // rows are read from the GGUF on demand into a bounded host row cache
        uint32_t ple_cache_rows;  // row cache slots (0 = disabled)
        bool     ple_direct_io;   // use O_DIRECT for PLE reads (bypass page cache); falls back if unsupported
        bool     ple_stream;      // stream the PLE n-gram table from disk on demand
```

- [ ] **Step 2: Add the defaults**

In `src/llama-model.cpp`, `llama_model_default_params` (currently around line 2796), after the `.moe_stream = false,` line:

```cpp
        /*.ple_cache_rows           =*/ 1 << 20,
        /*.ple_direct_io            =*/ false,
        /*.ple_stream               =*/ false,
```

- [ ] **Step 3: Add the fields to `common_params`**

In `common/common.h`, after the `moe_stream_direct` field (currently around line 581):

```cpp
    uint32_t ple_cache_rows      = 1 << 20; // PLE n-gram row cache slots (0 = disabled)
    bool     ple_direct_io       = false;   // use O_DIRECT for PLE table reads (bypass page cache)
    bool     ple_stream          = false;   // stream the PLE n-gram table from disk on demand
```

- [ ] **Step 4: Map the fields**

In `common/common.cpp`, after the `mparams.moe_stream_direct` line (currently around line 1692):

```cpp
    mparams.ple_cache_rows      = params.ple_cache_rows;
    mparams.ple_direct_io       = params.ple_direct_io;
    mparams.ple_stream          = params.ple_stream;
```

- [ ] **Step 5: Add the CLI args**

In `common/arg.cpp`, insert these three `add_opt` blocks directly after the `--moe-stream-direct` block (currently around line 2812):

```cpp
    add_opt(common_arg(
        {"--ple-stream"},
        "stream the PLE n-gram table from the GGUF on demand instead of loading it (qwen4exp); "
        "keeps the table out of RAM (use with --load-mode none or dio); bypasses the mmap lazy-read path",
        [](common_params & params) {
            params.ple_stream = true;
        }
    ).set_env("LLAMA_ARG_PLE_STREAM"));
    add_opt(common_arg(
        {"--ple-cache-rows"}, "N",
        "PLE row cache slots for --ple-stream (0 = disabled, default: 1M rows); each slot is one "
        "quantized table row; implies --ple-stream",
        [](common_params & params, const std::string & value) {
            params.ple_stream = true;
            const uint64_t n = std::stoul(value);
            if (n > UINT32_MAX) {
                throw std::invalid_argument("PLE cache rows out of range");
            }
            params.ple_cache_rows = (uint32_t) n;
        }
    ).set_env("LLAMA_ARG_PLE_CACHE_ROWS"));
    add_opt(common_arg(
        {"--ple-direct-io"},
        "use O_DIRECT for --ple-stream table reads (bypass the page cache); "
        "falls back to buffered reads if O_DIRECT is unsupported by the OS or filesystem; implies --ple-stream",
        [](common_params & params) {
            params.ple_stream = true;
            params.ple_direct_io = true;
        }
    ).set_env("LLAMA_ARG_PLE_DIRECT_IO"));
```

- [ ] **Step 6: Write the registration test**

In `tests/test-arg-parser.cpp`, inside `test()`, after the existing loop over examples (currently around line 60), add:

```cpp
    {
        // the PLE streaming flags must be registered (server example)
        auto ctx_arg = common_params_parser_init(params, LLAMA_EXAMPLE_SERVER);
        common_params_add_preset_options(ctx_arg.options);
        std::unordered_set<std::string> args;
        for (const auto & opt : ctx_arg.options) {
            for (const auto & arg : opt.get_args()) {
                args.insert(arg);
            }
        }
        assert(args.count("--ple-stream") == 1);
        assert(args.count("--ple-cache-rows") == 1);
        assert(args.count("--ple-direct-io") == 1);
    }
```

- [ ] **Step 7: Run the tests**

```bash
cmake --build build-tests --target test-arg-parser test-ple-stream --config Release
ctest --test-dir build-tests -R "test-arg-parser|test-ple-stream" --output-on-failure
```

Expected: both pass. The duplicate-argument scan in `test-arg-parser` also guards the new flag names against collisions.

- [ ] **Step 8: Commit**

```bash
git add include/llama.h src/llama-model.cpp common/common.h common/common.cpp common/arg.cpp tests/test-arg-parser.cpp
git commit -m "common: add --ple-stream, --ple-cache-rows and --ple-direct-io"
```

---

### Task 4: End-to-end verification on the Strix Halo box + docs

**Files:**
- Modify: none (verification only). Optional: `src/models/qwen4exp.cpp` comment fix if measurements disagree with the plan's expectations.
- Test: manual on the target machine (Ryzen AI MAX+ 395 / Radeon 8060S, gfx1151, Vulkan RADV).

**Interfaces:**
- Consumes: Tasks 1-3 (module, arch wiring, flags).

- [ ] **Step 1: Build the Vulkan server on the target box**

```bash
cmake -S . -B build-vulkan -DGGML_VULKAN=ON -DLLAMA_BUILD_SERVER=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-vulkan -j --target llama-server
```

Expected: builds; `./build-vulkan/bin/llama-server --list-devices` shows the Radeon 8060S.

- [ ] **Step 2: Load-check with `--load-mode none --ple-stream`**

Run (start with a smaller context; the harness repo observed a cold-boot OOM at the full 262144 allocation on Vulkan in the upload transient):

```bash
./build-vulkan/bin/llama-server \
  -m <model.gguf> -md <mtp-sidecar.gguf> \
  -ngl 999 -c 65536 --load-mode none \
  --ple-stream --ple-cache-rows 1048576 \
  -fa on -ctk q8_0 -ctv q8_0 --parallel 1 --jinja --no-webui
```

Expected: the load completes without OOM. `RssAnon` stays near the resident model size and does NOT grow by the table size (~48-100 GiB). The server log shows `PLE table streaming enabled, ... rows, 1048576 row cache slots (... MiB)` and (with `--ple-direct-io` added) `uses O_DIRECT`.

- [ ] **Step 3: Correctness vs the reference**

Send the same greedy prompt (temperature 0) to this server and to a reference run of the same build with `--load-mode mmap --tensor-read-lazy on`. The streaming gather dequantizes the same file bytes with the same `to_float`, so outputs must be byte-identical.

Expected: identical outputs. If they differ, stop and debug (the gather layout or row offset is wrong) - do not ship.

- [ ] **Step 4: Memory bounding**

With `--ple-direct-io`, sample `/proc/<pid>/status` VmRSS and the GPU residency during a 24K-token prefill and during decode. Expected: no page-cache growth from the table (O_DIRECT bypasses it); RSS grows only by the row cache (1048576 slots * row_size, ~2-8 GiB depending on the table quant). The exit log prints hit/miss/file-read stats; a sane decode workload should show mostly row hits.

- [ ] **Step 5: Throughput sanity**

Run the harness repo's `server-bench.py` or `llama-bench` on prose/code/structured. Expected: decode within ~20-40% of the mmap+lazy reference (the extra cost is the per-step row reads); prefill shows some SSD I/O cost on the first pass (rows are read once per distinct n-gram). If decode regresses hard, first try a larger `--ple-cache-rows`; if prefill regresses hard, that is a known v1 limitation (synchronous reads in set_input; async readahead is the documented follow-up, do not add it in this plan).

- [ ] **Step 6: Commit any comment/doc fixes**

If any expectation in Steps 2-5 was wrong, fix the stale comment in `load_arch_tensors` (it says "See --tensor-read-lazy") to mention `--ple-stream`, and commit:

```bash
git commit -m "qwen4exp: document the --ple-stream SSD path for the PLE table"
```

---

## Self-Review

**1. Spec coverage:**
- "Keep PLE table on disk, full model resident, bounded small hot cache" -> Task 1 (bounded direct-mapped row cache + O_DIRECT), Task 2 (table never loaded), Task 4 (verification). Covered.
- "Only stream PLE, not general any-expert/any-layer" -> the module is PLE-specific and only the PER_LAYER_TOKEN_EMBD tensor is skipped; MoE streaming untouched. Covered.
- Works with `--load-mode none`/`dio` (no mmap) -> the stream reads the GGUF directly and needs no mmap; `TENSOR_READ_LAZY` (mmap-only) is bypassed. Covered.
- Memory control for co-loading small models -> explicit buffers + explicit cache only; O_DIRECT keeps the file out of the page cache. Covered by design; verified in Task 4 Step 4.
- User constraint (no full Vulkan build on this PC) -> all build steps here are the CPU-only `build-tests` target; Task 4 builds on the target box. Covered.
- Bit-identical outputs vs the resident gather -> same `to_float` on the same bytes; asserted in Task 4 Step 3. Covered.

**2. Placeholder scan:** no TBD/TODO; every step has concrete code or commands; the only open-ended item (Task 4 Step 5 tuning) is a measurement step with explicit pass/fail criteria, not an implementation placeholder.

**3. Type consistency:** `llama_ple_stream::gather(const int32_t*, size_t, float*)` is declared in Task 1 and called identically in Task 2 Step 5. `params.ple_stream/ple_cache_rows/ple_direct_io` are added in Task 3 and consumed in Task 2 with the same names. `TENSOR_STREAMED` and `ml.create_tensor` match `src/llama-model-loader.h`. The `stats` struct and `get_stats()` names match between Task 1 and the Task 1 test. No mismatches found.

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-08-29-ple-table-ssd-streaming.md`. Two execution options:

**1. Subagent-Driven (recommended)** - I dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** - Execute tasks in this session using executing-plans, batch execution with checkpoints

Which approach?
