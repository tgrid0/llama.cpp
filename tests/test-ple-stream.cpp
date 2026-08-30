#include "ggml.h"
#include "ggml-cpp.h"
#include "llama.h"

#include "../src/llama-ple-stream.h"
#include "../src/llama-ple-sidecar.h"

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <stdexcept>
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

// the ctor rejects bad arguments with an exception
template <typename F>
static void expect_throws(F f) {
    bool threw = false;
    try {
        f();
    } catch (const std::exception &) {
        threw = true;
    }
    assert(threw);
}

// file truncated after open: rows past the new end make gather() fail
static void test_truncated() {
    const std::string path = "ple-test-trunc.bin";
    const int64_t head_dim = 4, n_rows = 8;
    write_f16_rows(path, n_rows, head_dim);

    {
        llama_ple_stream s(path.c_str(), 0, n_rows, head_dim, GGML_TYPE_F16, 0, false);
        const std::vector<int32_t> idx = {0};
        std::vector<float> dst(head_dim);
        assert(s.gather(idx.data(), idx.size(), dst.data()));
        check_dst(dst, idx, head_dim);

        // shrink to 1 row after open: row 5 is past the new end
        write_f16_rows(path, 1, head_dim);
        const std::vector<int32_t> idx2 = {5};
        std::vector<float> dst2(head_dim);
        assert(!s.gather(idx2.data(), idx2.size(), dst2.data()));
    }
    std::remove(path.c_str());
}

static void test_ctor_errors() {
    const std::string path = "ple-test-ctor.bin";
    const int64_t head_dim = 4, n_rows = 8;
    write_f16_rows(path, n_rows, head_dim);

    expect_throws([&] { llama_ple_stream s("", 0, n_rows, head_dim, GGML_TYPE_F16, 0, false); });
    expect_throws([&] { llama_ple_stream s("ple-test-missing.bin", 0, n_rows, head_dim, GGML_TYPE_F16, 0, false); });
    expect_throws([&] { llama_ple_stream s(path.c_str(), 0, 0, head_dim, GGML_TYPE_F16, 0, false); });
    expect_throws([&] { llama_ple_stream s(path.c_str(), 0, n_rows, -1, GGML_TYPE_F16, 0, false); });
    // data out of file bounds
    expect_throws([&] { llama_ple_stream s(path.c_str(), 0, n_rows * 100, head_dim, GGML_TYPE_F16, 0, false); });

    std::remove(path.c_str());
}

// captures every log line emitted while installed, to check log statement order
static void capture_log_lines(enum ggml_log_level level, const char * text, void * user_data) {
    (void) level;
    static_cast<std::vector<std::string> *>(user_data)->push_back(text);
}

// index of the first captured line containing needle, or -1
static int log_find(const std::vector<std::string> & logs, const char * needle) {
    for (size_t i = 0; i < logs.size(); ++i) {
        if (logs[i].find(needle) != std::string::npos) {
            return (int) i;
        }
    }
    return -1;
}

// regression test: a prior refactor reordered the single-file constructor's log
// statements so the --ple-cache-rows clamp WARN could print after the
// O_DIRECT-enabled INFO log instead of before it. head_dim is large enough that
// the file is >= 4096 bytes so the O_DIRECT probe (a 4096-byte read at offset 0)
// succeeds and O_DIRECT stays enabled, exercising both log lines together.
static void test_ctor_log_order_clamp_before_direct_io() {
    const std::string path = "ple-test-log-order.bin";
    const int64_t head_dim = 1024, n_rows = 8;
    write_f16_rows(path, n_rows, head_dim);

    std::vector<std::string> logs;
    ggml_log_callback prev_cb = nullptr;
    void * prev_ud = nullptr;
    ggml_log_get(&prev_cb, &prev_ud);
    llama_log_set(capture_log_lines, &logs);

    {
        llama_ple_stream s(path.c_str(), 0, n_rows, head_dim, GGML_TYPE_F16, (uint32_t) n_rows + 10, true);
    }

    llama_log_set(prev_cb, prev_ud);

    const int i_clamp   = log_find(logs, "clamping");
    const int i_direct  = log_find(logs, "uses O_DIRECT");
    const int i_enabled = log_find(logs, "streaming enabled");
    assert(i_clamp >= 0 && i_direct >= 0 && i_enabled >= 0);
    assert(i_clamp < i_direct);
    assert(i_direct < i_enabled);

    std::remove(path.c_str());
}

// regression test: the same refactor made the "uses O_DIRECT" INFO log print
// unconditionally inside open_files(), before the bounds check that runs after it
// in the constructor - so it printed even on a path where the bounds check then
// throws and the object is never built. n_rows is set far beyond what the file
// holds so the bounds check throws.
static void test_ctor_log_order_no_log_before_throw() {
    const std::string path = "ple-test-log-order-throw.bin";
    const int64_t head_dim = 1024, n_rows = 8;
    write_f16_rows(path, n_rows, head_dim);

    std::vector<std::string> logs;
    ggml_log_callback prev_cb = nullptr;
    void * prev_ud = nullptr;
    ggml_log_get(&prev_cb, &prev_ud);
    llama_log_set(capture_log_lines, &logs);

    expect_throws([&] { llama_ple_stream s(path.c_str(), 0, n_rows * 100, head_dim, GGML_TYPE_F16, 0, true); });

    llama_log_set(prev_cb, prev_ud);

    assert(log_find(logs, "uses O_DIRECT") < 0);
    assert(log_find(logs, "streaming enabled") < 0);

    std::remove(path.c_str());
}

// builds a 2-file sidecar manifest by hand (bypassing llama_ple_sidecar_load's file
// I/O) to test llama_ple_stream's multi-file addressing directly
static llama_ple_sidecar_manifest make_two_file_manifest(
        const std::string & path_a, const std::string & path_b, int64_t head_dim) {
    llama_ple_sidecar_manifest m;
    m.storage_dtype = "F16";
    m.row_dim       = head_dim;
    m.row_stride    = ggml_row_size(GGML_TYPE_F16, head_dim);
    m.files         = { { path_a, 4 * m.row_stride }, { path_b, 4 * m.row_stride } };
    m.segments      = {
        { 0, 4, 0, 0 },
        { 4, 4, 1, 0 },
    };
    m.n_rows = 8;
    return m;
}

static void test_sidecar_two_files() {
    const std::string path_a = "ple-test-sidecar-a.bin";
    const std::string path_b = "ple-test-sidecar-b.bin";
    const int64_t head_dim = 4;

    // file a holds rows [0,4) with values 0..3, file b holds rows [4,8) with values 4..7
    write_f16_rows(path_a, 4, head_dim);
    {
        FILE * f = fopen(path_b.c_str(), "wb");
        assert(f != nullptr);
        std::vector<ggml_fp16_t> row(head_dim);
        for (int64_t r = 0; r < 4; ++r) {
            for (int64_t e = 0; e < head_dim; ++e) {
                row[e] = ggml_fp32_to_fp16((float) (r + 4));
            }
            fwrite(row.data(), sizeof(ggml_fp16_t), (size_t) head_dim, f);
        }
        fclose(f);
    }

    const auto manifest = make_two_file_manifest(path_a, path_b, head_dim);
    llama_ple_stream s(manifest, head_dim, GGML_TYPE_F16, 8, false);

    const std::vector<int32_t> idx = { 0, 4, 3, 7 };
    std::vector<float> dst(head_dim * idx.size());
    assert(s.gather(idx.data(), idx.size(), dst.data()));
    check_dst(dst, idx, head_dim);

    std::remove(path_a.c_str());
    std::remove(path_b.c_str());
}

static void test_sidecar_dim_mismatch() {
    const std::string path_a = "ple-test-sidecar-mismatch.bin";
    write_f16_rows(path_a, 4, 4);
    auto manifest = make_two_file_manifest(path_a, path_a, 4);

    // caller-expected head_dim (8) does not match the manifest's row_dim (4)
    expect_throws([&] { llama_ple_stream s(manifest, 8, GGML_TYPE_F16, 0, false); });

    std::remove(path_a.c_str());
}

// a hand-built manifest (not run through llama_ple_sidecar_load's own validation)
// with a segment pointing past the end of the files list must not index out of bounds
static void test_sidecar_bad_file_index() {
    const std::string path_a = "ple-test-sidecar-bad-index.bin";
    write_f16_rows(path_a, 4, 4);

    auto manifest = make_two_file_manifest(path_a, path_a, 4);
    manifest.segments[1].file_index = 2; // only files[0] and files[1] exist

    expect_throws([&] { llama_ple_stream s(manifest, 4, GGML_TYPE_F16, 0, false); });

    std::remove(path_a.c_str());
}

int main() {
    test_f16();
    test_q8();
    test_truncated();
    test_ctor_errors();
    test_sidecar_two_files();
    test_sidecar_dim_mismatch();
    test_sidecar_bad_file_index();
    test_ctor_log_order_clamp_before_direct_io();
    test_ctor_log_order_no_log_before_throw();
    printf("test-ple-stream: all tests passed\n");
    return 0;
}
