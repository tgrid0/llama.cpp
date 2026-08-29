#include "ggml.h"
#include "ggml-cpp.h"

#include "../src/llama-ple-stream.h"

#include <cassert>
#include <cstdint>
#include <cstdio>
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

int main() {
    test_f16();
    test_q8();
    test_truncated();
    test_ctor_errors();
    printf("test-ple-stream: all tests passed\n");
    return 0;
}
