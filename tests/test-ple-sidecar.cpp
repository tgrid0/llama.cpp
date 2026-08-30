#include "../src/llama-ple-sidecar.h"

#include "ggml.h"

#include <cassert>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

#undef NDEBUG
#include <cassert>

// writes a zero-filled file of exactly `bytes` length
static void write_sized_file(const std::string & path, uint64_t bytes) {
    std::ofstream f(path, std::ios::binary);
    assert(f);
    std::vector<char> zeros(4096, 0);
    uint64_t left = bytes;
    while (left > 0) {
        const uint64_t chunk = left < zeros.size() ? left : zeros.size();
        f.write(zeros.data(), (std::streamsize) chunk);
        left -= chunk;
    }
}

static void write_manifest(const std::string & path, const std::string & json) {
    std::ofstream f(path, std::ios::binary);
    assert(f);
    f << json;
}

// 2 physical files, 3 rows each, row_stride 8 bytes, embedding_row_dimension 4 (F16)
static const char * k_manifest_2files = R"JSON({
  "storage_dtype": "F16",
  "embedding_row_dimension": 4,
  "physical_files": [
    { "index": 0, "path": "ple/ple-bf16-00001-of-00002.bin", "file_bytes": 24 },
    { "index": 1, "path": "ple/ple-bf16-00002-of-00002.bin", "file_bytes": 24 }
  ],
  "logical_parts": [
    { "logical_part": 0, "physical_file_index": 0, "physical_file": "ple/ple-bf16-00001-of-00002.bin",
      "global_row_start": 0, "rows": 3, "file_offset": 0, "row_stride_bytes": 8, "embedding_row_dimension": 4 },
    { "logical_part": 1, "physical_file_index": 1, "physical_file": "ple/ple-bf16-00002-of-00002.bin",
      "global_row_start": 3, "rows": 3, "file_offset": 0, "row_stride_bytes": 8, "embedding_row_dimension": 4 }
  ]
})JSON";

static void test_parse_two_files() {
    const std::string dir = "ple-sidecar-test-dir";
    std::remove((dir + "/ple-bf16-00001-of-00002.bin").c_str());
    std::remove((dir + "/ple-bf16-00002-of-00002.bin").c_str());
    std::remove((dir + "/ple-manifest.json").c_str());
#if defined(_WIN32)
    system(("mkdir \"" + dir + "\" 2>NUL").c_str());
#else
    system(("mkdir -p \"" + dir + "\"").c_str());
#endif

    write_manifest(dir + "/ple-manifest.json", k_manifest_2files);
    write_sized_file(dir + "/ple-bf16-00001-of-00002.bin", 24);
    write_sized_file(dir + "/ple-bf16-00002-of-00002.bin", 24);

    // path names the directory
    {
        const auto m = llama_ple_sidecar_load(dir);
        assert(m.n_rows == 6);
        assert(m.row_dim == 4);
        assert(m.row_stride == 8);
        assert(m.storage_dtype == "F16");
        assert(m.files.size() == 2);
        assert(m.segments.size() == 2);
        assert(m.segments[0].global_row_start == 0 && m.segments[0].rows == 3 && m.segments[0].file_index == 0);
        assert(m.segments[1].global_row_start == 3 && m.segments[1].rows == 3 && m.segments[1].file_index == 1);
        assert(llama_ple_sidecar_dtype(m.storage_dtype) == GGML_TYPE_F16);
    }

    // path names the manifest file directly
    {
        const auto m = llama_ple_sidecar_load(dir + "/ple-manifest.json");
        assert(m.n_rows == 6);
    }

    // path names one of the .bin shards: the manifest sits next to it
    {
        const auto m = llama_ple_sidecar_load(dir + "/ple-bf16-00001-of-00002.bin");
        assert(m.n_rows == 6);
    }

    std::remove((dir + "/ple-bf16-00001-of-00002.bin").c_str());
    std::remove((dir + "/ple-bf16-00002-of-00002.bin").c_str());
    std::remove((dir + "/ple-manifest.json").c_str());
}

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

static void test_errors() {
    expect_throws([] { llama_ple_sidecar_load("ple-sidecar-test-missing-dir"); });

    const std::string dir = "ple-sidecar-test-badcases";
#if defined(_WIN32)
    system(("mkdir \"" + dir + "\" 2>NUL").c_str());
#else
    system(("mkdir -p \"" + dir + "\"").c_str());
#endif

    // manifest with a gap: row 3 is missing between the two segments
    write_manifest(dir + "/ple-manifest.json", R"JSON({
      "storage_dtype": "F16", "embedding_row_dimension": 4,
      "physical_files": [ { "index": 0, "path": "a.bin", "file_bytes": 16 } ],
      "logical_parts": [
        { "physical_file_index": 0, "global_row_start": 0, "rows": 1, "file_offset": 0, "row_stride_bytes": 8 },
        { "physical_file_index": 0, "global_row_start": 4, "rows": 1, "file_offset": 8, "row_stride_bytes": 8 }
      ]
    })JSON");
    write_sized_file(dir + "/a.bin", 16);
    expect_throws([&] { llama_ple_sidecar_load(dir); });

    // physical file shorter than the manifest expects
    write_manifest(dir + "/ple-manifest.json", R"JSON({
      "storage_dtype": "F16", "embedding_row_dimension": 4,
      "physical_files": [ { "index": 0, "path": "a.bin", "file_bytes": 16 } ],
      "logical_parts": [
        { "physical_file_index": 0, "global_row_start": 0, "rows": 2, "file_offset": 0, "row_stride_bytes": 8 }
      ]
    })JSON");
    write_sized_file(dir + "/a.bin", 8); // needs 16
    expect_throws([&] { llama_ple_sidecar_load(dir); });

    std::remove((dir + "/a.bin").c_str());
    std::remove((dir + "/ple-manifest.json").c_str());

    expect_throws([] { llama_ple_sidecar_dtype("NOT_A_TYPE"); });
}

int main() {
    test_parse_two_files();
    test_errors();
    printf("test-ple-sidecar: all tests passed\n");
    return 0;
}
