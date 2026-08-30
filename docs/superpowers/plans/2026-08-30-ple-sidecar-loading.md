# PLE Sidecar Loading (--ple) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Add a `--ple <path>` flag so qwen4exp models can load their PLE n-gram table from an external ds4-dfm-style SSD-PLE sidecar (a JSON manifest plus a few large raw-row `.bin` files) instead of requiring the table to be a tensor inside the model's own GGUF, with a three-way fallback (sidecar -> embedded GGUF tensor -> none, warn and continue) so every qwen4exp variant loads without crashing.

**Architecture:** A new arch-specific, dependency-light manifest parser (`llama_ple_sidecar_load`, `src/llama-ple-sidecar.h/.cpp`) resolves `--ple <path>` to a `ple-manifest.json`, parses and validates it with the already-vendored `nlohmann::json`, and produces a plain description of physical files and row segments. `llama_ple_stream` (from the existing SSD-streaming feature) gains a second constructor that consumes that manifest directly, generalizing its single-file row addressing into a small sorted list of `(global_row_start, rows, file_index, file_offset)` segments so the existing cache/O_DIRECT/gather code needs no changes. `qwen4exp.cpp`'s `load_arch_tensors` tries the sidecar first, falls back to the embedded tensor, and - if neither exists - logs a warning and disables the PLE gating block for that model instead of asserting.

**Tech Stack:** C++17 (llama.cpp fork), `nlohmann::json` (already vendored at `vendor/nlohmann/json.hpp`, currently only wired into `common`), `std::filesystem` (already used in `common/`, new to `src/` but part of the C++17 baseline), the existing `llama_ple_stream` module from the prior `--ple-stream` plan.

**Spec:** this plan's own "New feature spec" section below (verbatim decisions from the user, gathered via clarifying questions and a live fetch of the reference manifest schema from https://huggingface.co/Baekpica/Qwen3.8-Flash-Next-Mixed-Quant-SSD-PLE-GGUF/tree/main).

## New feature spec (verbatim decisions - do not re-litigate)

- Support a sidecar PLE table matching the **exact on-disk format ds4-dfm consumes**, exposed via `--ple <path>` (not ds4-dfm's implicit manifest-relative-to-model-dir lookup).
- Confirmed real manifest schema (`MQ-Q6-SSD-PLE-BF16/ple/ple-manifest.json` on the HF repo above): top-level `embedding_row_dimension` (160), `storage_dtype` ("BF16"), plus two arrays:
  - `physical_files[]`: `{ index, path, file_bytes, payload_bytes, sha256, verified_bytes }`, path e.g. `"ple/ple-bf16-00001-of-00004.bin"`.
  - `logical_parts[]`: `{ logical_part, physical_file_index, physical_file, global_row_start, rows, file_offset, payload_bytes, row_stride_bytes, embedding_row_dimension, sha256, source_name, source_shard }`, `row_stride_bytes` 320 = 160 BF16 elements, every logical-shard start 4 KiB aligned.
  - On disk: `<dir>/ple-manifest.json` plus N `.bin` files named `ple-bf16-XXXXX-of-NNNNN.bin`, always found next to the manifest regardless of what the manifest's own `path`/`physical_file` fields say (verified against the real repo: the fields carry an artifact-root-relative path that does not match a `--ple <path>` pointed straight at the `ple/` directory, so resolution must not trust them literally - see Task 1).
  - Do not hash 95 GiB of payload at load time; do validate structure (segments cover `[0, n_rows)` with no gaps/overlaps, files exist and are large enough) and fail loudly on mismatch, same rigor as `llama_ple_stream`'s existing bounds check.
- **Locked to the qwen4exp arch only** - no generic/reusable sidecar-table abstraction.
- Resolution/fallback order in `load_arch_tensors`, only reached when `hparams.ple_n_heads > 0` (the model declares PLE-gated layers):
  1. `--ple <path>` given: try to load the sidecar. Success -> stream from it. Failure (bad path, malformed manifest, missing/short file) -> log a warning, fall through to step 2.
  2. No `--ple <path>`, or it failed: look for the PLE tensor embedded in the model's own GGUF (existing behavior, gated by `--ple-stream`/`--ple-cache-rows`/`--ple-direct-io` exactly as before).
  3. Neither found: log a warning and continue without a PLE table - the per-layer PLE gating weights (`ple_key`/`ple_value`/`ple_norm_*`/`ple_conv1d`) still load normally (they are small and always part of the checkpoint), but the gating block itself is skipped at graph-build time so the model runs as if those layers were not PLE-gated.
  4. `--ple <path>` given but the model has no PLE layers at all (`hparams.ple_n_heads == 0`): warn and ignore the flag.
- `--ple <path>` implies `--ple-stream`, mirroring `--ple-cache-rows`/`--ple-direct-io` - a sidecar table is definitionally SSD-streamed, never loaded resident.

## Global Constraints

- No unicode characters in code or comments: ASCII only (`->`, `x`, `...`). No em-dashes, no arrows.
- Comments: concise (1-2 lines), plain wording, no restating the code. Match surrounding style.
- Follow existing patterns, do not restructure: mirror `llama-ple-stream.h/.cpp`'s style, the `--ple-stream`/`--ple-cache-rows`/`--ple-direct-io` arg blocks, and `llama_model_qwen4exp::load_arch_tensors`'s existing PLE block.
- Do not split a line mid-sentence to fit a column width.
- Reuse `nlohmann::json` (already vendored at `vendor/nlohmann/json.hpp`); do not add a new JSON dependency.
- Do not hash the sidecar's payload bytes at load time (95+ GiB); validate structure only.
- Every task's requirements implicitly include this section.

---

### Task 1: Sidecar manifest parser (`llama_ple_sidecar_load`)

**Files:**
- Create: `src/llama-ple-sidecar.h`
- Create: `src/llama-ple-sidecar.cpp`
- Modify: `src/CMakeLists.txt` (add the .cpp after `llama-ple-stream.cpp`; add `../vendor` to the `llama` target's include dirs so it can see `nlohmann/json.hpp`, which today is only wired into `common`)
- Create: `tests/test-ple-sidecar.cpp`
- Modify: `tests/CMakeLists.txt` (register the test next to `test-ple-stream.cpp`, same Windows-shared-build guard)

**Interfaces:**
- Consumes: `nlohmann::json` from `vendor/nlohmann/json.hpp`; `std::filesystem` for path resolution and file-size checks.
- Produces: `struct llama_ple_sidecar_file { std::string path; uint64_t file_bytes; }`, `struct llama_ple_sidecar_segment { int64_t global_row_start; int64_t rows; uint32_t file_index; uint64_t file_offset; }`, `struct llama_ple_sidecar_manifest { std::vector<llama_ple_sidecar_file> files; std::vector<llama_ple_sidecar_segment> segments; int64_t n_rows; int64_t row_dim; size_t row_stride; std::string storage_dtype; }`, function `llama_ple_sidecar_manifest llama_ple_sidecar_load(const std::string & user_path)` (throws `std::runtime_error` on any failure), and `enum ggml_type llama_ple_sidecar_dtype(const std::string & storage_dtype)` (throws on an unrecognized dtype string). Task 2 consumes the manifest struct and `llama_ple_sidecar_dtype`. Task 3 consumes `llama_ple_sidecar_load` and the exception contract.

- [x] **Step 1: Write the failing test**

`tests/test-ple-sidecar.cpp` (assert/printf style, mirrors `tests/test-ple-stream.cpp`):

```cpp
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
```

- [x] **Step 2: Run test to verify it fails**

```bash
cmake -S . -B build-tests -DGGML_VULKAN=OFF -DLLAMA_BUILD_TESTS=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-tests --target test-ple-sidecar --config Release
```

Expected: build fails, `llama-ple-sidecar.h: No such file or directory`.

- [x] **Step 3: Write the header**

`src/llama-ple-sidecar.h`:

```cpp
#pragma once

#include "ggml.h"

#include <cstdint>
#include <string>
#include <vector>

// Parses and validates a ds4-dfm-style SSD-PLE sidecar manifest (ple-manifest.json):
// the qwen4exp PLE n-gram table stored as raw rows across a few large physical files,
// addressed by a JSON index instead of GGUF tensor metadata. See --ple.
// Reference format: https://huggingface.co/Baekpica/Qwen3.8-Flash-Next-Mixed-Quant-SSD-PLE-GGUF

struct llama_ple_sidecar_file {
    std::string path;        // resolved filesystem path, ready to open
    uint64_t    file_bytes = 0;
};

// one contiguous run of global rows stored in one physical file
struct llama_ple_sidecar_segment {
    int64_t  global_row_start = 0;
    int64_t  rows             = 0;
    uint32_t file_index       = 0; // into llama_ple_sidecar_manifest::files
    uint64_t file_offset      = 0; // byte offset of global_row_start within that file
};

struct llama_ple_sidecar_manifest {
    std::vector<llama_ple_sidecar_file>    files;
    std::vector<llama_ple_sidecar_segment> segments; // sorted by global_row_start, contiguous, no gaps
    int64_t     n_rows     = 0; // sum of segments[*].rows
    int64_t     row_dim    = 0; // embedding_row_dimension
    size_t      row_stride = 0; // bytes per row, from the manifest
    std::string storage_dtype;  // e.g. "BF16"
};

// Resolves `user_path` to a manifest file (the path itself if it names a JSON file,
// `user_path/ple-manifest.json` if it names a directory, or the manifest found next
// to `user_path` if it names some other file, e.g. one of the .bin shards), parses
// it, and validates internal consistency: segments cover [0, n_rows) with no gaps
// or overlaps, and every referenced physical file exists and is large enough.
// Throws std::runtime_error with a descriptive message on any failure.
llama_ple_sidecar_manifest llama_ple_sidecar_load(const std::string & user_path);

// maps a manifest's storage_dtype string ("BF16", "F16", "F32", "Q8_0") to a ggml_type.
// Throws std::runtime_error on an unrecognized value.
enum ggml_type llama_ple_sidecar_dtype(const std::string & storage_dtype);
```

- [x] **Step 4: Write the implementation**

`src/llama-ple-sidecar.cpp`:

```cpp
#include "llama-ple-sidecar.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <stdexcept>

namespace fs = std::filesystem;
using json = nlohmann::json;

static fs::path find_manifest(const std::string & user_path) {
    fs::path p(user_path);
    std::error_code ec;
    if (fs::is_directory(p, ec)) {
        return p / "ple-manifest.json";
    }
    if (p.filename() == "ple-manifest.json") {
        return p;
    }
    // any other file (e.g. one of the .bin shards): the manifest sits next to it
    return p.parent_path() / "ple-manifest.json";
}

// the manifest's own path/physical_file fields record a path relative to whatever
// artifact root the producing tool used, which is not necessarily where --ple points;
// every physical file is expected to sit next to the manifest itself, so resolve by
// basename first and only fall back to the manifest's literal relative path
static fs::path resolve_physical_file(const fs::path & manifest_dir, const std::string & field) {
    std::error_code ec;

    fs::path by_basename = manifest_dir / fs::path(field).filename();
    if (fs::exists(by_basename, ec)) {
        return by_basename;
    }
    fs::path by_relative = manifest_dir / field;
    if (fs::exists(by_relative, ec)) {
        return by_relative;
    }
    fs::path as_given(field);
    if (as_given.is_absolute() && fs::exists(as_given, ec)) {
        return as_given;
    }
    throw std::runtime_error("PLE sidecar manifest references a missing physical file: " + field);
}

llama_ple_sidecar_manifest llama_ple_sidecar_load(const std::string & user_path) {
    const fs::path manifest_path = find_manifest(user_path);

    std::ifstream f(manifest_path, std::ios::binary);
    if (!f) {
        throw std::runtime_error("PLE sidecar manifest not found: " + manifest_path.string());
    }

    json j;
    try {
        f >> j;
    } catch (const json::exception & e) {
        throw std::runtime_error("PLE sidecar manifest is not valid JSON (" + manifest_path.string() + "): " + e.what());
    }

    if (!j.contains("physical_files") || !j.contains("logical_parts")) {
        throw std::runtime_error("PLE sidecar manifest is missing physical_files or logical_parts: " + manifest_path.string());
    }

    llama_ple_sidecar_manifest out;
    out.row_dim       = j.value("embedding_row_dimension", (int64_t) 0);
    out.storage_dtype = j.value("storage_dtype", std::string());

    const fs::path manifest_dir = manifest_path.parent_path();

    for (const auto & pf : j.at("physical_files")) {
        llama_ple_sidecar_file file;
        file.path       = resolve_physical_file(manifest_dir, pf.at("path").get<std::string>()).string();
        file.file_bytes = pf.value("file_bytes", (uint64_t) 0);
        out.files.push_back(std::move(file));
    }

    for (const auto & lp : j.at("logical_parts")) {
        llama_ple_sidecar_segment seg;
        seg.global_row_start = lp.at("global_row_start").get<int64_t>();
        seg.rows             = lp.at("rows").get<int64_t>();
        seg.file_index       = lp.at("physical_file_index").get<uint32_t>();
        seg.file_offset      = lp.at("file_offset").get<uint64_t>();

        if (seg.file_index >= out.files.size()) {
            throw std::runtime_error("PLE sidecar manifest logical_part references an out-of-range physical_file_index");
        }

        const uint64_t stride = lp.value("row_stride_bytes", (uint64_t) 0);
        if (out.row_stride == 0) {
            out.row_stride = stride;
        } else if (stride != 0 && stride != out.row_stride) {
            throw std::runtime_error("PLE sidecar manifest has inconsistent row_stride_bytes across logical_parts");
        }

        out.segments.push_back(seg);
    }

    if (out.segments.empty() || out.row_stride == 0) {
        throw std::runtime_error("PLE sidecar manifest has no usable logical_parts: " + manifest_path.string());
    }

    std::sort(out.segments.begin(), out.segments.end(),
            [](const llama_ple_sidecar_segment & a, const llama_ple_sidecar_segment & b) {
                return a.global_row_start < b.global_row_start;
            });

    int64_t expect_row = 0;
    for (const auto & seg : out.segments) {
        if (seg.global_row_start != expect_row) {
            throw std::runtime_error("PLE sidecar manifest logical_parts have a gap or overlap at row " + std::to_string(expect_row));
        }
        const uint64_t need = seg.file_offset + (uint64_t) seg.rows * out.row_stride;
        if (need > out.files[seg.file_index].file_bytes) {
            throw std::runtime_error("PLE sidecar manifest logical_part exceeds its physical file's recorded size");
        }
        expect_row += seg.rows;
    }
    out.n_rows = expect_row;

    for (const auto & file : out.files) {
        std::error_code ec;
        const uint64_t actual = (uint64_t) fs::file_size(file.path, ec);
        if (ec || actual < file.file_bytes) {
            throw std::runtime_error("PLE sidecar physical file is missing or smaller than the manifest expects: " + file.path);
        }
    }

    return out;
}

enum ggml_type llama_ple_sidecar_dtype(const std::string & storage_dtype) {
    if (storage_dtype == "BF16") return GGML_TYPE_BF16;
    if (storage_dtype == "F16")  return GGML_TYPE_F16;
    if (storage_dtype == "F32")  return GGML_TYPE_F32;
    if (storage_dtype == "Q8_0") return GGML_TYPE_Q8_0;
    throw std::runtime_error("PLE sidecar manifest has an unrecognized storage_dtype: " + storage_dtype);
}
```

- [x] **Step 5: Register the source and the test**

In `src/CMakeLists.txt`:
- Add `llama-ple-sidecar.cpp` directly after the `llama-ple-stream.cpp` line (currently line 40).
- Add a line after `target_include_directories(llama PRIVATE .)` (currently line 61):
  ```cmake
  target_include_directories(llama PRIVATE ../vendor)
  ```
  (`nlohmann/json.hpp` is header-only and already vendored there; today only `common` sees it.)

In `tests/CMakeLists.txt`, inside the same guard as `test-ple-stream.cpp` (currently lines 283-286):

```cmake
if (NOT WIN32 OR NOT BUILD_SHARED_LIBS)
    llama_build_and_test(test-ple-stream.cpp)
    llama_build_and_test(test-ple-sidecar.cpp)
endif()
```

- [x] **Step 6: Run test to verify it passes**

```bash
cmake --build build-tests --target test-ple-sidecar --config Release
ctest --test-dir build-tests -R test-ple-sidecar --output-on-failure
```

Expected: `test-ple-sidecar: all tests passed`.

- [x] **Step 7: Commit**

```bash
git add src/llama-ple-sidecar.h src/llama-ple-sidecar.cpp src/CMakeLists.txt tests/test-ple-sidecar.cpp tests/CMakeLists.txt
git commit -m "ple: parse and validate ds4-dfm-style SSD-PLE sidecar manifests"
```

---

### Task 2: Multi-file row addressing in `llama_ple_stream`

**Files:**
- Modify: `src/llama-ple-stream.h`
- Modify: `src/llama-ple-stream.cpp`
- Modify: `tests/test-ple-stream.cpp` (append sidecar-manifest tests)

**Interfaces:**
- Consumes: Task 1's `llama_ple_sidecar_manifest`, `llama_ple_sidecar_file`, `llama_ple_sidecar_segment`.
- Produces: a second `llama_ple_stream` constructor, `llama_ple_stream(const llama_ple_sidecar_manifest & manifest, int64_t head_dim, enum ggml_type type, uint32_t n_cache_rows, bool use_direct_io)`; `gather()`, `row_size()`, `n_rows()`, `get_stats()` behave identically regardless of which constructor built the stream. Task 3 consumes this constructor.

This task generalizes the existing single-file `offs_ + row*row_size_` addressing into a small sorted segment list, so both constructors share one row-resolution path. The existing single-file constructor's behavior and the existing tests must not change.

- [x] **Step 1: Write the failing test**

Append to `tests/test-ple-stream.cpp` (before `int main()`), and add `#include "../src/llama-ple-sidecar.h"` and `#include <fstream>` to its includes:

```cpp
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
```

And add both calls to `main()`:

```cpp
    test_sidecar_two_files();
    test_sidecar_dim_mismatch();
```

- [x] **Step 2: Run test to verify it fails**

```bash
cmake --build build-tests --target test-ple-stream --config Release
```

Expected: fails to compile, no such constructor.

- [x] **Step 3: Refactor the header to a segment-list internal layout**

Replace `src/llama-ple-stream.h` in full:

```cpp
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
```

- [x] **Step 4: Rewrite the implementation around the segment list**

Replace `src/llama-ple-stream.cpp` in full:

```cpp
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

    if (use_direct_io_) {
        LLAMA_LOG_INFO("%s: PLE table streaming uses O_DIRECT (page cache bypassed)\n", __func__);
    }
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

    // opened and validated before row_buf_/cache_ are allocated: both can throw, and
    // row_buf_ is a raw pointer (no RAII), so allocating it first would leak on those paths
    open_files({ path }, use_direct_io);

    if (offs + (size_t) n_rows_ * row_size_ > files_[0]->size()) {
        throw std::runtime_error("PLE table data is not within the file bounds, model is corrupted or incomplete");
    }
    segments_.push_back({ 0, n_rows_, 0, offs });

    init_cache(n_cache_rows);

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

    init_cache(n_cache_rows);

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
    // a direct read failure may have switched the file to buffered I/O
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
```

Note the `use_direct_io_ && !files_[0]->has_direct_io()` check in `read_row`: with a single file this is unchanged from before; with a sidecar, all files were opened (and, on failure, reopened buffered) together in `open_files`, so checking `files_[0]` is representative of every file's I/O mode.

- [x] **Step 5: Run all ple-stream tests**

```bash
cmake --build build-tests --target test-ple-stream test-ple-sidecar --config Release
ctest --test-dir build-tests -R "test-ple-stream|test-ple-sidecar" --output-on-failure
```

Expected: both pass, including the pre-existing single-file tests (`test_f16`, `test_q8`, `test_truncated`, `test_ctor_errors`) unchanged.

- [x] **Step 6: Commit**

```bash
git add src/llama-ple-stream.h src/llama-ple-stream.cpp tests/test-ple-stream.cpp
git commit -m "ple: generalize llama_ple_stream to multi-file sidecar row addressing"
```

---

### Task 3: Wire `--ple <path>` and the three-way fallback into qwen4exp

**Files:**
- Modify: `include/llama.h` (`llama_model_params.ple_path`)
- Modify: `src/llama-model.cpp` (`llama_model_default_params` default)
- Modify: `common/common.h` (`common_params.ple_path`)
- Modify: `common/common.cpp` (map `common_params.ple_path` -> `llama_model_params.ple_path`)
- Modify: `common/arg.cpp` (new `--ple` block; fix the now-stale `--ple-stream` help text)
- Modify: `src/models/models.h` (`llama_model_qwen4exp::ple_table_available`)
- Modify: `src/models/qwen4exp.cpp` (`load_arch_tensors` three-way fallback; graph-build guard)
- Modify: `tests/test-arg-parser.cpp` (registration assertion for `--ple`)

**Interfaces:**
- Consumes: Task 1's `llama_ple_sidecar_load`, `llama_ple_sidecar_dtype` (and their `std::runtime_error` contract); Task 2's sidecar `llama_ple_stream` constructor.
- Produces: `common_params::ple_path` (`std::string`, default empty); `llama_model_params::ple_path` (`const char *`, default `nullptr`); `llama_model_qwen4exp::ple_table_available` (`bool`).

- [x] **Step 1: Add the field to `llama_model_params`**

In `include/llama.h`, inside `struct llama_model_params`, after `bool ple_stream;` (currently line 363):

```cpp
        // external SSD-PLE sidecar directory/manifest (qwen4exp); NULL or empty falls
        // back to the model's embedded PLE table. Implies ple_stream = true.
        const char * ple_path;
```

- [x] **Step 2: Add the default**

In `src/llama-model.cpp`, `llama_model_default_params`, after `/*.ple_stream =*/ false,` (currently line 2822):

```cpp
        /*.ple_path                 =*/ nullptr,
```

- [x] **Step 3: Add the field to `common_params`**

In `common/common.h`, after `bool ple_stream = false;` (currently line 585):

```cpp
    std::string ple_path; // external SSD-PLE sidecar directory/manifest (qwen4exp)
```

- [x] **Step 4: Map the field**

In `common/common.cpp`, after `mparams.ple_stream = params.ple_stream;` (currently line 1695):

```cpp
    mparams.ple_path = params.ple_path.empty() ? nullptr : params.ple_path.c_str();
```

`params` outlives `mparams` here (same call, `common_model_params_to_llama` returns before `params.ple_path` could be mutated or destroyed), matching the lifetime the rest of this function already relies on.

- [x] **Step 5: Add the CLI arg, and fix the stale `--ple-stream` help text**

In `common/arg.cpp`, replace the `--ple-stream` block's help string (currently around line 2814-2818, the "not a separate sidecar file" line is no longer true):

```cpp
    add_opt(common_arg(
        {"--ple-stream"},
        "stream the PLE n-gram table from the GGUF on demand instead of loading it (qwen4exp); "
        "keeps the table out of RAM (use with --load-mode none or dio); bypasses the mmap lazy-read path",
        [](common_params & params) {
            params.ple_stream = true;
        }
    ).set_env("LLAMA_ARG_PLE_STREAM"));
```

Then add a new block right after the `--ple-direct-io` block (currently ending around line 2844):

```cpp
    add_opt(common_arg(
        {"--ple"}, "PATH",
        "load the PLE n-gram table (qwen4exp) from an external SSD-PLE sidecar directory "
        "or manifest (a ple-manifest.json plus its .bin shards, the ds4-dfm SSD-PLE format) "
        "instead of the model's own GGUF; falls back to the model's embedded table if the "
        "sidecar cannot be loaded, and to running without a PLE table if neither is found; "
        "implies --ple-stream",
        [](common_params & params, const std::string & value) {
            params.ple_stream = true;
            params.ple_path = value;
        }
    ).set_env("LLAMA_ARG_PLE_PATH"));
```

- [x] **Step 6: Write the registration test**

In `tests/test-arg-parser.cpp`, next to the existing `--ple-stream` assertions (currently around line 118-120):

```cpp
        assert(args.count("--ple") == 1);
```

- [x] **Step 7: Run test to verify it fails (arg not yet registered)**

```bash
cmake --build build-tests --target test-arg-parser --config Release
ctest --test-dir build-tests -R test-arg-parser --output-on-failure
```

Expected: fails until Step 5's edit to `arg.cpp` is in place; run this after Step 5, not before, since Steps 5-6 land together in the same source pass. Confirm PASS once both are done.

- [x] **Step 8: Add the `ple_table_available` member**

In `src/models/models.h`, inside `struct llama_model_qwen4exp`, next to the existing `ple_stream` member:

```cpp
        // true once load_arch_tensors resolved a PLE table (sidecar or embedded); false
        // means the model declares PLE-gated layers but no table was found - the gating
        // block is skipped at graph-build time instead of crashing (PLE is optional).
        bool ple_table_available = false;
```

- [x] **Step 9: Implement the three-way fallback in `load_arch_tensors`**

In `src/models/qwen4exp.cpp`, add the include:

```cpp
#include "llama-ple-sidecar.h"
```

Replace the PLE-table block (Task 2 of the prior plan put this around lines 135-163; re-locate it by searching for `hparams.ple_n_heads > 0` in `load_arch_tensors`):

```cpp
    // flat [ple_head_dim, n_rows] gather target; n_rows is padded, so read it back
    if (hparams.ple_n_heads > 0) {
        const std::string ple_name = tn(LLM_TENSOR_PER_LAYER_TOKEN_EMBD, "weight").str();

        if (params.ple_path != nullptr && params.ple_path[0] != '\0') {
            try {
                const llama_ple_sidecar_manifest manifest = llama_ple_sidecar_load(params.ple_path);
                const enum ggml_type sidecar_type = llama_ple_sidecar_dtype(manifest.storage_dtype);

                ple_stream = std::make_unique<llama_ple_stream>(
                        manifest, hparams.ple_head_dim, sidecar_type,
                        params.ple_cache_rows, params.ple_direct_io);

                LLAMA_LOG_INFO("%s: PLE n-gram table loaded from sidecar %s (%" PRId64 " rows across %zu files)\n",
                        __func__, params.ple_path, manifest.n_rows, manifest.files.size());
                ple_table_available = true;
            } catch (const std::exception & e) {
                LLAMA_LOG_WARN("%s: --ple %s could not be loaded (%s), falling back to the model's embedded PLE table\n",
                        __func__, params.ple_path, e.what());
            }
        }

        if (!ple_table_available) {
            const auto * ple_w = ml.get_weight(ple_name.c_str());
            if (ple_w == nullptr) {
                LLAMA_LOG_WARN("%s: qwen4exp declares PLE layers but no PLE n-gram table was found "
                        "(no --ple sidecar, no embedded '%s' tensor); continuing without it, "
                        "output quality will be reduced\n", __func__, ple_name.c_str());
            } else {
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
                ple_table_available = true;
            }
        }
    } else if (params.ple_path != nullptr && params.ple_path[0] != '\0') {
        LLAMA_LOG_WARN("%s: --ple was given but this model has no PLE layers, ignoring it\n", __func__);
    }
```

- [x] **Step 10: Skip the PLE gating block when no table is available**

In `src/models/qwen4exp.cpp`, `llama_model_qwen4exp::graph::graph` (the constructor that builds the trunk graph), change the guard around the `build_ple` call (currently around line 375-377):

```cpp
        if (hparams.is_ple(il)) {
            res_hc = build_ple(inp->get_recr(), mctx_hyb, res_hc, il);
        }
```

to:

```cpp
        if (hparams.is_ple(il) && static_cast<const llama_model_qwen4exp &>(model).ple_table_available) {
            res_hc = build_ple(inp->get_recr(), mctx_hyb, res_hc, il);
        }
```

`res_hc` unchanged is the correct "no PLE contribution" behavior: `build_ple` only ever adds a gated correction on top of `res_hc` (see its final `ggml_add`-based combine), so skipping it entirely is equivalent to that gate being closed everywhere.

- [x] **Step 11: Compile and run the regression tests**

```bash
cmake --build build-tests --target test-ple-stream test-ple-sidecar test-llama-archs test-arg-parser --config Release
ctest --test-dir build-tests -R "test-ple-stream|test-ple-sidecar|test-llama-archs|test-arg-parser" --output-on-failure
```

Expected: all pass. `test-llama-archs` exercises the synthetic qwen4exp arch with every PLE flag defaulting off, so `ple_table_available` should end up matching whatever that fixture's GGUF already provides (no regression).

- [x] **Step 12: Commit**

```bash
git add include/llama.h src/llama-model.cpp common/common.h common/common.cpp common/arg.cpp src/models/models.h src/models/qwen4exp.cpp tests/test-arg-parser.cpp
git commit -m "qwen4exp: load the PLE table from an external SSD-PLE sidecar (--ple)"
```

---

### Task 4: End-to-end verification on the Strix Halo box + docs

**Files:**
- Modify: none (verification only).
- Test: manual on the target machine (Ryzen AI MAX+ 395 / Radeon 8060S, gfx1151, Vulkan RADV).

**Interfaces:**
- Consumes: Tasks 1-3.

- [ ] **Step 1: Build the Vulkan server on the target box**

```bash
cmake -S . -B build-vulkan -DGGML_VULKAN=ON -DLLAMA_BUILD_SERVER=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-vulkan -j --target llama-server
```

- [ ] **Step 2: Load-check against a real sidecar**

Download (or point at an already-downloaded copy of) one `MQ-Q*-SSD-PLE-BF16` variant from
`https://huggingface.co/Baekpica/Qwen3.8-Flash-Next-Mixed-Quant-SSD-PLE-GGUF`, keeping its
`ple/` subfolder alongside the GGUF as published. Run:

```bash
./build-vulkan/bin/llama-server \
  -m <MQ-Q6-SSD-PLE-BF16>/Qwen3.8-Flash-Next-MQ-Q6-SSD-PLE-BF16-*.gguf \
  --ple <MQ-Q6-SSD-PLE-BF16>/ple \
  -ngl 999 -c 65536 --load-mode none \
  --ple-cache-rows 1048576 \
  -fa on -ctk q8_0 -ctv q8_0 --parallel 1 --jinja --no-webui
```

Expected: the log shows `PLE n-gram table loaded from sidecar ... (N rows across 4 files)` and
`PLE sidecar streaming enabled, ...`. RssAnon does not grow by the ~95 GiB sidecar size.

- [ ] **Step 3: Fallback checks**

- Same command with `--ple` pointed at a nonexistent path: expect a `could not be loaded ... falling back` warning, then either the embedded-table path (if the GGUF has one) or the no-PLE warning, and the server still starts and answers a prompt.
- Same command against a GGUF that embeds its own PLE table and no `--ple` flag: expect the existing `--ple-stream` behavior, unchanged from the prior plan.
- Same command against a GGUF with no PLE layers at all (`hparams.ple_n_heads == 0`) plus `--ple <path>`: expect the "ignoring it" warning and a normal load.
- Same command against a GGUF that still embeds `per_layer_token_embd.weight` (not stripped) plus a working `--ple <sidecar>`: expect the log to show "PLE n-gram table loaded from sidecar", the load to succeed (no "wrong number of tensors" error), and decoding to work normally. This is the one code path added by this plan's final-review fix round with no automated test coverage (see final-review-fix-report.md) - it must be checked by hand here.
- Same command with `--ple` pointed at a sidecar with fewer rows than this model's PLE head ranges require (e.g. a sidecar built for a different/smaller qwen4exp variant): expect a "PLE sidecar has fewer rows than the model's PLE head ranges require" warning at load time, then fallback per the normal rules (embedded table if present, otherwise a clean no-PLE warning) - NOT a crash or abort during decode. This guards against the final-review-caught GGML_ABORT bug (see final-review-fix-report.md); if this instead aborts, the fix from that report did not land correctly and must be re-checked before shipping.

- [ ] **Step 4: Correctness vs. the reference**

Send the same greedy prompt (temperature 0) to the sidecar-streamed server and to ds4-dfm running
the same artifact. Exact byte-for-byte identity is not expected (different engines, different
attention/quant kernels elsewhere in the graph), but the PLE-gated behavior should be directionally
consistent - sanity-check with `LLAMA_PLE_HOST_GATHER` stats logging that hit/miss counts look
reasonable and outputs are coherent.

- [ ] **Step 5: Commit any comment/doc fixes**

If any expectation in Steps 2-4 was wrong, fix the stale comment/log text and commit:

```bash
git commit -m "qwen4exp: document the --ple sidecar path"
```

---

## Self-Review

**1. Spec coverage:**
- Sidecar format matching ds4-dfm's, exposed via `--ple <path>` -> Task 1 (manifest parser matching the confirmed real schema) + Task 3 Step 5 (flag). Covered.
- Multi-file row addressing reusing the existing cache/O_DIRECT machinery -> Task 2 (segment-list generalization of `llama_ple_stream`, single code path for both constructors). Covered.
- Locked to qwen4exp, no generic abstraction -> both new modules (`llama_ple_sidecar_*`, the new `llama_ple_stream` ctor) are qwen4exp-specific files/APIs, not registered anywhere generic. Covered.
- Resolution order sidecar -> embedded -> none, with warnings, not crashes -> Task 3 Step 9. Covered.
- `hparams.ple_n_heads == 0` plus `--ple` given -> warn and ignore -> Task 3 Step 9 (the `else if` branch). Covered.
- Graceful "continue without PLE" at the graph level -> Task 3 Step 10 (`ple_table_available` guard on the `build_ple` call site). Covered.
- `--ple` implies `--ple-stream` -> Task 3 Step 5 (`params.ple_stream = true;` in the new arg's callback). Covered.
- Do not trust checksums, do validate structure -> Task 1's `llama_ple_sidecar_load` (gap/overlap and file-size checks, no `sha256` read). Covered.

**2. Placeholder scan:** no TBD/TODO; every step has concrete code; Task 4's steps are manual verification with explicit expected log lines/behaviors, not implementation placeholders.

**3. Type consistency:** `llama_ple_sidecar_manifest`/`llama_ple_sidecar_file`/`llama_ple_sidecar_segment` are defined in Task 1 and consumed with the same field names in Task 2's constructor and Task 3's `load_arch_tensors` block. `llama_ple_stream`'s two constructors and `segment` struct are defined in Task 2 and used identically (single- and multi-file) in Task 3. `common_params::ple_path` (std::string) and `llama_model_params::ple_path` (const char*) match the mapping written in Task 3 Step 4. `ple_table_available` is declared in Task 3 Step 8 and read in Steps 9-10 with the same name. No mismatches found.

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-08-30-ple-sidecar-loading.md`. Two execution options:

**1. Subagent-Driven (recommended)** - I dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** - Execute tasks in this session using executing-plans, batch execution with checkpoints

Which approach?
