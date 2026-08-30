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
