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
