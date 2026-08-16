#pragma once

#include "common.cuh"

// DeepSeek V4 CSA sparse-attention prefill: see
// docs/superpowers/specs/2026-08-15-dsv4-sparse-attn-prefill-design.md

bool ggml_cuda_flash_attn_ext_top_k_supported(const ggml_tensor * dst);

void ggml_cuda_flash_attn_ext_top_k(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
