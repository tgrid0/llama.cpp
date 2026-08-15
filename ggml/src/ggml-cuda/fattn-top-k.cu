#include "common.cuh"
#include "convert.cuh"
#include "fattn-common.cuh"
#include "fattn-top-k.cuh"

// DeepSeek V4 CSA sparse-attention prefill kernel: attends only to the n_kv_raw dense/recent
// keys plus the n_top_k keys selected by the lightning indexer, instead of the whole KV range.
// Shape is fixed by the architecture: head_dim 512, 64 query heads sharing one KV head (MQA),
// K and V are the same latent tensor. See the design doc referenced above.

#define DSV4_HEAD_SIZE       512
#define DSV4_HEADS_PER_BLOCK 8
#define DSV4_KEYS_PER_TILE   16

static __global__ void flash_attn_ext_top_k(
        const float * __restrict__ q,
        const half  * __restrict__ k,
        const half  * __restrict__ mask,
        const float * __restrict__ sinks,
        const int   * __restrict__ top_k,
        float       * __restrict__ dst,
        const float scale,
        const int32_t n_kv, const int32_t n_kv_raw, const int32_t n_top_k,
        const int32_t nbq1, const int32_t nbq2, const int32_t nbq3,
        const int32_t nbk1, const int32_t nbk3,
        const int32_t nbm1, const int32_t nbm3,
        const int32_t nbt1, const int32_t nbt3,
        const int32_t nb1,  const int32_t nb2,  const int32_t nb3) {
    constexpr int regs_per_lane = DSV4_HEAD_SIZE / WARP_SIZE;

    const int tid     = threadIdx.x;
    const int lane     = tid % WARP_SIZE;
    const int warp_id  = tid / WARP_SIZE;
    const int head    = blockIdx.y * DSV4_HEADS_PER_BLOCK + warp_id;
    const int token   = blockIdx.x;
    const int stream  = blockIdx.z;

    float accum[regs_per_lane];
#pragma unroll
    for (int i = 0; i < regs_per_lane; ++i) {
        accum[i] = 0.0f;
    }

    float row_max = -FLT_MAX;
    float row_sum = 0.0f;

    const float * q_row    = q    + (int64_t) stream*nbq3 + (int64_t) head*nbq2 + (int64_t) token*nbq1;
    const half  * mask_row = mask + (int64_t) stream*nbm3 + (int64_t) token*nbm1;
    const int   * top_row  = top_k + (int64_t) stream*nbt3 + (int64_t) token*nbt1;

    __shared__ half key_sh[DSV4_KEYS_PER_TILE * DSV4_HEAD_SIZE];
    __shared__ int  key_idx[DSV4_KEYS_PER_TILE];

    const int total_keys = n_kv_raw + n_top_k;

    for (int kb = 0; kb < total_keys; kb += DSV4_KEYS_PER_TILE) {
        if (tid < DSV4_KEYS_PER_TILE) {
            const int selected = kb + tid;
            int key = n_kv; // out-of-range sentinel, skipped below
            if (selected < n_kv_raw) {
                key = selected;
            } else if (selected < total_keys) {
                const int compressed = top_row[selected - n_kv_raw];
                if (compressed >= 0 && compressed < n_kv - n_kv_raw) {
                    key = n_kv_raw + compressed;
                }
            }
            key_idx[tid] = key;
        }
        __syncthreads();

        for (int idx = tid; idx < DSV4_KEYS_PER_TILE * DSV4_HEAD_SIZE; idx += blockDim.x) {
            const int col = idx / DSV4_HEAD_SIZE;
            const int dim = idx % DSV4_HEAD_SIZE;
            const int key = key_idx[col];
            key_sh[idx] = key < n_kv ? k[(int64_t) key*nbk1 + dim] : __float2half(0.0f);
        }
        __syncthreads();

        for (int col = 0; col < DSV4_KEYS_PER_TILE; ++col) {
            const int selected = kb + col;
            const int key = key_idx[col];
            if (selected >= total_keys || key >= n_kv) {
                continue;
            }

            float partial = 0.0f;
#pragma unroll
            for (int i = 0; i < regs_per_lane; ++i) {
                const int dim = lane + i*WARP_SIZE;
                partial += q_row[dim] * __half2float(key_sh[col*DSV4_HEAD_SIZE + dim]);
            }
            const float mask_val = __half2float(mask_row[key]);
            const float score = warp_reduce_sum<WARP_SIZE>(partial)*scale + mask_val;
            if (mask_val < -65000.0f) {
                continue;
            }

            const float new_max   = fmaxf(row_max, score);
            const float old_scale = row_sum == 0.0f ? 0.0f : expf(row_max - new_max);
            const float val_scale = expf(score - new_max);
            row_sum = row_sum*old_scale + val_scale;
            row_max = new_max;

#pragma unroll
            for (int i = 0; i < regs_per_lane; ++i) {
                const int dim = lane + i*WARP_SIZE;
                accum[i] = accum[i]*old_scale + val_scale*__half2float(key_sh[col*DSV4_HEAD_SIZE + dim]);
            }
        }
        __syncthreads();
    }

    if (sinks != nullptr) {
        const float sink      = sinks[head];
        const float new_max   = fmaxf(row_max, sink);
        const float old_scale = row_sum == 0.0f ? 0.0f : expf(row_max - new_max);
        row_sum = row_sum*old_scale + expf(sink - new_max);
#pragma unroll
        for (int i = 0; i < regs_per_lane; ++i) {
            accum[i] *= old_scale;
        }
    }

    const float inv_sum = row_sum == 0.0f ? 0.0f : 1.0f/row_sum;
    float * dst_row = dst + (int64_t) stream*nb3 + (int64_t) token*nb2 + (int64_t) head*nb1;
#pragma unroll
    for (int i = 0; i < regs_per_lane; ++i) {
        dst_row[lane + i*WARP_SIZE] = accum[i]*inv_sum;
    }
}

bool ggml_cuda_flash_attn_ext_top_k_supported(const ggml_tensor * dst) {
    const ggml_tensor * Q     = dst->src[0];
    const ggml_tensor * K     = dst->src[1];
    const ggml_tensor * V     = dst->src[2];
    const ggml_tensor * mask  = dst->src[3];
    const ggml_tensor * top_k = dst->src[5];

    if (!top_k) {
        return false;
    }

    float max_bias = 0.0f;
    float logit_softcap = 0.0f;
    memcpy(&max_bias,      (const float *) dst->op_params + 1, sizeof(float));
    memcpy(&logit_softcap, (const float *) dst->op_params + 2, sizeof(float));
    if (max_bias != 0.0f || logit_softcap != 0.0f) {
        return false;
    }

    if (Q->type != GGML_TYPE_F32 || top_k->type != GGML_TYPE_I32 ||
        !mask || mask->type != GGML_TYPE_F16 ||
        Q->ne[0] != DSV4_HEAD_SIZE || K->ne[0] != DSV4_HEAD_SIZE || V->ne[0] != DSV4_HEAD_SIZE ||
        Q->ne[1] < 64 || Q->ne[2] != 64 || K->ne[2] != 1 || V->ne[2] != 1 ||
        K->data != V->data || K->ne[1] != V->ne[1] ||
        !ggml_is_contiguous(mask) || !ggml_is_contiguous(top_k)) {
        return false;
    }

    const int32_t n_kv_raw = ggml_get_op_params_i32(dst, 4);
    if (n_kv_raw < 0 || n_kv_raw > K->ne[1] || top_k->ne[0] > K->ne[1] - n_kv_raw) {
        return false;
    }

    // Only worth dispatching once the KV cache is well past the active window - below that,
    // the dense path is already cheap and the gather overhead isn't paid off.
    const int64_t n_kv_active = n_kv_raw + top_k->ne[0];
    if (K->ne[1] < 3*n_kv_active) {
        return false;
    }

    return true;
}

void ggml_cuda_flash_attn_ext_top_k(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * Q     = dst->src[0];
    const ggml_tensor * K     = dst->src[1];
    const ggml_tensor * mask  = dst->src[3];
    const ggml_tensor * sinks = dst->src[4];
    const ggml_tensor * top_k = dst->src[5];

    GGML_ASSERT(Q->type == GGML_TYPE_F32);
    GGML_ASSERT(mask && mask->type == GGML_TYPE_F16);
    GGML_ASSERT(top_k && top_k->type == GGML_TYPE_I32);

    const int32_t n_kv_raw = ggml_get_op_params_i32(dst, 4);

    cudaStream_t stream = ctx.stream();

    const char * K_data = (const char *) K->data;
    size_t nb11 = K->nb[1];
    size_t nb13 = K->nb[3];

    if (K->type != GGML_TYPE_F16) {
        const ggml_cuda_flash_attn_ext_f16_extra_data f16_extra =
            ggml_cuda_flash_attn_ext_get_f16_extra_data(dst, /*need_f16_K=*/true, /*need_f16_V=*/false);
        GGML_ASSERT(f16_extra.K != 0);
        GGML_ASSERT(ggml_is_contiguously_allocated(K));

        const size_t bs = ggml_blck_size(K->type);
        const size_t ts = ggml_type_size(K->type);

        to_fp16_cuda_t to_fp16 = ggml_get_to_fp16_cuda(K->type);
        to_fp16(K_data, (half *) f16_extra.K, ggml_nelements(K), stream);

        nb11 = nb11*bs*sizeof(half)/ts;
        nb13 = nb13*bs*sizeof(half)/ts;
        K_data = (const char *) f16_extra.K;
    }

    float scale = 0.0f;
    memcpy(&scale, (const float *) dst->op_params + 0, sizeof(float));

    const dim3 block((unsigned) (WARP_SIZE * DSV4_HEADS_PER_BLOCK));
    const dim3 grid((unsigned) Q->ne[1], (unsigned) (Q->ne[2] / DSV4_HEADS_PER_BLOCK), (unsigned) Q->ne[3]);

    flash_attn_ext_top_k<<<grid, block, 0, stream>>>(
        (const float *) Q->data, (const half *) K_data, (const half *) mask->data,
        sinks ? (const float *) sinks->data : nullptr, (const int *) top_k->data,
        (float *) dst->data, scale,
        (int32_t) K->ne[1], n_kv_raw, (int32_t) top_k->ne[0],
        (int32_t) (Q->nb[1]/sizeof(float)), (int32_t) (Q->nb[2]/sizeof(float)), (int32_t) (Q->nb[3]/sizeof(float)),
        (int32_t) (nb11/sizeof(half)), (int32_t) (nb13/sizeof(half)),
        (int32_t) (mask->nb[1]/sizeof(half)), (int32_t) (mask->nb[3]/sizeof(half)),
        (int32_t) (top_k->nb[1]/sizeof(int32_t)), (int32_t) (top_k->nb[3]/sizeof(int32_t)),
        (int32_t) (dst->nb[1]/sizeof(float)), (int32_t) (dst->nb[2]/sizeof(float)), (int32_t) (dst->nb[3]/sizeof(float)));
}
