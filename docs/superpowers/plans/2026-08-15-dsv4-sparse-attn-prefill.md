# DeepSeek-V4 Sparse-Attention Prefill Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the dense masked flash-attention fallback in DeepSeek-V4's compressed sparse-attention (CSA) prefill path with a real sparse kernel that only computes attention over the lightning indexer's selected keys, so prefill throughput stops degrading with context depth on this box's ROCm/HIP backend.

**Architecture:** An additive, optional hint (`ggml_flash_attn_ext_add_top_k`) is added to the existing `GGML_OP_FLASH_ATTN_EXT` op (mirrors the existing `ggml_flash_attn_ext_add_sinks` pattern, using the previously-unused `src[5]` slot). `build_attn_mha` and `build_csa_lid_attention` are extended to set this hint, while keeping the existing dense mask as the correctness fallback. A new HIP/CUDA kernel implements a fast path for the hint at DeepSeek V4's fixed CSA shape (head_dim 512, 64-head MQA, K==V single latent, batched prefill queries); it reuses the existing F16-dequant-once scratch-buffer mechanism (`ggml_cuda_flash_attn_ext_get_f16_extra_data`, already used by `launch_fattn`) so it works under this box's `--cache-type-k q8_0` setting without hand-written quant math. Every other backend/shape/model is byte-for-byte unaffected: the hint is inert unless a narrow gate matches.

**Tech Stack:** C/C++, ggml (CUDA/HIP shared backend, compiled for both via the existing hipify path), llama.cpp model-graph code. HIP/ROCm compilation and GPU execution require the actual Strix Halo box (gfx1151, ROCm 7.14) - this plan's dev/edit environment (Windows + WSL Ubuntu 24.04) has no ROCm toolchain, so Tasks 4-6's build/run steps must be executed on that box.

**Spec:** `docs/superpowers/specs/2026-08-15-dsv4-sparse-attn-prefill-design.md`

## Global Constraints

- Every change outside the new kernel file must be purely additive: existing callers of `build_attn_mha`, `ggml_flash_attn_ext`, and every other backend's FLASH_ATTN_EXT handling must see zero behavior change. Default parameter values must preserve current behavior when omitted.
- The new kernel only ever activates for the exact DeepSeek V4 CSA shape (head_dim 512, 64 query heads, single KV head, K and V sharing one buffer, batched queries `ne[1] >= 64`) on devices with 32-lane warps (gfx1151/RDNA and CUDA; CDNA/gfx8-9's 64-lane warps are explicitly out of scope - the gate falls through to the existing dense path there, no crash, no regression).
- No unicode characters in code or comments (`-`, `->`, `x` instead of em-dash, arrow, multiplication sign per AGENTS.md).
- Comments stay to 1-2 lines, explain non-obvious invariants only, no restating what the code says.
- Never commit or push without explicit user approval for each commit (this plan stages commits per task; confirm with the user before any `git push`).

---

### Task 1: ggml core hint API

**Files:**
- Modify: `ggml/include/ggml.h:2436-2438` (right after `ggml_flash_attn_ext_add_sinks`'s declaration)
- Modify: `ggml/src/ggml.c:5466-5480` (right after `ggml_flash_attn_ext_add_sinks`'s definition)
- Test: `tests/test-backend-ops.cpp` (extended in Task 3; this task only needs the API to compile and link)

**Interfaces:**
- Produces: `void ggml_flash_attn_ext_add_top_k(struct ggml_tensor * a, struct ggml_tensor * top_k, int64_t n_kv_raw)` - sets `a->src[5] = top_k` and stores `n_kv_raw` via `ggml_set_op_params_i32(a, 4, ...)`. Consumed by Task 2's `build_attn_mha` and Task 4/5's kernel dispatch.

- [ ] **Step 1: Add the declaration to `ggml.h`**

In `ggml/include/ggml.h`, immediately after the existing `ggml_flash_attn_ext_add_sinks` declaration (around line 2438), add:

```c
    // sparse attention hint: attend only to the first n_kv_raw keys (dense prefix) plus the
    // keys selected by top_k. top_k is I32 [n_top_k, n_tokens, 1, n_streams]; each index i
    // selects absolute key n_kv_raw + i. Negative or out-of-range indices are ignored.
    // Backends may ignore the hint: the kq_mask must still encode the same selection, so a
    // dense fallback computes the identical result.
    GGML_API void ggml_flash_attn_ext_add_top_k(
            struct ggml_tensor * a,
            struct ggml_tensor * top_k,
                     int64_t     n_kv_raw);
```

- [ ] **Step 2: Add the definition to `ggml.c`**

In `ggml/src/ggml.c`, immediately after the existing `ggml_flash_attn_ext_add_sinks` definition (around line 5480), add:

```c
void ggml_flash_attn_ext_add_top_k(
        struct ggml_tensor * a,
        struct ggml_tensor * top_k,
                 int64_t     n_kv_raw) {
    GGML_ASSERT(a->op == GGML_OP_FLASH_ATTN_EXT);
    GGML_ASSERT(a->src[5] == NULL);
    GGML_ASSERT(top_k->type == GGML_TYPE_I32);
    GGML_ASSERT(top_k->ne[1] == a->src[0]->ne[1]);
    GGML_ASSERT(n_kv_raw >= 0 && n_kv_raw <= a->src[1]->ne[1]);

    a->src[5] = top_k;
    ggml_set_op_params_i32(a, 4, (int32_t) n_kv_raw);
}
```

- [ ] **Step 3: Build ggml only, verify it compiles**

Run (in WSL, or wherever this branch is normally built for a quick CPU-only sanity compile):
```bash
cmake --build build --config Release -j 8 --target ggml
```
Expected: builds cleanly, no errors referencing `ggml_flash_attn_ext_add_top_k`.

- [ ] **Step 4: Commit**

```bash
git add ggml/include/ggml.h ggml/src/ggml.c
git commit -m "$(cat <<'EOF'
ggml: add sparse-attention top_k hint to flash_attn_ext

Additive-only: sets the previously-unused src[5] slot on a
FLASH_ATTN_EXT node plus an op_param for n_kv_raw, mirroring the
existing add_sinks pattern. No backend reads it yet, so this has no
runtime effect until a consumer is wired up.
EOF
)"
```

---

### Task 2: Wire the hint through build_attn_mha into DeepSeek V4's graph

**Files:**
- Modify: `src/llama-graph.h` (near `build_attn_mha`'s declaration, currently ending at what will become line ~1079 after Task 1's unrelated line-number drift; search for `build_attn_mha` instead of using a fixed line number)
- Modify: `src/llama-graph.cpp:2395-2445` (`build_attn_mha`'s definition, the `use_flash_attn` branch)
- Modify: `src/models/deepseek4.cpp:734-793` (`build_csa_lid_attention`)

**Interfaces:**
- Consumes: `ggml_flash_attn_ext_add_top_k` from Task 1.
- Produces: `build_attn_mha(..., ggml_tensor * top_k = nullptr, int64_t n_kv_raw = 0)` - two new optional trailing params, default-preserving for every existing call site. Consumed by `build_csa_lid_attention` in this task, and indirectly exercised by Task 3's test and Task 4/5's kernel.

- [ ] **Step 1: Extend `build_attn_mha`'s declaration in `src/llama-graph.h`**

Find the existing declaration (it currently ends with `float kq_scale, int il) const;`) and add two optional trailing parameters:

```cpp
    ggml_tensor * build_attn_mha(
             ggml_tensor * q,
             ggml_tensor * k,
             ggml_tensor * v,
             ggml_tensor * kq_b,
             ggml_tensor * kq_mask,
             ggml_tensor * sinks,
             ggml_tensor * v_mla,
                    float   kq_scale,
                      int   il,
             ggml_tensor * top_k = nullptr,
                 int64_t   n_kv_raw = 0) const;
```

- [ ] **Step 2: Extend `build_attn_mha`'s definition in `src/llama-graph.cpp`**

Find the function definition (`ggml_tensor * llm_graph_context::build_attn_mha(...)` around line 2395). Add the same two trailing parameters to the signature, and inside the `use_flash_attn` branch, right after the existing `ggml_flash_attn_ext_add_sinks(cur, sinks);` call, add:

```cpp
        ggml_flash_attn_ext_add_sinks(cur, sinks);
        if (top_k) {
            ggml_flash_attn_ext_add_top_k(cur, top_k, n_kv_raw);
        }
        ggml_flash_attn_ext_set_prec (cur, GGML_PREC_F32);
```

(The `ggml_flash_attn_ext_set_prec` line already exists immediately after `add_sinks` - just insert the new `if (top_k)` block between them.)

Update the function's parameter list at its definition site to match the header:

```cpp
ggml_tensor * llm_graph_context::build_attn_mha(
         ggml_tensor * q,
         ggml_tensor * k,
         ggml_tensor * v,
         ggml_tensor * kq_b,
         ggml_tensor * kq_mask,
         ggml_tensor * sinks,
         ggml_tensor * v_mla,
                float   kq_scale,
                  int   il,
         ggml_tensor * top_k,
             int64_t   n_kv_raw) const {
```

- [ ] **Step 3: Wire it into `build_csa_lid_attention` in `src/models/deepseek4.cpp`**

Find the existing call (around line 786):

```cpp
    ggml_tensor * out = build_attn_mha(q, k_all, k_all, nullptr, kq_mask, sinks, nullptr, kq_scale, il);
```

Change it to pass the top_k tensor already computed earlier in the same function (from `build_lid_top_k`, stored in the local `top_k` variable) plus the raw-key count:

```cpp
    ggml_tensor * out = build_attn_mha(q, k_all, k_all, nullptr, kq_mask, sinks, nullptr, kq_scale, il, top_k, raw_k->ne[2]);
```

`raw_k` is already in scope (defined a few lines earlier as `mctx_raw->get_k(ctx0, il)`), and `top_k` is the tensor returned by the `build_lid_top_k` call at the top of this function. No other lines in `build_csa_lid_attention` change - `build_top_k_mask` and the dense-mask construction stay exactly as they are today, since they remain the correctness fallback for every backend that doesn't implement the fast path.

- [ ] **Step 4: Build and verify no other call sites broke**

```bash
cmake --build build --config Release -j 8 --target llama
```
Expected: builds cleanly. Since the two new parameters are optional/defaulted, every other `build_attn_mha` call site in the codebase (grep `build_attn_mha(` to confirm none pass a 10th/11th positional argument accidentally) continues to compile unchanged.

- [ ] **Step 5: Manual verification that the hint reaches the graph**

There is no automated unit-test harness for full model-graph construction in this codebase (the `deepseek4` architecture's graph-building logic is only exercised by loading a real model). Verify manually:

1. Load the antirez DeepSeek-V4-Flash GGUF with `GGML_SCHED_DEBUG=2` set (as used in the earlier decode-perf investigation, see `docs/superpowers/specs/2026-08-10-dsv4-strix-halo-perf-investigation.md`), at a context depth past 1024 tokens so the lightning-indexer top-k path is definitely active.
2. Confirm the run completes and produces output textually identical to a pre-change build at the same prompt/seed (the hint is inert until Task 4/5's kernel exists, so behavior must be unchanged at this point - this step only confirms the new `ggml_flash_attn_ext_add_top_k` call doesn't trip any `GGML_ASSERT` during real graph construction, e.g. the `top_k->ne[1] == a->src[0]->ne[1]` assert).

- [ ] **Step 6: Commit**

```bash
git add src/llama-graph.h src/llama-graph.cpp src/models/deepseek4.cpp
git commit -m "$(cat <<'EOF'
llama: wire sparse-attention hint through build_attn_mha into DSV4 CSA

build_attn_mha gains optional top_k/n_kv_raw params that forward to
ggml_flash_attn_ext_add_top_k when set. build_csa_lid_attention now
passes its already-computed lightning-indexer top_k selection through.
The dense mask fallback (build_top_k_mask) is untouched, so this has
no effect on any backend until a kernel consumes the hint.
EOF
)"
```

---

### Task 3: Correctness-gate test case in test-backend-ops

**Files:**
- Modify: `tests/test-backend-ops.cpp:6812-6932` (`test_flash_attn_ext` struct)
- Modify: `tests/test-backend-ops.cpp` near line 9616 (test case registration, alongside the other large-shape FA cases)

**Interfaces:**
- Consumes: `ggml_flash_attn_ext_add_top_k` (Task 1), the DSV4 CSA shape assumptions from the design doc (head_dim 512, 64 heads, K==V).
- Produces: two registered `test_flash_attn_ext` cases (F16 and Q8_0 K) that exercise the top_k-hinted path. These are what Task 5's build must pass on the `ROCm0` backend.

- [ ] **Step 1: Add `top_k`/`n_kv_raw`/`n_top_k` fields to `test_flash_attn_ext`, and a `VARS_TO_STR17` macro**

The existing `VARS_TO_STR*` family (near line 429-435) tops out at `VARS_TO_STR16`. Add one more, immediately after the existing `VARS_TO_STR16` definition, following its exact pattern:

```cpp
#define VARS_TO_STR17(a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q) VAR_TO_STR(a) + "," + VARS_TO_STR16(b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q)
```

In `tests/test-backend-ops.cpp`, extend the `test_flash_attn_ext` struct (around line 6812-6850):

```cpp
    const bool mask; // use mask
    const bool sinks; // use sinks
    const bool top_k; // use the sparse-attention top_k hint (mask must also be true)
    const int64_t n_kv_raw; // dense prefix length when top_k is used
    const int64_t n_top_k; // number of gathered indices when top_k is used (must be <= kv - n_kv_raw)

    const float max_bias; // ALiBi
    const float logit_softcap; // Gemma 2

    const ggml_prec prec;
    const ggml_type type_K;
    const ggml_type type_V;
    std::array<int32_t, 4> permute;

    std::string vars() override {
        return VARS_TO_STR17(hsk, hsv, nh, nr23, kv, nb, mask, sinks, top_k, n_kv_raw, n_top_k, max_bias, logit_softcap, prec, type_K, type_V, permute);
    }
```

Update the constructor to accept the three new parameters with backward-compatible defaults, inserted right after `sinks`:

```cpp
    test_flash_attn_ext(int64_t hsk = 128, int64_t hsv = 128, int64_t nh = 32, std::array<int64_t, 2> nr23 = {1, 1}, int64_t kv = 96, int64_t nb = 8,
                        bool mask = true, bool sinks = false, bool top_k = false, int64_t n_kv_raw = 0, int64_t n_top_k = 0,
                        float max_bias = 0.0f, float logit_softcap = 0.0f, ggml_prec prec = GGML_PREC_F32,
                        ggml_type type_K = GGML_TYPE_F16, ggml_type type_V = GGML_TYPE_F16, std::array<int32_t, 4> permute = {0, 1, 2, 3})
        : hsk(hsk), hsv(hsv), nh(nh), nr23(nr23), kv(kv), nb(nb), mask(mask), sinks(sinks), top_k(top_k), n_kv_raw(n_kv_raw), n_top_k(n_top_k), max_bias(max_bias), logit_softcap(logit_softcap), prec(prec),
          type_K(type_K), type_V(type_V), permute(permute) {
        // the hint requires the mask fallback to still encode the selection, and a valid,
        // genuinely sparse (non-empty, in-range) index count
        GGML_ASSERT(!top_k || (mask && n_top_k > 0 && n_kv_raw >= 0 && n_kv_raw + n_top_k <= kv));
    }
```

**Note on `nh`/`nr23` for MQA shapes (relevant to Step 4 below):** in `build_graph`, `q`'s head count is `nh*nr23[0]` while `k`/`v`'s head count is `nh` directly - so the query:KV head ratio (GQA/MQA ratio) is controlled by `nr23[0]`, not `nh`. To get DeepSeek V4's 64 query heads over a single KV head, use `nh=1, nr23={64, 1}` (q heads = 1*64 = 64, k/v heads = 1) - not `nh=64, nr23={1,1}` (which would build 64 separate KV heads, i.e. plain MHA, and fail the kernel's `K->ne[2] == 1` gate).

- [ ] **Step 2: Alias V onto K when top_k is used, build the top_k tensor, and call the hint in `build_graph`**

DeepSeek V4's real CSA attention passes the same tensor as both K and V (`build_csa_lid_attention` calls `build_attn_mha(q, k_all, k_all, ...)`), and the kernel's dispatch gate (Task 4) requires `K->data == V->data`. The existing `v` construction in `build_graph` (around line 6881-6893) only aliases V onto K for one specific hardcoded shape (`hsk_padded == 576 && hsv_padded == 512`, an unrelated MLA case). Extend that condition to also alias when `top_k` is set, rather than broadening the hardcoded shape check (which would silently change already-registered, unrelated test cases at hsk=hsv=512):

```cpp
        ggml_tensor * v = nullptr;
        if (top_k || (type_K == type_V && hsk_padded == 576 && hsv_padded == 512)) {
            // in this branch, the V cache is sub-view of the K cache. this is used by some
            // MLA-based models, and required whenever the sparse-attention top_k hint is used
            // since DeepSeek V4's CSA attention passes the same tensor as both K and V.
            v = ggml_view_4d(ctx, k, hsv_padded, kv, nh, nr23[1], k->nb[1], k->nb[2], k->nb[3], 0);
        } else {
            v = create_permuted(type_V,        hsv_padded, kv, nh,         nr23[1], true); // the V tensor is usually a view of the V cache
        }
```

(Only the `if` condition changes - the two branch bodies are unchanged from the existing code.)

Right after the existing `sinks` tensor construction and before `ggml_flash_attn_ext`, add:

```cpp
        ggml_tensor * tk = nullptr;
        if (top_k) {
            tk = ggml_new_tensor_4d(ctx, GGML_TYPE_I32, n_top_k, nb, 1, nr23[1]);
            ggml_set_name(tk, "top_k");
        }
```

Then change the existing:

```cpp
        ggml_tensor * out = ggml_flash_attn_ext(ctx, q, k, v, m, 1.0f/sqrtf(hsk), max_bias, logit_softcap);
        ggml_flash_attn_ext_add_sinks(out, s);
```

to:

```cpp
        ggml_tensor * out = ggml_flash_attn_ext(ctx, q, k, v, m, 1.0f/sqrtf(hsk), max_bias, logit_softcap);
        ggml_flash_attn_ext_add_sinks(out, s);
        if (tk) {
            ggml_flash_attn_ext_add_top_k(out, tk, n_kv_raw);
        }
```

- [ ] **Step 3: Fill the top_k tensor with valid indices in `initialize_tensors`**

In the same struct's `initialize_tensors` override (around line 6916-6927), add a branch for the new tensor name, following the same random-valid-index pattern used elsewhere in this file for `GGML_OP_GET_ROWS` (around line 7541-7550):

```cpp
    void initialize_tensors(ggml_context * ctx) override {
        for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != NULL; t = ggml_get_next_tensor(ctx, t)) {
            if (strcmp(t->name, "s") == 0) {
                // make the sink values more noticeable in order to trigger a test failure when the implementation is wrong
                init_tensor_uniform(t, -10.0f, 10.0f);
            } else if (strcmp(t->name, "m") == 0) {
                init_tensor_kq_mask(t);
            } else if (strcmp(t->name, "top_k") == 0) {
                const int64_t n_compressed = kv - n_kv_raw;
                const int64_t nels = ggml_nelements(t);
                std::vector<int32_t> data(nels);
                std::uniform_int_distribution<int32_t> dist(0, (int32_t) n_compressed - 1);
                for (int64_t i = 0; i < nels; i++) {
                    data[i] = dist(rng);
                }
                ggml_backend_tensor_set(t, data.data(), 0, nels * sizeof(int32_t));
            } else {
                init_tensor_uniform(t);
            }
        }
    }
```

- [ ] **Step 4: Register the two DSV4-shaped test cases**

Near line 9616 (alongside the other large-KV FA cases such as `test_cases.emplace_back(new test_flash_attn_ext(256, 256, 4, {6, 1}, kv, 512, ...))`), add:

```cpp
    // DeepSeek V4 CSA sparse-attention prefill: head_dim 512, 64 query heads over a single KV
    // head (nh=1, nr23={64,1} -> q heads = nh*nr23[0] = 64, k/v heads = nh = 1), K==V latent.
    // n_kv_raw=256 dense prefix + n_top_k=2048 gathered, kv=16384 total: comfortably past the
    // fast kernel's 3x-active-window gate (3*(256+2048) = 6912 <= 16384), so this actually
    // dispatches to the sparse kernel on ROCm0/CUDA0, not just the dense CPU fallback.
    for (ggml_type type_KV : {GGML_TYPE_F16, GGML_TYPE_Q8_0}) {
        test_cases.emplace_back(new test_flash_attn_ext(
            512, 512, 1, {64, 1}, 16384, 128, true, false, true, 256, 2048, 0.0f, 0.0f, GGML_PREC_F32, type_KV, type_KV));
    }
```

- [ ] **Step 5: Build test-backend-ops and run the new cases**

```bash
cmake --build build --config Release -j 8 --target test-backend-ops
./build/bin/test-backend-ops test -b CPU0 -o FLASH_ATTN_EXT
```

Expected: PASS for all `FLASH_ATTN_EXT` cases, including the two new `top_k=1` ones. On CPU, `src[5]` is never read, so this confirms the ggml-core API and graph construction are structurally correct (no assert failures, no crash) and that adding the hint does not change output versus the equivalent hint-free case - the dense mask already fully encodes the same selection.

- [ ] **Step 6: Commit**

```bash
git add tests/test-backend-ops.cpp
git commit -m "$(cat <<'EOF'
tests: add FLASH_ATTN_EXT top_k-hint coverage at the DSV4 CSA shape

Registers two test_flash_attn_ext cases (F16 and Q8_0 K) matching
DeepSeek V4's compressed-attention shape with the sparse-attention
hint enabled. Currently exercises only the dense CPU fallback (no
backend implements the hint yet); this becomes the correctness gate
for the HIP/CUDA kernel added next.
EOF
)"
```

---

### Task 4: Sparse FA kernel + CUDA/HIP dispatch wiring

**Files:**
- Create: `ggml/src/ggml-cuda/fattn-top-k.cuh`
- Create: `ggml/src/ggml-cuda/fattn-top-k.cu`
- Modify: `ggml/src/ggml-cuda/fattn.cu` (enum, gate, alloc-size, dispatch switch, include)

**Interfaces:**
- Consumes: `dst->src[5]` (top_k) and `op_params[4]` (n_kv_raw) set by Task 2's graph wiring; `ggml_cuda_flash_attn_ext_get_f16_extra_data` and `ggml_get_to_fp16_cuda` (existing, reused unchanged from `fattn-common.cuh`/`convert.cuh`).
- Produces: `void ggml_cuda_flash_attn_ext_top_k(ggml_backend_cuda_context & ctx, ggml_tensor * dst)` and `bool ggml_cuda_flash_attn_ext_top_k_supported(const ggml_tensor * dst)`, both declared in `fattn-top-k.cuh` and called from `fattn.cu`.

**Design notes carried over from research (not in the original spec doc, discovered while planning):**
- DeepSeek V4's CSA attention uses K and V as literally the same latent tensor (`build_csa_lid_attention` calls `build_attn_mha(q, k_all, k_all, ...)` - both K and V args are `k_all`). The kernel only ever reads one buffer.
- Rather than hand-writing Q8_0 dequant math inside the kernel (higher risk), this reuses the exact dequant-once-to-F16-scratch mechanism `launch_fattn` already uses for other kernels (`ggml_cuda_flash_attn_ext_get_f16_extra_data` + `ggml_get_to_fp16_cuda`). This means the kernel body only ever handles F16 K/V - simpler and lower-risk than a from-scratch quantized kernel, while still working under this box's `--cache-type-k q8_0` setting.
- The kernel is gated to devices with 32-lane warps (`ggml_cuda_info().devices[device].warp_size == 32`) - true for gfx1151/RDNA and all CUDA GPUs, false for CDNA/gfx8-9 (64-lane). CDNA falls through to the existing dense path unchanged; this is an explicit scope reduction versus the Vulkan source (which pinned to 64 lanes), justified because this box's actual hardware is RDNA3.5/32-lane and a 64-lane variant would need separate register-count tuning with no way to test it here.
- **This kernel must be reserved scratch space at allocation time.** `ggml_cuda_flash_attn_ext_get_alloc_size` (in `fattn.cu`) decides `need_f16_K`/`need_f16_V` from the *kernel enum* before any compute runs - missing the new enum case there would silently under-allocate the buffer this kernel dequantizes into, an out-of-bounds write. This is covered explicitly in Step 4 below.

- [ ] **Step 1: Write `ggml/src/ggml-cuda/fattn-top-k.cuh`**

```cpp
#pragma once

#include "common.cuh"

// DeepSeek V4 CSA sparse-attention prefill: see
// docs/superpowers/specs/2026-08-15-dsv4-sparse-attn-prefill-design.md

bool ggml_cuda_flash_attn_ext_top_k_supported(const ggml_tensor * dst);

void ggml_cuda_flash_attn_ext_top_k(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
```

- [ ] **Step 2: Write `ggml/src/ggml-cuda/fattn-top-k.cu`**

```cpp
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
```

- [ ] **Step 3: Add the new enum value and dispatch gate in `fattn.cu`**

Add the include near the top of `ggml/src/ggml-cuda/fattn.cu` (alongside the existing `fattn-*.cuh` includes):

```cpp
#include "fattn-top-k.cuh"
```

Extend the enum (around line 331-336):

```cpp
enum best_fattn_kernel {
    BEST_FATTN_KERNEL_NONE    =   0,
    BEST_FATTN_KERNEL_TILE    = 200,
    BEST_FATTN_KERNEL_VEC     = 100,
    BEST_FATTN_KERNEL_MMA_F16 = 400,
    BEST_FATTN_KERNEL_TOP_K   = 900,
};
```

At the very top of `ggml_cuda_get_best_fattn_kernel`, right after the `#ifndef FLASH_ATTN_AVAILABLE` guard (around line 362) and before any existing shape-dispatch logic, add the early gate:

```cpp
    if (ggml_cuda_flash_attn_ext_top_k_supported(dst) && ggml_cuda_info().devices[device].warp_size == 32) {
        return BEST_FATTN_KERNEL_TOP_K;
    }
```

If the gate doesn't match, execution falls through unchanged into the existing logic below - no other line in this function changes.

- [ ] **Step 4: Add the alloc-size case (critical - prevents an out-of-bounds write)**

In `ggml_cuda_flash_attn_ext_get_alloc_size` (around line 550-562), add a case to the switch so the Q8_0-K dequant scratch buffer actually gets reserved:

```cpp
    switch (kernel) {
        case BEST_FATTN_KERNEL_TILE:
        case BEST_FATTN_KERNEL_MMA_F16:
            need_f16_K = true;
            need_f16_V = true;
            break;
        case BEST_FATTN_KERNEL_VEC:
            need_f16_K = K->type == GGML_TYPE_F32;
            need_f16_V = V->type == GGML_TYPE_F32;
            break;
        case BEST_FATTN_KERNEL_TOP_K:
            need_f16_K = K->type != GGML_TYPE_F16;
            need_f16_V = false; // V shares K's buffer for this op (see design doc)
            break;
        case BEST_FATTN_KERNEL_NONE:
            break;
    }
```

- [ ] **Step 5: Add the dispatch switch case**

In `ggml_cuda_flash_attn_ext` (around line 570-585):

```cpp
void ggml_cuda_flash_attn_ext(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_set_device(ctx.device);
    switch (ggml_cuda_get_best_fattn_kernel(ggml_cuda_get_device(), dst)) {
        case BEST_FATTN_KERNEL_NONE:
            GGML_ABORT("fatal error");
        case BEST_FATTN_KERNEL_TILE:
            ggml_cuda_flash_attn_ext_tile(ctx, dst);
            break;
        case BEST_FATTN_KERNEL_VEC:
            ggml_cuda_flash_attn_ext_vec(ctx, dst);
            break;
        case BEST_FATTN_KERNEL_MMA_F16:
            ggml_cuda_flash_attn_ext_mma_f16(ctx, dst);
            break;
        case BEST_FATTN_KERNEL_TOP_K:
            ggml_cuda_flash_attn_ext_top_k(ctx, dst);
            break;
    }
}
```

- [ ] **Step 6: Add the new source file to the build**

Check `ggml/src/ggml-cuda/CMakeLists.txt` for how existing `fattn-*.cu` files are picked up (likely a glob pattern that already covers this - confirm rather than assume) and add `fattn-top-k.cu` explicitly if the build doesn't glob `.cu` files automatically.

- [ ] **Step 7: Commit**

```bash
git add ggml/src/ggml-cuda/fattn-top-k.cuh ggml/src/ggml-cuda/fattn-top-k.cu ggml/src/ggml-cuda/fattn.cu
git commit -m "$(cat <<'EOF'
ggml-cuda: sparse-attention prefill kernel for DeepSeek V4 CSA

New fattn-top-k.cu implements the FLASH_ATTN_EXT top_k hint at the
fixed DSV4 CSA shape (head_dim 512, 64-head MQA, K==V, 32-lane warps
only). Reuses the existing dequant-once-to-F16-scratch mechanism for
non-F16 K instead of hand-written quant math. Gated narrowly in
fattn.cu's kernel selector; falls through to the existing dense path
for every other shape/backend/device.
EOF
)"
```

---

### Task 5: Build and correctness-validate on the Strix Halo ROCm box

**Files:** none (build/test only - this task produces no code changes unless Task 4's first-draft kernel needs fixes, in which case loop back into `fattn-top-k.cu`/`fattn.cu` before re-committing)

**Interfaces:**
- Consumes: Task 4's kernel and Task 3's registered test cases.
- Produces: a green `test-backend-ops -b ROCm0 -o FLASH_ATTN_EXT` run, the correctness gate the design doc calls for before this touches a real model.

This task must run on the actual Strix Halo box (gfx1151, ROCm 7.14 toolbox) - this plan's editing environment has no ROCm/HIP compiler available. Expect this to be an iterate-until-green loop: first-draft GPU kernel code written without a compiler in the loop commonly has real compile errors (type mismatches, missing casts) and possibly numerical bugs (indexing off-by-ones) on the first attempt. Treat build/test failures here as expected iteration, not a sign the plan is wrong - fix forward in `fattn-top-k.cu`/`fattn.cu` and re-run.

- [ ] **Step 1: Build with the HIP/ROCm backend enabled**

Use this branch's normal ROCm build invocation on the Strix Halo box (the same one used for every prior session on this branch - check recent shell history or the project's build docs if not memorized; do not guess flags). Watch specifically for compile errors in `fattn-top-k.cu` and `fattn.cu`.

- [ ] **Step 2: Run the targeted test-backend-ops cases**

```bash
./build/bin/test-backend-ops test -b ROCm0 -o FLASH_ATTN_EXT
```

Expected: every `FLASH_ATTN_EXT` case passes, including the two `top_k=1` DSV4-shaped cases from Task 3 - these are the ones that now actually exercise `ggml_cuda_flash_attn_ext_top_k` instead of falling through to the dense path, since the gate's shape/warp-size conditions match on this hardware. A failure here (wrong output, not a crash) points to an indexing or online-softmax bug in the kernel; a crash points to a launch-config or memory-safety bug (check Task 4 Step 4's alloc-size wiring first if it's an out-of-bounds/segfault).

- [ ] **Step 3: Confirm the fast path actually dispatches (not just passes by accident)**

Add a one-shot `GGML_LOG_DEBUG` line inside `ggml_cuda_flash_attn_ext_top_k` (temporary, or gated behind existing verbose-logging conventions in this file) printing tensor shapes and `n_kv_raw`/`n_top_k`, or use `GGML_SCHED_DEBUG=2` if the op name is distinguishable there. Confirm it fires during the test run. Per the "Verify engagement from the executed graph, not from the gate" lesson already recorded in the sibling-repo notes (`strix-halo-llamacpp`'s `EXPLORING.md`), don't trust the test passing alone as proof the new code path ran - a gate that silently falls through to the dense path would also pass this test, just without the intended speedup.

- [ ] **Step 4: Fix forward and re-commit if needed**

If Steps 1-3 required changes to `fattn-top-k.cu`, `fattn-top-k.cuh`, or `fattn.cu`, commit those fixes separately from Task 4's original commit (do not amend):

```bash
git add ggml/src/ggml-cuda/fattn-top-k.cuh ggml/src/ggml-cuda/fattn-top-k.cu ggml/src/ggml-cuda/fattn.cu
git commit -m "$(cat <<'EOF'
ggml-cuda: fix sparse-attention top_k kernel issues found on gfx1151

EOF
)"
```
(Fill in the actual issue found - do not use this as a placeholder message; describe what was actually wrong.)

---

### Task 6: Real-model validation

**Files:** none - this is a manual runtime check, matching the design doc's testing section and the same A/B pattern used for the earlier decode-perf fix.

**Interfaces:**
- Consumes: the passing build from Task 5.
- Produces: a confirmation (recorded back into this plan file or a follow-up note, not code) that prefill throughput improves at long context with identical output.

- [ ] **Step 1: Baseline run (fast path effectively disabled)**

Since the fast path only engages once `K->ne[1] >= 3*(n_kv_raw + n_top_k)` (a property of context depth, not a flag), get a "before" comparison point either from a pre-Task-4 build, or by testing at a context depth below that threshold on the new build (confirms the gate correctly falls through at short context too).

- [ ] **Step 2: Long-context A/B on the antirez DeepSeek-V4-Flash GGUF**

Using the launch command from the perf-investigation doc (`--ctx-size` large enough to push well past the 3x active-window threshold, `--cache-type-k q8_0 --cache-type-v q8_0` to match the box's real config and exercise the Q8_0-dequant-once path specifically), run the same long prompt on the pre-Task-4 build and the new build. Confirm:
- Generated text is identical (or logits match within float tolerance) between the two builds - correctness first.
- Prefill t/s improves at long context depth (compare against the ~110/~30/~12 t/s at 0/128k/470k baseline numbers from this conversation).

- [ ] **Step 3: Record the result**

Append a short "Validation results" section to this plan file (or a new dated follow-up doc under `docs/superpowers/specs/` if the findings are substantial) with the actual before/after numbers and confirmation of output parity, then commit:

```bash
git add docs/superpowers/plans/2026-08-15-dsv4-sparse-attn-prefill.md
git commit -m "docs: record DSV4 sparse-attention prefill validation results"
```
