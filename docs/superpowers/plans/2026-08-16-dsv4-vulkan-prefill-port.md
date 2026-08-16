# DeepSeek-V4 Vulkan sparse-attention prefill port Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port the Vulkan lightning-indexer, DSV4 HC pre/comb/post fusion, and sparse top-k
flash-attention kernels from `Nathanw1014/llama.cpp`'s `strix-halo-vulkan-beta3` branch into
our fork, closing the gap between our fork's current ~125 t/s ROCm prefill and the ~290 t/s
beta3 achieves on Vulkan, and root-causing the Vulkan device-lost/OOM crash at large context.

**Architecture:** Three independent Vulkan backend-kernel additions to ops our fork already has
cross-backend (CPU/CUDA/Metal/SYCL) implementations for, followed by an investigation task that
depends on all three landing. Each kernel task is a fresh port of beta3's current file state
(not a per-commit replay), reusing the `ggml_flash_attn_ext_add_top_k` hint API already
committed on the `dsv4-sparse-attn-prefill` worktree.

**Tech Stack:** GLSL compute shaders (Vulkan), C++ (ggml-vulkan.cpp dispatch/pipeline code),
glslc (Windows Vulkan SDK) for shader syntax validation, WSL Ubuntu 24.04 (`ninja-build`,
`glslc`, `libvulkan-dev`, `spirv-headers`, `glslang-tools`, `vulkan-tools` - already installed
this session) for full build+link validation, `test-backend-ops` for correctness regression.

**Spec:** `docs/superpowers/specs/2026-08-16-dsv4-vulkan-prefill-port-design.md`

## Global Constraints

- Branch from the tip of the `dsv4-sparse-attn-prefill` worktree/branch, not from `b10331_zgfs`
  directly - the new Vulkan kernels must consume the same `src[5]`/`op_params[4]` hint the
  already-committed HIP kernel and ggml-core API use.
- Reference source is `W:\projects\llm\nathanw-llamacpp`, branch `strix-halo-vulkan-beta3`
  (already checked out at HEAD). Port beta3's **current file state**, not individual commits.
- Scope is the prefill cluster only: lightning indexer, DSV4 HC pre/comb/post, sparse top-k
  flash-attention (scalar + coopmat2), and the tiling/splitting/multi-sequence work already
  baked into those kernels' current state. The decode small-batch quantized-gather cluster
  (`lightning_indexer_decode_cm.comp`, `flash_attn_top_k`'s decode-batch gather-to-compact
  paths) is explicitly out of scope - do not port it even if encountered while reading beta3's
  source.
- No local Vulkan device is reachable for execution testing, in either Windows (real GPU, but
  no build toolchain wired to it from this session) or WSL (build toolchain present, but only
  Mesa's software `llvmpipe` device, which `ggml-vulkan.cpp` deliberately excludes - confirmed
  via a real build reporting "No devices found"). Every task's testing is: glslc shader syntax
  validation + WSL build/link validation + `test-backend-ops -b CPU -o <OP>` regression (CPU
  backend is unaffected by these changes, so this only catches build breakage, not Vulkan
  functional bugs). Real functional/perf validation on Vulkan happens once, at the end, on the
  user's Strix Halo box - do not claim a task's Vulkan kernel is functionally correct before then.
- Comments stay short (usually 1-2 lines) unless documenting a genuinely non-obvious invariant
  (matches this repo's AGENTS.md guidance, and the prior plan's ruling on this point).
- No unicode in code or comments.

---

## Task 1: Lightning indexer Vulkan kernel

**Files:**
- Create: `ggml/src/ggml-vulkan/vulkan-shaders/lightning_indexer.comp`
- Create: `ggml/src/ggml-vulkan/vulkan-shaders/lightning_indexer_cm.comp`
- Modify: `ggml/src/ggml-vulkan/ggml-vulkan.cpp`
- Modify: `ggml/src/ggml-vulkan/vulkan-shaders/vulkan-shaders-gen.cpp`

**Interfaces:**
- Consumes: `ggml_lightning_indexer(ctx, q, k, weights, mask)` - already exists in
  `ggml/include/ggml.h:2597` and `ggml/src/ggml.c:6313`, unchanged by this task. Produces a
  `GGML_OP_LIGHTNING_INDEXER` node with `src[0..3] = q, k, weights, mask`.
- Produces: a working Vulkan dispatch path for `GGML_OP_LIGHTNING_INDEXER`, so
  `llama_context::resolve_fused_ops` (`src/llama-context.cpp:582-584`) stops falling this op
  back to CPU on Vulkan. No signature or call-site changes anywhere outside `ggml-vulkan.cpp`.

- [ ] **Step 1: Copy the shader files from beta3**

Copy these two files verbatim from the reference repo:

```bash
cp /w/projects/llm/nathanw-llamacpp/ggml/src/ggml-vulkan/vulkan-shaders/lightning_indexer.comp \
   ggml/src/ggml-vulkan/vulkan-shaders/lightning_indexer.comp
cp /w/projects/llm/nathanw-llamacpp/ggml/src/ggml-vulkan/vulkan-shaders/lightning_indexer_cm.comp \
   ggml/src/ggml-vulkan/vulkan-shaders/lightning_indexer_cm.comp
```

Do not modify the constants `K_PER_GROUP`, `N_HEAD`, `TILE`, `HEAD_SIZE`, `N_WAVES`, or
`HEADS_PER_TILE` in either file - `lightning_indexer_cm.comp`'s shared-memory usage is tuned
right at the device limit (comment near the top of the file explains: exceeding it makes the
pipeline silently fail to create, not error). Do not port `lightning_indexer_decode_cm.comp` -
out of scope (decode cluster).

- [ ] **Step 2: Validate shader syntax with glslc**

Run (Windows, this session has the Vulkan SDK):
```bash
"/c/VulkanSDK/1.3.246.0/Bin/glslc.exe" -fshader-stage=compute \
  ggml/src/ggml-vulkan/vulkan-shaders/lightning_indexer.comp -o /tmp/li.spv
"/c/VulkanSDK/1.3.246.0/Bin/glslc.exe" -fshader-stage=compute \
  -DN_WAVES=8 -DHEADS_PER_TILE=4 \
  ggml/src/ggml-vulkan/vulkan-shaders/lightning_indexer_cm.comp -o /tmp/li_cm.spv
```
Expected: both compile cleanly (no errors). If `lightning_indexer_cm.comp` fails because your
glslc build lacks `GL_KHR_cooperative_matrix` support, note this in your report rather than
editing the shader to work around it - the WSL build's own glslc will be the real gate for that
extension (confirmed present this session: "GL_KHR_cooperative_matrix supported by glslc").

- [ ] **Step 3: Add the push-constant struct to ggml-vulkan.cpp**

Find the other `vk_op_*_push_constants` struct definitions in `ggml-vulkan.cpp` (search for
`_push_constants {` to find the neighborhood) and add, following the same pattern:

```cpp
struct vk_op_lightning_indexer_push_constants {
    uint32_t n_kv, n_batch, n_stream, nem3;
    uint32_t nb1, nb3;
    uint32_t nbq1, nbq2, nbq3;
    uint32_t nbk2, nbk3;
    uint32_t nbw1, nbw3;
    uint32_t nbm1, nbm3;
};
static_assert(sizeof(vk_op_lightning_indexer_push_constants) <= 128);
```

Read beta3's `ggml-vulkan.cpp:1804-1812` first to confirm this matches exactly (field order
matters - it must match `lightning_indexer.comp`'s `Parameters` push-constant block byte for
byte).

- [ ] **Step 4: Add pipeline fields and creation**

In the `vk_device` struct, find where `pipeline_rwkv_wkv7` (or a similarly-placed custom-op
pipeline field) is declared and add, immediately after:

```cpp
vk_pipeline pipeline_lightning_indexer_f16;
vk_pipeline pipeline_lightning_indexer_cm_f16;
vk_pipeline pipeline_lightning_indexer_cm_small_f16;
```

Find where that field's pipeline is created (the `ggml_vk_create_pipeline(...)` call for
`pipeline_rwkv_wkv7` or similar) and add pipeline creation mirroring beta3's
`ggml-vulkan.cpp:6150-6168`: the scalar pipeline is created unconditionally (workgroup denom
`{8,1,1}`); the two coopmat variants are gated on
`device->subgroup_arithmetic && device->subgroup_size == 64`, further gated on
`defined(VK_KHR_cooperative_matrix) && defined(GGML_VULKAN_COOPMAT_GLSLC_SUPPORT)` plus a
runtime check (`device->coopmat_support && device->coopmat_support_16x16x16_f32acc &&
device->subgroup_size_control`), and the full-size `_cm_f16` variant additionally requires
`maxComputeWorkGroupInvocations >= 512`, `maxComputeWorkGroupSize[0] >= 512`,
`maxComputeSharedMemorySize >= 64*1024`. Copy beta3's exact condition, don't rederive it -
these thresholds were tuned against real hardware limits.

- [ ] **Step 5: Add the dispatch function**

Add a new static function, placed near other single-op dispatch functions (e.g. near wherever
`ggml_vk_rwkv_wkv7` or `ggml_vk_gated_delta_net` is defined):

```cpp
static void ggml_vk_lightning_indexer(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_tensor * dst) {
    const ggml_tensor * q = dst->src[0];
    const ggml_tensor * k = dst->src[1];
    const ggml_tensor * w = dst->src[2];
    const ggml_tensor * m = dst->src[3];
    vk_pipeline pipeline = ggml_vk_op_get_pipeline(ctx, q, k, w, dst, dst->op);
    GGML_ASSERT(pipeline != nullptr);
    ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);
    // build push constants from q/k/w/m/dst ne and nb fields - read beta3's
    // ggml-vulkan.cpp:14039-14072 for the exact field mapping before writing this
}
```

Read beta3's `ggml-vulkan.cpp:14039-14072` for the exact push-constant field mapping and the
`ggml_vk_dispatch_pipeline(...)` call's buffer/workgroup-count arguments (grid is
`{k->ne[2], q->ne[2], q->ne[3]}`) - copy it precisely rather than re-deriving from the struct
alone, since a transposed field is a silent-wrong-answer bug, not a compile error.

- [ ] **Step 6: Add the pipeline-selection case**

In `ggml_vk_op_get_pipeline`, find `case GGML_OP_GATED_DELTA_NET:` (`ggml-vulkan.cpp:11557`)
and add a new case immediately after that case block ends (before the next `case`). Requires
`q->type==F32, k->type==F16, w->type==F32, dst->type==F32`; picks the cm/cm_small coopmat
pipeline when available and `q->ne[2] >= 16`, else falls back to the scalar pipeline. Read
beta3's `ggml-vulkan.cpp:12917-12926` for the exact selection logic (decode-cm branch is out of
scope - never select it; treat that condition as always false).

- [ ] **Step 7: Add the main-dispatch-switch case**

In the main graph-execution switch, find
`case GGML_OP_GATED_DELTA_NET: ggml_vk_gated_delta_net(ctx, compute_ctx, node); break;`
(`ggml-vulkan.cpp:15626-15628`) and add immediately after:

```cpp
case GGML_OP_LIGHTNING_INDEXER:
    ggml_vk_lightning_indexer(ctx, compute_ctx, node);
    break;
```

- [ ] **Step 8: Add the supports_op case**

In `ggml_backend_vk_device_supports_op`, find `case GGML_OP_GATED_DELTA_NET:`
(`ggml-vulkan.cpp:18388`) and add a new case after its block. Read beta3's full gate at
`ggml-vulkan.cpp:19943` through the next `case` statement (the prior research pass did not
confirm it read the complete block - read it fully yourself before porting). Known conditions:
`subgroup_size==64 && subgroup_arithmetic`; dtype checks `q==F32,k==F16,w==F32,m==F16,dst==F32`;
shape checks `q->ne[0]==128, q->ne[1]==64, k->ne[0]==128, k->ne[1]==1`; broadcast/shape
consistency `w->ne[0]==q->ne[1], w->ne[1]==q->ne[2], m->ne[0]==k->ne[2], m->ne[1]==q->ne[2],
op->ne[0]==k->ne[2], op->ne[1]==q->ne[2]`.

- [ ] **Step 9: Register the shaders**

In `vulkan-shaders-gen.cpp`, find where other shaders are registered via `string_to_spv(...)`
and add, mirroring beta3's `vulkan-shaders-gen.cpp:803-808`:

```cpp
string_to_spv("lightning_indexer_f16", "lightning_indexer.comp", {});
#if defined(GGML_VULKAN_COOPMAT_GLSLC_SUPPORT)
string_to_spv("lightning_indexer_cm_f16", "lightning_indexer_cm.comp", {{"N_WAVES", "8"}, {"HEADS_PER_TILE", "4"}});
string_to_spv("lightning_indexer_cm_small_f16", "lightning_indexer_cm.comp", {{"N_WAVES", "1"}, {"HEADS_PER_TILE", "1"}});
#endif
```
(Note: no CMakeLists.txt edit needed - `vulkan-shaders/CMakeLists.txt:204` globs `*.comp`
automatically; confirmed identical in both forks.)

- [ ] **Step 10: Build and test**

```bash
wsl.exe -e bash -lc "cd /mnt/w/projects/llm/llama.cpp/build && make -j16 test-backend-ops 2>&1 | tail -60"
```
Expected: clean build, no errors.

```bash
wsl.exe -e bash -lc "cd /mnt/w/projects/llm/llama.cpp/build && ./bin/test-backend-ops test -b CPU -o LIGHTNING_INDEXER 2>&1 | tail -20"
```
Expected: all cases still pass (CPU backend unaffected by this change - this is a build-sanity
regression check, not a Vulkan functional test). Also run
`./bin/test-backend-ops test -b Vulkan0 -o LIGHTNING_INDEXER` and confirm it reports "No devices
found" (expected in this environment, not a failure) rather than a crash or assertion failure -
a crash here would indicate a bug in the supports_op gate itself, reachable even without a
device.

- [ ] **Step 11: Commit**

```bash
git add ggml/src/ggml-vulkan/vulkan-shaders/lightning_indexer.comp \
        ggml/src/ggml-vulkan/vulkan-shaders/lightning_indexer_cm.comp \
        ggml/src/ggml-vulkan/ggml-vulkan.cpp \
        ggml/src/ggml-vulkan/vulkan-shaders/vulkan-shaders-gen.cpp
git commit -m "vulkan: port DSV4 lightning indexer kernel from strix-halo-vulkan-beta3"
```

---

## Task 2: DSV4 HC pre/comb/post fusion Vulkan kernels

**Files:**
- Create: `ggml/src/ggml-vulkan/vulkan-shaders/dsv4_hc_pre.comp`
- Create: `ggml/src/ggml-vulkan/vulkan-shaders/dsv4_hc_comb.comp`
- Create: `ggml/src/ggml-vulkan/vulkan-shaders/dsv4_hc_post.comp`
- Modify: `ggml/src/ggml-vulkan/ggml-vulkan.cpp`
- Modify: `ggml/src/ggml-vulkan/vulkan-shaders/vulkan-shaders-gen.cpp`
- Reference (read-only): `ggml/src/ggml-cuda/dsv4-hc.cu`, `ggml/src/ggml-cuda/dsv4-hc.cuh`

**Interfaces:**
- Consumes: `ggml_dsv4_hc_pre/comb/post(...)` - already exist in `ggml/include/ggml.h:2614-2641`
  and `ggml/src/ggml.c:6347-6471`, unchanged by this task. `hc_pre` sets `src[0]=x, src[1]=weights`;
  `hc_comb` sets `src[0]=mixes, src[1]=scale, src[2]=base` plus op_params
  `[0]=eps (f32), [1]=n_iter (i32)`; `hc_post` sets `src[0]=x, src[1]=residual, src[2]=post,
  src[3]=comb`.
- Produces: working Vulkan dispatch for `GGML_OP_DSV4_HC_PRE/COMB/POST`, so
  `llama_context::resolve_fused_ops` (`src/llama-context.cpp:587-591`) stops disabling these
  three fusions on Vulkan.

This task depends on Task 1 landing first (shares insertion anchors in the same three
`ggml-vulkan.cpp` switch statements - add these cases after Task 1's `LIGHTNING_INDEXER` cases,
not after `GATED_DELTA_NET` directly, to avoid a merge conflict within the task).

- [ ] **Step 1: Copy the shader files from beta3**

```bash
cp /w/projects/llm/nathanw-llamacpp/ggml/src/ggml-vulkan/vulkan-shaders/dsv4_hc_pre.comp \
   ggml/src/ggml-vulkan/vulkan-shaders/dsv4_hc_pre.comp
cp /w/projects/llm/nathanw-llamacpp/ggml/src/ggml-vulkan/vulkan-shaders/dsv4_hc_comb.comp \
   ggml/src/ggml-vulkan/vulkan-shaders/dsv4_hc_comb.comp
cp /w/projects/llm/nathanw-llamacpp/ggml/src/ggml-vulkan/vulkan-shaders/dsv4_hc_post.comp \
   ggml/src/ggml-vulkan/vulkan-shaders/dsv4_hc_post.comp
```

These are small, flat elementwise kernels (37/101/44 lines) explicitly commented in beta3 as
ports of `ggml-cuda/dsv4-hc.cu` - read that CUDA file first (it's in our own fork already) to
understand the math each one performs before porting; it is the ground truth for correctness,
the GLSL is just a restatement of it in a different language. None use shared memory, subgroup
ops, or coopmat - no `_cm` variants exist for these three.

- [ ] **Step 2: Validate shader syntax with glslc**

```bash
for f in dsv4_hc_pre dsv4_hc_comb dsv4_hc_post; do
  "/c/VulkanSDK/1.3.246.0/Bin/glslc.exe" -fshader-stage=compute \
    ggml/src/ggml-vulkan/vulkan-shaders/$f.comp -o /tmp/$f.spv
done
```
Expected: all three compile cleanly.

- [ ] **Step 3: Add push-constant structs**

Read beta3's `ggml-vulkan.cpp:1814-1841` for the three `Parameters`-mirroring push-constant
structs (one per shader, each `<=128` bytes per its `static_assert`) and add them verbatim,
adjacent to the struct added in Task 1.

- [ ] **Step 4: Add pipeline fields and creation**

Add three fields (`pipeline_dsv4_hc_pre_f32`, `_comb_f32`, `_post_f32`) alongside Task 1's
fields. Creation is unconditional (no coopmat gating - these are plain elementwise kernels):
read beta3's `ggml-vulkan.cpp:6209-6217` for the exact `ggml_vk_create_pipeline(...)` calls,
workgroup `{256,1,1}`.

- [ ] **Step 5: Add the three dispatch functions**

Add `ggml_vk_dsv4_hc_pre`, `ggml_vk_dsv4_hc_comb`, `ggml_vk_dsv4_hc_post`, each pulling `ne`/`nb`
off the relevant `dst->src[N]` and building push constants. `hc_comb` additionally reads its two
scalar params via `ggml_get_op_params_f32(dst, 0)` and `ggml_get_op_params_i32(dst, 1)` (same
op_params slots `ggml.c:6387-6388` sets - do not swap the index or the type). Read beta3's
`ggml-vulkan.cpp:14077` onward for the exact three function bodies.

- [ ] **Step 6: Add pipeline-selection, main-dispatch, and supports_op cases**

Same three switch statements as Task 1 (`ggml_vk_op_get_pipeline`, the main graph-execution
switch, `ggml_backend_vk_device_supports_op`), adding one case per op (3 cases x 3 switches = 9
new case blocks), inserted after Task 1's `LIGHTNING_INDEXER` cases in each switch.
`supports_op`'s gate (beta3 `ggml-vulkan.cpp:19933-19942`) is a plain F32-type check per src, no
shape/contiguity gate - these ops are generic/contiguous by construction upstream, unlike
Task 1's shape-heavy gate.

- [ ] **Step 7: Register the shaders**

In `vulkan-shaders-gen.cpp`, add three unconditional `string_to_spv(...)` registrations (no
coopmat guard needed), mirroring the pattern from Task 1's Step 9 but without the `#if` block.

- [ ] **Step 8: Build and test**

```bash
wsl.exe -e bash -lc "cd /mnt/w/projects/llm/llama.cpp/build && make -j16 test-backend-ops 2>&1 | tail -60"
wsl.exe -e bash -lc "cd /mnt/w/projects/llm/llama.cpp/build && ./bin/test-backend-ops test -b CPU -o DSV4_HC_PRE -o DSV4_HC_COMB -o DSV4_HC_POST 2>&1 | tail -30"
```
Expected: clean build; all CPU-backend cases still pass. As in Task 1, confirm
`-b Vulkan0` reports "No devices found" cleanly rather than crashing.

- [ ] **Step 9: Commit**

```bash
git add ggml/src/ggml-vulkan/vulkan-shaders/dsv4_hc_pre.comp \
        ggml/src/ggml-vulkan/vulkan-shaders/dsv4_hc_comb.comp \
        ggml/src/ggml-vulkan/vulkan-shaders/dsv4_hc_post.comp \
        ggml/src/ggml-vulkan/ggml-vulkan.cpp \
        ggml/src/ggml-vulkan/vulkan-shaders/vulkan-shaders-gen.cpp
git commit -m "vulkan: port DSV4 HC pre/comb/post fusion kernels from strix-halo-vulkan-beta3"
```

---

## Task 3: Sparse top-k flash-attention Vulkan kernel (scalar path)

**Files:**
- Create: `ggml/src/ggml-vulkan/vulkan-shaders/flash_attn_top_k.comp`
- Modify: `ggml/src/ggml-vulkan/ggml-vulkan.cpp`
- Modify: `ggml/src/ggml-vulkan/vulkan-shaders/vulkan-shaders-gen.cpp`

**Interfaces:**
- Consumes: the `src[5]`/`op_params[4]` hint produced by `ggml_flash_attn_ext_add_top_k`
  (already committed on the `dsv4-sparse-attn-prefill` worktree this plan branches from -
  unchanged by this task).
- Produces: a scalar-only Vulkan fast path for DSV4-shaped sparse flash-attention. The coopmat2
  variant and tiling/split-k integration are Task 4, not this task - this task's kernel is
  functionally complete but not the top-performance path.

- [ ] **Step 1: Copy the shader file from beta3**

```bash
cp /w/projects/llm/nathanw-llamacpp/ggml/src/ggml-vulkan/vulkan-shaders/flash_attn_top_k.comp \
   ggml/src/ggml-vulkan/vulkan-shaders/flash_attn_top_k.comp
```

Do not modify `LANES` (hardcoded 64, deliberately not the `SUBGROUP_SIZE` spec constant -
there's an in-file comment explaining this is intentional for array-bound folding) or
`HEADS_PER_GROUP`/`KEYS_PER_BLOCK`. Do not remove the barrier-divergence guard comment near the
top - it explains why there is deliberately no early-return bounds check inside the main loop.

- [ ] **Step 2: Validate shader syntax with glslc**

```bash
"/c/VulkanSDK/1.3.246.0/Bin/glslc.exe" -fshader-stage=compute \
  ggml/src/ggml-vulkan/vulkan-shaders/flash_attn_top_k.comp -o /tmp/fatk.spv
```

- [ ] **Step 3: Add the push-constant struct**

```cpp
struct vk_op_flash_attn_top_k_push_constants {
    uint32_t n_batch, n_kv, n_kv_raw, n_top_k, n_head;
    uint32_t nbq1, nbq2, nbq3, nbk1, nbk3, nbm1, nbm3, nbt1, nbt3, nb1, nb2, nb3;
    float scale;
    uint32_t has_sinks, split_mode;
};
static_assert(sizeof(vk_op_flash_attn_top_k_push_constants) <= 128);
```
This field order must match `flash_attn_top_k.comp`'s `Parameters` push-constant block exactly
(confirmed via direct read of the shader this session). For this task, always write
`split_mode = 0` (no split-k) - Task 4 wires up the nonzero value.

- [ ] **Step 4: Add the pipeline field and creation**

Add `pipeline_flash_attn_top_k_f16` next to the other new pipeline fields. Create it
unconditionally (no coopmat gating needed yet - Task 4 adds the gated `_cm_f16` variant).

- [ ] **Step 5: Add the dispatch/gate function**

Find `ggml_vk_flash_attn(...)` (entry point at `ggml-vulkan.cpp:11899` in our fork - confirmed
this session) and read it in full first, to find where its initial `GGML_ASSERT`s end and the
dense-path pipeline-selection logic begins. Add a new static function above it:

```cpp
static bool ggml_vk_flash_attn_top_k(ggml_backend_vk_context * ctx, vk_context& subctx,
                                      const ggml_tensor * q, const ggml_tensor * k,
                                      const ggml_tensor * v, const ggml_tensor * mask,
                                      ggml_tensor * dst) {
    const ggml_tensor * top_k = dst->src[5];
    if (top_k == nullptr) {
        return false;
    }
    // shape/dtype gate - see conditions below
    // ...
    // dispatch pipeline_flash_attn_top_k_f16 with split_mode=0
    return true;
}
```

Gate conditions (all must hold, else `return false` and let the existing dense path handle it):
- `q->type==F32, k->type==F16, v->type==F16, mask->type==F16, top_k->type==I32`
- `q->ne[0]==512 && k->ne[0]==512 && v->ne[0]==512 && q->ne[1]>=64 && q->ne[2]==64`
- `k->ne[2]==1 && v->ne[2]==1` (MQA)
- `q->ne[1]==top_k->ne[1] && q->ne[3]==top_k->ne[3]` (broadcast dims match - our fork's HIP
  kernel gate is missing this check; beta3's Vulkan gate has it; include it here)
- `k->ne[1]==v->ne[1] && k->buffer==v->buffer && k->data==v->data` (K==V aliasing)
- `ggml_is_contiguous(mask) && ggml_is_contiguous(top_k)`
- `max_bias==0 && logit_softcap==0` (read these off `dst->op_params`, same slots the existing
  dense FA path already uses)
- `n_kv_raw>=0 && n_kv_raw<=k->ne[1] && top_k->ne[0]<=k->ne[1]-n_kv_raw` (read `n_kv_raw` from
  `dst->op_params[4]`, the field `ggml_flash_attn_ext_add_top_k` sets)
- Crossover: `k->ne[1] >= 3*(n_kv_raw+top_k->ne[0])` (fixed 3x threshold for this task, matching
  our fork's already-reviewed HIP kernel's gate - Task 4 loosens this once the coopmat pipeline
  exists, since beta3's variable threshold assumes a coopmat path is available)

Call this function from `ggml_vk_flash_attn`, right after its initial asserts and before the
dense-path pipeline-selection logic: `if (ggml_vk_flash_attn_top_k(ctx, subctx, q, k, v, mask,
dst)) { return; }`. This mirrors the existing pattern for `sinks` (`src[4]`), which
`ggml_vk_flash_attn` already receives as an explicit parameter from its one call site
(`ggml-vulkan.cpp:17103`) - `top_k` (`src[5]`) is read directly inside the new function instead,
matching beta3's own choice not to widen `ggml_vk_flash_attn`'s signature.

- [ ] **Step 6: Add the supports_op case**

`GGML_OP_FLASH_ATTN_EXT` already has a `supports_op` case (it's a long-existing op) - do not add
a new case, extend the existing one so it still returns `true` for the dense path when the
top-k gate doesn't match, and additionally accepts the new shape when it does. Read the existing
`case GGML_OP_FLASH_ATTN_EXT:` block in `ggml_backend_vk_device_supports_op` before touching it.

- [ ] **Step 7: Register the shader**

```cpp
string_to_spv("flash_attn_top_k_f16", "flash_attn_top_k.comp", {});
```
(Unconditional, no coopmat guard - Task 4 adds the guarded `_cm` line.)

- [ ] **Step 8: Build and test**

```bash
wsl.exe -e bash -lc "cd /mnt/w/projects/llm/llama.cpp/build && make -j16 test-backend-ops 2>&1 | tail -60"
wsl.exe -e bash -lc "cd /mnt/w/projects/llm/llama.cpp/build && ./bin/test-backend-ops test -b CPU -o FLASH_ATTN_EXT 2>&1 | tail -10"
```
Expected: clean build; CPU-backend FLASH_ATTN_EXT cases (including the DSV4 top-k shapes
already registered by the prior plan's Task 3) still pass. This confirms the new code doesn't
break anything reachable without a Vulkan device; it does not confirm the new kernel is
functionally correct on real hardware - that's the final hardware-validation handoff.

- [ ] **Step 9: Commit**

```bash
git add ggml/src/ggml-vulkan/vulkan-shaders/flash_attn_top_k.comp \
        ggml/src/ggml-vulkan/ggml-vulkan.cpp \
        ggml/src/ggml-vulkan/vulkan-shaders/vulkan-shaders-gen.cpp
git commit -m "vulkan: port DSV4 sparse top-k flash-attention kernel (scalar path)"
```

---

## Task 4: Sparse top-k flash-attention Vulkan kernel (coopmat2 + tiling/split-k)

**This is the highest-risk task in this plan.** Unlike Tasks 1-3, it requires modifying shared
Vulkan FA infrastructure files that our fork has independently diverged from beta3's ancestor
substantially (measured this session: 244-1522 changed lines per file between the shared
ancestor commit `ddd4ec1428a6201e18975ea52b07c71e0f9aef26` and our fork's current versions). A
blind copy of beta3's files would silently discard whatever our fork independently changed in
its own Vulkan dense-FA path since that ancestor. Use the diff-and-reconcile method below, not
a file copy, for the shared files.

**Files:**
- Create: `ggml/src/ggml-vulkan/vulkan-shaders/flash_attn_top_k_cm.comp`
- Modify (reconcile, not copy): `ggml/src/ggml-vulkan/vulkan-shaders/flash_attn_base.glsl`
- Modify (reconcile, not copy): `ggml/src/ggml-vulkan/vulkan-shaders/flash_attn_cm1.comp`
- Modify (reconcile, not copy): `ggml/src/ggml-vulkan/vulkan-shaders/flash_attn_cm2.comp`
- Modify (reconcile, not copy): `ggml/src/ggml-vulkan/vulkan-shaders/flash_attn_split_k_reduce.comp`
- Modify: `ggml/src/ggml-vulkan/ggml-vulkan.cpp` (extends Task 3's `ggml_vk_flash_attn_top_k`)
- Modify: `ggml/src/ggml-vulkan/vulkan-shaders/vulkan-shaders-gen.cpp`

**Interfaces:**
- Consumes: Task 3's `ggml_vk_flash_attn_top_k` function and `pipeline_flash_attn_top_k_f16` -
  this task extends both, it does not replace them.
- Produces: coopmat2 pipeline selection and split-k tiling for the sparse top-k kernel, matching
  beta3's actual performance characteristics (this is where the real speedup lives - Task 3's
  scalar-only kernel is functionally complete but not fast).

- [ ] **Step 1: Isolate beta3's DSV4-specific delta on the shared files**

In the beta3 checkout, compute the isolated diff against the shared ancestor commit (this
excludes general upstream churn unrelated to DSV4 sparse-FA):

```bash
cd /w/projects/llm/nathanw-llamacpp
git diff ddd4ec1428a6201e18975ea52b07c71e0f9aef26 HEAD -- \
  ggml/src/ggml-vulkan/vulkan-shaders/flash_attn_base.glsl \
  ggml/src/ggml-vulkan/vulkan-shaders/flash_attn_cm1.comp \
  ggml/src/ggml-vulkan/vulkan-shaders/flash_attn_cm2.comp \
  ggml/src/ggml-vulkan/vulkan-shaders/flash_attn.comp \
  ggml/src/ggml-vulkan/vulkan-shaders/flash_attn_split_k_reduce.comp \
  > /tmp/dsv4-shared-fa-delta.diff
```

Read this diff in full. It contains exactly the semantic changes beta3 made to shared dense-FA
infrastructure to support the sparse top-k kernel (new uniform/push-constant fields, new
branches for split/tile handling, any signature changes to shared helper functions) - this is
your specification for what needs to exist in our fork's versions of these files, not a patch to
apply mechanically.

- [ ] **Step 2: Read our fork's current versions of the same files**

Read `ggml/src/ggml-vulkan/vulkan-shaders/flash_attn_base.glsl`,
`ggml/src/ggml-vulkan/vulkan-shaders/flash_attn_cm1.comp`,
`ggml/src/ggml-vulkan/vulkan-shaders/flash_attn_cm2.comp`,
`ggml/src/ggml-vulkan/vulkan-shaders/flash_attn.comp`, and
`ggml/src/ggml-vulkan/vulkan-shaders/flash_attn_split_k_reduce.comp` in full, in our fork.
Identify where each semantic change from Step 1's diff should land given our fork's current
structure (function names, field layout, or control flow may differ from beta3's ancestor
version even where the underlying logic is equivalent).

- [ ] **Step 3: Apply the semantic changes to our fork's files**

Manually re-implement each change from Step 1's diff against our fork's current file content,
adapting to our fork's structure. This is engineering judgment, not mechanical patching - if a
change's target location or rationale is unclear after reading both versions, or a change
conflicts with something our fork independently added, **stop and report BLOCKED** with the
specific conflict rather than guessing. Getting this wrong risks silently breaking dense
flash-attention for every non-DSV4 model on Vulkan, not just DSV4 - treat any uncertainty here
as a reason to escalate, not push through.

- [ ] **Step 4: Copy the new coopmat2 shader file**

```bash
cp /w/projects/llm/nathanw-llamacpp/ggml/src/ggml-vulkan/vulkan-shaders/flash_attn_top_k_cm.comp \
   ggml/src/ggml-vulkan/vulkan-shaders/flash_attn_top_k_cm.comp
```

- [ ] **Step 5: Validate shader syntax with glslc**

```bash
"/c/VulkanSDK/1.3.246.0/Bin/glslc.exe" -fshader-stage=compute \
  ggml/src/ggml-vulkan/vulkan-shaders/flash_attn_top_k_cm.comp -o /tmp/fatkcm.spv
```
Also re-run glslc on the five reconciled shared files from Steps 1-3 to confirm they still
compile after your edits.

- [ ] **Step 6: Extend the pipeline field, creation, and dispatch function**

Add `pipeline_flash_attn_top_k_cm_f16`, gated identically to Task 1's coopmat pipelines
(`device->coopmat_support && device->coopmat_support_16x16x16_f32acc &&
device->subgroup_size_control`, plus the `GGML_VULKAN_COOPMAT_GLSLC_SUPPORT` /
`VK_KHR_cooperative_matrix` macro guards).

Extend Task 3's `ggml_vk_flash_attn_top_k` to: (a) select the coopmat pipeline when available
(env override `GGML_VK_FA_TOPK_CM` to force-disable, matching beta3's pattern), (b) loosen the
crossover threshold when coopmat is available to `k->ne[1] >= (n_kv_raw+top_k->ne[0])+1` instead
of Task 3's fixed `3*`, since the coopmat path is cheap enough to win as soon as anything is
pruned, and (c) add the split/tile branch, reusing the existing shared split-k-reduce
infrastructure (`get_fa_tuning_params`/`get_fa_pipeline_state` or whatever your fork's current
equivalent is named after Steps 1-3's reconciliation) rather than writing new tiling logic from
scratch. Read beta3's **current** `ggml_vk_flash_attn_top_k` function in full (search
`ggml-vulkan.cpp` for that function name in the beta3 checkout) as the authoritative reference
for this step - the version referenced during this plan's research predates several of the
tiling commits, so line-number citations from that research are stale for this step; the
function's current HEAD state is what matters.

- [ ] **Step 7: Register the coopmat shader**

```cpp
#if defined(GGML_VULKAN_COOPMAT_GLSLC_SUPPORT)
string_to_spv("flash_attn_top_k_cm_f16", "flash_attn_top_k_cm.comp", {});
#endif
```

- [ ] **Step 8: Build and test - broad regression, not just DSV4**

```bash
wsl.exe -e bash -lc "cd /mnt/w/projects/llm/llama.cpp/build && make -j16 test-backend-ops 2>&1 | tail -60"
wsl.exe -e bash -lc "cd /mnt/w/projects/llm/llama.cpp/build && ./bin/test-backend-ops test -b CPU -o FLASH_ATTN_EXT 2>&1 | tail -10"
```
Because this task touches files shared with every model's dense Vulkan flash-attention, also
run the full `test-backend-ops -b CPU` suite (not filtered to FLASH_ATTN_EXT) as an extra
build/link/basic-sanity pass:
```bash
wsl.exe -e bash -lc "cd /mnt/w/projects/llm/llama.cpp/build && ./bin/test-backend-ops test -b CPU 2>&1 | tail -30"
```
Expected: clean build, all CPU cases pass (again, this only proves the C++/shader compiles and
CPU-backend behavior is unaffected - it cannot prove the shared-file changes didn't regress
Vulkan's dense FA path for non-DSV4 models, since there's no local device to run against).
**Flag this explicitly in your report** as the top-priority item for the final hardware
handoff: the user must run broad Vulkan FLASH_ATTN_EXT regression (not just the new DSV4 shapes)
on real hardware before trusting this change.

- [ ] **Step 9: Commit**

```bash
git add ggml/src/ggml-vulkan/vulkan-shaders/flash_attn_top_k_cm.comp \
        ggml/src/ggml-vulkan/vulkan-shaders/flash_attn_base.glsl \
        ggml/src/ggml-vulkan/vulkan-shaders/flash_attn_cm1.comp \
        ggml/src/ggml-vulkan/vulkan-shaders/flash_attn_cm2.comp \
        ggml/src/ggml-vulkan/vulkan-shaders/flash_attn_split_k_reduce.comp \
        ggml/src/ggml-vulkan/ggml-vulkan.cpp \
        ggml/src/ggml-vulkan/vulkan-shaders/vulkan-shaders-gen.cpp
git commit -m "vulkan: add coopmat2 + split-k tiling for DSV4 sparse top-k flash-attention"
```

---

## Task 5: Device-lost/OOM crash investigation

Depends on Tasks 1-4. Tests the design doc's hypothesis: running HC pre/comb/post and the
indexer unfused (today's Vulkan fallback) allocates larger intermediate buffers at huge `n_kv`
than the fused kernels from Tasks 1-2 do, and that this - not something unrelated - is why
`--ctx 65536`/`--ctx 1048576` crash with `ErrorOutOfDeviceMemory`/`ErrorDeviceLost` on our
fork's Vulkan backend today.

This task cannot be completed by a subagent alone - there is no Vulkan device reachable from
this development environment (confirmed this session on both Windows and WSL). Its deliverable
is a concrete, ready-to-run test procedure plus written findings from actually running it,
handed to the user to execute on the Strix Halo box.

**Files:**
- Create: `docs/superpowers/specs/2026-08-16-dsv4-vulkan-crash-investigation.md` (findings +
  procedure, written by the implementer, filled in with the user's results once available)

- [ ] **Step 1: Confirm the crash site and existing instrumentation**

Read `ggml/src/ggml-vulkan/ggml-vulkan.cpp:16038-16047` (the pinned-memory-allocation fallback
path already confirmed this session - it warns and falls back to a CPU buffer on
`ErrorOutOfDeviceMemory`, it does not itself crash) and confirm the actual crash in the user's
logs (`ErrorDeviceLost` at queue submit time, not the pinned-memory warning) is a separate,
later failure - almost certainly the GPU running out of actual VRAM/UMA memory during the
prefill graph's execution, not the pinned-memory staging-buffer fallback path itself. Write this
distinction into the findings doc so it isn't re-litigated.

- [ ] **Step 2: Prepare a before/after memory-footprint comparison procedure**

The build already supports `-DGGML_VULKAN_MEMORY_DEBUG=ON` (confirmed present in this session's
CMakeCache as an existing, currently-off option). Write a procedure into the findings doc:
1. Build with `-DGGML_VULKAN_MEMORY_DEBUG=ON` at the commit just before Task 1 (i.e. at the
   `dsv4-sparse-attn-prefill` worktree tip, before any Vulkan kernel work) and at the tip of
   this plan's branch (after Task 4).
2. Run the same DSV4 prefill workload at `--ctx 65536` (the smaller of the two contexts that
   crashed - start here, escalate to `--ctx 1048576` only if the smaller one succeeds) against
   both builds, capturing the memory-debug log output.
3. Compare total allocated buffer size and the largest single allocation between the two runs.
   If the "before" build's peak allocation is substantially larger and the "after" build no
   longer crashes at `--ctx 65536`, the hypothesis is confirmed - report this as the fix Tasks
   1-4 already delivered as a side effect, no further crash-specific code change needed.
4. If the "after" build still crashes, or the memory delta doesn't explain it, escalate to
   checking `ggml_vk_get_max_memory_allocation_size`/`suballocation_block_size` handling
   (`ggml-vulkan.cpp`, search for `max_memory_allocation_size`) against the KV-cache tensor
   sizes at `--ctx 1048576` - a single tensor exceeding the device's
   `maxMemoryAllocationSize` would explain a hard crash independent of total footprint.

- [ ] **Step 3: Hand off to the user**

Write the exact commands (build invocation, workload invocation matching the user's earlier
`llama-server` command line, log-capture instructions) into the findings doc, commit it, and
report back to the user with a summary of what to run and what to send back (memory-debug log
output, whether the crash still reproduces at each context size).

- [ ] **Step 4: Commit**

```bash
git add docs/superpowers/specs/2026-08-16-dsv4-vulkan-crash-investigation.md
git commit -m "docs: DSV4 Vulkan crash investigation procedure and findings"
```
