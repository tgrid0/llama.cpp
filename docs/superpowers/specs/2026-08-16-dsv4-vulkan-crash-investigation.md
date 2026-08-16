# DeepSeek-V4 Vulkan device-lost/OOM crash: investigation procedure

Status: procedure written and grounded in code by an agent session with no reachable Vulkan
device (confirmed on both Windows and WSL - WSL only exposes Mesa's software llvmpipe, which
`ggml-vulkan.cpp` deliberately excludes from device enumeration). Steps 1-4 below are ready to
run; the "Results" section at the bottom is a template to fill in once you've run them on the
Strix Halo box.

## Background

Running our fork's Vulkan backend at large context (`--ctx 65536`, `--ctx 1048576`) with the
DeepSeek-V4-Flash workload crashes with `ErrorOutOfDeviceMemory`/`ErrorDeviceLost`. This was
confirmed to be unrelated to the sparse-attention prefill work in this plan (Tasks 1-4, branch
`dsv4-vulkan-prefill-port`): Vulkan's flash-attention path never reads the sparse-attention hint
(`src[5]`/`op_params[4]`) that plan is built around, and the crash reproduces on paths that plan
never touches.

Working hypothesis (from `docs/superpowers/specs/2026-08-16-dsv4-vulkan-prefill-port-design.md`):
before Tasks 1-2 landed, our fork's Vulkan backend had no kernels for `GGML_OP_LIGHTNING_INDEXER`
or `GGML_OP_DSV4_HC_PRE`/`COMB`/`POST`, so those ops ran **unfused** - decomposed into many more,
larger primitive ggml ops - on every layer, every token. Unfused execution may have allocated much
larger intermediate Vulkan buffers at huge `n_kv` than the now-landed fused kernels do, and that
- not something else - may be the actual cause of the OOM/device-lost crash.

This document is the ready-to-run procedure to test that hypothesis, plus the escalation path if
it doesn't hold.

## Step 1: Confirm the crash site is two separate things (do not re-litigate this)

There are two distinct Vulkan failure points in the logs, and they must not be confused:

**1a. Pinned-memory allocation fallback - NOT fatal, just a warning.**

`ggml/src/ggml-vulkan/ggml-vulkan.cpp`, function `ggml_backend_vk_host_buffer_type_alloc_buffer`
(around line 16493 on this branch's tip; the brief's line numbers, ~16038-16047, were from before
Tasks 1-4 added new ops earlier in the file and shifted everything down - same code, same
behavior):

```cpp
static ggml_backend_buffer_t ggml_backend_vk_host_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    ...
    try {
        ptr = ggml_vk_host_malloc(vk_instance.devices[0], size);
    } catch (vk::SystemError& e) {
        GGML_LOG_WARN("ggml_vulkan: Failed to allocate pinned memory (%s)\n", e.what());
        // fallback to cpu buffer
        return ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(), size);
    }
    ...
}
```

If pinned (host-visible, device-local-preferred) memory allocation fails with
`ErrorOutOfDeviceMemory`, this catches it, logs a `WARNING: Failed to allocate pinned memory`
line, and **falls back to a regular CPU-heap buffer** for that staging allocation. Execution
continues - this is slower (no pinned-memory fast path for that transfer) but does not crash the
process by itself. If this is the only Vulkan-related line in the log, the run did not crash here.

**1b. `ErrorDeviceLost` - fatal, this is the actual crash.**

Thrown as `vk::DeviceLostError` from the `VK_CHECK` macro (`ggml-vulkan.cpp:189-206`) at a Vulkan
API call site - most likely `ggml_vk_wait_for_fence` (`ggml-vulkan.cpp:2644-2663`, calling
`getFenceStatus` while waiting on queued GPU work) or a `vkQueueSubmit`-adjacent `VK_CHECK`. `VK_CHECK`
logs via `ggml_vk_print_device_lost_info` (`ggml-vulkan.cpp:2226`) - look for a line like
`ggml_vulkan: device lost on <device name>` in the log - then **rethrows**, which propagates
uncaught out of the backend and takes the whole `llama-server` process down. This is the actual
crash.

**The distinction that matters:** the pinned-memory warning happens first, at buffer-allocation
time (e.g. while setting up KV-cache or staging buffers), and is recoverable. `ErrorDeviceLost`
happens later, during actual graph execution / queue-submit-wait, and is fatal. If your log shows
the pinned-memory warning followed by normal operation for a while and *then* a device-lost crash,
those are two separate events - do not treat the warning as the cause of the crash. The most
likely root cause of the `ErrorDeviceLost` itself is the GPU (Strix Halo's unified memory) running
out of usable memory during the prefill graph's execution - this document's job is to determine
whether that's because of unfused-op buffer bloat (Step 2) or something else (Step 3).

## Step 2: Before/after memory-footprint comparison

### 2.0 What actually drives the fused vs. unfused code path (read this first)

At context-init time, `llama_context::resolve_fused_ops` (`src/llama-context.cpp:568-593`) probes
each backend device for support of the fused ops and sets `cparams.fused_lid` and
`cparams.fused_dsv4_hc_pre`/`fused_dsv4_hc_comb`/`fused_dsv4_hc_post` accordingly, logging one of:

```
resolving fused Lightning Indexer support:
resolving fused DeepSeek V4 HC support:
<op name> enabled
<op name> not supported, set to disabled
```

`src/models/deepseek4.cpp` branches on these flags in two places: `build_hc_pre`/`build_hc_comb`/
`build_hc_post` (lines ~295, 388, 416) check `cparams.fused_dsv4_hc_pre`/`comb`/`post`, and the
lightning-indexer build function separately checks `cparams.fused_lid` at line ~672. When the
relevant flag is true, the code emits a single fused `ggml_dsv4_hc_pre`/`comb`/`post` or
lightning-indexer op; when false (today's Vulkan situation pre-Task-1), it falls through to a
decomposed chain of primitive ops (`dsv4_hc_affine`, `ggml_scale_bias`, mean, a Sinkhorn-iteration
loop, etc. for HC; a separate primitive chain for the indexer) - many more, and larger,
intermediate tensors per layer per token. This is the exact mechanism the hypothesis is about,
and it's a clean, greppable signal independent of buffer sizes:
**grep each build's startup log for the four lines above.** On the "before" build you should see
`not supported, set to disabled` for the lightning-indexer and all three HC probes on Vulkan; on
the "after" build (this branch) you should see `enabled` for all four.

### 2.1 A note on `-DGGML_VULKAN_MEMORY_DEBUG=ON`

This CMake option (`ggml/CMakeLists.txt:226`, wired in `ggml/src/ggml-vulkan/CMakeLists.txt:132-134`)
only adds a compile-time `#define GGML_VULKAN_MEMORY_DEBUG` - but as of this branch, nothing in
`ggml-vulkan.cpp` actually tests that macro with `#ifdef` any more (confirmed by grep: zero hits
outside the two `CMakeLists.txt` files). Git history on `ggml-vulkan.cpp` shows this used to gate
the memory-logging code directly with `#ifdef GGML_VULKAN_MEMORY_DEBUG`, but it was refactored
upstream to a **runtime** toggle: `vk_memory_logger_enabled` (`ggml-vulkan.cpp:2128`) is set from
the environment variable `GGML_VK_MEMORY_LOGGER` (checked once at Vulkan instance init,
`ggml-vulkan.cpp:7495`), and the `VK_LOG_MEMORY` macro (`ggml-vulkan.cpp:2130`) checks that bool.
The CMake option is harmless to pass (it's still a valid option, just currently inert) - you don't
need it, but it's included below for completeness since you may want it as a hedge in case a
different code path than the one read here ends up mattering. **The actual switch you need is the
environment variable `GGML_VK_MEMORY_LOGGER=1` at run time - no special build flag required.**

Memory-debug output format (once `GGML_VK_MEMORY_LOGGER=1` is set), one line per allocation/free:

```
ggml_vulkan memory: <device>: +<size> <type> at <buffer addr>. Total device: <X>, total host: <Y>
ggml_vulkan memory: <device>: -<size> <type> at <buffer addr>. Total device: <X>, total host: <Y>
```

plus, on every resize of the shared scratch buffers used for op intermediates:

```
ggml_vk_preallocate_buffers(x_size: <N>)
ggml_vk_preallocate_buffers(y_size: <N>)
ggml_vk_preallocate_buffers(split_k_size: <N>)
ggml_vk_preallocate_buffers(add_partials_size: <N>)
```

`x_size`/`y_size` are the shared scratch buffers used for MUL_MAT-family and general op
intermediates - this is exactly where the unfused HC/indexer decomposition would show up as a
larger `x_size`/`y_size` request than the fused kernels need, and it's a single, directly
comparable number between the two builds.

### 2.2 Build both commits

These two commits are currently local-only in the Windows dev worktree they were built in - they
are not on any remote (`origin` or the `tgrid` fork). Get them onto your Strix Halo box first
(push to your fork, `git fetch` from the dev machine, rsync the repo, whatever you normally use
to move branches over), then build each one separately. Set `REPO` once to your checkout root
below - adjust the path if your checkout lives somewhere else - everything after that is a
straight copy-paste.

```bash
export REPO=/home/walker/llm/llama.cpp   # adjust if your checkout lives elsewhere
```

**Build A - "before" (pre-Task-1, no Vulkan kernels for indexer/HC fusion), commit
`93c26caa5766ab1377d7dba0623da207d71e8956` (branch `dsv4-sparse-attn-prefill`):**

```bash
cd "$REPO"
git fetch --all
git checkout 93c26caa5766ab1377d7dba0623da207d71e8956
cmake -B build-vk-before -DGGML_VULKAN=ON -DGGML_VULKAN_MEMORY_DEBUG=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-vk-before --config Release -j "$(nproc)" --target llama-server
```

**Build B - "after" (this plan's tip, fused Vulkan kernels for indexer + HC pre/comb/post),
commit `935ba54762a52cb11b4c55a364a582bd96fe7f0e` (branch `dsv4-vulkan-prefill-port`):**

```bash
cd "$REPO"
git checkout 935ba54762a52cb11b4c55a364a582bd96fe7f0e
cmake -B build-vk-after -DGGML_VULKAN=ON -DGGML_VULKAN_MEMORY_DEBUG=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-vk-after --config Release -j "$(nproc)" --target llama-server
```

(Later commits on `dsv4-vulkan-prefill-port` past `935ba5476`, if any, are this investigation doc
itself and are docs-only - safe to build at the branch tip instead of the exact SHA if you prefer,
it won't change the binary.)

### 2.3 Run the workload against each build

Same workload both times - this is the exact `llama-server` invocation that was reported to
crash, with `--ctx-size 65536` (the smaller of the two contexts that crashed - start here; only
move to `--ctx-size 1048576` if 65536 succeeds on Build B). Set `GGML_VK_MEMORY_LOGGER=1` in the
environment and redirect stderr to a log file (the memory-debug lines and all `GGML_LOG_*` output
go to stderr):

```bash
GGML_VK_MEMORY_LOGGER=1 "$REPO/build-vk-before/bin/llama-server" --port 8090 \
  --threads 32 \
  --flash-attn on \
  --fit off \
  --no-warmup \
  --batch-size 2048 \
  --ubatch-size 2048 \
  --cache-type-k q8_0 \
  --cache-type-v q8_0 \
  --jinja \
  --load-mode none \
  --cache-prompt \
  --cache-ram 0 \
  --parallel 1 \
  --n-gpu-layers all \
  --gpu-layers-draft all \
  --ctx-checkpoints 4 \
  --model /home/walker/llm/models/deepseek-v4-flash-0731/antirez/DeepSeek-V4-Flash-Layers37-42Q4KExperts-OtherExpertLayersIQ2XXSGateUp-Q2KDown-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-fixed-0731.gguf \
  --chat-template-file /home/walker/llm/chat-templates/deepseek-v4-flash-0731/default-llamacpp.jinja \
  --chat-template-kwargs "{\"reasoning_effort\": \"max\"}" \
  --spec-type draft-dspark \
  --spec-draft-model /home/walker/llm/models/deepseek-v4-flash-0731/antirez/DeepSeek-V4-Flash-DSpark-dflash-0731.gguf \
  --spec-draft-n-max 64 \
  --ctx-size 65536 \
  --temp 1.0 \
  --reasoning on \
  --reasoning-preserve \
  --alias antirez/ds4flash-284b-a13b-q2 \
  2>&1 | tee /tmp/vk-crash-before-ctx65536.log
```

**Stop the first server before starting the second one** - both bind the same port (8090) and
only one can run at a time. Go back to the terminal running `build-vk-before/bin/llama-server`
(or the crashed/finished command above) and press Ctrl-C to make sure it has exited, then repeat
identically against `build-vk-after/bin/llama-server`, logging to
`/tmp/vk-crash-after-ctx65536.log`:

```bash
GGML_VK_MEMORY_LOGGER=1 "$REPO/build-vk-after/bin/llama-server" --port 8090 \
  --threads 32 \
  --flash-attn on \
  --fit off \
  --no-warmup \
  --batch-size 2048 \
  --ubatch-size 2048 \
  --cache-type-k q8_0 \
  --cache-type-v q8_0 \
  --jinja \
  --load-mode none \
  --cache-prompt \
  --cache-ram 0 \
  --parallel 1 \
  --n-gpu-layers all \
  --gpu-layers-draft all \
  --ctx-checkpoints 4 \
  --model /home/walker/llm/models/deepseek-v4-flash-0731/antirez/DeepSeek-V4-Flash-Layers37-42Q4KExperts-OtherExpertLayersIQ2XXSGateUp-Q2KDown-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-fixed-0731.gguf \
  --chat-template-file /home/walker/llm/chat-templates/deepseek-v4-flash-0731/default-llamacpp.jinja \
  --chat-template-kwargs "{\"reasoning_effort\": \"max\"}" \
  --spec-type draft-dspark \
  --spec-draft-model /home/walker/llm/models/deepseek-v4-flash-0731/antirez/DeepSeek-V4-Flash-DSpark-dflash-0731.gguf \
  --spec-draft-n-max 64 \
  --ctx-size 65536 \
  --temp 1.0 \
  --reasoning on \
  --reasoning-preserve \
  --alias antirez/ds4flash-284b-a13b-q2 \
  2>&1 | tee /tmp/vk-crash-after-ctx65536.log
```

For each run, once the server has finished loading (watch for the "server is listening" line in
the terminal running the command above), **open a second terminal** - the server above runs in
the foreground, piped through `tee`, so it needs the terminal it's in - and send it one
prefill-heavy request from there. The crash was reported during prefill, not idle startup, so an
idle/clean-startup server does not rule anything out; you need to actually push a long prompt
through it.

Generate a filler prompt long enough to approach `--ctx-size 65536`, build the JSON request with
`jq` (handles escaping so you don't have to hand-quote 200KB of text), and POST it to the
chat-completions endpoint on port 8090:

```bash
# ~60k tokens of filler text (rough estimate for English prose: ~4 chars/token, so ~240000 chars;
# for --ctx-size 1048576 scale the repeat count by 16x, e.g. 96000 instead of 6000):
python3 -c "print('The quick brown fox jumps over the lazy dog. ' * 6000)" > /tmp/vk-crash-prompt.txt
wc -c /tmp/vk-crash-prompt.txt

jq -n --rawfile prompt /tmp/vk-crash-prompt.txt \
  '{model: "antirez/ds4flash-284b-a13b-q2", messages: [{role: "user", content: $prompt}], max_tokens: 64, stream: false}' \
  > /tmp/vk-crash-request.json

curl -s -X POST http://127.0.0.1:8090/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d @/tmp/vk-crash-request.json
```

Run this same three-command block (regenerating the prompt file isn't necessary between runs -
reuse `/tmp/vk-crash-prompt.txt` and `/tmp/vk-crash-request.json`) against each of the two servers
in turn, from the second terminal, while the first terminal's `tee` log captures the server-side
output. If `curl` returns a normal completion, the request went through cleanly; if the server
process dies mid-request, `curl` will show a connection-reset/empty-reply error and the crash
should appear in the first terminal's log.

### 2.4 Compare the two logs

```bash
grep -E "resolving fused|enabled|not supported, set to disabled" /tmp/vk-crash-before-ctx65536.log
grep -E "resolving fused|enabled|not supported, set to disabled" /tmp/vk-crash-after-ctx65536.log

grep "ggml_vk_preallocate_buffers" /tmp/vk-crash-before-ctx65536.log | tail -20
grep "ggml_vk_preallocate_buffers" /tmp/vk-crash-after-ctx65536.log | tail -20

grep "Total device:" /tmp/vk-crash-before-ctx65536.log | tail -5
grep "Total device:" /tmp/vk-crash-after-ctx65536.log | tail -5

grep -E "Failed to allocate pinned memory|device lost|ErrorDeviceLost|ErrorOutOfDeviceMemory" /tmp/vk-crash-before-ctx65536.log
grep -E "Failed to allocate pinned memory|device lost|ErrorDeviceLost|ErrorOutOfDeviceMemory" /tmp/vk-crash-after-ctx65536.log
```

**If the hypothesis holds:** Build A's log shows `not supported, set to disabled` for the
indexer/HC probes, a visibly larger `ggml_vk_preallocate_buffers(x_size/y_size: ...)` and/or peak
`Total device:` figure than Build B, and Build A crashes with `ErrorDeviceLost` while Build B
completes the same `--ctx-size 65536` request cleanly. That confirms the fused kernels landed by
Tasks 1-4 already fix the crash as a side effect - report this, no further crash-specific code
change is needed. Then retest Build B at `--ctx-size 1048576` to see how far the fix extends.

**If Build B still crashes, or the memory delta doesn't explain the gap:** move to Step 3.

## Step 3: Escalation path if the hypothesis doesn't hold

Check whether a single tensor is exceeding the device's allocation limit, independent of total
footprint - this would explain a hard, sudden crash that doesn't correlate with overall memory
pressure. Relevant code: `ggml-vulkan.cpp:6430-6458`, where the device's
`max_memory_allocation_size` (from `VkPhysicalDeviceMaintenance3Properties::maxMemoryAllocationSize`,
clamped by `maxBufferSize` when `maintenance4` is supported), `max_buffer_size`, and
`suballocation_block_size` (batched-allocation cap, 1GB by default) are established. All three are
runtime-overridable via environment variables, no rebuild needed:

```bash
# Query what the "after" build's log already reported for pipeline setup/limits, or force smaller
# caps to see whether a smaller cap avoids the crash (would confirm a single-allocation overflow):
GGML_VK_FORCE_MAX_ALLOCATION_SIZE=<bytes>   # overrides max_memory_allocation_size
GGML_VK_FORCE_MAX_BUFFER_SIZE=<bytes>       # overrides max_buffer_size
GGML_VK_SUBALLOCATION_BLOCK_SIZE=<bytes>    # overrides the 1GB suballocation batching cap
```

Compare the largest single `+<size>` line in the `GGML_VK_MEMORY_LOGGER=1` log against the
device's reported `maxMemoryAllocationSize` (visible via `vulkaninfo | grep -i maxMemoryAllocation`
or `vulkaninfo | grep -i maxBufferSize` on the Strix Halo box) - at `--ctx-size 1048576` in
particular, a single KV-cache tensor or attention intermediate could plausibly exceed a
driver-reported allocation limit even if total VRAM usage is fine. If a single allocation is at or
above the limit, that's a distinct bug from the unfused-buffer-bloat hypothesis and needs its own
fix (e.g. splitting that tensor's allocation, or an existing `suballocation_block_size`-based
chunking path not being applied to it) - out of scope for this plan, file as a follow-up.

If neither the total-footprint comparison nor the single-allocation-limit check explains the
crash, get node-level attribution before going further: rerun either binary (no rebuild needed)
with `GGML_VK_SERIALIZE_SUBMISSIONS=1` set in the environment - it's also a runtime env var
(`ggml-vulkan.cpp:7141`). This makes `ggml_vk_print_device_lost_info`
(`ggml-vulkan.cpp:2226-2235`) report the exact graph node range that was in flight when the
device was lost, instead of just the device name - that names the specific op to investigate next,
rather than guessing from the model architecture.

## Step 4: What to send back

After running Steps 2.3-2.4 (and Step 3 if needed), report:

1. Whether Build A (before) and Build B (after) each crashed or completed cleanly at
   `--ctx-size 65536`, and if Build B succeeded, the same for `--ctx-size 1048576`.
2. The four `resolving fused .../ enabled / not supported` log lines from each build's startup
   log (confirms which code path - fused vs. unfused - was actually active).
3. The full `/tmp/vk-crash-{before,after}-ctx*.log` files, or at minimum the
   `ggml_vk_preallocate_buffers` lines, the last few `Total device:` lines before any crash, and
   the exact error/warning lines around the crash (or the tail of the log if it completed
   cleanly).
4. If Step 3 was needed: the `vulkaninfo` `maxMemoryAllocationSize`/`maxBufferSize` values for
   your Strix Halo device, and the largest single `+<size>` allocation line from the log.

## Results

*(Fill in once run on the Strix Halo box - not yet run as of this writing, since no Vulkan device
is reachable from the dev environment this procedure was written in.)*

- Build A (`93c26caa5`, before) at `--ctx-size 65536`: _pending_
- Build B (`935ba5476`, after) at `--ctx-size 65536`: _pending_
- Build B at `--ctx-size 1048576` (only if the above succeeded): _pending_
- Conclusion: _pending_
