# Porting agentic-session features from CachyLLama (server-only, no client changes)

Status: plan for a future agent session. No code changed yet. Scope constraint from the
user: only features that work without any client-side modification (no new headers/params
the client must send to get the benefit), and explicitly **not** the Strix Halo/Vulkan/RDNA3
hardware-tuning parts of CachyLLama (those are out of scope here — separate concern from
"long agentic session" behavior).

Source: `CachyLLama/README.md` (root README, CachyLLama-specific sections at
the top, lines 1-216) plus `docs/moe-expert-residency.md` and
`docs/development/user-isolation-design.md` in that repo.

## Headline finding: our disk cache is architecturally ahead in one way, but has a concrete gap that specifically hurts long sessions

This branch's own disk-prompt-cache (`tools/server/server-disk-cache.h/.cpp`,
`server-context.cpp:1798-1839`) is **content-addressed** (SHA256 of the token sequence,
`server-disk-cache.h:116`) with a genuine longest-common-prefix search across *all* stored
entries (`find_best_prefix`, `server-disk-cache.h:99-102`). This is actually more general
than CachyLLama's "system prompt cache" (README lines 21-23: a separate, fixed-size,
8-entries-by-default cache keyed on "the first N tokens of any prompt") — our design doesn't
need a special-cased second cache tier to get cross-conversation reuse of a shared system
prompt, because any two requests sharing an identical prefix will hash-match regardless of
which "tier" conceptually owns that prefix.

**However, there is a real, concrete gap that undoes this advantage specifically in long
agentic sessions.** `server-context.cpp:1818-1822`:
```cpp
// Exact miss - try prefix search. Require at least 50% of the
// request tokens to be covered by the cached prefix.
const size_t n_req = task.tokens.get_text_tokens().size();
const size_t min_prefix_len = std::max((size_t)1, n_req / 2);
const std::string prefix_hash = disk_cache->find_best_prefix(task.tokens, min_prefix_len);
```
The prefix search only fires if the matched prefix covers **at least half** of the current
request's total token count. Consider the exact scenario CachyLLama's README calls out
(README line 19: "persistent, multi-turn agentic sessions with 18-30K-token static
prefixes"): an 8-15K-token system prompt + tool definitions, shared across many turns. Early
in a conversation this 50% threshold is easily satisfied. But once the conversation grows
past roughly 2x the shared-prefix length (very normal for a long agentic session -
accumulated tool outputs, multi-turn history), **the threshold silently stops firing** and
the entire prompt gets re-evaluated from scratch every request, even though the large,
genuinely reusable system-prompt segment is still sitting in the cache. This is the inverse
of CachyLLama's design: their dedicated system-prompt tier is *always* checked (it's keyed
on the fixed prefix boundary, not scaled by total request length), so the fixed-size saving
persists no matter how long the conversation grows.

This directly matches the user's own stated pain point (agentic sessions with long-running
context) and is a small, surgical, well-understood fix - not a rewrite:

**Proposed fix**: change the threshold from proportional (`n_req / 2`) to a policy that
still tries the prefix match when the matched prefix is large in *absolute* terms even if
it's a small fraction of a long request - e.g. `min(n_req / 2, some_absolute_floor)`, or a
two-stage search (try a generous absolute floor first, fall back to the proportional
threshold only if that yields no match) so a genuinely large, valuable shared prefix isn't
silently ignored just because the conversation grew. Needs a decision on the right floor
value and whether it should be a new CLI flag (default preserving today's 50% behavior,
opt-in to the floor-based variant) or a straight behavior change — check with the user before
picking, since this changes default caching behavior. Validate with a synthetic test:
save a ~10K-token prefix entry, then issue a request with a ~9K-token prefix hit plus 15K
tokens of divergent tail (prefix is ~35% of total) and confirm the fix causes the prefix hit
to be used where today it would be skipped. Existing test scaffolding:
`tools/server/tests/unit/test_disk_cache.py`.

## Second gap found while reading this code: no model-compatibility check on checkpoint restore

CachyLLama's README (line 21) specifically calls out "Conversation hash and model
compatibility hash prevent mismatched checkpoint restoration" as a named safety feature. This
branch's `server-disk-cache.cpp` has **no equivalent check** (grepped for
model/arch/vocab/fingerprint validation on load - none found). A disk-cache entry saved under
one model/quant and restored against a different one loaded later (very plausible with
llama-swap juggling multiple models, or during the re-quantization work planned in
`2026-08-10-dsv4-strix-halo-perf-fix.md`'s Branch A) would silently attempt to restore
mismatched KV state. This subsystem already has a history of subtle correctness bugs (see the
prior design docs in this directory:
`2026-05-10-disk-kv-cache-design.md`, `2026-05-29-disk-cache-shutdown-prefix-design.md`)  so
treat this as worth closing, not hypothetical. Minimal fix: store a hash of `n_vocab` +
`general.architecture` + a content hash of key hparams (or just the model file's own path/
mtime/size, cheaper but weaker) alongside each cache entry, and reject/evict a load whose
recorded fingerprint doesn't match the currently-loaded model. Scope this as its own small,
independently-testable change, not bundled with the prefix-threshold fix above.

## Third candidate: MoE expert activation tracking API (net-new, doesn't exist here at all)

Grepped this branch for `expert_tracking`/`expert-stats`/`llama_expert_stats` - **no matches,
this feature genuinely doesn't exist here yet.** CachyLLama's version (README lines 29,
135-184): `GET /expert-stats` and `POST /expert-tracking` HTTP endpoints plus a small C API
(`llama_expert_tracking_enable`, `llama_expert_stats_get`, `llama_expert_stats_reset`,
`llama_model_n_expert`, `llama_model_n_expert_used`) exposing per-layer, per-expert
activation counts and frequencies, computed from data the MoE forward pass already has
(which experts got selected per token). Purely additive observability - no client changes
needed to use the server without it, and no impact on inference if tracking is left disabled
(check CachyLLama's implementation for the actual per-token overhead when enabled, since even
a counter increment per selected expert per layer per token adds up at scale - confirm it's
gated cheaply, e.g. behind a single branch, before assuming "free").

This is lower priority for "long agentic sessions" specifically, but two things make it worth
including in this doc rather than a separate backlog item:
1. It's genuinely useful for the DeepSeek-V4-Flash perf investigation
   (`2026-08-10-dsv4-strix-halo-perf-investigation.md`) - deepseek4 has an unusual hash-based
   MoE router override for the first `hash_layer_count` layers (`ffn_gate_tid2eid`, see
   `src/models/deepseek4.cpp:1334-1337`) alongside the normal learned router for later layers;
   being able to directly observe per-layer expert activation patterns would help confirm
   whether that hash-routing path behaves as expected, independent of the perf question.
2. It's small, self-contained, and low-risk relative to the caching-policy changes above -
   a reasonable first PR-sized chunk to port if the agent wants a low-risk starting point
   before tackling the disk-cache changes.

Port by reading CachyLLama's actual diff (not just the README) for these symbols - find the
commits that introduced `llama_expert_tracking_enable` etc. in that repo's history, since the
README describes the interface but the instrumentation points (where in the MoE forward pass
the counter increments happen) need to be located precisely and checked against this branch's
current `build_moe_ffn`/expert-selection code, which may have drifted from whatever CachyLLama
based it on.

## Explicitly excluded, and why

- **Strix Halo/Vulkan/RDNA3 tuning** (APU Vulkan `nodes_per_submit`, quantized-KV FA scratch,
  CPU ISA auto-detection): out of scope per the user's explicit instruction - covered instead
  by the sibling-repo survey doc (`2026-08-10-strix-halo-sibling-repo-survey.md`) as
  Strix-Halo-specific work, not bundled here.
- **User isolation** (`user_id`, `u/` namespace, per-user concurrency cap, slot affinity):
  excluded because the useful part requires the client to send `llama_user_id` (via
  `extra_body` or a request field) to get any benefit - this is exactly the "client-side
  modification" the user asked to avoid. Worth revisiting only if the user later runs
  multiple concurrent agents/tenants through one server and is willing to update client
  config.
- **MoE expert SSD residency** (`--moe-expert-residency`) and **`--cpu-moe`/`--n-cpu-moe`**:
  `--cpu-moe`/`--n-cpu-moe` already exist in this branch's own `common/arg.cpp` (confirmed
  during the perf-investigation research) - nothing to port. Full SSD expert residency solves
  "model doesn't fit in RAM," which doesn't apply to the user's current 91GB-model/128GB-RAM
  setup; worth reconsidering only if the user later runs a model that doesn't fit.
- **Hybrid MoE checkpoint restore** (Qwen3.5/3.6, Gemma 4, GLM-4.7 recurrent-state handling):
  irrelevant to DeepSeek-V4-Flash (no recurrent/SSM state in this architecture) and not
  currently relevant to any model the user runs on this branch. Noted for future reference
  only if the user starts running hybrid/recurrent models.
- **Persistent SSD cache's hot/warm/cold RAM tiering and `posix_fadvise` readahead**: the
  tiering concept is largely superseded by this branch's simpler content-addressed design
  (see headline finding above), but the specific low-level trick of calling
  `posix_fadvise(POSIX_FADV_WILLNEED)` (or the platform equivalent) on a cache entry's `.bin`
  file just before a blocking read, to let the kernel start I/O while other work proceeds, is
  a cheap, low-risk, purely-additive perf tweak independent of the cache-policy question
  above. Worth a quick look at `server-disk-cache.cpp`'s read path as a small optional
  follow-up, not a priority.

## Suggested order of operations

1. Read the actual CachyLLama commits/diffs for each numbered item above (this doc is based
   on the README plus this branch's own source, not CachyLLama's diffs - confirm
   implementation details, especially the expert-tracking overhead question and the exact
   "conversation hash" scheme, before writing code).
2. Ship the model-compatibility check on checkpoint restore first - smallest, most clearly a
   pure bug fix, no behavior-change judgment calls needed, and directly protects the
   re-quantization work planned elsewhere in this project.
3. Discuss the prefix-threshold policy change with the user before implementing (it changes
   default caching behavior) - present it as a specific proposal (absolute floor vs. new
   flag) rather than assuming which the user wants.
4. Port expert activation tracking as a self-contained addition whenever convenient - it has
   no interaction with the other two items.
