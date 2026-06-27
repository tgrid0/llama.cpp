# Hold-until-EOS resolution for in-`<think>` tool calls (Qwen family)

Date: 2026-07-01
Status: Approved design, pending implementation plan
Scope gate: `analyze_reasoning::recover_inline_tool_calls` (Qwen3.5/3.6 family + Nemotron-3 XML tool format)

## Problem

Qwen-family models sometimes emit a `<tool_call>...</tool_call>` block **inside** the
`<think>` reasoning region, closing `</think>` only afterwards or not at all. The current
scoped fix (branch `b9821_disk_cache`, commits `037165166` / `ecb044ff8`) extracts such a
call **eagerly** during streaming. Two problems follow from eager extraction:

1. **Streaming crash.** A partial parse at step *N* commits the in-think block as a tool
   call; a later partial parse re-interprets it and finds fewer tool calls. This trips the
   monotonicity guard in `common_chat_msg_diff::compute_diffs`
   (`common/chat.cpp:283`, `Invalid diff: now finding less tool calls!`).
2. **Stray `</think>` leak.** Duplicate/stray `</think>` tags fall into the content region.

The root constraint is **diff monotonicity**: the streaming server (`task_result_state::update_chat_msg`,
`tools/server/server-task.cpp:158`) re-parses the full accumulated text on every token and
emits only the delta via `compute_diffs`. Each channel (`reasoning_content`, `content`) may
only be *appended* to (`string_diff`, `common/chat.cpp:54`), and `tool_calls.size()` may only
*grow*. Any reclassification of already-committed bytes is illegal.

A `<tool_call>` block inside a closed `<think>` is byte-identical whether it is a real action
or an illustration. The disambiguating signal is not inside the block — it is **what follows**.
During streaming that signal has not been sampled yet, which is why eager extraction has to
guess and sometimes guesses wrong.

## Decision summary

| Decision | Choice |
|----------|--------|
| Release timing | **Hold until EOS only.** Partial parses stream reasoning up to the in-think tool marker, then withhold. The final (`is_partial=false`) parse commits everything in one burst. |
| Resolution rule | The in-think call is **real (extract)** only if it is *terminal* — followed by (optional `</think>` +) EOS. Any real payload after `</think>` (text or a tool call) means the in-think call was **illustrative** and stays reasoning. |
| Scope | Behind the existing `recover_inline_tool_calls` flag. Non-recovery models are byte-for-byte unchanged. |

Consequence accepted by design: under hold-until-EOS the entire tail after the in-think
marker (including an illustrative-case text answer) is buffered and bursts at EOS. Client sees
pause → burst. This is the accepted latency tradeoff.

## Why holding is monotonic

Held partial state = `reasoning_content` up to the first in-think tool marker, zero tool
calls, empty content. Every possible final resolution only **appends** to reasoning and/or
**grows** `tool_calls` from 0 upward:

- Illustrative (post-think call): reasoning grows to include the in-think block text; tool
  calls 0 → N (the post-think calls).
- Illustrative (post-think text): reasoning grows to include the in-think block text; content
  0 → text.
- Extract: reasoning stays (or grows by trailing pre-`</think>` reasoning); tool calls 0 → N.

No channel ever shrinks and `tool_calls` never decreases, so `compute_diffs` cannot throw.

## Architecture / components

| File | Change |
|------|--------|
| `common/peg-parser.h` | Add `COMMON_PEG_PARSE_FLAG_PARTIAL`; add `ctx.is_partial()`; add a parse-time guard primitive `p.hold_when_partial()` — succeeds as `eps` on a final parse, fails at the current position on a partial parse. |
| `common/chat.cpp` | In `common_chat_peg_parse`, set `COMMON_PEG_PARSE_FLAG_PARTIAL` when `is_partial`. The existing `is_partial && result.end > 0` branch (`common/chat.cpp:2713`) already returns the held prefix AST, so no server-side change is needed. |
| `common/chat-auto-parser-generator.cpp` | Restructure the recovery-mode composition into the A/B ordered choice below, factored into **one shared helper** applied at all three tool-format composition sites (they are structurally identical — see below). Rework the now-subsumed recovery branches in `build_trailing_end_parser` (`:206`), `build_content_before_tools` (`:220`), and `analyze_content::build_parser` (`:253`). |
| `common/chat-auto-parser.h` | Declarations for any new reasoning sub-parser helpers (reasoning-up-to-marker, absorb-until-`</think>`). |
| `tests/test-chat.cpp` | Streaming + full-parse tests per the edge-case table. |

### Composition sites (verified)

The recovery-flagged models — Qwen3.5 (`models/templates/Qwen3.5-4B.jinja`) and Nemotron-3
(`models/templates/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16.jinja`) — use a fully **tagged** tool
format (`<tool_call><function=name><parameter=k>v</parameter></function></tool_call>`, no JSON
args), which detects as `tool_format::TAG_WITH_TAGGED` → `build_tool_parser_tag_tagged`
(`:445`, composed at `:582`).

All three tool-format composition sites share the identical shape
`ctx.reasoning_parser + optional(content(content_before_tools)) + tool_calls + trailing + end`:

- `build_tool_parser_json_native` — `:332`
- `build_tool_parser_tag_json` — `:442`
- `build_tool_parser_tag_tagged` — `:582` (the Qwen3.5 / Nemotron-3 path)

Because they are identical, the A/B ordered choice is implemented **once** and applied at all
three, gated on `recover_inline_tool_calls`. This avoids depending on the format detection
picking a specific mode.

Verifying which path a template hits is testable today: `tests/test-chat.cpp` `peg_tester`
already drives the Qwen3.5 (`:2037`) and Nemotron-3 (`:2664`) templates end-to-end, and
`tests/test-chat-auto-parser.cpp` loads them and can assert the detected
`analysis.tools.format.mode`. Add an explicit `TAG_WITH_TAGGED` assertion for both to pin the
path against future template drift.

The decision must be a **parse-time** check: the parser tree is built once per template
(`build_parser` runs at analysis time), so partial-vs-final cannot be a build-time branch —
it is read from the context flag at parse time via `hold_when_partial()`.

The hold lives in the **parser layer**, not the server/diff layer. The diff layer has no
notion of "this tool call originated inside reasoning," so suppressing diffs there would be
guesswork; the parser already knows the structure.

## Parse-time grammar

```
recovery_parser = choice({
  // A — reasoning closed normally, no in-think call
  reasoning(until </think>) + </think> + content_before_tools + tool_calls + trailing + end,

  // B — reasoning stopped at an in-think tool marker
  reasoning(until_one_of([</think>, tool_markers]))   // streams up to the marker
    + hold_when_partial()                             // PARTIAL: fail → hold; FINAL: eps
    + choice({
        // B1 illustrative, post-think TOOL CALL (matches only if a post-</think> call exists)
        reasoning(absorb until </think>) + </think> + optional(content) + tool_calls(>=1) + trailing + end,
        // B2 illustrative, post-think TEXT (reverts to pre-change behavior)
        reasoning(absorb until </think>) + </think> + content(non-empty, non-marker) + end,
        // B3 extract — fallback: in-think call is terminal
        tool_calls + absorb-trailing-reasoning / swallow-stray-</think> + end
      })
})
```

- **Partial parse:** path A streams reasoning normally (identical to standard tag-based
  behavior). Path B streams reasoning up to the marker, then `hold_when_partial()` fails; the
  partial branch in `common_chat_peg_parse` returns the held prefix. Nothing after the marker
  streams.
- **Final parse:** `hold_when_partial()` is `eps`. Ordered choice resolves: B1 matches only if
  a post-`</think>` call exists; B2 matches only if real text follows `</think>`; if the call
  is terminal, both fail and B3 extracts.

## Edge cases (each becomes a test)

1. In-think call, `</think>`, **EOS**, no post-think payload → **extract** (B3). The real case
   — 100% of the originally reported real occurrences.
2. In-think call, `</think>` **never closes**, EOS → **extract** (B3, terminal/unclosed).
3. In-think call + `</think>` + post-think **tool call** → **illustrative**, emit post-think
   only (B1). This is the exotic double-call pattern that was previously unsupported / a
   streaming crash; it is now stable.
4. In-think call + `</think>` + **text answer**, no tool call → **illustrative**, in-think
   stays reasoning, text becomes content (B2). Pre-change behavior, restored.
5. Stray/duplicate `</think>` around the call → swallowed. B2's content requirement must be
   genuinely non-empty and non-marker so stray tags are not mistaken for real content and do
   not force a wrong illustrative match.
6. Parallel in-think calls: extract all if terminal (B3), else illustrative (B1/B2).
7. Non-recovery model → untouched.

Grammar note for B3: the observed real pattern is the in-think call sitting immediately
before `</think>`, so B3's core job is simply extract-call + swallow trailing `</think>` /
whitespace / stray tags + `end`. Preserving any reasoning text that appears *between* the
in-think call and `</think>` as a second `reasoning_content` segment is a cheap defensive
extra, not a Qwen-observed requirement; implement it only if it falls out naturally.

## Error handling

- **Hard fallback / no infinite hold:** `hold_when_partial()` only fails when `is_partial`.
  The final parse always commits, so a stream can never hang waiting for resolution.
- If B1/B2/B3 all fail on malformed input, fall through to the existing lenient content
  parsing path — no new crash paths are introduced.

## Testing

- Extend `tests/test-chat.cpp` with **incremental/streaming** tests: feed the text growing
  token by token, assert `common_chat_msg_diff::compute_diffs` never throws, and assert the
  final message is correct — for edge cases 1–6, alongside full-parse assertions. Reinstate
  the previously-unsupported exotic double-call case (old `test-chat.cpp:2206`) as a *stable*
  streaming test.
- Regression: `test-chat-auto-parser` (499 assertions) and `test-chat-peg-parser` (198
  assertions) must stay green — non-recovery paths are unchanged.

## Out of scope

- Early (pre-EOS) release of the held span. Considered and rejected in favor of the simpler
  hold-until-EOS; it would require a monotonicity proof per early-release rule.
- Extending the behavior to non-recovery models.
- Any change to the disk-cache subsystem.
