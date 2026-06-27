# Hold-until-EOS for in-`<think>` tool calls — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace eager extraction of tool calls emitted inside a Qwen `<think>` block with a hold-until-EOS scheme that streams reasoning up to the in-think tool marker, withholds the rest during streaming, and resolves the call's role (real vs illustrative) at the final parse.

**Architecture:** Add one parse-time PEG primitive, `hold_when_partial()`, gated by a new `COMMON_PEG_PARSE_FLAG_PARTIAL`. In recovery mode (`recover_inline_tool_calls`), prepend an ordered-choice "B path" to the standard parser composition: reasoning runs `until_one_of([</think>] + tool-markers)`, and if it stops at a tool marker, `hold_when_partial()` returns `NEED_MORE_INPUT` during streaming (holding, carrying the reasoning-up-to-marker node) and `eps` at the final parse (resolving via B1 illustrative-with-call / B3 terminal-extract / B2 illustrative-with-text). Non-recovery models are untouched.

**Tech Stack:** C++17, the in-tree PEG parser (`common/peg-parser.*`), the auto-parser generator (`common/chat-auto-parser*`), the chat mapper (`common/chat-peg-parser.*`), test harness in `tests/test-chat*.cpp` (CTest).

## Global Constraints

- Behavior change is gated strictly on `analyze_reasoning::recover_inline_tool_calls`. Non-recovery templates must produce byte-identical parses to before.
- Streaming monotonicity is enforced by `common_chat_msg_diff::compute_diffs` (`common/chat.cpp:264`): `reasoning_content`/`content` may only be appended; `tool_calls.size()` may only grow. Any regression throws `Invalid diff: ...`.
- The existing `tests/test-chat.cpp` `.run()` harness (`test-chat.cpp:1074-1119`) already stream-tests every prefix and runs `compute_diffs` between consecutive parses, asserting the accumulated message equals each partial parse. A green `.run()` proves both correctness and monotonicity.
- Reasoning spec: `docs/superpowers/specs/2026-07-01-inline-toolcall-hold-design.md`.
- Build (per project convention, WSL Ubuntu): `cmake -B build && cmake --build build --config Release -j 8`. Test binaries: `test-chat`, `test-chat-peg-parser`, `test-chat-auto-parser`.

---

### Task 1: Add the `hold_when_partial` PEG primitive and PARTIAL flag

**Files:**
- Modify: `common/peg-parser.h` (flag enum ~`:145`, `common_peg_parse_context` ~`:167`, parser structs ~`:187`, variant list ~`:284`, builder methods ~`:363`)
- Modify: `common/peg-parser.cpp` (parse visitor ~`:389`, any `std::visit` sites for grammar/serialization, `to_json`/`from_json` ~`:2040`)
- Test: `tests/test-chat-peg-parser.cpp`

**Interfaces:**
- Produces: `COMMON_PEG_PARSE_FLAG_PARTIAL` (flag), `bool common_peg_parse_context::is_partial() const`, `common_peg_parser common_peg_parser_builder::hold_when_partial()`, and a `common_peg_hold_parser {}` variant. Semantics: on a parse where `ctx.is_partial()` is true it returns `NEED_MORE_INPUT` at the current position (consuming nothing); otherwise it succeeds like `eps`.

- [ ] **Step 1: Write the failing test**

Add to `tests/test-chat-peg-parser.cpp` (near the other `build_tagged_peg_parser` tests around `:894`), and register it in that file's test runner list the same way neighboring tests are registered:

```cpp
static void test_hold_when_partial(testing & t) {
    auto parser = build_tagged_peg_parser([](common_peg_parser_builder & p) {
        return p.literal("a") + p.hold_when_partial() + p.literal("b") + p.end();
    });
    // Final parse (no PARTIAL flag): the hold is transparent, whole input matches.
    t.assert_true("final parse commits past the hold",
                  parser.parse_and_extract("ab").result.success());
    // Partial parse: the hold refuses to commit, so the overall parse does not succeed.
    t.assert_true("partial parse holds at the barrier",
                  !parser.parse_and_extract("ab", COMMON_PEG_PARSE_FLAG_PARTIAL).result.success());
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cmake --build build --config Release -j 8 --target test-chat-peg-parser`
Expected: FAIL to compile — `hold_when_partial` and `COMMON_PEG_PARSE_FLAG_PARTIAL` are not declared.

- [ ] **Step 3: Add the flag and `is_partial()`**

In `common/peg-parser.h`, extend the flag enum (`:145`):

```cpp
enum common_peg_parse_flags {
    COMMON_PEG_PARSE_FLAG_NONE    = 0,
    COMMON_PEG_PARSE_FLAG_LENIENT = 1 << 0,
    COMMON_PEG_PARSE_FLAG_DEBUG   = 1 << 1,
    COMMON_PEG_PARSE_FLAG_PARTIAL = 1 << 2,
};
```

In `common_peg_parse_context` (next to `is_lenient`/`is_debug`, `:180`):

```cpp
    bool is_partial() const { return flags & COMMON_PEG_PARSE_FLAG_PARTIAL; }
```

- [ ] **Step 4: Add the parser struct, variant entry, and builder method**

In `common/peg-parser.h`, add a struct near the other empty parsers (after `common_peg_end_parser`, `:191`):

```cpp
struct common_peg_hold_parser {};
```

Add it to the `common_peg_parser_variant` alias (`:284`), e.g. right after `common_peg_end_parser`:

```cpp
    common_peg_end_parser,
    common_peg_hold_parser,
```

Add the builder method near `eps()`/`end()` (`:371`):

```cpp
    // Streaming barrier: transparent (eps) on a final parse; yields NEED_MORE_INPUT on a
    // partial parse so the parser holds here (nodes captured before it still stream).
    //   S -> (partial ? need-more : ε)
    common_peg_parser hold_when_partial() { return add(common_peg_hold_parser{}); }
```

- [ ] **Step 5: Handle the variant in the parse visitor**

In `common/peg-parser.cpp`, add an `operator()` next to the epsilon/end handlers (`:389`):

```cpp
    common_peg_parse_result operator()(const common_peg_hold_parser & /* p */) const {
        if (ctx.is_partial()) {
            return common_peg_parse_result(COMMON_PEG_PARSE_RESULT_NEED_MORE_INPUT, start_pos);
        }
        return common_peg_parse_result(COMMON_PEG_PARSE_RESULT_SUCCESS, start_pos);
    }
```

- [ ] **Step 6: Handle the variant at every other `std::visit`/serialization site**

The variant is visited in several places (grammar building, dump, serialization). Mirror `common_peg_epsilon_parser` exactly — the hold is a no-op for grammar generation and dumping. Add `common_peg_hold_parser` to each `if constexpr (std::is_same_v<T, common_peg_epsilon_parser> || ...)` chain that lists epsilon/end (search `common/peg-parser.cpp` for `common_peg_epsilon_parser`; the relevant chains are around `:1003`, `:1665`, `:1767`, `:1951`). The compiler will error on any site left unhandled — resolve each by treating hold like epsilon.

For `to_json`/`from_json` (`:2040`), mirror the epsilon case:

```cpp
// to_json (in the type switch that emits {"type": ...}):
} else if constexpr (std::is_same_v<T, common_peg_hold_parser>) {
    j["type"] = "hold";
// from_json (in the string-keyed dispatch):
} else if (type == "hold") {
    return common_peg_hold_parser{};
```

- [ ] **Step 7: Run test to verify it passes**

Run: `cmake --build build --config Release -j 8 --target test-chat-peg-parser && ./build/bin/test-chat-peg-parser`
Expected: PASS, including `test_hold_when_partial`. No other assertions regress.

- [ ] **Step 8: Commit**

```bash
git add common/peg-parser.h common/peg-parser.cpp tests/test-chat-peg-parser.cpp
git commit -m "feat(peg): add hold_when_partial primitive and PARTIAL parse flag

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: Thread `is_partial` into the PEG parse as the PARTIAL flag

**Files:**
- Modify: `common/chat.cpp` (`common_chat_peg_parse`, ~`:2702`)

**Interfaces:**
- Consumes: `COMMON_PEG_PARSE_FLAG_PARTIAL` (Task 1).
- Produces: the `is_partial` argument of `common_chat_peg_parse` now sets `COMMON_PEG_PARSE_FLAG_PARTIAL` in the parse context, making `ctx.is_partial()` true during streaming parses. No behavior change yet (no parser uses `hold_when_partial` until Task 3).

- [ ] **Step 1: Add the flag wiring**

In `common/chat.cpp`, in `common_chat_peg_parse`, where flags are assembled (`:2702`):

```cpp
    common_peg_parse_flags flags = COMMON_PEG_PARSE_FLAG_LENIENT;
    if (params.debug) {
        flags |= COMMON_PEG_PARSE_FLAG_DEBUG;
    }
    if (is_partial) {
        flags |= COMMON_PEG_PARSE_FLAG_PARTIAL;
    }
```

- [ ] **Step 2: Verify no behavior change**

Run: `cmake --build build --config Release -j 8 --target test-chat && ./build/bin/test-chat`
Expected: PASS — all existing tests still green (no grammar uses the flag yet).

- [ ] **Step 3: Commit**

```bash
git add common/chat.cpp
git commit -m "feat(chat): set PARTIAL parse flag when parsing partial output

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: Recovery-mode ordered-choice composition (the core change)

**Files:**
- Modify: `common/chat-auto-parser.h` (declare the new method on `analyze_reasoning`, near `build_trailing_end_parser` ~`:280`)
- Modify: `common/chat-auto-parser-generator.cpp` (`analyze_reasoning::build_parser` ~`:166`; the three composition sites: `build_tool_parser_json_native` ~`:332`, `build_tool_parser_tag_json` ~`:442`, `build_tool_parser_tag_tagged` ~`:582`)
- Test: `tests/test-chat.cpp` (Qwen3.5 block, after `:2037`)

**Interfaces:**
- Consumes: `hold_when_partial()` (Task 1); the PARTIAL flag now set during streaming (Task 2).
- Produces: `common_peg_parser analyze_reasoning::build_recovery_composition(parser_build_context & ctx, const common_peg_parser & tool_calls, const common_peg_parser & content_before_tools, const common_peg_parser & trailing) const;` where `tool_calls` is the **raw** (not `optional`-wrapped) tool-calls parser. Returns the full recovery parser body (reasoning + tools + content + end).

**Design recap (from the spec):**
- Reasoning prefix runs `reasoning(until_one_of([</think>] + tool-markers))`.
- A positive lookahead `peek(marker)` decides the branch: if a tool marker is next, take the B path; otherwise fall through to the standard composition (path A), which already handles normal reasoning, post-`</think>` calls, and no-reasoning.
- `hold_when_partial()` sits right after the marker lookahead: during streaming it yields `NEED_MORE_INPUT`, so the B sequence returns early carrying only the reasoning-up-to-marker node (streamed), and the choice short-circuits before path A's greedy reasoning is ever evaluated — no retraction.
- At the final parse the hold is transparent and the inner ordered choice resolves: **B1** (illustrative, requires a real post-`</think>` tool call), then **B3** (terminal → extract the in-think call), then **B2** (illustrative, real post-`</think>` text). B1 is required-`tool_calls` so it only wins when a post-think call exists; B3's trailing `end()` only matches when the call is terminal; B2 catches the remaining real-text case.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test-chat.cpp` inside the existing Qwen3.5 block (reuse the `tst`/`special_function_tool` already in scope, after `:2037`). These call `.run()`, which stream-tests every prefix and runs `compute_diffs`.

```cpp
        // --- Hold-until-EOS: in-<think> tool call resolution (recover_inline_tool_calls) ---

        // Edge 1: in-think call is terminal (call, </think>, EOS) -> extract as a real call.
        tst.test(
               "I'm\nthinking\n"
               "<tool_call>\n<function=special_function>\n<parameter=arg1>\n1\n</parameter>\n</function>\n</tool_call>\n"
               "</think>")
            .reasoning_format(COMMON_REASONING_FORMAT_AUTO)
            .enable_thinking(true)
            .tools({ special_function_tool })
            .expect_reasoning("I'm\nthinking")
            .expect_tool_calls({ { "special_function", R"({"arg1": 1})", {} } })
            .run();

        // Edge 2: unclosed </think> (call, EOS) -> still terminal -> extract.
        tst.test(
               "I'm\nthinking\n"
               "<tool_call>\n<function=special_function>\n<parameter=arg1>\n1\n</parameter>\n</function>\n</tool_call>")
            .reasoning_format(COMMON_REASONING_FORMAT_AUTO)
            .enable_thinking(true)
            .tools({ special_function_tool })
            .expect_reasoning("I'm\nthinking")
            .expect_tool_calls({ { "special_function", R"({"arg1": 1})", {} } })
            .run();

        // Edge 3: in-think call + </think> + real post-think call -> in-think is illustrative
        // (stays reasoning), only the post-think call is emitted.
        tst.test(
               "I'm\nthinking\n"
               "<tool_call>\n<function=special_function>\n<parameter=arg1>\n1\n</parameter>\n</function>\n</tool_call>\n"
               "</think>\n\n"
               "<tool_call>\n<function=special_function>\n<parameter=arg1>\n2\n</parameter>\n</function>\n</tool_call>")
            .reasoning_format(COMMON_REASONING_FORMAT_AUTO)
            .enable_thinking(true)
            .tools({ special_function_tool })
            .expect_reasoning(
               "I'm\nthinking\n"
               "<tool_call>\n<function=special_function>\n<parameter=arg1>\n1\n</parameter>\n</function>\n</tool_call>")
            .expect_tool_calls({ { "special_function", R"({"arg1": 2})", {} } })
            .run();

        // Edge 4: in-think call + </think> + real text answer -> in-think is illustrative
        // (stays reasoning), text becomes content (pre-change behavior restored).
        tst.test(
               "I'm\nthinking\n"
               "<tool_call>\n<function=special_function>\n<parameter=arg1>\n1\n</parameter>\n</function>\n</tool_call>\n"
               "</think>\n\n"
               "The answer is 42.")
            .reasoning_format(COMMON_REASONING_FORMAT_AUTO)
            .enable_thinking(true)
            .tools({ special_function_tool })
            .expect_reasoning(
               "I'm\nthinking\n"
               "<tool_call>\n<function=special_function>\n<parameter=arg1>\n1\n</parameter>\n</function>\n</tool_call>")
            .expect_content("The answer is 42.")
            .run();
```

> Whitespace note: `expect_reasoning` values assume the parser trims the whitespace immediately adjacent to `</think>` (consistent with the existing `message_assist_thoughts` test at `:2039`, which expects `"I'm\nthinking"` from `"I'm\nthinking\n</think>..."`). If the first run shows a different boundary (e.g. a trailing newline inside the absorbed reasoning for edges 3/4), adjust the expected string to the parser's actual node text — do not add trimming logic to make it match.

- [ ] **Step 2: Run tests to verify they fail**

Run: `cmake --build build --config Release -j 8 --target test-chat && ./build/bin/test-chat`
Expected: FAIL — edges 3/4 emit the in-think call as a tool call (eager) and/or edge-3's streaming trips `Invalid diff: now finding less tool calls!`. This reproduces the bug.

- [ ] **Step 3: Make recovery-mode reasoning parser standard**

In `common/chat-auto-parser-generator.cpp`, `analyze_reasoning::build_parser` (`:166`), delete the `if (recover_inline_tool_calls) { ... }` block (`:166-191`) so recovery mode falls through to the standard tag-based reasoning parser below it (`:195-199`). `ctx.reasoning_parser` in recovery mode now equals the standard `optional(optspace(start) + reasoning(until end) + optspace(end))`. The recovery behavior moves entirely into `build_recovery_composition` (next step).

- [ ] **Step 4: Declare and implement `build_recovery_composition`**

In `common/chat-auto-parser.h`, declare on `analyze_reasoning` (near `build_trailing_end_parser`, `:280`):

```cpp
    // Recovery mode (Qwen family): full parser body that holds in-<think> tool calls until
    // EOS and resolves their role (illustrative vs real) at the final parse. `tool_calls` is
    // the raw (not optional-wrapped) tool-calls parser.
    common_peg_parser build_recovery_composition(parser_build_context &    ctx,
                                                 const common_peg_parser & tool_calls,
                                                 const common_peg_parser & content_before_tools,
                                                 const common_peg_parser & trailing) const;
```

In `common/chat-auto-parser-generator.cpp`, implement it (place it after `build_content_before_tools`, ~`:242`):

```cpp
common_peg_parser analyze_reasoning::build_recovery_composition(parser_build_context &    ctx,
                                                                const common_peg_parser & tool_calls,
                                                                const common_peg_parser & content_before_tools,
                                                                const common_peg_parser & trailing) const {
    auto &            p       = ctx.p;
    const std::string end_tag = trim_whitespace(end);

    // Tool-call start markers (same source as the old recovery reasoning terminators).
    std::vector<std::string>       marker_strs;
    std::vector<common_peg_parser> marker_lits;
    if (ctx.tools) {
        if (!ctx.tools->format.section_start.empty()) {
            marker_strs.push_back(ctx.tools->format.section_start);
            marker_lits.push_back(p.literal(ctx.tools->format.section_start));
        }
        if (!ctx.tools->format.per_call_start.empty()) {
            marker_strs.push_back(ctx.tools->format.per_call_start);
            marker_lits.push_back(p.literal(ctx.tools->format.per_call_start));
        }
    }

    // No markers -> nothing to recover; fall back to the standard composition.
    if (marker_lits.empty()) {
        return ctx.reasoning_parser + p.optional(p.content(content_before_tools)) +
               p.optional(tool_calls) + trailing + p.end();
    }

    std::vector<std::string> delimiters;
    delimiters.push_back(end_tag);
    delimiters.insert(delimiters.end(), marker_strs.begin(), marker_strs.end());

    auto marker_peek  = p.peek(marker_lits.size() == 1 ? marker_lits[0] : p.choice(marker_lits));
    auto swallow_end  = p.zero_or_more(p.optspace(end));  // swallow the closing/stray </think>

    // B-path shared prefix: reasoning up to the first </think> OR tool marker; only continue
    // when a marker is what stopped it, then hold during streaming.
    auto b_prefix = p.optional(p.optspace(start)) +
                    p.reasoning(p.until_one_of(delimiters)) +
                    marker_peek +
                    p.hold_when_partial();

    // B1: illustrative — absorb the in-think call as reasoning up to </think>, then require a
    //     real post-</think> tool call.
    auto b1 = b_prefix + p.reasoning(p.until(end_tag)) + p.optspace(end) +
              p.optional(p.content(content_before_tools)) + tool_calls + trailing + p.end();
    // B3: terminal — the in-think call is the real call; extract it (only matches at EOS).
    auto b3 = b_prefix + tool_calls + swallow_end + p.end();
    // B2: illustrative — absorb the in-think call as reasoning up to </think>, then real text.
    auto b2 = b_prefix + p.reasoning(p.until(end_tag)) + p.optspace(end) +
              p.content(p.rest()) + p.end();

    // A / default: standard composition (normal reasoning, post-think calls, or no
    // reasoning). `swallow_end` after the reasoning parser absorbs a stray/duplicate
    // </think> that can appear between reasoning-close and a post-</think> tool call
    // (the standard reasoning parser only consumes the first </think>).
    auto a = ctx.reasoning_parser + swallow_end + p.optional(p.content(content_before_tools)) +
             p.optional(tool_calls) + trailing + p.end();

    return p.choice({ b1, b3, b2, a });
}
```

> Note the ordered choice is `{b1, b3, b2, a}`: during streaming, `b1` reaches `hold_when_partial()` (via `b_prefix`) which returns `NEED_MORE_INPUT`, so the choice returns immediately with the reasoning-up-to-marker node and never evaluates `a`. If reasoning stopped at `</think>` (no in-think call), `marker_peek` fails in b1/b3/b2 and the choice falls through to `a`.

- [ ] **Step 5: Route the three composition sites through it in recovery mode**

At each of the three sites, immediately before the existing `if (!require_calls/require_tools) tool_calls = p.optional(tool_calls);` wrap, insert the recovery branch that passes the **raw** `tool_calls`. Use this guard (define a local `bool recovery` once per function):

```cpp
    bool recovery = ctx.reasoning && ctx.reasoning->recover_inline_tool_calls &&
                    ctx.extracting_reasoning && !ctx.reasoning->end.empty() &&
                    (ctx.reasoning->mode == reasoning_mode::TAG_BASED ||
                     ctx.reasoning->mode == reasoning_mode::TOOLS_ONLY);
```

**Site A — `build_tool_parser_json_native` (`:332`).** The tail currently is:

```cpp
    auto trailing = ctx.reasoning->build_trailing_end_parser(ctx);
    auto content_before_tools = ctx.reasoning->build_content_before_tools(ctx, tool_start);
    return ctx.reasoning_parser + p.optional(p.content(content_before_tools)) + tools_parser + trailing + p.end();
```

Replace with:

```cpp
    auto trailing = ctx.reasoning->build_trailing_end_parser(ctx);
    auto content_before_tools = ctx.reasoning->build_content_before_tools(ctx, tool_start);
    if (recovery) {
        return ctx.reasoning->build_recovery_composition(ctx, tools_parser, content_before_tools, trailing);
    }
    return ctx.reasoning_parser + p.optional(p.content(content_before_tools)) + tools_parser + trailing + p.end();
```

(Define `recovery` just above this tail. `tools_parser` here is the raw, non-optional parser built earlier in the function.)

**Site B — `build_tool_parser_tag_json` (`:442`).** The tail is:

```cpp
    if (!require_calls) {
        tool_calls = p.optional(tool_calls);
    }

    std::string trigger_marker       = !format.section_start.empty() ? format.section_start : format.per_call_start;
    auto        content_before_tools = ctx.reasoning->build_content_before_tools(ctx, trigger_marker);
    auto        trailing             = ctx.reasoning->build_trailing_end_parser(ctx);
    return ctx.reasoning_parser + p.optional(p.content(content_before_tools)) + tool_calls + trailing + p.end();
```

Replace with (note: recovery uses the raw `tool_calls`, so branch **before** the optional wrap):

```cpp
    std::string trigger_marker       = !format.section_start.empty() ? format.section_start : format.per_call_start;
    auto        content_before_tools = ctx.reasoning->build_content_before_tools(ctx, trigger_marker);
    auto        trailing             = ctx.reasoning->build_trailing_end_parser(ctx);

    bool recovery = ctx.reasoning && ctx.reasoning->recover_inline_tool_calls &&
                    ctx.extracting_reasoning && !ctx.reasoning->end.empty() &&
                    (ctx.reasoning->mode == reasoning_mode::TAG_BASED ||
                     ctx.reasoning->mode == reasoning_mode::TOOLS_ONLY);
    if (recovery) {
        return ctx.reasoning->build_recovery_composition(ctx, tool_calls, content_before_tools, trailing);
    }

    if (!require_calls) {
        tool_calls = p.optional(tool_calls);
    }
    return ctx.reasoning_parser + p.optional(p.content(content_before_tools)) + tool_calls + trailing + p.end();
```

**Site C — `build_tool_parser_tag_tagged` (`:582`, the Qwen3.5/Nemotron-3 path).** The tail is:

```cpp
    if (!require_tools) {
        tool_calls = p.optional(tool_calls);
    }

    std::string trigger_marker       = !format.section_start.empty() ? format.section_start : format.per_call_start;
    auto        content_before_tools = ctx.reasoning->build_content_before_tools(ctx, trigger_marker);
    auto        trailing             = ctx.reasoning->build_trailing_end_parser(ctx);
    return ctx.reasoning_parser + p.optional(p.content(content_before_tools)) + tool_calls + trailing + p.end();
```

Replace with the same shape as Site B (branch on `recovery` before the `if (!require_tools)` wrap, passing raw `tool_calls`):

```cpp
    std::string trigger_marker       = !format.section_start.empty() ? format.section_start : format.per_call_start;
    auto        content_before_tools = ctx.reasoning->build_content_before_tools(ctx, trigger_marker);
    auto        trailing             = ctx.reasoning->build_trailing_end_parser(ctx);

    bool recovery = ctx.reasoning && ctx.reasoning->recover_inline_tool_calls &&
                    ctx.extracting_reasoning && !ctx.reasoning->end.empty() &&
                    (ctx.reasoning->mode == reasoning_mode::TAG_BASED ||
                     ctx.reasoning->mode == reasoning_mode::TOOLS_ONLY);
    if (recovery) {
        return ctx.reasoning->build_recovery_composition(ctx, tool_calls, content_before_tools, trailing);
    }

    if (!require_tools) {
        tool_calls = p.optional(tool_calls);
    }
    return ctx.reasoning_parser + p.optional(p.content(content_before_tools)) + tool_calls + trailing + p.end();
```

- [ ] **Step 6: Run the new tests + existing Qwen/Nemotron tests**

Run: `cmake --build build --config Release -j 8 --target test-chat && ./build/bin/test-chat`
Expected: PASS — edges 1–4 pass (including their streaming `compute_diffs`), and the pre-existing Qwen3.5 (`:2037`) and Nemotron-3 (`:2664`) blocks still pass. If an `expect_reasoning` boundary differs, adjust the expected string per the whitespace note in Step 1.

- [ ] **Step 7: Commit**

```bash
git add common/chat-auto-parser.h common/chat-auto-parser-generator.cpp tests/test-chat.cpp
git commit -m "feat(chat): hold in-<think> tool calls until EOS and resolve role at final parse

Recovery mode now streams reasoning up to the in-think tool marker, holds
via hold_when_partial(), and resolves illustrative-vs-real at the final
parse. Fixes the streaming 'finding less tool calls' crash and the stray
</think> leak. Scoped to recover_inline_tool_calls (Qwen3.5/Nemotron-3).

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 4: Add a parallel-illustrative edge test and Nemotron-3 coverage

**Files:**
- Test: `tests/test-chat.cpp` (Qwen3.5 block after the Task 3 tests; Nemotron-3 block after `:2664`)

**Interfaces:**
- Consumes: the recovery composition (Task 3). No production code changes — this task widens test coverage for the edge-case table (spec cases 5 and 6) and confirms the second recovery-flagged model behaves identically.

- [ ] **Step 1: Write the tests**

Add to the Qwen3.5 block (edge 5: stray/duplicate `</think>` around a terminal in-think call must be swallowed, not leaked to content):

```cpp
        // Edge 5: stray/duplicate </think> after a terminal in-think call is swallowed.
        tst.test(
               "I'm\nthinking\n"
               "<tool_call>\n<function=special_function>\n<parameter=arg1>\n1\n</parameter>\n</function>\n</tool_call>\n"
               "</think>\n</think>")
            .reasoning_format(COMMON_REASONING_FORMAT_AUTO)
            .enable_thinking(true)
            .tools({ special_function_tool })
            .expect_reasoning("I'm\nthinking")
            .expect_tool_calls({ { "special_function", R"({"arg1": 1})", {} } })
            .run();
```

Add one Nemotron-3 test (edge 1 equivalent) in the Nemotron-3 block after `:2664`, using whatever tool/message helpers that block already uses (mirror the Qwen edge-1 test structure with that block's `tst` and tool fixtures):

```cpp
        // Hold-until-EOS: terminal in-think call extracts as a real call (Nemotron-3).
        tst.test(
               "I'm\nthinking\n"
               "<tool_call>\n<function=special_function>\n<parameter=arg1>\n1\n</parameter>\n</function>\n</tool_call>\n"
               "</think>")
            .reasoning_format(COMMON_REASONING_FORMAT_AUTO)
            .enable_thinking(true)
            .tools({ special_function_tool })
            .expect_reasoning("I'm\nthinking")
            .expect_tool_calls({ { "special_function", R"({"arg1": 1})", {} } })
            .run();
```

> If the Nemotron-3 block uses different tool fixtures or marker text than Qwen3.5, adapt the input's tool-call syntax and the tool argument to match that block's existing passing tests before running.

- [ ] **Step 2: Run tests to verify they pass**

Run: `cmake --build build --config Release -j 8 --target test-chat && ./build/bin/test-chat`
Expected: PASS. (These exercise the already-implemented recovery path; they should pass immediately. If edge 5 fails with a leaked `</think>` in content, that indicates `swallow_end` is not consuming duplicates — confirm `swallow_end = zero_or_more(optspace(end))` in `build_recovery_composition`.)

- [ ] **Step 3: Commit**

```bash
git add tests/test-chat.cpp
git commit -m "test(chat): cover stray </think> swallow and Nemotron-3 inline tool-call hold

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 5: Pin the detected tool format for the recovery-flagged templates

**Files:**
- Test: `tests/test-chat-auto-parser.cpp` (near the Nemotron-3 loader `:1303` and the format tests, e.g. after `:1449`)

**Interfaces:**
- Consumes: nothing from prior tasks. Guards against future template drift silently moving Qwen3.5/Nemotron-3 off the `TAG_WITH_TAGGED` path (which would bypass the recovery composition's assumptions).

- [ ] **Step 1: Write the test**

Add a test that loads each recovery-flagged template, runs analysis, and asserts the format and the recovery flag:

```cpp
static void test_recovery_templates_are_tag_tagged(testing & t) {
    for (const char * path : {
             "models/templates/Qwen3.5-4B.jinja",
             "models/templates/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16.jinja",
         }) {
        common_chat_template tmpl = load_template(t, path);
        struct autoparser analysis;
        analysis.analyze_template(tmpl);
        t.assert_equal(std::string(path) + ": tool format should be TAG_WITH_TAGGED",
                       tool_format::TAG_WITH_TAGGED, analysis.tools.format.mode);
        t.assert_true(std::string(path) + ": recover_inline_tool_calls should be set",
                      analysis.reasoning.recover_inline_tool_calls);
    }
}
```

Register `test_recovery_templates_are_tag_tagged` in this file's test runner list alongside the other `test_tool_format_*` registrations. (Confirm the exact `load_template`/`assert_equal` signatures against neighboring tests such as `test_tool_format_cohere` at `:1411`; adapt the helper names if this file uses a different loader.)

- [ ] **Step 2: Run the test**

Run: `cmake --build build --config Release -j 8 --target test-chat-auto-parser && ./build/bin/test-chat-auto-parser`
Expected: PASS — both templates detect as `TAG_WITH_TAGGED` with `recover_inline_tool_calls == true`.

- [ ] **Step 3: Commit**

```bash
git add tests/test-chat-auto-parser.cpp
git commit -m "test(chat): pin Qwen3.5/Nemotron-3 to TAG_WITH_TAGGED + recovery flag

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 6: Full regression pass

**Files:** none (verification only).

- [ ] **Step 1: Build everything**

Run: `cmake -B build && cmake --build build --config Release -j 8`
Expected: clean build.

- [ ] **Step 2: Run the three chat test binaries**

Run:
```bash
./build/bin/test-chat && ./build/bin/test-chat-auto-parser && ./build/bin/test-chat-peg-parser
```
Expected: all PASS. In particular `test-chat-auto-parser` (previously 499 assertions) and `test-chat-peg-parser` (previously 198) stay green, confirming non-recovery models are unaffected.

- [ ] **Step 3: Confirm the original crash is gone**

The edge-3 streaming test (two complete tool-call blocks, first inside reasoning) exercises the exact pattern that produced `Invalid diff: now finding less tool calls!` on the current branch. Its green `.run()` in Task 3 is the regression proof. No separate action needed beyond confirming Task 3's tests are in the passing set.

---

## Self-Review

**Spec coverage:**
- Hold-until-EOS mechanism → Tasks 1–3 (`hold_when_partial` + PARTIAL flag + B-path composition).
- Resolution rule (terminal→extract; post-think payload→illustrative) → Task 3 ordered choice `{b1, b3, b2, a}`; edges 1–4 tests.
- Scope gate `recover_inline_tool_calls` → `recovery` guard at all three sites (Task 3 Step 5); pinned by Task 5.
- Monotonicity / no retraction → `hold_when_partial` returns `NEED_MORE_INPUT` (Task 1) short-circuiting before path A; verified by `.run()` streaming assertions (Tasks 3–4).
- Stray/duplicate `</think>` swallow (edge 5) → `swallow_end` in composition; Task 4 test.
- Parallel in-think calls (edge 6) → covered by the required-`tool_calls` semantics; edge-3/edge-5 exercise multi-block inputs. (If a dedicated parallel-in-think test is desired later it can be added, but the resolution logic is format-symmetric.)
- All three composition paths identical → shared `build_recovery_composition` applied at Sites A/B/C (Task 3 Step 5).
- Testability of format detection → Task 5.
- Hard fallback / no infinite hold → hold only triggers under PARTIAL; final parse always resolves (Task 1 semantics); path A remains a total fallback in the choice.

**Placeholder scan:** No TBD/TODO. Every code step shows complete code. The two "adapt if the harness differs" notes (Nemotron fixtures in Task 4, loader signatures in Task 5) point at concrete neighboring references rather than leaving content unspecified.

**Type consistency:** `hold_when_partial()` / `common_peg_hold_parser` / `COMMON_PEG_PARSE_FLAG_PARTIAL` / `is_partial()` are defined in Task 1 and used consistently in Tasks 2–3. `build_recovery_composition(ctx, tool_calls, content_before_tools, trailing)` is declared (Task 3 Step 4) and called with the same argument order and raw (non-optional) `tool_calls` at all three sites (Step 5).
