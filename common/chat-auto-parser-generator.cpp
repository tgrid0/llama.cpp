#include "chat-auto-parser-helpers.h"
#include "chat-auto-parser.h"
#include "chat-peg-parser.h"
#include "chat.h"
#include "common.h"
#include "json-schema-to-grammar.h"
#include "log.h"
#include "nlohmann/json.hpp"
#include "peg-parser.h"

#include <stdexcept>
#include <string>

using json = nlohmann::ordered_json;

// Helper to iterate over tools/functions
static void foreach_function(const json & tools, const std::function<void(const json &)> & fn) {
    for (const auto & tool : tools) {
        if (!tool.contains("type") || tool.at("type") != "function" || !tool.contains("function")) {
            continue;
        }
        fn(tool);
    }
}

namespace autoparser {

parser_build_context::parser_build_context(common_chat_peg_builder & p, const generation_params & inputs) :
    p(p),
    inputs(inputs),
    reasoning_parser(p.eps()) {}

common_chat_params peg_generator::generate_parser(const common_chat_template &    tmpl,
                                                  const struct generation_params & inputs) {
    // Run differential analysis to extract template structure
    struct autoparser autoparser;
    autoparser.analyze_template(tmpl);
    return generate_parser(tmpl, inputs, autoparser);
}

common_chat_params peg_generator::generate_parser(const common_chat_template &    tmpl,
                                                  const struct generation_params & inputs,
                                                  const autoparser &              autoparser) {
    // Create the result structure
    common_chat_params data;
    data.prompt            = common_chat_template_direct_apply(tmpl, inputs);
    data.generation_prompt = common_chat_template_generation_prompt(tmpl, inputs);
    data.format            = COMMON_CHAT_FORMAT_PEG_NATIVE;
    data.preserved_tokens  = autoparser.preserved_tokens;
    data.additional_stops.insert(data.additional_stops.end(),
        autoparser.additional_stops.begin(), autoparser.additional_stops.end());

    std::string parser_generation_prompt = data.generation_prompt;

    if (inputs.continue_final_message != COMMON_CHAT_CONTINUATION_NONE && !inputs.continue_msg.empty()) {
        // Build up generation prompt manually
        const auto & msg = inputs.continue_msg;

        if (!autoparser.reasoning.start.empty()) {
            data.generation_prompt = data.generation_prompt.substr(0, data.generation_prompt.find(autoparser.reasoning.start));
            data.generation_prompt += autoparser.reasoning.start + msg.reasoning_content;
            if (inputs.continue_final_message == COMMON_CHAT_CONTINUATION_CONTENT) {
                data.generation_prompt += autoparser.reasoning.end;
            }
        }

        if (inputs.continue_final_message == COMMON_CHAT_CONTINUATION_CONTENT) {
            data.generation_prompt += msg.render_content();
        }

        data.prompt += data.generation_prompt;
    }

    auto parser = autoparser.build_parser(inputs, parser_generation_prompt);
    data.parser = parser.save();

    // Build grammar if tools are present
    bool has_tools =
        autoparser.tools.format.mode != tool_format::NONE && inputs.tools.is_array() && !inputs.tools.empty();
    std::string trigger_marker = !autoparser.tools.format.section_start.empty() ? autoparser.tools.format.section_start :
                                                                                  autoparser.tools.format.per_call_start;

    bool has_response_format = !inputs.json_schema.empty() && inputs.json_schema.is_object();
    bool include_grammar = has_response_format || (has_tools &&
            ((inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_AUTO && !trigger_marker.empty()) ||
              inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_REQUIRED));

    if (include_grammar) {
        data.grammar_lazy = !has_response_format && inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_AUTO;
        data.grammar      = build_grammar([&](const common_grammar_builder & builder) {
            foreach_function(inputs.tools, [&](const json & tool) {
                const auto & function = tool.at("function");
                auto         schema   = function.contains("parameters") ? function.at("parameters") : json::object();
                builder.resolve_refs(schema);
            });
            if (has_response_format) {
                auto schema = inputs.json_schema;
                builder.resolve_refs(schema);
            }
            parser.build_grammar(builder, data.grammar_lazy);
        });

        // Set grammar triggers based on tool section markers (fall back to per-call markers)
        if (data.grammar_lazy) {
            data.grammar_triggers = {
                { COMMON_GRAMMAR_TRIGGER_TYPE_WORD, trigger_marker }
            };
            if (autoparser.tools.format.openai_wrapper_trigger) {
                // model emits the OpenAI function wrapper, trigger on it
                data.grammar_triggers.push_back({ COMMON_GRAMMAR_TRIGGER_TYPE_WORD, "{\"type\": \"function\"," });
            }
        }
    }

    return data;
}

common_peg_arena autoparser::build_parser(const generation_params & inputs, const std::string & generation_prompt) const {
    if (!analysis_complete) {
        throw std::invalid_argument("Cannot call build_parser on autoparser without performing analysis first, call analyze_template(...)");
    }
    return build_chat_peg_parser([&](common_chat_peg_builder & p) {
        parser_build_context ctx(p, inputs);
        bool                 extract_reasoning = inputs.reasoning_format != COMMON_REASONING_FORMAT_NONE;

        ctx.extracting_reasoning = extract_reasoning && reasoning.mode != reasoning_mode::NONE;
        ctx.content              = &content;
        ctx.reasoning            = &reasoning;
        ctx.tools                = &tools;

        // Build reasoning parser
        ctx.reasoning_parser = reasoning.build_parser(ctx);

        auto parser = p.eps();

        bool has_tools           = inputs.tools.is_array() && !inputs.tools.empty();
        bool has_response_format = inputs.json_schema.is_object() && !inputs.json_schema.empty();
        bool pure_content        = reasoning.mode == reasoning_mode::NONE;

        if (has_response_format) {
            auto response_format = p.rule("response-format", p.content(p.schema(p.json(), "response-format-schema", inputs.json_schema)));
            parser = ctx.reasoning_parser + p.space() + p.choice({
                p.literal("```json") + p.space() + response_format + p.space() + p.literal("```"),
                p.space() + response_format  + p.space()
            }) + p.end();
            pure_content = false;
        } else if (has_tools && inputs.tool_choice != COMMON_CHAT_TOOL_CHOICE_NONE && jinja_caps.supports_tool_calls) {
            parser = tools.build_parser(ctx);
            pure_content = false;
        } else {
            parser = content.build_parser(ctx);
        }
        const std::string reasoning_start = trim_whitespace(reasoning.start);
        return pure_content ? p.prefix(generation_prompt, reasoning_start) + parser : p.prefix(generation_prompt, reasoning_start) << parser;
    });
}

common_peg_parser analyze_reasoning::build_parser(parser_build_context & ctx) const {
    auto & p = ctx.p;

    if (!ctx.extracting_reasoning) {
        return p.eps();
    }

    if (mode == reasoning_mode::TAG_BASED || mode == reasoning_mode::TOOLS_ONLY) {
        if (!end.empty()) {
            const std::string end_tag = trim_whitespace(end);

            // Standard tag-based: optional(<think>reasoning</think>). Reasoning runs up
            // to the closing tag, which is consumed.
            if (!start.empty()) {
                return p.optional(p.optspace(start) + p.reasoning(p.until(end_tag)) + p.optspace(end));
            }
            // Delimiter-style (empty start)
            return p.optional(p.reasoning(p.until(end_tag)) + p.optspace(end));
        }
    }

    return p.eps();
}

common_peg_parser analyze_reasoning::build_trailing_end_parser(parser_build_context & ctx) const {
    auto & p = ctx.p;
    if (!ctx.extracting_reasoning || end.empty() || !recover_inline_tool_calls) {
        return p.eps();
    }
    if (mode == reasoning_mode::TAG_BASED || mode == reasoning_mode::TOOLS_ONLY) {
        // Some models (e.g. Qwen) emit </think> after tool calls even though they already
        // closed the reasoning block before the tool call. Swallow any such stray or
        // duplicate closing tags so they are not emitted as content.
        return p.zero_or_more(p.optspace(end));
    }
    return p.eps();
}

common_peg_parser analyze_reasoning::build_content_before_tools(parser_build_context & ctx,
                                                                const std::string &    trigger_marker) const {
    auto & p = ctx.p;

    std::vector<std::string> delimiters;
    if (!trigger_marker.empty()) {
        delimiters.push_back(trigger_marker);
    }
    if (recover_inline_tool_calls && ctx.extracting_reasoning && !end.empty() &&
        (mode == reasoning_mode::TAG_BASED || mode == reasoning_mode::TOOLS_ONLY)) {
        // Stop content at a stray reasoning end tag so it does not leak into content;
        // the trailing parser then swallows it.
        delimiters.push_back(trim_whitespace(end));
    }

    if (delimiters.empty()) {
        return p.eps();
    }
    if (delimiters.size() == 1) {
        return p.until(delimiters[0]);
    }
    return p.until_one_of(delimiters);
}

common_peg_parser analyze_reasoning::build_recovery_content_loop(parser_build_context &           ctx,
                                                                 const common_peg_parser *        tool_calls,
                                                                 const std::vector<std::string> & marker_strs,
                                                                 const std::string &              rule_name) const {
    auto &            p       = ctx.p;
    const std::string end_tag = trim_whitespace(end);

    auto swallow_end = p.zero_or_more(p.optspace(end));

    // Content stops at any tool-call marker or at a reasoning end tag; the choice below
    // then decides whether the marker is real (terminal tool calls / trailing close) or
    // illustrative (consumed back into content, loop continues).
    std::vector<std::string> delimiters = marker_strs;
    delimiters.push_back(end_tag);

    return p.rule(rule_name, [&]() {
        std::vector<common_peg_parser> after;

        if (tool_calls != nullptr && !marker_strs.empty()) {
            std::vector<common_peg_parser> marker_lits;
            marker_lits.reserve(marker_strs.size());
            for (const auto & m : marker_strs) {
                marker_lits.push_back(p.literal(m));
            }
            auto marker_peek = p.peek(marker_lits.size() == 1 ? marker_lits[0] : p.choice(marker_lits));
            // Terminal: the marker starts real tool calls that run to EOS. The hold must
            // come before the tool-call parser: a partial parse short-circuits at the hold,
            // so no tool nodes are captured (and streamed) before EOS resolves whether the
            // block is a real call or an illustrative mention followed by more content.
            after.push_back(marker_peek + p.hold_when_partial() + *tool_calls + swallow_end + p.end());
        }

        // Trailing stray/duplicate </think> tags at EOS (also matches plain EOS).
        after.push_back(swallow_end + p.end());

        // Illustrative: the marker is plain text; consume it as content and continue.
        std::vector<common_peg_parser> as_content;
        as_content.reserve(delimiters.size());
        for (const auto & d : delimiters) {
            as_content.push_back(p.literal(d));
        }
        after.push_back(p.content(as_content.size() == 1 ? as_content[0] : p.choice(as_content)) + p.ref(rule_name));

        return p.content(p.until_one_of(delimiters)) + p.choice(after);
    });
}

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

    // Post-</think> body shared by the illustrative-B path and the standard A path:
    // content segments that may contain illustrative markers (kept as content),
    // terminated by real tool calls at EOS or by EOS with stray </think> swallowed.
    auto body = build_recovery_content_loop(ctx, &tool_calls, marker_strs, "recovery-content");

    // B3: terminal — the in-think call is the real call; extract it (only matches at EOS).
    auto b3 = b_prefix + tool_calls + swallow_end + p.end();
    // B12: illustrative — absorb the in-think call as reasoning up to </think>, then parse
    // the remainder like a normal post-reasoning body (content and/or real tool calls).
    auto b12 = b_prefix + p.reasoning(p.until(end_tag)) + p.optspace(end) + body;

    // A / default: standard composition (normal reasoning, post-think calls, or no
    // reasoning). `swallow_end` after the reasoning parser absorbs a stray/duplicate
    // </think> that can appear between reasoning-close and a post-</think> tool call
    // (the standard reasoning parser only consumes the first </think>).
    auto a = ctx.reasoning_parser + swallow_end + body;

    // B3 is tried first: in LENIENT mode, Literal at EOS returns NEED_MORE_INPUT (not FAIL),
    // so B12 placed first would return NMI and short-circuit the choice before B3 could be
    // tried. B3 uses end() which always returns hard FAIL (not NMI) when not at EOS, so B3
    // cleanly fails for non-terminal inputs and lets the choice fall through to B12/A.
    return p.choice({ b3, b12, a });
}

common_peg_parser analyze_content::build_parser(parser_build_context & ctx) const {
    auto & p = ctx.p;

    if (is_always_wrapped()) {
        if (ctx.extracting_reasoning) {
            return ctx.reasoning_parser + start + p.content(p.until(end)) + end + p.end();
        }
        return p.content(p.until(start)) + start + p.content(p.until(end)) + end + p.end();
    }
    if (ctx.reasoning && ctx.reasoning->recover_inline_tool_calls && ctx.extracting_reasoning &&
        !ctx.reasoning->end.empty()) {
        // Recovery mode (Qwen family): a reasoning end tag mentioned mid-content stays
        // content; only trailing stray/duplicate tags at EOS are swallowed.
        return ctx.reasoning_parser +
               ctx.reasoning->build_recovery_content_loop(ctx, nullptr, {}, "recovery-content");
    }
    return ctx.reasoning_parser + p.content(p.rest()) + p.end();
}

common_peg_parser analyze_content::build_optional_wrapped(parser_build_context & ctx) const {
    auto & p = ctx.p;

    if (is_always_wrapped()) {
        return p.optional(start + p.content(p.until(end)) + end);
    }
    return p.eps();
}

common_peg_parser analyze_tools::build_parser(parser_build_context & ctx) const {
    switch (format.mode) {
        case tool_format::JSON_NATIVE:
            return build_tool_parser_json_native(ctx);
        case tool_format::TAG_WITH_JSON:
            return build_tool_parser_tag_json(ctx);
        case tool_format::TAG_WITH_TAGGED:
            return build_tool_parser_tag_tagged(ctx);
        default:
            LOG_ERR("[ERROR] Template seems to support tool calls, but failed to determine tool format. Tool calling will not work properly. "
                "Check for a fixed template for your model in the models/templates directory of your llama.cpp installation or "
                "report an issue at https://github.com/ggml-org/llama.cpp/issues\n");
            return ctx.p.eps();
    }
}

common_peg_parser analyze_tools::build_tool_parser_json_native(parser_build_context & ctx) const {
    auto &       p           = ctx.p;
    const auto & inputs      = ctx.inputs;

    // Build effective field names with dot notation if function_field is set
    std::string name_field = format.name_field;
    std::string args_field = format.args_field;

    if (!format.function_field.empty() && format.function_field != "function" &&
        name_field.find('.') == std::string::npos) {
        name_field = format.function_field + "." + name_field;
        args_field = format.function_field + "." + args_field;
    }

    auto tools_parser = p.eps();
    if (format.section_start.empty() && !format.per_call_start.empty()) {
        auto single_tool_parser = p.standard_json_tools(
            format.per_call_start, format.per_call_end, inputs.tools, inputs.parallel_tool_calls,
            inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_REQUIRED, name_field, args_field, format.tools_array_wrapped,
            format.fun_name_is_key, format.id_field, format.gen_id_field, format.parameter_order, format.openai_wrapper_trigger);
        tools_parser = p.trigger_rule("tool-calls", p.one_or_more(single_tool_parser + p.space()));
    } else {
        tools_parser = p.standard_json_tools(
            format.section_start, format.section_end, inputs.tools, inputs.parallel_tool_calls,
            inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_REQUIRED, name_field, args_field, format.tools_array_wrapped,
            format.fun_name_is_key, format.id_field, format.gen_id_field, format.parameter_order, format.openai_wrapper_trigger);
    }

    // Handle content wrappers if present
    if (ctx.content && ctx.content->is_always_wrapped()) {
        auto wrapped_content = ctx.content->build_optional_wrapped(ctx);
        auto trailing        = ctx.reasoning->build_trailing_end_parser(ctx);
        return ctx.reasoning_parser + wrapped_content + tools_parser + trailing + p.end();
    }

    std::string tool_start = "{";
    if (!format.section_start.empty()) {
        tool_start = format.section_start;
    } else if (!format.per_call_start.empty()) {
        tool_start = format.per_call_start;
    }

    auto trailing = ctx.reasoning->build_trailing_end_parser(ctx);
    auto content_before_tools = ctx.reasoning->build_content_before_tools(ctx, tool_start);
    bool recovery = ctx.reasoning && ctx.reasoning->recover_inline_tool_calls &&
                    ctx.extracting_reasoning && !ctx.reasoning->end.empty() &&
                    (ctx.reasoning->mode == reasoning_mode::TAG_BASED ||
                     ctx.reasoning->mode == reasoning_mode::TOOLS_ONLY);
    if (recovery) {
        return ctx.reasoning->build_recovery_composition(ctx, tools_parser, content_before_tools, trailing);
    }
    return ctx.reasoning_parser + p.optional(p.content(content_before_tools)) + tools_parser + trailing + p.end();
}

common_peg_parser analyze_tools::build_func_parser(common_chat_peg_builder & p, const std::string & name,
                                                    const common_peg_parser & call_id_section, bool have_call_id,
                                                    const common_peg_parser & args,
                                                    std::optional<common_peg_parser> atomic_peek) const {
    auto              open           = p.tool_open(function.name_prefix + p.tool_name(p.literal(name)) + function.name_suffix);
    bool              matched_atomic = false;
    common_peg_parser func_parser    = p.eps();

    if (!function.args_separator.empty()) {
        open = open + p.space() + p.literal(function.args_separator);
    }

    if (!function.name_suffix.empty()) {
        func_parser    = open + call_id_section + p.space() + args;
        matched_atomic = true;
    } else if (have_call_id) {
        func_parser    = p.atomic(open + call_id_section) + p.space() + args;
        matched_atomic = true;
    } else if (atomic_peek.has_value()) {
        func_parser    = p.atomic(open + call_id_section + p.space() + *atomic_peek) + args;
        matched_atomic = true;
    } else {
        func_parser = open + call_id_section + p.space() + args;
    }

    if (!function.close.empty()) {
        func_parser = func_parser + p.space() + p.tool_close(p.literal(function.close));
    } else if (!format.per_call_end.empty()) {
        // When there's no func_close but there is a per_call_end marker, use peek() to ensure
        // we only emit tool_close when we can actually see the closing marker. This prevents
        // premature closing during partial parsing when we've seen e.g. "</" which could be
        // either "</tool_call>" (end) or "<arg_key>" prefix that failed to match.
        // Laguna (v4): the model may emit whitespace between the last </arg_value> and
        // </tool_call> even though the template renders them tight. Tolerate optional
        // leading space in the close lookahead so the tool call still closes.
        auto close_peek = arguments.tolerate_intertag_whitespace
                              ? p.peek(p.space() + p.literal(format.per_call_end))
                              : p.peek(p.literal(format.per_call_end));
        func_parser = func_parser + p.tool_close(close_peek);
    } else {
        func_parser = func_parser + p.tool_close(p.space());  // force this to process tool closing callbacks in mapper
    }
    if (!matched_atomic) {
        func_parser = p.atomic(func_parser);
    }
    return func_parser;
}

common_peg_parser analyze_tools::build_tool_parser_tag_json(parser_build_context & ctx) const {
    auto &       p           = ctx.p;
    const auto & inputs      = ctx.inputs;

    common_peg_parser tool_choice = p.choice();

    foreach_function(inputs.tools, [&](const json & tool) {
        const auto & func   = tool.at("function");
        std::string  name   = func.at("name");
        const auto & schema = func.contains("parameters") ? func.at("parameters") : json::object();

        // Build call_id parser based on position (if supported)
        bool have_call_id = false;
        common_peg_parser call_id_section = p.eps();
        if (call_id.pos == call_id_position::BETWEEN_FUNC_AND_ARGS && !call_id.prefix.empty() &&
            (!call_id.suffix.empty() || !arguments.start.empty())) {
            if (!call_id.suffix.empty()) {
                call_id_section = p.optional(call_id.prefix + p.tool_id(p.until(call_id.suffix))) + call_id.suffix;
            } else {
                call_id_section = p.optional(call_id.prefix + p.tool_id(p.until(arguments.start)));
            }
            have_call_id = true;
        }
        auto args_parser = p.tool_args(p.schema(p.json(), "tool-" + name + "-schema", schema));
        if (!arguments.start.empty()) {
            args_parser = p.literal(arguments.start) + args_parser;
        }
        if (!arguments.end.empty()) {
            args_parser = args_parser + p.literal(arguments.end);
        }

        auto atomic_peek = !arguments.start.empty() ? std::optional(p.peek(p.literal(arguments.start))) : std::nullopt;
        auto func_parser = build_func_parser(p, name, call_id_section, have_call_id, args_parser, atomic_peek);
        tool_choice |= p.rule("tool-" + name, func_parser);
    });

    auto require_calls = inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_REQUIRED;

    common_peg_parser tool_calls = p.eps();

    if (!format.per_call_start.empty()) {
        auto wrapped_call = format.per_call_start + tool_choice + format.per_call_end;
        if (inputs.parallel_tool_calls) {
            tool_calls = p.trigger_rule("tool-call", wrapped_call + p.zero_or_more(p.space() + wrapped_call));
        } else {
            tool_calls = p.trigger_rule("tool-call", wrapped_call);
        }
        if (!format.section_start.empty()) {
            tool_calls = p.trigger_rule("tool-calls",
                                        p.literal(format.section_start) + p.space() + tool_calls + p.space() +
                                            (format.section_end.empty() ? p.end() : p.literal(format.section_end)));
        }
    } else {
        std::string separator = ", ";  // Default
        if (inputs.parallel_tool_calls) {
            tool_calls = p.trigger_rule("tool-call", format.section_start + tool_choice +
                                                         p.zero_or_more(separator + tool_choice) + format.section_end);
        } else {
            tool_calls = p.trigger_rule("tool-call", format.section_start + tool_choice + format.section_end);
        }
    }

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
}

common_peg_parser analyze_tools::build_tool_parser_tag_tagged(parser_build_context & ctx) const {
    auto &       p           = ctx.p;
    const auto & inputs      = ctx.inputs;

    auto until_suffix = p.rule("until-suffix", p.until(arguments.value_suffix));

    common_peg_parser tool_choice = p.choice();

    foreach_function(inputs.tools, [&](const json & tool) {
        const auto &          func       = tool.at("function");
        std::string           name       = func.at("name");
        auto                  params     = func.contains("parameters") ? func.at("parameters") : json::object();
        const auto &          properties = params.contains("properties") ? params.at("properties") : json::object();

        std::set<std::string> required;
        if (params.contains("required")) {
            params.at("required").get_to(required);
        }

        auto schema_info = common_schema_info();
        schema_info.resolve_refs(params);

        // Build parser for each argument, separating required and optional
        std::vector<common_peg_parser> required_parsers;
        std::vector<common_peg_parser> optional_parsers;
        for (const auto & [param_name, param_schema] : properties.items()) {
            bool is_required = required.find(param_name) != required.end();

            auto arg =
                p.tool_arg(p.tool_arg_open(arguments.name_prefix + p.tool_arg_name(p.literal(param_name)) +
                                           arguments.name_suffix) +
                           arguments.value_prefix +
                           (schema_info.resolves_to_string(param_schema) ?
                                p.ac(p.tool_arg_string_value(until_suffix) +
                                    p.tool_arg_close(p.literal(arguments.value_suffix)), arguments.value_suffix) :
                                (p.tool_arg_json_value(p.schema(
                                    p.json(), "tool-" + name + "-arg-" + param_name + "-schema", param_schema, false)) +
                                    p.tool_arg_close(p.literal(arguments.value_suffix)))));

            auto named_arg = p.rule("tool-" + name + "-arg-" + param_name, arg);
            if (is_required) {
                required_parsers.push_back(named_arg);
            } else {
                optional_parsers.push_back(named_arg);
            }
        }

        // Build required arg sequence in definition order
        common_peg_parser args_seq = p.eps();
        for (size_t i = 0; i < required_parsers.size(); i++) {
            if (i > 0) {
                args_seq = args_seq + p.space();
            }
            args_seq = args_seq + required_parsers[i];
        }

        // Build optional args with flexible ordering
        if (!optional_parsers.empty()) {
            common_peg_parser any_opt = p.choice();
            for (const auto & opt : optional_parsers) {
                any_opt |= opt;
            }
            args_seq = args_seq + p.repeat(p.space() + any_opt, 0, -1);
        }

        if (!arguments.start.empty()) {
            args_seq = p.literal(arguments.start) + args_seq;
        }
        if (!arguments.end.empty()) {
            args_seq = args_seq + p.literal(arguments.end);
        }

        // Build call_id parser based on position (if supported)
        common_peg_parser call_id_section = p.eps();
        bool have_call_id = false;
        if (call_id.pos == call_id_position::BETWEEN_FUNC_AND_ARGS && !call_id.prefix.empty() &&
            (!call_id.suffix.empty() || !arguments.start.empty())) {
            have_call_id = true;
            if (!call_id.suffix.empty()) {
                call_id_section = p.optional(call_id.prefix + p.tool_id(p.until(call_id.suffix)) + call_id.suffix);
            } else {
                call_id_section = p.optional(call_id.prefix + p.tool_id(p.until(arguments.start)));
            }
        }

        // Only peek for an arg tag when there are required args that must follow.
        // When all args are optional, the model may emit no arg tags at all (#20650).
        auto atomic_peek = (!arguments.name_prefix.empty() && !required_parsers.empty()) ?
            std::optional(p.peek(p.literal(arguments.name_prefix))) : std::nullopt;
        auto func_parser = build_func_parser(p, name, call_id_section, have_call_id, args_seq, atomic_peek);
        tool_choice |= p.rule("tool-" + name, func_parser);
    });

    auto require_tools = inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_REQUIRED;

    common_peg_parser tool_calls = p.eps();

    if (!format.per_call_start.empty()) {
        auto wrapped_call = format.per_call_start + p.space() + tool_choice + p.space() + format.per_call_end;
        // Recovery mode (Qwen family): the model may close </think> between an in-think
        // tool call and a following one. Swallow any stray closing tags between calls so
        // both are extracted (stable, monotonic tool count) instead of stalling the parse.
        auto between_calls = p.space();
        if (ctx.reasoning && ctx.reasoning->recover_inline_tool_calls && ctx.extracting_reasoning &&
            !ctx.reasoning->end.empty()) {
            between_calls = p.space() + p.zero_or_more(p.optspace(ctx.reasoning->end));
        }
        if (inputs.parallel_tool_calls) {
            tool_calls = p.trigger_rule("tool-call", wrapped_call + p.zero_or_more(between_calls + wrapped_call) + between_calls);
        } else {
            tool_calls = p.trigger_rule("tool-call", wrapped_call + between_calls);
        }
        if (!format.section_start.empty()) {
            tool_calls = p.trigger_rule("tool-calls",
                                        p.literal(format.section_start) + p.space() + tool_calls + p.space() +
                                            (format.section_end.empty() ? p.end() : p.literal(format.section_end) + p.space()));
        }
    } else {
        std::string separator = ", ";  // Default

        if (inputs.parallel_tool_calls) {
            tool_calls = p.trigger_rule("tool-call", format.section_start + p.space() + tool_choice +
                                                         p.zero_or_more(separator + tool_choice) + p.space() +
                                                         format.section_end);
        } else {
            tool_calls = p.trigger_rule(
                "tool-call", format.section_start + p.space() + tool_choice + p.space() + format.section_end);
        }
    }

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
}

}  // namespace autoparser
