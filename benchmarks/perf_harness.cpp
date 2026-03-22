/**
 * @file perf_harness.cpp
 * @brief Repo-local microbenchmarks for the performance-first runtime surface.
 *
 * Build with:
 *   scripts/build -DZOO_BUILD_BENCHMARKS=ON
 *
 * Run with:
 *   build/benchmarks/zoo_benchmarks
 *   ZOO_BENCHMARK_MODEL=/path/to/model.gguf build/benchmarks/zoo_benchmarks
 */

#include "zoo/core/model.hpp"
#include "zoo/internal/agent/runtime.hpp"
#include "zoo/internal/core/stream_filter.hpp"

#include <common.h>

#include <array>
#include <chrono>
#include <cstdlib>
#include <deque>
#include <functional>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;
using zoo::CancellationCallback;
using zoo::Error;
using zoo::ErrorCode;
using zoo::Expected;
using zoo::GenerationOptions;
using zoo::HistorySnapshot;
using zoo::Message;
using zoo::MessageView;
using zoo::ModelConfig;
using zoo::Role;
using zoo::TextResponse;
using zoo::TokenAction;
using zoo::TokenCallback;
using zoo::core::ToolCallTriggerMatcher;
using zoo::internal::agent::AgentBackend;
using zoo::internal::agent::AgentRuntime;
using zoo::internal::agent::GenerationResult;
using zoo::internal::agent::ParsedToolResponse;

volatile std::size_t g_benchmark_sink = 0;

struct BenchmarkResult {
    std::string name;
    int iterations = 0;
    double total_ms = 0.0;

    [[nodiscard]] double average_us() const {
        return iterations == 0 ? 0.0 : (total_ms * 1000.0) / static_cast<double>(iterations);
    }
};

template <typename Func>
BenchmarkResult benchmark_case(std::string name, int iterations, Func&& func) {
    const auto start = Clock::now();
    for (int iteration = 0; iteration < iterations; ++iteration) {
        func();
    }
    const auto end = Clock::now();
    return BenchmarkResult{
        std::move(name),
        iterations,
        std::chrono::duration<double, std::milli>(end - start).count(),
    };
}

void print_result(const BenchmarkResult& result) {
    std::cout << std::left << std::setw(36) << result.name << "  iterations=" << std::setw(6)
              << result.iterations << " total_ms=" << std::fixed << std::setprecision(3)
              << std::setw(10) << result.total_ms << " avg_us=" << std::setprecision(2)
              << result.average_us() << '\n';
}

class BenchmarkBackend final : public AgentBackend {
  public:
    using GenerationAction =
        std::function<Expected<GenerationResult>(TokenCallback, const CancellationCallback&)>;

    void push_generation(GenerationAction action) {
        std::lock_guard<std::mutex> lock(mutex_);
        generations_.push_back(std::move(action));
    }

    Expected<void> add_message(MessageView message) override {
        std::lock_guard<std::mutex> lock(mutex_);
        history_.push_back(Message::from_view(message));
        return {};
    }

    Expected<GenerationResult> generate_from_history(const GenerationOptions&, TokenCallback on_token,
                                                     CancellationCallback should_cancel) override {
        GenerationAction action;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (generations_.empty()) {
                return std::unexpected(
                    Error{ErrorCode::InferenceFailed, "No scripted generation available"});
            }
            action = std::move(generations_.front());
            generations_.pop_front();
        }
        return action(on_token, should_cancel);
    }

    void finalize_response() override {}

    void set_system_prompt(std::string_view prompt) override {
        std::lock_guard<std::mutex> lock(mutex_);
        Message system_message = Message::system(std::string(prompt));
        if (!history_.empty() && history_.front().role == Role::System) {
            history_.front() = std::move(system_message);
        } else {
            history_.insert(history_.begin(), std::move(system_message));
        }
    }

    HistorySnapshot get_history() const override {
        std::lock_guard<std::mutex> lock(mutex_);
        return HistorySnapshot{history_};
    }

    void clear_history() override {
        std::lock_guard<std::mutex> lock(mutex_);
        history_.clear();
    }

    void replace_history(HistorySnapshot snapshot) override {
        std::lock_guard<std::mutex> lock(mutex_);
        history_ = std::move(snapshot.messages);
    }

    HistorySnapshot swap_history(HistorySnapshot snapshot) override {
        std::lock_guard<std::mutex> lock(mutex_);
        HistorySnapshot previous{std::move(history_)};
        history_ = std::move(snapshot.messages);
        return previous;
    }

    bool set_tool_calling(const std::vector<zoo::CoreToolInfo>& tools) override {
        std::lock_guard<std::mutex> lock(mutex_);
        tool_calling_enabled_ = !tools.empty();
        return tool_calling_enabled_;
    }

    ParsedToolResponse parse_tool_response(std::string_view text) const override {
        ParsedToolResponse result;

        const std::string open_tag = "<tool_call>";
        const std::string close_tag = "</tool_call>";
        const std::string value(text);
        const auto start = value.find(open_tag);
        const auto end = value.find(close_tag);
        if (start == std::string::npos || end == std::string::npos) {
            result.content = value;
            return result;
        }

        auto json_str = value.substr(start + open_tag.size(), end - start - open_tag.size());
        auto parsed = nlohmann::json::parse(json_str, nullptr, false);
        if (!parsed.is_discarded()) {
            zoo::OwnedToolCall tool_call;
            tool_call.id = parsed.value("id", "");
            tool_call.name = parsed.value("name", "");
            if (parsed.contains("arguments")) {
                tool_call.arguments_json = parsed["arguments"].dump();
            }
            result.tool_calls.push_back(std::move(tool_call));
        }

        result.content = value.substr(0, start);
        if (end + close_tag.size() < value.size()) {
            result.content += value.substr(end + close_tag.size());
        }
        return result;
    }

    const char* tool_calling_format_name() const noexcept override {
        return "benchmark";
    }

    bool set_schema_grammar(const std::string&) override {
        return true;
    }

    void clear_tool_grammar() override {}

  private:
    mutable std::mutex mutex_;
    std::deque<GenerationAction> generations_;
    std::vector<Message> history_;
    bool tool_calling_enabled_ = false;
};

ModelConfig make_model_config(std::string model_path = "unused.gguf") {
    ModelConfig config;
    config.model_path = std::move(model_path);
    return config;
}

zoo::AgentConfig make_agent_config() {
    zoo::AgentConfig config;
    config.request_queue_capacity = 32;
    config.max_history_messages = 8;
    config.max_tool_iterations = 4;
    config.max_tool_retries = 1;
    return config;
}

GenerationResult tool_call_generation(const std::string& tool_name, const nlohmann::json& arguments,
                                      std::string id = "bench-call") {
    const nlohmann::json payload = {{"id", std::move(id)},
                                    {"name", tool_name},
                                    {"arguments", arguments}};
    return GenerationResult{"<tool_call>" + payload.dump() + "</tool_call>", 0, true, "", {}};
}

template <typename Result>
void require_success(const Expected<Result>& result, std::string_view benchmark_name) {
    if (!result) {
        throw std::runtime_error(std::string(benchmark_name) + " failed: " +
                                 result.error().to_string());
    }
}

void run_stream_filter_benchmark() {
    std::vector<common_grammar_trigger> triggers;
    triggers.push_back({COMMON_GRAMMAR_TRIGGER_TYPE_WORD, "[TOOL_CALLS]"});
    triggers.push_back({COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN,
                        R"(\s*<\|tool_call_start\|>\s*\[)"});
    ToolCallTriggerMatcher matcher(triggers);
    const std::string text = "assistant: [TOOL_CALLS] <|tool_call_start|> [";

    auto result = benchmark_case("stream_filter.trigger_matcher", 200000, [&] {
        g_benchmark_sink += matcher.is_detected(text) ? 1u : 0u;
    });
    print_result(result);
}

void run_runtime_complete_benchmark() {
    auto backend = std::make_unique<BenchmarkBackend>();
    auto* backend_ptr = backend.get();
    AgentRuntime runtime(make_model_config(), make_agent_config(), GenerationOptions{},
                         std::move(backend));

    const std::array<MessageView, 4> messages = {
        MessageView{Role::System, "Benchmark system prompt."},
        MessageView{Role::User, "Message one"},
        MessageView{Role::Assistant, "Reply one"},
        MessageView{Role::User, "Message two"},
    };

    auto result = benchmark_case("runtime.complete.stateless", 500, [&] {
        backend_ptr->push_generation([](TokenCallback, const CancellationCallback&) {
            return Expected<GenerationResult>(GenerationResult{"ok", 0, false, "", {}});
        });

        auto handle =
            runtime.complete(zoo::ConversationView{std::span<const MessageView>(messages)});
        auto response = handle.await_result();
        require_success(response, "runtime.complete.stateless");
        g_benchmark_sink += response->text.size();
    });
    print_result(result);
}

void run_runtime_extract_benchmark() {
    auto backend = std::make_unique<BenchmarkBackend>();
    auto* backend_ptr = backend.get();
    AgentRuntime runtime(make_model_config(), make_agent_config(), GenerationOptions{},
                         std::move(backend));

    const nlohmann::json schema = {{"type", "object"},
                                   {"properties", {{"count", {{"type", "integer"}}}}},
                                   {"required", nlohmann::json::array({"count"})},
                                   {"additionalProperties", false}};

    auto result = benchmark_case("runtime.extract.stateless", 400, [&] {
        backend_ptr->push_generation([](TokenCallback, const CancellationCallback&) {
            return Expected<GenerationResult>(GenerationResult{R"({"count":7})", 0, false, "",
                                                               {}});
        });

        auto handle = runtime.extract(schema, "There are 7 apples.");
        auto response = handle.await_result();
        require_success(response, "runtime.extract.stateless");
        g_benchmark_sink += response->data["count"].get<int>();
    });
    print_result(result);
}

void run_runtime_tool_loop_benchmark() {
    auto backend = std::make_unique<BenchmarkBackend>();
    auto* backend_ptr = backend.get();
    AgentRuntime runtime(make_model_config(), make_agent_config(), GenerationOptions{},
                         std::move(backend));

    auto definition = zoo::tools::detail::make_tool_definition(
        "double", "Double a number", std::vector<std::string>{"value"},
        [](int value) { return value * 2; });
    if (!definition) {
        throw std::runtime_error(definition.error().to_string());
    }
    auto registered = runtime.register_tool(std::move(*definition));
    require_success(registered, "runtime.tool_loop.register");

    auto result = benchmark_case("runtime.chat.tool_loop", 300, [&] {
        backend_ptr->push_generation([](TokenCallback, const CancellationCallback&) {
            return Expected<GenerationResult>(tool_call_generation("double", {{"value", 21}}));
        });
        backend_ptr->push_generation([](TokenCallback, const CancellationCallback&) {
            return Expected<GenerationResult>(GenerationResult{"42", 0, false, "", {}});
        });

        auto handle = runtime.chat("double 21");
        auto response = handle.await_result();
        require_success(response, "runtime.chat.tool_loop");
        g_benchmark_sink += response->text.size();
    });
    print_result(result);
}

std::optional<std::string> benchmark_model_path(int argc, char** argv) {
    if (argc > 1 && argv[1] != nullptr && argv[1][0] != '\0') {
        return std::string(argv[1]);
    }

    if (const char* env = std::getenv("ZOO_BENCHMARK_MODEL")) {
        if (*env != '\0') {
            return std::string(env);
        }
    }

    return std::nullopt;
}

void run_live_model_benchmark(const std::string& model_path) {
    ModelConfig model_config;
    model_config.model_path = model_path;
    model_config.context_size = 2048;
    model_config.n_gpu_layers = 0;

    GenerationOptions generation;
    generation.max_tokens = 16;
    generation.sampling.temperature = 0.0f;
    generation.sampling.top_p = 1.0f;
    generation.sampling.top_k = 1;
    generation.sampling.seed = 7;

    auto model_result = zoo::core::Model::load(model_config, generation);
    require_success(model_result, "live_model.load");
    auto& model = *model_result;

    HistorySnapshot history;
    history.messages.reserve(17);
    history.messages.push_back(Message::system("You are benchmarking prompt rendering."));
    for (int index = 0; index < 8; ++index) {
        history.messages.push_back(Message::user("User turn " + std::to_string(index)));
        history.messages.push_back(Message::assistant("Assistant turn " + std::to_string(index)));
    }

    auto result = benchmark_case("live_model.generate_history", 3, [&] {
        model->replace_history(history);
        auto generation_result = model->generate_from_history();
        require_success(generation_result, "live_model.generate_history");
        g_benchmark_sink += generation_result->text.size();
    });
    print_result(result);
}

} // namespace

int main(int argc, char** argv) {
    try {
        std::cout << "Zoo-Keeper benchmark harness\n";
        std::cout << "sizeof(RequestHandle<TextResponse>)=" << sizeof(zoo::RequestHandle<TextResponse>)
                  << " sizeof(MessageView)=" << sizeof(MessageView)
                  << " sizeof(OwnedMessage)=" << sizeof(zoo::OwnedMessage) << '\n';

        run_stream_filter_benchmark();
        run_runtime_complete_benchmark();
        run_runtime_extract_benchmark();
        run_runtime_tool_loop_benchmark();

        const auto model_path = benchmark_model_path(argc, argv);
        if (model_path) {
            run_live_model_benchmark(*model_path);
        } else {
            std::cout << "live_model.generate_history           skipped    set ZOO_BENCHMARK_MODEL or pass a model path\n";
        }

        std::cout << "benchmark_sink=" << g_benchmark_sink << '\n';
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "benchmark failed: " << e.what() << '\n';
        return 1;
    }
}
