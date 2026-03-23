/**
 * @file perf_harness.cpp
 * @brief Repo-local benchmarks for the real GGUF-backed runtime surface.
 *
 * Build with:
 *   scripts/build -DZOO_BUILD_BENCHMARKS=ON
 *
 * Run with:
 *   build/benchmarks/zoo_benchmarks /path/to/model.gguf
 *   ZOO_BENCHMARK_MODEL=/path/to/model.gguf build/benchmarks/zoo_benchmarks
 */

#include "zoo/agent.hpp"
#include "zoo/core/model.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;
using zoo::Expected;
using zoo::GenerationOptions;
using zoo::HistorySnapshot;
using zoo::Message;
using zoo::ModelConfig;
using zoo::TextResponse;

volatile std::size_t g_benchmark_sink = 0;
constexpr int kBenchmarkIterations = 3;

struct BenchmarkSample {
    double latency_ms = 0.0;
    double ttft_ms = 0.0;
    double decode_tokens_per_second = 0.0;
    double effective_prefill_tokens_per_second = 0.0;
    int prompt_tokens = 0;
    int completion_tokens = 0;
};

struct MetricSummary {
    double average = 0.0;
    double p50 = 0.0;
    double p95 = 0.0;
};

struct TokenSummary {
    double average = 0.0;
    int min = 0;
    int max = 0;
};

struct BenchmarkResult {
    std::string name;
    std::vector<BenchmarkSample> samples;
    MetricSummary latency_ms;
    MetricSummary ttft_ms;
    MetricSummary decode_tokens_per_second;
    MetricSummary effective_prefill_tokens_per_second;
    TokenSummary prompt_tokens;
    TokenSummary completion_tokens;
};

template <typename Func>
BenchmarkResult benchmark_case(std::string name, int iterations, Func&& func) {
    BenchmarkResult result;
    result.name = std::move(name);
    result.samples.reserve(static_cast<std::size_t>(iterations));
    for (int iteration = 0; iteration < iterations; ++iteration) {
        result.samples.push_back(func());
    }
    return result;
}

double percentile(std::vector<double> values, double fraction) {
    if (values.empty()) {
        return 0.0;
    }
    std::sort(values.begin(), values.end());
    const auto clamped = std::clamp(fraction, 0.0, 1.0);
    const auto rank = std::max<std::size_t>(
        1, static_cast<std::size_t>(std::ceil(clamped * static_cast<double>(values.size()))));
    const auto index = std::min(values.size() - 1, rank - 1);
    return values[index];
}

MetricSummary summarize_metric(const std::vector<BenchmarkSample>& samples,
                               double BenchmarkSample::*member) {
    MetricSummary summary;
    if (samples.empty()) {
        return summary;
    }

    std::vector<double> values;
    values.reserve(samples.size());
    double total = 0.0;
    for (const auto& sample : samples) {
        const double value = sample.*member;
        values.push_back(value);
        total += value;
    }

    summary.average = total / static_cast<double>(samples.size());
    summary.p50 = percentile(values, 0.50);
    summary.p95 = percentile(values, 0.95);
    return summary;
}

TokenSummary summarize_tokens(const std::vector<BenchmarkSample>& samples,
                              int BenchmarkSample::*member) {
    TokenSummary summary;
    if (samples.empty()) {
        return summary;
    }

    int total = 0;
    summary.min = samples.front().*member;
    summary.max = samples.front().*member;
    for (const auto& sample : samples) {
        const int value = sample.*member;
        total += value;
        summary.min = std::min(summary.min, value);
        summary.max = std::max(summary.max, value);
    }
    summary.average = static_cast<double>(total) / static_cast<double>(samples.size());
    return summary;
}

BenchmarkResult finalize_result(BenchmarkResult result) {
    result.latency_ms = summarize_metric(result.samples, &BenchmarkSample::latency_ms);
    result.ttft_ms = summarize_metric(result.samples, &BenchmarkSample::ttft_ms);
    result.decode_tokens_per_second =
        summarize_metric(result.samples, &BenchmarkSample::decode_tokens_per_second);
    result.effective_prefill_tokens_per_second =
        summarize_metric(result.samples, &BenchmarkSample::effective_prefill_tokens_per_second);
    result.prompt_tokens = summarize_tokens(result.samples, &BenchmarkSample::prompt_tokens);
    result.completion_tokens =
        summarize_tokens(result.samples, &BenchmarkSample::completion_tokens);
    return result;
}

double tokens_per_second(int tokens, double elapsed_ms) {
    if (tokens <= 0 || elapsed_ms <= 0.0) {
        return 0.0;
    }
    return (static_cast<double>(tokens) * 1000.0) / elapsed_ms;
}

void print_result(const BenchmarkResult& result) {
    const auto print_metric = [](std::string_view label, const MetricSummary& summary,
                                 std::string_view unit) {
        std::cout << "  " << std::left << std::setw(14) << label << " avg=" << std::fixed
                  << std::setprecision(2) << std::setw(9) << summary.average
                  << " p50=" << std::setw(9) << summary.p50 << " p95=" << std::setw(9)
                  << summary.p95 << " " << unit << '\n';
    };

    const auto print_tokens = [](std::string_view label, const TokenSummary& summary) {
        std::cout << "  " << std::left << std::setw(14) << label << " avg=" << std::fixed
                  << std::setprecision(2) << std::setw(9) << summary.average
                  << " min=" << std::setw(5) << summary.min << " max=" << std::setw(5)
                  << summary.max << " tokens\n";
    };

    std::cout << result.name << "  iterations=" << result.samples.size() << '\n';
    print_metric("latency_ms", result.latency_ms, "ms");
    print_metric("ttft_ms", result.ttft_ms, "ms");
    print_metric("decode_tps", result.decode_tokens_per_second, "tok/s");
    print_metric("prefill_tps", result.effective_prefill_tokens_per_second, "tok/s");
    print_tokens("prompt_tokens", result.prompt_tokens);
    print_tokens("completion", result.completion_tokens);
}

template <typename Result>
void require_success(const Expected<Result>& result, std::string_view benchmark_name) {
    if (!result) {
        throw std::runtime_error(std::string(benchmark_name) +
                                 " failed: " + result.error().to_string());
    }
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

ModelConfig make_model_config(const std::string& model_path) {
    ModelConfig config;
    config.model_path = model_path;
    config.context_size = 2048;
    config.n_gpu_layers = 0;
    return config;
}

GenerationOptions make_generation_options() {
    GenerationOptions generation;
    generation.max_tokens = 16;
    generation.sampling.temperature = 0.0f;
    generation.sampling.top_p = 1.0f;
    generation.sampling.top_k = 1;
    generation.sampling.seed = 7;
    return generation;
}

HistorySnapshot make_history_snapshot(std::string final_user_prompt) {
    HistorySnapshot history;
    history.messages.reserve(16);
    history.messages.push_back(Message::system("You are benchmarking real-model generation."));
    for (int index = 0; index < 7; ++index) {
        history.messages.push_back(Message::user("User turn " + std::to_string(index)));
        history.messages.push_back(Message::assistant("Assistant turn " + std::to_string(index)));
    }
    history.messages.push_back(Message::user(std::move(final_user_prompt)));
    return history;
}

void run_live_model_benchmarks(const std::string& model_path) {
    auto model_result =
        zoo::core::Model::load(make_model_config(model_path), make_generation_options());
    require_success(model_result, "live_model.load");
    auto& model = *model_result;

    const HistorySnapshot history =
        make_history_snapshot("Summarize the conversation in one short sentence.");

    auto generate_result = benchmark_case("live_model.generate", kBenchmarkIterations, [&] {
        model->clear_history();
        model->set_system_prompt("You are benchmarking stateful generation.");
        auto response = model->generate("Reply in one short sentence.");
        require_success(response, "live_model.generate");
        g_benchmark_sink += response->text.size();
        return BenchmarkSample{
            .latency_ms = static_cast<double>(response->metrics.latency_ms.count()),
            .ttft_ms = static_cast<double>(response->metrics.time_to_first_token_ms.count()),
            .decode_tokens_per_second = response->metrics.tokens_per_second,
            .effective_prefill_tokens_per_second = tokens_per_second(
                response->usage.prompt_tokens,
                static_cast<double>(response->metrics.time_to_first_token_ms.count())),
            .prompt_tokens = response->usage.prompt_tokens,
            .completion_tokens = response->usage.completion_tokens,
        };
    });
    print_result(finalize_result(std::move(generate_result)));

    auto history_result = benchmark_case("live_model.generate_history", kBenchmarkIterations, [&] {
        model->replace_history(history);
        std::optional<Clock::time_point> first_token_time;
        int completion_tokens = 0;
        const auto start_time = Clock::now();
        auto on_token = [&](std::string_view) {
            if (!first_token_time.has_value()) {
                first_token_time = Clock::now();
            }
            ++completion_tokens;
            return zoo::TokenAction::Continue;
        };
        auto response = model->generate_from_history({}, on_token);
        const auto end_time = Clock::now();
        require_success(response, "live_model.generate_history");
        g_benchmark_sink += response->text.size();
        const double latency_ms =
            std::chrono::duration<double, std::milli>(end_time - start_time).count();
        const double ttft_ms =
            first_token_time.has_value()
                ? std::chrono::duration<double, std::milli>(*first_token_time - start_time).count()
                : 0.0;
        const double decode_tps =
            first_token_time.has_value()
                ? tokens_per_second(completion_tokens, std::chrono::duration<double, std::milli>(
                                                           end_time - *first_token_time)
                                                           .count())
                : 0.0;
        return BenchmarkSample{
            .latency_ms = latency_ms,
            .ttft_ms = ttft_ms,
            .decode_tokens_per_second = decode_tps,
            .effective_prefill_tokens_per_second =
                tokens_per_second(response->prompt_tokens, ttft_ms),
            .prompt_tokens = response->prompt_tokens,
            .completion_tokens = completion_tokens,
        };
    });
    print_result(finalize_result(std::move(history_result)));
}

} // namespace

int main(int argc, char** argv) {
    try {
        const auto model_path = benchmark_model_path(argc, argv);
        if (!model_path) {
            std::cerr << "benchmark failed: provide a GGUF path as argv[1] or set "
                         "ZOO_BENCHMARK_MODEL\n";
            return 1;
        }

        std::cout << "Zoo-Keeper benchmark harness\n";
        std::cout << "sizeof(RequestHandle<TextResponse>)="
                  << sizeof(zoo::RequestHandle<TextResponse>)
                  << " sizeof(MessageView)=" << sizeof(zoo::MessageView)
                  << " sizeof(OwnedMessage)=" << sizeof(zoo::OwnedMessage) << '\n';

        run_live_model_benchmarks(*model_path);

        std::cout << "benchmark_sink=" << g_benchmark_sink << '\n';
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "benchmark failed: " << e.what() << '\n';
        return 1;
    }
}
