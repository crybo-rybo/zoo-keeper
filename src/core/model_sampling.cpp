/**
 * @file model_sampling.cpp
 * @brief Sampler-chain and tool-grammar management for `zoo::core::Model`.
 */

#include "zoo/core/model.hpp"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <llama.h>
#include <random>

namespace zoo::core {

namespace {

uint32_t make_sampler_seed(int configured_seed) {
    if (configured_seed >= 0) {
        return static_cast<uint32_t>(configured_seed);
    }

    static std::atomic<uint64_t> counter{0};

    uint64_t seed =
        static_cast<uint64_t>(std::chrono::high_resolution_clock::now().time_since_epoch().count());
    seed ^= counter.fetch_add(0x9e3779b97f4a7c15ull, std::memory_order_relaxed);

    std::random_device rd;
    seed ^= (static_cast<uint64_t>(rd()) << 32);
    seed ^= static_cast<uint64_t>(rd());

    return static_cast<uint32_t>(seed ^ (seed >> 32));
}

} // namespace

bool Model::set_tool_grammar(const std::string& grammar_str) {
    tool_grammar_str_ = grammar_str;
    return rebuild_sampler_with_grammar();
}

bool Model::set_schema_grammar(const std::string& grammar_str) {
    tool_grammar_str_ = grammar_str;
    return rebuild_sampler_with_schema_grammar();
}

void Model::clear_tool_grammar() noexcept {
    if (grammar_mode_ == GrammarMode::None) {
        return;
    }

    tool_grammar_str_.clear();
    grammar_mode_ = GrammarMode::None;
    sampler_ = create_sampler_chain();
}

bool Model::rebuild_sampler_with_grammar() {
    auto chain_params = llama_sampler_chain_default_params();
    chain_params.no_perf = false;
    auto chain = LlamaSamplerHandle(llama_sampler_chain_init(chain_params));
    if (!chain) {
        return false;
    }

    add_sampling_stages(chain.get());

    const char* trigger = "<tool_call>";
    const char* trigger_patterns[] = {trigger};
    auto* grammar_sampler = llama_sampler_init_grammar_lazy_patterns(
        vocab_, tool_grammar_str_.c_str(), "root", trigger_patterns, 1, nullptr, 0);
    if (!grammar_sampler) {
        return false;
    }
    llama_sampler_chain_add(chain.get(), grammar_sampler);

    add_dist_sampler(chain.get());

    sampler_ = std::move(chain);
    grammar_mode_ = GrammarMode::ToolCall;
    return true;
}

bool Model::rebuild_sampler_with_schema_grammar() {
    auto chain_params = llama_sampler_chain_default_params();
    chain_params.no_perf = false;
    auto chain = LlamaSamplerHandle(llama_sampler_chain_init(chain_params));
    if (!chain) {
        return false;
    }

    // Grammar must filter logits before top-k/top-p narrow the candidate set,
    // otherwise top-k=1 can select a token the grammar rejects.
    auto* grammar_sampler = llama_sampler_init_grammar(vocab_, tool_grammar_str_.c_str(), "root");
    if (!grammar_sampler) {
        return false;
    }
    llama_sampler_chain_add(chain.get(), grammar_sampler);

    add_sampling_stages(chain.get());
    add_dist_sampler(chain.get());

    sampler_ = std::move(chain);
    grammar_mode_ = GrammarMode::Schema;
    return true;
}

void Model::add_sampling_stages(llama_sampler* chain) const {
    const auto& sp = config_.sampling;
    if (sp.repeat_penalty != 1.0f) {
        if (auto* p = llama_sampler_init_penalties(sp.repeat_last_n, sp.repeat_penalty, 0.0f, 0.0f))
            llama_sampler_chain_add(chain, p);
    }
    if (sp.top_k > 0) {
        if (auto* p = llama_sampler_init_top_k(sp.top_k))
            llama_sampler_chain_add(chain, p);
    }
    if (sp.top_p < 1.0f) {
        if (auto* p = llama_sampler_init_top_p(sp.top_p, 1))
            llama_sampler_chain_add(chain, p);
    }
    if (sp.temperature > 0.0f) {
        if (auto* p = llama_sampler_init_temp(sp.temperature))
            llama_sampler_chain_add(chain, p);
    }
}

void Model::add_dist_sampler(llama_sampler* chain) const {
    const uint32_t seed = make_sampler_seed(config_.sampling.seed);
    if (auto* d = llama_sampler_init_dist(seed)) {
        llama_sampler_chain_add(chain, d);
    } else if (auto* g = llama_sampler_init_greedy()) {
        llama_sampler_chain_add(chain, g);
    }
}

Model::LlamaSamplerHandle Model::create_sampler_chain() {
    auto chain_params = llama_sampler_chain_default_params();
    chain_params.no_perf = false;
    auto chain = LlamaSamplerHandle(llama_sampler_chain_init(chain_params));
    if (!chain) {
        return nullptr;
    }

    add_sampling_stages(chain.get());
    add_dist_sampler(chain.get());

    return chain;
}

} // namespace zoo::core
