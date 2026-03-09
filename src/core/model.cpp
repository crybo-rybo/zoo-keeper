/**
 * @file model.cpp
 * @brief Implementation of the llama.cpp-backed `zoo::core::Model`.
 */

#include "zoo/core/model.hpp"
#include "zoo/core/batch.hpp"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <ctime>
#include <llama.h>
#include <mutex>
#include <random>

namespace zoo::core {

// ============================================================================
// Static initialization
// ============================================================================

static std::once_flag g_init_flag;

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

Expected<TokenAction> invoke_token_callback(const TokenCallback& callback, std::string_view token) {
    try {
        return callback(token);
    } catch (const std::exception& e) {
        return std::unexpected(
            Error{ErrorCode::InferenceFailed, "Token callback threw an exception", e.what()});
    } catch (...) {
        return std::unexpected(
            Error{ErrorCode::InferenceFailed, "Token callback threw an unknown exception"});
    }
}

} // namespace

void Model::initialize_global() {
    std::call_once(g_init_flag, []() {
        llama_backend_init();
        ggml_backend_load_all();
    });
}

// ============================================================================
// Construction / Destruction
// ============================================================================

Model::Model(const Config& config) : config_(config) {}

Model::~Model() {
    if (sampler_) {
        llama_sampler_free(sampler_);
        sampler_ = nullptr;
    }
    if (ctx_) {
        llama_free(ctx_);
        ctx_ = nullptr;
    }
    if (llama_model_) {
        llama_model_free(llama_model_);
        llama_model_ = nullptr;
    }
}

// ============================================================================
// Factory
// ============================================================================

Expected<std::unique_ptr<Model>> Model::load(const Config& config) {
    if (auto result = config.validate(); !result) {
        return std::unexpected(result.error());
    }

    auto model = std::unique_ptr<Model>(new Model(config));
    if (auto result = model->initialize(); !result) {
        return std::unexpected(result.error());
    }

    if (config.system_prompt) {
        model->set_system_prompt(*config.system_prompt);
    }

    return model;
}

// ============================================================================
// Initialization (llama.cpp setup)
// ============================================================================

Expected<void> Model::initialize() {
    initialize_global();

    llama_log_set(
        [](enum ggml_log_level level, const char* text, void*) {
            if (level >= GGML_LOG_LEVEL_WARN) {
                fprintf(stderr, "%s", text);
            }
        },
        nullptr);

    auto model_params = llama_model_default_params();
    model_params.n_gpu_layers = config_.n_gpu_layers;
    model_params.use_mmap = config_.use_mmap;
    model_params.use_mlock = config_.use_mlock;

    llama_model_ = llama_model_load_from_file(config_.model_path.c_str(), model_params);
    if (!llama_model_) {
        return std::unexpected(Error{ErrorCode::ModelLoadFailed,
                                     "Failed to load model from path: " + config_.model_path});
    }

    auto ctx_params = llama_context_default_params();
    ctx_params.n_ctx = static_cast<uint32_t>(config_.context_size);
    ctx_params.n_batch = static_cast<uint32_t>(config_.context_size);
    ctx_params.n_ubatch = 512;
    ctx_params.n_threads = -1;
    ctx_params.n_threads_batch = -1;
    ctx_params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    // Use 8-bit KV cache to reduce memory footprint vs the upstream F16 default.
    ctx_params.type_k = GGML_TYPE_Q8_0;
    ctx_params.type_v = GGML_TYPE_Q8_0;

    ctx_ = llama_init_from_model(llama_model_, ctx_params);
    if (!ctx_) {
        return std::unexpected(
            Error{ErrorCode::ContextCreationFailed, "Failed to create llama context"});
    }

    context_size_ = static_cast<int>(llama_n_ctx(ctx_));

    vocab_ = llama_model_get_vocab(llama_model_);
    if (!vocab_) {
        return std::unexpected(
            Error{ErrorCode::BackendInitFailed, "Failed to get model vocabulary"});
    }

    sampler_ = create_sampler_chain();
    if (!sampler_) {
        return std::unexpected(
            Error{ErrorCode::BackendInitFailed, "Failed to create sampler chain"});
    }

    tmpl_ = llama_model_chat_template(llama_model_, nullptr);
    formatted_.clear();
    prev_len_ = 0;

    if (!tmpl_) {
        return std::unexpected(
            Error{ErrorCode::TemplateRenderFailed, "Model has no chat template"});
    }

    return {};
}

// ============================================================================
// Tokenization
// ============================================================================

Expected<std::vector<int>> Model::tokenize(const std::string& text) {
    static_assert(sizeof(int) == sizeof(llama_token));
    static_assert(alignof(int) == alignof(llama_token));

    const bool is_first = llama_memory_seq_pos_max(llama_get_memory(ctx_), 0) == -1;
    const int32_t raw =
        llama_tokenize(vocab_, text.c_str(), text.length(), nullptr, 0, is_first, true);
    if (raw == INT32_MIN) {
        return std::unexpected(Error{ErrorCode::TokenizationFailed, "Tokenization overflow"});
    }
    const int n = (raw < 0) ? -raw : raw;
    if (n == 0)
        return std::vector<int>{};

    std::vector<int> tokens(n);
    if (llama_tokenize(vocab_, text.c_str(), text.length(),
                       reinterpret_cast<llama_token*>(tokens.data()), tokens.size(), is_first,
                       true) < 0) {
        return std::unexpected(Error{ErrorCode::TokenizationFailed, "Tokenization failed"});
    }
    return tokens;
}

// ============================================================================
// Inference
// ============================================================================

Expected<std::string> Model::run_inference(const std::vector<int>& prompt_tokens, int max_tokens,
                                           const std::vector<std::string>& stop_sequences,
                                           const std::optional<TokenCallback>& on_token,
                                           const CancellationCallback& should_cancel) {
    std::string generated_text;
    const int effective_max = (max_tokens > 0) ? max_tokens : context_size_;
    generated_text.reserve(std::min(static_cast<size_t>(effective_max) * 8, size_t{65536}));
    int token_count = 0;
    bool in_tool_call = false;

    const int n_batch = static_cast<int>(llama_n_batch(ctx_));
    const int base_pos = llama_memory_seq_pos_max(llama_get_memory(ctx_), 0) + 1;

    // --- Chunked prefill ---
    auto chunks = compute_prefill_chunks(static_cast<int>(prompt_tokens.size()), n_batch);

    for (const auto& chunk : chunks) {
        if (should_cancel && should_cancel()) {
            return std::unexpected(
                Error{ErrorCode::RequestCancelled, "Request cancelled during prompt prefill"});
        }

        int n_ctx_used = base_pos + chunk.offset + chunk.count;
        if (n_ctx_used > context_size_) {
            return std::unexpected(
                Error{ErrorCode::ContextWindowExceeded, "Prompt tokens exceed context size"});
        }

        llama_batch batch = llama_batch_init(chunk.count, 0, 1);
        for (int i = 0; i < chunk.count; ++i) {
            batch.token[i] = static_cast<llama_token>(prompt_tokens[chunk.offset + i]);
            batch.pos[i] = static_cast<llama_pos>(base_pos + chunk.offset + i);
            batch.n_seq_id[i] = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i] = (chunk.emit_logits && i == chunk.count - 1);
        }
        batch.n_tokens = chunk.count;

        int rc = llama_decode(ctx_, batch);
        llama_batch_free(batch);
        if (rc != 0) {
            return std::unexpected(
                Error{ErrorCode::InferenceFailed, "Failed to decode prefill batch"});
        }
    }

    // --- Autoregressive generation ---
    int current_pos = base_pos + static_cast<int>(prompt_tokens.size());
    llama_token new_token;

    // Pre-allocate a single-token batch and reuse it for every decode step.
    llama_batch ar_batch = llama_batch_init(1, 0, 1);

    while (true) {
        if (should_cancel && should_cancel()) {
            return std::unexpected(
                Error{ErrorCode::RequestCancelled, "Request cancelled during generation"});
        }

        new_token = llama_sampler_sample(sampler_, ctx_, -1);
        if (llama_vocab_is_eog(vocab_, new_token))
            break;

        char buff[256];
        const int n = llama_token_to_piece(vocab_, new_token, buff, sizeof(buff), 0, true);
        if (n < 0) {
            return std::unexpected(Error{ErrorCode::Unknown, "Failed to convert token"});
        }

        generated_text.append(buff, static_cast<size_t>(n));
        ++token_count;

        // Sentinel detection: stop streaming to callback when <tool_call> appears.
        // Only check the tail region that could contain the newly completed tag.
        if (grammar_active_ && !in_tool_call) {
            static constexpr std::string_view kOpenTag = "<tool_call>";
            const size_t check_start =
                generated_text.size() > kOpenTag.size() + static_cast<size_t>(n)
                    ? generated_text.size() - kOpenTag.size() - static_cast<size_t>(n)
                    : 0;
            if (generated_text.find(kOpenTag.data(), check_start) != std::string::npos) {
                in_tool_call = true;
            }
        }

        // Stop generation when sentinel closing tag is complete
        if (in_tool_call && generated_text.ends_with("</tool_call>")) {
            break;
        }

        if (!stop_sequences.empty()) {
            size_t match_len = find_stop_sequence(generated_text, stop_sequences);
            if (match_len > 0) {
                generated_text.resize(generated_text.size() - match_len);
                break;
            }
        }

        // Only stream to user callback when not inside a tool call
        if (on_token && !in_tool_call) {
            auto action =
                invoke_token_callback(*on_token, std::string_view(buff, static_cast<size_t>(n)));
            if (!action) {
                return std::unexpected(action.error());
            }
            if (*action == TokenAction::Stop)
                break;
        }

        if (token_count >= effective_max)
            break;

        if (current_pos >= context_size_)
            break;

        // Decode the new token using the pre-allocated batch
        ar_batch.token[0] = new_token;
        ar_batch.pos[0] = static_cast<llama_pos>(current_pos);
        ar_batch.n_seq_id[0] = 1;
        ar_batch.seq_id[0][0] = 0;
        ar_batch.logits[0] = true;
        ar_batch.n_tokens = 1;

        int rc = llama_decode(ctx_, ar_batch);
        if (rc != 0) {
            llama_batch_free(ar_batch);
            return std::unexpected(Error{ErrorCode::InferenceFailed, "Failed to decode token"});
        }
        ++current_pos;
    }

    llama_batch_free(ar_batch);
    return generated_text;
}

// ============================================================================
// Prompt Formatting
// ============================================================================

const std::vector<llama_chat_message>& Model::build_llama_messages() {
    if (llama_msgs_cache_size_ == messages_.size()) {
        return llama_msgs_cache_;
    }
    llama_msgs_cache_.clear();
    llama_msgs_cache_.reserve(messages_.size());
    for (const auto& msg : messages_) {
        llama_msgs_cache_.push_back({role_to_string(msg.role), msg.content.c_str()});
    }
    llama_msgs_cache_size_ = messages_.size();
    return llama_msgs_cache_;
}

Expected<std::string> Model::format_prompt() {
    const auto& llama_msgs = build_llama_messages();
    int new_len =
        llama_chat_apply_template(tmpl_, llama_msgs.data(), llama_msgs.size(), true, nullptr, 0);

    if (new_len < 0) {
        return std::unexpected(
            Error{ErrorCode::TemplateRenderFailed, "llama_chat_apply_template failed"});
    }

    if (new_len == 0) {
        return std::string{};
    }

    if (new_len > static_cast<int>(formatted_.size())) {
        formatted_.resize(static_cast<size_t>(new_len));
    }

    new_len = llama_chat_apply_template(tmpl_, llama_msgs.data(), llama_msgs.size(), true,
                                        formatted_.data(), formatted_.size());
    if (new_len < 0) {
        return std::unexpected(
            Error{ErrorCode::TemplateRenderFailed, "llama_chat_apply_template failed"});
    }

    if (new_len < prev_len_)
        clear_kv_cache();

    return std::string(formatted_.begin() + prev_len_, formatted_.begin() + new_len);
}

void Model::finalize_response() {
    const auto& llama_msgs = build_llama_messages();
    int new_prev_len =
        llama_chat_apply_template(tmpl_, llama_msgs.data(), llama_msgs.size(), false, nullptr, 0);
    if (new_prev_len > 0)
        prev_len_ = new_prev_len;
}

// ============================================================================
// KV Cache
// ============================================================================

void Model::clear_kv_cache() {
    if (ctx_) {
        llama_memory_clear(llama_get_memory(ctx_), false);
        prev_len_ = 0;
    }
}

// ============================================================================
// High-Level Generate
// ============================================================================

Expected<Response> Model::generate(const std::string& user_message,
                                   std::optional<TokenCallback> on_token,
                                   CancellationCallback should_cancel) {
    auto start_time = std::chrono::steady_clock::now();

    auto add_result = add_message(Message::user(user_message));
    if (!add_result) {
        return std::unexpected(add_result.error());
    }

    std::chrono::steady_clock::time_point first_token_time;
    bool first_token_received = false;
    int completion_tokens = 0;

    const std::optional<TokenCallback> effective_callback =
        on_token ? std::move(on_token) : config_.on_token;

    auto wrapped_callback = [&](std::string_view token) -> TokenAction {
        if (!first_token_received) {
            first_token_time = std::chrono::steady_clock::now();
            first_token_received = true;
        }
        ++completion_tokens;
        if (effective_callback) {
            return (*effective_callback)(token);
        }
        return TokenAction::Continue;
    };

    auto prompt_result = format_prompt();
    if (!prompt_result) {
        messages_.pop_back();
        return std::unexpected(prompt_result.error());
    }

    auto tokens_result = tokenize(*prompt_result);
    if (!tokens_result) {
        messages_.pop_back();
        return std::unexpected(tokens_result.error());
    }

    int prompt_tokens = static_cast<int>(tokens_result->size());

    auto generate_result =
        run_inference(*tokens_result, config_.max_tokens, config_.stop_sequences,
                      std::optional<TokenCallback>(wrapped_callback), should_cancel);

    if (!generate_result) {
        messages_.pop_back();
        return std::unexpected(generate_result.error());
    }

    std::string generated_text = std::move(*generate_result);
    messages_.push_back(Message::assistant(generated_text));
    estimated_tokens_ += estimate_tokens(generated_text) + kTemplateOverheadPerMessage;
    finalize_response();

    auto end_time = std::chrono::steady_clock::now();

    Response response;
    response.text = std::move(generated_text);
    response.usage.prompt_tokens = prompt_tokens;
    response.usage.completion_tokens = completion_tokens;
    response.usage.total_tokens = prompt_tokens + completion_tokens;

    response.metrics.latency_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    if (first_token_received) {
        response.metrics.time_to_first_token_ms =
            std::chrono::duration_cast<std::chrono::milliseconds>(first_token_time - start_time);
        auto generation_time =
            std::chrono::duration_cast<std::chrono::milliseconds>(end_time - first_token_time);
        if (generation_time.count() > 0) {
            response.metrics.tokens_per_second =
                (completion_tokens * 1000.0) / generation_time.count();
        }
    }

    return response;
}

// ============================================================================
// Low-Level Generate (for Agent tool loop)
// ============================================================================

Expected<Model::GenerationResult>
Model::generate_from_history(std::optional<TokenCallback> on_token,
                             CancellationCallback should_cancel) {
    // Reset lazy grammar sampler so the trigger state is fresh for each
    // generation pass. Without this, the grammar stays "triggered" after
    // a tool call and constrains the follow-up natural language response.
    if (grammar_active_) {
        rebuild_sampler_with_grammar();
    }

    auto prompt_result = format_prompt();
    if (!prompt_result) {
        return std::unexpected(prompt_result.error());
    }

    auto tokens_result = tokenize(*prompt_result);
    if (!tokens_result) {
        return std::unexpected(tokens_result.error());
    }

    int prompt_tokens = static_cast<int>(tokens_result->size());

    auto text_result = run_inference(*tokens_result, config_.max_tokens, config_.stop_sequences,
                                     on_token, should_cancel);

    if (!text_result) {
        return std::unexpected(text_result.error());
    }

    bool tool_detected = grammar_active_ && (text_result->find("<tool_call>") != std::string::npos);

    return GenerationResult{std::move(*text_result), prompt_tokens, tool_detected};
}

// ============================================================================
// History Management
// ============================================================================

void Model::set_system_prompt(const std::string& prompt) {
    Message sys_msg = Message::system(prompt);

    if (!messages_.empty() && messages_[0].role == Role::System) {
        estimated_tokens_ -= estimate_tokens(messages_[0].content) + kTemplateOverheadPerMessage;
        messages_[0] = sys_msg;
    } else {
        messages_.insert(messages_.begin(), sys_msg);
    }

    estimated_tokens_ += estimate_tokens(prompt) + kTemplateOverheadPerMessage;
    llama_msgs_cache_size_ = 0; // Invalidate — content changed even if size didn't
}

Expected<void> Model::add_message(const Message& message) {
    auto err = validate_role_sequence(messages_, message.role);
    if (!err) {
        return std::unexpected(err.error());
    }

    messages_.push_back(message);
    estimated_tokens_ += estimate_tokens(message.content) + kTemplateOverheadPerMessage;
    trim_history_to_fit();
    return {};
}

std::vector<Message> Model::get_history() const {
    return messages_;
}

void Model::clear_history() {
    messages_.clear();
    estimated_tokens_ = 0;
    clear_kv_cache();
}

// ============================================================================
// Context Info
// ============================================================================

int Model::context_size() const noexcept {
    return config_.context_size;
}
int Model::estimated_tokens() const noexcept {
    return estimated_tokens_;
}
bool Model::is_context_exceeded() const noexcept {
    return estimated_tokens_ > config_.context_size;
}

int Model::estimate_tokens(const std::string& text) const {
    // If llama.cpp is initialized, use real tokenization
    if (vocab_) {
        static_assert(sizeof(int) == sizeof(llama_token));
        const int32_t raw =
            llama_tokenize(vocab_, text.c_str(), text.length(), nullptr, 0, false, true);
        const int n = (raw < 0) ? -raw : raw;
        if (n > 0)
            return n;
    }
    return std::max(1, static_cast<int>(text.length() / 4));
}

// ============================================================================
// Tool Grammar
// ============================================================================

bool Model::set_tool_grammar(const std::string& grammar_str) {
    tool_grammar_str_ = grammar_str;
    return rebuild_sampler_with_grammar();
}

void Model::clear_tool_grammar() noexcept {
    if (!grammar_active_)
        return;
    tool_grammar_str_.clear();
    grammar_active_ = false;

    if (sampler_) {
        llama_sampler_free(sampler_);
    }
    sampler_ = create_sampler_chain();
}

bool Model::rebuild_sampler_with_grammar() {
    auto chain_params = llama_sampler_chain_default_params();
    chain_params.no_perf = false;
    llama_sampler* chain = llama_sampler_chain_init(chain_params);
    if (!chain)
        return false;

    add_sampling_stages(chain);

    // Lazy grammar sampler — activates when "<tool_call>" appears in output
    const char* trigger = "<tool_call>";
    const char* trigger_patterns[] = {trigger};
    auto* grammar_sampler = llama_sampler_init_grammar_lazy_patterns(
        vocab_, tool_grammar_str_.c_str(), "root", trigger_patterns, 1, nullptr, 0);
    if (!grammar_sampler) {
        llama_sampler_free(chain);
        return false;
    }
    llama_sampler_chain_add(chain, grammar_sampler);

    add_dist_sampler(chain);

    // Only replace the old sampler after the new one is fully built
    if (sampler_) {
        llama_sampler_free(sampler_);
    }
    sampler_ = chain;
    grammar_active_ = true;
    return true;
}

// ============================================================================
// Sampler Chain
// ============================================================================

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
    uint32_t seed = make_sampler_seed(config_.sampling.seed);
    if (auto* d = llama_sampler_init_dist(seed)) {
        llama_sampler_chain_add(chain, d);
    } else if (auto* g = llama_sampler_init_greedy()) {
        llama_sampler_chain_add(chain, g);
    }
}

llama_sampler* Model::create_sampler_chain() {
    auto chain_params = llama_sampler_chain_default_params();
    chain_params.no_perf = false;
    llama_sampler* chain = llama_sampler_chain_init(chain_params);
    if (!chain)
        return nullptr;

    add_sampling_stages(chain);
    add_dist_sampler(chain);

    return chain;
}

// ============================================================================
// Stop Sequence Detection
// ============================================================================

size_t Model::find_stop_sequence(const std::string& generated_text,
                                 const std::vector<std::string>& stop_sequences) const {
    for (const auto& s : stop_sequences) {
        if (s.empty())
            continue;
        if (generated_text.size() >= s.size() &&
            generated_text.compare(generated_text.size() - s.size(), s.size(), s) == 0) {
            return s.size();
        }
    }
    return 0;
}

void Model::trim_history_to_fit() {
    const size_t system_offset =
        (!messages_.empty() && messages_.front().role == Role::System) ? 1u : 0u;
    const size_t max_messages = config_.max_history_messages;

    if (messages_.size() <= system_offset + max_messages) {
        return;
    }

    size_t erase_end = messages_.size() - max_messages;
    if (erase_end < system_offset) {
        erase_end = system_offset;
    }

    while (erase_end < messages_.size() && messages_[erase_end].role != Role::User) {
        ++erase_end;
    }

    if (erase_end <= system_offset) {
        return;
    }

    for (size_t index = system_offset; index < erase_end; ++index) {
        estimated_tokens_ -=
            estimate_tokens(messages_[index].content) + kTemplateOverheadPerMessage;
    }
    if (estimated_tokens_ < 0)
        estimated_tokens_ = 0;

    messages_.erase(messages_.begin() + static_cast<std::ptrdiff_t>(system_offset),
                    messages_.begin() + static_cast<std::ptrdiff_t>(erase_end));
    clear_kv_cache();
}

} // namespace zoo::core
