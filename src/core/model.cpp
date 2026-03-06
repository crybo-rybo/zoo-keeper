#include "zoo/core/model.hpp"
#include <llama.h>
#include <algorithm>
#include <chrono>
#include <mutex>
#include <ctime>

namespace zoo::core {

// ============================================================================
// Static initialization
// ============================================================================

static std::once_flag g_init_flag;

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
    if (sampler_) { llama_sampler_free(sampler_); sampler_ = nullptr; }
    if (ctx_) { llama_free(ctx_); ctx_ = nullptr; }
    if (llama_model_) { llama_model_free(llama_model_); llama_model_ = nullptr; }
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

    llama_log_set([](enum ggml_log_level level, const char* text, void*) {
        if (level >= GGML_LOG_LEVEL_WARN) {
            fprintf(stderr, "%s", text);
        }
    }, nullptr);

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

    ctx_ = llama_init_from_model(llama_model_, ctx_params);
    if (!ctx_) {
        llama_model_free(llama_model_); llama_model_ = nullptr;
        return std::unexpected(Error{ErrorCode::ContextCreationFailed,
            "Failed to create llama context"});
    }

    context_size_ = static_cast<int>(llama_n_ctx(ctx_));

    vocab_ = llama_model_get_vocab(llama_model_);
    if (!vocab_) {
        llama_free(ctx_); llama_model_free(llama_model_);
        ctx_ = nullptr; llama_model_ = nullptr;
        return std::unexpected(Error{ErrorCode::BackendInitFailed,
            "Failed to get model vocabulary"});
    }

    sampler_ = create_sampler_chain();
    if (!sampler_) {
        llama_free(ctx_); llama_model_free(llama_model_);
        ctx_ = nullptr; llama_model_ = nullptr;
        return std::unexpected(Error{ErrorCode::BackendInitFailed,
            "Failed to create sampler chain"});
    }

    tmpl_ = llama_model_chat_template(llama_model_, nullptr);
    formatted_.resize(context_size_ * 4);
    prev_len_ = 0;
    return {};
}

// ============================================================================
// Tokenization
// ============================================================================

Expected<std::vector<int>> Model::tokenize(const std::string& text) {
    static_assert(sizeof(int) == sizeof(llama_token));
    static_assert(alignof(int) == alignof(llama_token));

    const bool is_first = llama_memory_seq_pos_max(llama_get_memory(ctx_), 0) == -1;
    const int32_t raw = llama_tokenize(vocab_, text.c_str(), text.length(), nullptr, 0, is_first, true);
    if (raw == INT32_MIN) {
        return std::unexpected(Error{ErrorCode::TokenizationFailed, "Tokenization overflow"});
    }
    const int n = (raw < 0) ? -raw : raw;
    if (n == 0) return std::vector<int>{};

    std::vector<int> tokens(n);
    if (llama_tokenize(vocab_, text.c_str(), text.length(),
            reinterpret_cast<llama_token*>(tokens.data()), tokens.size(), is_first, true) < 0) {
        return std::unexpected(Error{ErrorCode::TokenizationFailed, "Tokenization failed"});
    }
    return tokens;
}

// ============================================================================
// Inference
// ============================================================================

Expected<std::string> Model::run_inference(
    const std::vector<int>& prompt_tokens, int max_tokens,
    const std::vector<std::string>& stop_sequences,
    const std::optional<std::function<void(std::string_view)>>& on_token
) {
    std::string generated_text;
    generated_text.reserve(max_tokens > 0 ? static_cast<size_t>(max_tokens) * 8 : 4096);
    int token_count = 0;

    llama_batch batch = llama_batch_get_one(
        const_cast<llama_token*>(reinterpret_cast<const llama_token*>(prompt_tokens.data())),
        static_cast<int32_t>(prompt_tokens.size()));
    llama_token new_token;

    while (true) {
        int n_ctx_used = llama_memory_seq_pos_max(llama_get_memory(ctx_), 0) + 1;
        if (n_ctx_used + batch.n_tokens > context_size_) {
            if (token_count > 0) break;
            return std::unexpected(Error{ErrorCode::ContextWindowExceeded,
                "Batch tokens exceed context size"});
        }

        if (llama_decode(ctx_, batch) != 0) {
            return std::unexpected(Error{ErrorCode::InferenceFailed, "Failed to decode batch"});
        }

        new_token = llama_sampler_sample(sampler_, ctx_, -1);
        if (llama_vocab_is_eog(vocab_, new_token)) break;

        char buff[256];
        const int n = llama_token_to_piece(vocab_, new_token, buff, sizeof(buff), 0, true);
        if (n < 0) {
            return std::unexpected(Error{ErrorCode::Unknown, "Failed to convert token"});
        }

        generated_text.append(buff, static_cast<size_t>(n));
        ++token_count;

        if (max_tokens > 0 && token_count >= max_tokens) {
            if (on_token) (*on_token)(std::string_view(buff, static_cast<size_t>(n)));
            break;
        }

        if (!stop_sequences.empty()) {
            size_t match_len = find_stop_sequence(generated_text, stop_sequences);
            if (match_len > 0) {
                generated_text.resize(generated_text.size() - match_len);
                break;
            }
        }

        if (on_token) (*on_token)(std::string_view(buff, static_cast<size_t>(n)));
        batch = llama_batch_get_one(&new_token, 1);
    }

    return generated_text;
}

// ============================================================================
// Prompt Formatting
// ============================================================================

std::vector<llama_chat_message> Model::build_llama_messages() const {
    std::vector<llama_chat_message> llama_msgs;
    llama_msgs.reserve(messages_.size());
    for (const auto& msg : messages_) {
        llama_msgs.push_back({role_to_string(msg.role), msg.content.c_str()});
    }
    return llama_msgs;
}

Expected<std::string> Model::format_prompt() {
    auto llama_msgs = build_llama_messages();
    int new_len = llama_chat_apply_template(
        tmpl_, llama_msgs.data(), llama_msgs.size(), true, formatted_.data(), formatted_.size());

    if (new_len > static_cast<int>(formatted_.size())) {
        formatted_.resize(new_len);
        new_len = llama_chat_apply_template(
            tmpl_, llama_msgs.data(), llama_msgs.size(), true, formatted_.data(), formatted_.size());
    }

    if (new_len < 0) {
        return std::unexpected(Error{ErrorCode::TemplateRenderFailed,
            "llama_chat_apply_template failed"});
    }

    if (new_len < prev_len_) clear_kv_cache();

    return std::string(formatted_.begin() + prev_len_, formatted_.begin() + new_len);
}

void Model::finalize_response() {
    auto llama_msgs = build_llama_messages();
    int new_prev_len = llama_chat_apply_template(
        tmpl_, llama_msgs.data(), llama_msgs.size(), false, nullptr, 0);
    if (new_prev_len > 0) prev_len_ = new_prev_len;
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

Expected<Response> Model::generate(
    const std::string& user_message,
    std::optional<std::function<void(std::string_view)>> on_token
) {
    auto start_time = std::chrono::steady_clock::now();

    auto add_result = add_message(Message::user(user_message));
    if (!add_result) {
        return std::unexpected(add_result.error());
    }

    std::chrono::steady_clock::time_point first_token_time;
    bool first_token_received = false;
    int completion_tokens = 0;

    auto wrapped_callback = [&](std::string_view token) {
        if (!first_token_received) {
            first_token_time = std::chrono::steady_clock::now();
            first_token_received = true;
        }
        ++completion_tokens;
        if (on_token) {
            (*on_token)(token);
        }
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

    auto generate_result = run_inference(
        *tokens_result,
        config_.max_tokens,
        config_.stop_sequences,
        std::optional<std::function<void(std::string_view)>>(wrapped_callback)
    );

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

    response.metrics.latency_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time);

    if (first_token_received) {
        response.metrics.time_to_first_token_ms =
            std::chrono::duration_cast<std::chrono::milliseconds>(
                first_token_time - start_time);
        auto generation_time = std::chrono::duration_cast<std::chrono::milliseconds>(
            end_time - first_token_time);
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

Expected<Model::GenerationResult> Model::generate_from_history(
    std::optional<std::function<void(std::string_view)>> on_token
) {
    auto prompt_result = format_prompt();
    if (!prompt_result) {
        return std::unexpected(prompt_result.error());
    }

    auto tokens_result = tokenize(*prompt_result);
    if (!tokens_result) {
        return std::unexpected(tokens_result.error());
    }

    int prompt_tokens = static_cast<int>(tokens_result->size());

    auto text_result = run_inference(
        *tokens_result,
        config_.max_tokens,
        config_.stop_sequences,
        on_token
    );

    if (!text_result) {
        return std::unexpected(text_result.error());
    }

    return GenerationResult{std::move(*text_result), prompt_tokens};
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
}

Expected<void> Model::add_message(const Message& message) {
    auto err = validate_role_sequence(messages_, message.role);
    if (!err) {
        return std::unexpected(err.error());
    }

    messages_.push_back(message);
    estimated_tokens_ += estimate_tokens(message.content) + kTemplateOverheadPerMessage;
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

int Model::context_size() const { return config_.context_size; }
int Model::estimated_tokens() const { return estimated_tokens_; }
bool Model::is_context_exceeded() const { return estimated_tokens_ > config_.context_size; }

int Model::estimate_tokens(const std::string& text) const {
    // If llama.cpp is initialized, use real tokenization
    if (vocab_) {
        static_assert(sizeof(int) == sizeof(llama_token));
        const int32_t raw = llama_tokenize(vocab_, text.c_str(), text.length(), nullptr, 0, false, true);
        const int n = (raw < 0) ? -raw : raw;
        if (n > 0) return n;
    }
    return std::max(1, static_cast<int>(text.length() / 4));
}

// ============================================================================
// Sampler Chain
// ============================================================================

llama_sampler* Model::create_sampler_chain() {
    auto chain_params = llama_sampler_chain_default_params();
    chain_params.no_perf = false;
    llama_sampler* chain = llama_sampler_chain_init(chain_params);
    if (!chain) return nullptr;

    const auto& sp = config_.sampling;
    if (sp.repeat_penalty != 1.0f) {
        if (auto* p = llama_sampler_init_penalties(sp.repeat_last_n, sp.repeat_penalty, 0.0f, 0.0f))
            llama_sampler_chain_add(chain, p);
    }
    if (sp.top_k > 0) {
        if (auto* p = llama_sampler_init_top_k(sp.top_k)) llama_sampler_chain_add(chain, p);
    }
    if (sp.top_p < 1.0f) {
        if (auto* p = llama_sampler_init_top_p(sp.top_p, 1)) llama_sampler_chain_add(chain, p);
    }
    if (sp.temperature > 0.0f) {
        if (auto* p = llama_sampler_init_temp(sp.temperature)) llama_sampler_chain_add(chain, p);
    }

    uint32_t seed = (sp.seed < 0) ? static_cast<uint32_t>(time(nullptr)) : static_cast<uint32_t>(sp.seed);
    if (auto* d = llama_sampler_init_dist(seed)) {
        llama_sampler_chain_add(chain, d);
    } else if (auto* g = llama_sampler_init_greedy()) {
        llama_sampler_chain_add(chain, g);
    }

    return chain;
}

// ============================================================================
// Stop Sequence Detection
// ============================================================================

size_t Model::find_stop_sequence(
    const std::string& generated_text,
    const std::vector<std::string>& stop_sequences) const {
    for (const auto& s : stop_sequences) {
        if (s.empty()) continue;
        if (generated_text.size() >= s.size() &&
            generated_text.compare(generated_text.size() - s.size(), s.size(), s) == 0) {
            return s.size();
        }
    }
    return 0;
}

} // namespace zoo::core
