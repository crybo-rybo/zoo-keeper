#include "zoo/core/model.hpp"
#include <llama.h>
#include <mutex>
#include <ctime>

namespace zoo::core {

// ============================================================================
// LlamaBackend — production IBackend using llama.cpp
// ============================================================================

class LlamaBackend : public IBackend {
public:
    LlamaBackend();
    ~LlamaBackend() override;

    LlamaBackend(const LlamaBackend&) = delete;
    LlamaBackend& operator=(const LlamaBackend&) = delete;

    Expected<void> initialize(const Config& config) override;
    Expected<std::vector<int>> tokenize(const std::string& text) override;
    Expected<std::string> generate(
        const std::vector<int>& prompt_tokens,
        int max_tokens,
        const std::vector<std::string>& stop_sequences,
        const std::optional<std::function<void(std::string_view)>>& on_token = std::nullopt
    ) override;
    int get_context_size() const override;
    void clear_kv_cache() override;
    Expected<std::string> format_prompt(const std::vector<Message>& messages) override;
    void finalize_response(const std::vector<Message>& messages) override;

private:
    static void initialize_global();

    size_t find_stop_sequence(const std::string& generated_text,
                              const std::vector<std::string>& stop_sequences) const;
    llama_sampler* create_sampler_chain(const Config& config);
    std::vector<llama_chat_message> build_llama_messages(const std::vector<Message>& messages) const;

    llama_model* model_ = nullptr;
    llama_context* ctx_ = nullptr;
    llama_sampler* sampler_ = nullptr;
    const llama_vocab* vocab_ = nullptr;

    int context_size_ = 0;
    int vocab_size_ = 0;
    const char* tmpl_ = nullptr;

    int prev_len_ = 0;
    std::vector<char> formatted_;
};

static std::once_flag g_init_flag;

LlamaBackend::LlamaBackend() {
    llama_log_set([](enum ggml_log_level level, const char* text, void*) {
        if (level >= GGML_LOG_LEVEL_WARN) {
            fprintf(stderr, "%s", text);
        }
    }, nullptr);
}

LlamaBackend::~LlamaBackend() {
    if (sampler_) { llama_sampler_free(sampler_); sampler_ = nullptr; }
    if (ctx_) { llama_free(ctx_); ctx_ = nullptr; }
    if (model_) { llama_model_free(model_); model_ = nullptr; }
}

void LlamaBackend::initialize_global() {
    std::call_once(g_init_flag, []() {
        llama_backend_init();
        ggml_backend_load_all();
    });
}

Expected<void> LlamaBackend::initialize(const Config& config) {
    initialize_global();

    auto validation = config.validate();
    if (!validation) {
        return std::unexpected(validation.error());
    }

    auto model_params = llama_model_default_params();
    model_params.n_gpu_layers = config.n_gpu_layers;
    model_params.use_mmap = config.use_mmap;
    model_params.use_mlock = config.use_mlock;

    model_ = llama_model_load_from_file(config.model_path.c_str(), model_params);
    if (!model_) {
        return std::unexpected(Error{ErrorCode::ModelLoadFailed,
            "Failed to load model from path: " + config.model_path});
    }

    auto ctx_params = llama_context_default_params();
    ctx_params.n_ctx = static_cast<uint32_t>(config.context_size);
    ctx_params.n_batch = 512;
    ctx_params.n_ubatch = 512;
    ctx_params.n_threads = -1;
    ctx_params.n_threads_batch = -1;
    ctx_params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;

    ctx_ = llama_init_from_model(model_, ctx_params);
    if (!ctx_) {
        llama_model_free(model_); model_ = nullptr;
        return std::unexpected(Error{ErrorCode::ContextCreationFailed,
            "Failed to create llama context"});
    }

    context_size_ = static_cast<int>(llama_n_ctx(ctx_));

    vocab_ = llama_model_get_vocab(model_);
    if (!vocab_) {
        llama_free(ctx_); llama_model_free(model_);
        ctx_ = nullptr; model_ = nullptr;
        return std::unexpected(Error{ErrorCode::BackendInitFailed,
            "Failed to get model vocabulary"});
    }
    vocab_size_ = llama_vocab_n_tokens(vocab_);

    sampler_ = create_sampler_chain(config);
    if (!sampler_) {
        llama_free(ctx_); llama_model_free(model_);
        ctx_ = nullptr; model_ = nullptr;
        return std::unexpected(Error{ErrorCode::BackendInitFailed,
            "Failed to create sampler chain"});
    }

    tmpl_ = llama_model_chat_template(model_, nullptr);
    formatted_.resize(context_size_ * 4);
    prev_len_ = 0;
    return {};
}

Expected<std::vector<int>> LlamaBackend::tokenize(const std::string& text) {
    if (!model_) {
        return std::unexpected(Error{ErrorCode::BackendInitFailed,
            "Backend not initialized"});
    }

    static_assert(sizeof(int) == sizeof(llama_token));
    static_assert(alignof(int) == alignof(llama_token));

    const bool is_first = llama_memory_seq_pos_max(llama_get_memory(ctx_), 0) == -1;
    const int32_t raw = llama_tokenize(vocab_, text.c_str(), text.length(), nullptr, 0, is_first, true);
    if (raw == INT32_MIN) {
        return std::unexpected(Error{ErrorCode::TokenizationFailed,
            "Tokenization overflow"});
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

Expected<std::string> LlamaBackend::generate(
    const std::vector<int>& prompt_tokens, int max_tokens,
    const std::vector<std::string>& stop_sequences,
    const std::optional<std::function<void(std::string_view)>>& on_token
) {
    if (!ctx_ || !model_ || !sampler_) {
        return std::unexpected(Error{ErrorCode::BackendInitFailed, "Backend not initialized"});
    }

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

int LlamaBackend::get_context_size() const { return context_size_; }

void LlamaBackend::clear_kv_cache() {
    if (ctx_) {
        llama_memory_clear(llama_get_memory(ctx_), false);
        prev_len_ = 0;
    }
}

std::vector<llama_chat_message> LlamaBackend::build_llama_messages(
    const std::vector<Message>& messages) const {
    std::vector<llama_chat_message> llama_msgs;
    llama_msgs.reserve(messages.size());
    for (const auto& msg : messages) {
        llama_msgs.push_back({role_to_string(msg.role), msg.content.c_str()});
    }
    return llama_msgs;
}

Expected<std::string> LlamaBackend::format_prompt(const std::vector<Message>& messages) {
    if (!model_) {
        return std::unexpected(Error{ErrorCode::BackendInitFailed, "Backend not initialized"});
    }

    auto llama_msgs = build_llama_messages(messages);
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

void LlamaBackend::finalize_response(const std::vector<Message>& messages) {
    if (!model_) return;
    auto llama_msgs = build_llama_messages(messages);
    int new_prev_len = llama_chat_apply_template(
        tmpl_, llama_msgs.data(), llama_msgs.size(), false, nullptr, 0);
    if (new_prev_len > 0) prev_len_ = new_prev_len;
}

size_t LlamaBackend::find_stop_sequence(
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

llama_sampler* LlamaBackend::create_sampler_chain(const Config& config) {
    auto chain_params = llama_sampler_chain_default_params();
    chain_params.no_perf = false;
    llama_sampler* chain = llama_sampler_chain_init(chain_params);
    if (!chain) return nullptr;

    const auto& sp = config.sampling;
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

// Factory function
std::unique_ptr<IBackend> create_llama_backend() {
    return std::make_unique<LlamaBackend>();
}

} // namespace zoo::core
