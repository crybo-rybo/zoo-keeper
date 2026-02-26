#include "zoo/backend/llama_backend.hpp"
#include <llama.h>
#include <atomic>
#include <mutex>

namespace zoo {
namespace backend {

// File-scope statics for idempotent global initialization
static std::once_flag g_init_flag;
static std::atomic<bool> g_initialized{false};

void LlamaBackend::initialize_global() {
    std::call_once(g_init_flag, []() {
        llama_backend_init();
        ggml_backend_load_all();
        g_initialized.store(true, std::memory_order_release);
    });
}

void LlamaBackend::shutdown_global() {
    llama_backend_free();
}

LlamaBackend::LlamaBackend() {
    // Suppress llama.cpp/ggml internal logging to prevent output noise
    // (Metal pipeline compilation, model metadata, tensor loading, etc.)
    llama_log_set([](enum ggml_log_level level, const char* text, void*) {
        if (level >= GGML_LOG_LEVEL_WARN) {
            fprintf(stderr, "%s", text);
        }
    }, nullptr);

    // Note: Global backend initialization (llama_backend_init / ggml_backend_load_all)
    // is no longer performed here. Call LlamaBackend::initialize_global() once at
    // program start, or rely on the lazy initialization in LlamaBackend::initialize().
}

LlamaBackend::~LlamaBackend() {
    // Free resources in reverse order of creation
    if (sampler_ != nullptr) {
        llama_sampler_free(sampler_);
        sampler_ = nullptr;
    }
    if (ctx_ != nullptr) {
        llama_free(ctx_);
        ctx_ = nullptr;
    }
    if (model_ != nullptr) {
        llama_model_free(model_);
        model_ = nullptr;
    }

    // Note: We don't call llama_backend_free() here as it should only be called
    // once at program exit via LlamaBackend::shutdown_global(), not per instance.
}

Expected<void> LlamaBackend::initialize(const Config& config) {
    // Ensure global backend is initialized (idempotent lazy-init for backward compatibility)
    initialize_global();

    // Validate config first
    auto validation = config.validate();
    if (!validation) {
        return tl::unexpected(validation.error());
    }

    // Set up model parameters
    auto model_params = llama_model_default_params();
    model_params.n_gpu_layers = config.n_gpu_layers;
    model_params.use_mmap = config.use_mmap;
    model_params.use_mlock = config.use_mlock;

    // Load model
    model_ = llama_model_load_from_file(config.model_path.c_str(), model_params);
    if (model_ == nullptr) {
        return tl::unexpected(Error{
            ErrorCode::ModelLoadFailed,
            "Failed to load model from path: " + config.model_path
        });
    }

    // Set up context parameters
    auto ctx_params = llama_context_default_params();
    ctx_params.n_ctx = static_cast<uint32_t>(config.context_size);
    ctx_params.n_batch = 512;  // Reasonable default for batch processing
    ctx_params.n_ubatch = 512; // Physical batch size
    ctx_params.n_threads = -1; // Auto-detect thread count
    ctx_params.n_threads_batch = -1; // Auto-detect
    ctx_params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    ctx_params.type_k = static_cast<ggml_type>(config.kv_cache_type_k);
    ctx_params.type_v = static_cast<ggml_type>(config.kv_cache_type_v);

    // Create context
    ctx_ = llama_init_from_model(model_, ctx_params);
    if (ctx_ == nullptr) {
        llama_model_free(model_);
        model_ = nullptr;
        return tl::unexpected(Error{
            ErrorCode::ContextCreationFailed,
            "Failed to create llama context"
        });
    }

    // Cache context size and vocab size
    context_size_ = static_cast<int>(llama_n_ctx(ctx_));

    // Get vocabulary from model
    vocab_ = llama_model_get_vocab(model_);
    if (vocab_ == nullptr) {
        llama_free(ctx_);
        llama_model_free(model_);
        ctx_ = nullptr;
        model_ = nullptr;
        return tl::unexpected(Error{
            ErrorCode::BackendInitFailed,
            "Failed to get model vocabulary"
        });
    }
    vocab_size_ = llama_vocab_n_tokens(vocab_);

    // Create sampler chain
    sampler_ = create_sampler_chain(config);
    if (sampler_ == nullptr) {
        llama_free(ctx_);
        llama_model_free(model_);
        ctx_ = nullptr;
        model_ = nullptr;
        return tl::unexpected(Error{
            ErrorCode::BackendInitFailed,
            "Failed to create sampler chain"
        });
    }

    // Cache the model's chat template pointer (model-lifetime: valid as long as model_ is alive).
    // May be nullptr if model has no embedded template — llama_chat_apply_template
    // falls back to ChatML format in that case.
    tmpl_ = llama_model_chat_template(model_, nullptr);

    // Initialize prompt formatting state
    // Use context_size * 4 since context_size is in tokens (~4 chars/token)
    formatted_.resize(context_size_ * 4);
    prev_len_ = 0;
    kv_cache_token_count_ = 0;
    return {};
}

Expected<std::vector<int>> LlamaBackend::tokenize(const std::string& text) {
    if (model_ == nullptr) {
        return tl::unexpected(Error{
            ErrorCode::BackendInitFailed,
            "Backend not initialized - model is null"
        });
    }

    static_assert(sizeof(int) == sizeof(llama_token), "int must match llama_token size");
    static_assert(alignof(int) == alignof(llama_token), "int must match llama_token alignment");

    const bool is_first = llama_memory_seq_pos_max(llama_get_memory(ctx_), 0) == -1;

    // Calculate the number of tokens needed for the text.
    // llama_tokenize returns negative required size when output buffer is null/too small,
    // or INT32_MIN on overflow.
    const int32_t raw = llama_tokenize(vocab_, text.c_str(), text.length(), nullptr, 0, is_first, true);
    if (raw == INT32_MIN) {
        return tl::unexpected(Error{
            ErrorCode::TokenizationFailed,
            "Tokenization overflow (input too large)"
        });
    }
    const int n_prompt_tokens = (raw < 0) ? -raw : raw;
    if (n_prompt_tokens == 0) {
        return std::vector<int>{};
    }
    std::vector<int> tokens(n_prompt_tokens);
    if(llama_tokenize(
        vocab_,
        text.c_str(),
        text.length(),
        reinterpret_cast<llama_token*>(tokens.data()),
        tokens.size(),
        is_first,
        true) < 0) {
        return tl::unexpected(Error{
            ErrorCode::TokenizationFailed,
            "Tokenization failed - output buffer too small or other error"
        });
    }

    return tokens;
}

Expected<std::string> LlamaBackend::generate(
    const std::vector<int>& prompt_tokens,
    int max_tokens,
    const std::vector<std::string>& stop_sequences,
    const std::optional<std::function<void(std::string_view)>>& on_token
) {
    if (ctx_ == nullptr || model_ == nullptr || sampler_ == nullptr) {
        return tl::unexpected(Error{
            ErrorCode::BackendInitFailed,
            "Backend not initialized"
        });
    }

    std::string generated_text;
    // Reserve capacity to reduce reallocations during generation.
    // Heuristic: ~8 bytes per token average for UTF-8 text.
    generated_text.reserve(max_tokens > 0 ? static_cast<size_t>(max_tokens) * 8 : 4096);

    int token_count = 0;

    // Create batch for the prompt.
    // const_cast is safe: llama_batch_get_one stores the pointer and llama_decode
    // only reads through it — the token buffer is never mutated.
    llama_batch batch = llama_batch_get_one(
        const_cast<llama_token*>(reinterpret_cast<const llama_token*>(prompt_tokens.data())),
        static_cast<int32_t>(prompt_tokens.size())
    );
    llama_token new_token;
    while(true)
    {
        // Ensure we have enough context to evaluate the batch
        int n_ctx_used = llama_memory_seq_pos_max(llama_get_memory(ctx_), 0) + 1;
        if (n_ctx_used + batch.n_tokens > context_size_) {
            return tl::unexpected(Error{
                ErrorCode::ContextWindowExceeded,
                "Batch tokens exceed context size",
                "batch_size=" + std::to_string(batch.n_tokens) +
                " context_size=" + std::to_string(context_size_)
            });
        }

        // Process prompt (prefill phase)
        if (llama_decode(ctx_, batch) != 0) {
            return tl::unexpected(Error{
                ErrorCode::InferenceFailed,
                "Failed to decode prompt batch"
            });
        }

        // Sample the next token
        new_token = llama_sampler_sample(sampler_, ctx_, -1);

        // Are we at the end of the generation?
        if(llama_vocab_is_eog(vocab_, new_token))
        {
            break;
        }

        // Convert token to string
        char buff[256];
        const int n = llama_token_to_piece(vocab_, new_token, buff, sizeof(buff), 0, true);
        if(n < 0)
        {
            return tl::unexpected(Error{
                ErrorCode::Unknown,
                "Failed to convert token to piece"
            });
        }

        generated_text.append(buff, static_cast<size_t>(n));
        ++token_count;

        // Enforce max_tokens limit
        if (max_tokens > 0 && token_count >= max_tokens) {
            // Fire callback for the final token before breaking
            if (on_token.has_value()) {
                (*on_token)(std::string_view(buff, static_cast<size_t>(n)));
            }
            break;
        }

        // Check stop sequences and trim the match from the end
        if (!stop_sequences.empty()) {
            size_t match_len = find_stop_sequence(generated_text, stop_sequences);
            if (match_len > 0) {
                generated_text.resize(generated_text.size() - match_len);
                break;  // Don't fire callback — this token was part of a stop sequence
            }
        }

        // Fire callback only after confirming this token is not part of a stop sequence
        if (on_token.has_value()) {
            (*on_token)(std::string_view(buff, static_cast<size_t>(n)));
        }

        // Prepare the next batch
        batch = llama_batch_get_one(&new_token, 1);
    }

    return generated_text;
}

int LlamaBackend::get_kv_cache_token_count() const {
    return kv_cache_token_count_;
}

void LlamaBackend::clear_kv_cache() {
    if (ctx_ != nullptr) {
        auto memory = llama_get_memory(ctx_);
        llama_memory_clear(memory, false);
        kv_cache_token_count_ = 0;
        prev_len_ = 0;
    }
}

std::vector<llama_chat_message> LlamaBackend::build_llama_messages(
    const std::vector<Message>& messages) const
{
    std::vector<llama_chat_message> llama_msgs;
    llama_msgs.reserve(messages.size());
    for (const auto& msg : messages) {
        llama_msgs.push_back({role_to_string(msg.role), msg.content.c_str()});
    }
    return llama_msgs;
}

Expected<std::string> LlamaBackend::format_prompt(const std::vector<Message>& messages)
{
    if (model_ == nullptr) {
        return tl::unexpected(Error{
            ErrorCode::BackendInitFailed,
            "Backend not initialized - model is null"
        });
    }

    // Build llama_chat_message vector (zero-allocation: pointers to static strings and source data)
    auto llama_msgs = build_llama_messages(messages);

    // Apply template with generation prompt (add_generation_prompt = true)
    int new_len = llama_chat_apply_template(
        tmpl_, llama_msgs.data(), llama_msgs.size(),
        true, formatted_.data(), formatted_.size());

    // Resize buffer if needed and retry
    if (new_len > static_cast<int>(formatted_.size())) {
        formatted_.resize(new_len);
        new_len = llama_chat_apply_template(
            tmpl_, llama_msgs.data(), llama_msgs.size(),
            true, formatted_.data(), formatted_.size());
    }

    if (new_len < 0) {
        return tl::unexpected(Error{
            ErrorCode::InvalidTemplate,
            "llama_chat_apply_template failed"
        });
    }

    // If history shrank (e.g. from clear_history() or an ephemeral context removal
    // that wasn't already caught), we must reset the prompt cache and KV cache
    // since the new prompt is shorter than what we previously processed.
    if (new_len < prev_len_) {
        clear_kv_cache();
    }

    // Return only the incremental portion (new text since last call)
    std::string prompt(formatted_.begin() + prev_len_, formatted_.begin() + new_len);
    return prompt;
}

void LlamaBackend::finalize_response(const std::vector<Message>& messages)
{
    if (model_ == nullptr) {
        return;
    }

    // Build llama_chat_message vector (zero-allocation)
    auto llama_msgs = build_llama_messages(messages);

    // Dry run with add_generation_prompt=false to measure full conversation length
    int new_prev_len = llama_chat_apply_template(
        tmpl_, llama_msgs.data(), llama_msgs.size(),
        false, nullptr, 0);

    if (new_prev_len > 0) {
        prev_len_ = new_prev_len;
    }
}

int LlamaBackend::get_context_size() const {
    return context_size_;
}

int LlamaBackend::get_vocab_size() const {
    return vocab_size_;
}

size_t LlamaBackend::find_stop_sequence(
    const std::string& generated_text,
    const std::vector<std::string>& stop_sequences
) const {
    for (const auto& stop_seq : stop_sequences) {
        if (stop_seq.empty()) continue;

        // Check if generated text ends with this stop sequence
        if (generated_text.size() >= stop_seq.size()) {
            if (generated_text.compare(
                    generated_text.size() - stop_seq.size(),
                    stop_seq.size(), stop_seq) == 0) {
                return stop_seq.size();
            }
        }
    }
    return 0;
}

llama_sampler* LlamaBackend::create_sampler_chain(const Config& config) {
    // Initialize sampler chain
    auto chain_params = llama_sampler_chain_default_params();
    chain_params.no_perf = false;  // Enable performance tracking

    llama_sampler* chain = llama_sampler_chain_init(chain_params);
    if (chain == nullptr) {
        return nullptr;
    }
    
    // Add samplers in the recommended order:
    // 1. Repetition penalty
    // 2. Top-K
    // 3. Top-P
    // 4. Temperature
    // 5. Distribution sampler

    const auto& sp = config.sampling;

    // Repetition penalty sampler
    if (sp.repeat_penalty != 1.0f) {
        auto* penalties = llama_sampler_init_penalties(
            sp.repeat_last_n,   // penalty_last_n
            sp.repeat_penalty,  // penalty_repeat
            0.0f,              // penalty_freq
            0.0f               // penalty_present
        );
        if (penalties != nullptr) {
            llama_sampler_chain_add(chain, penalties);
        }
    }

    // Top-K sampler
    if (sp.top_k > 0) {
        auto* top_k = llama_sampler_init_top_k(sp.top_k);
        if (top_k != nullptr) {
            llama_sampler_chain_add(chain, top_k);
        }
    }

    // Top-P sampler
    if (sp.top_p < 1.0f) {
        auto* top_p = llama_sampler_init_top_p(sp.top_p, 1);  // min_keep = 1
        if (top_p != nullptr) {
            llama_sampler_chain_add(chain, top_p);
        }
    }

    // Temperature sampler
    if (sp.temperature > 0.0f) {
        auto* temp = llama_sampler_init_temp(sp.temperature);
        if (temp != nullptr) {
            llama_sampler_chain_add(chain, temp);
        }
    }

    // Final distribution sampler
    uint32_t seed = (sp.seed < 0) ? static_cast<uint32_t>(time(nullptr)) : static_cast<uint32_t>(sp.seed);
    auto* dist = llama_sampler_init_dist(seed);
    if (dist != nullptr) {
        llama_sampler_chain_add(chain, dist);
    } else {
        // If we can't create the distribution sampler, fall back to greedy
        auto* greedy = llama_sampler_init_greedy();
        if (greedy != nullptr) {
            llama_sampler_chain_add(chain, greedy);
        }
    }

    return chain;

}

// Factory function implementation
std::unique_ptr<IBackend> create_backend() {
    return std::unique_ptr<IBackend>(new LlamaBackend());
}

} // namespace backend
} // namespace zoo
