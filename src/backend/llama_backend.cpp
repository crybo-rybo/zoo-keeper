#include "zoo/backend/llama_backend.hpp"
#include <llama.h>
#include <algorithm>
#include <cstring>

namespace zoo {
namespace backend {

LlamaBackend::LlamaBackend() {
    // Initialize llama.cpp backend (call once at program start)
    // This is idempotent and safe to call multiple times
    llama_backend_init();
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
    // once at program exit, not for each LlamaBackend instance
}

Expected<void> LlamaBackend::initialize(const Config& config) {
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

    // Get vocabulary from model
    vocab_ = llama_model_get_vocab(model_);

    // Set up context parameters
    auto ctx_params = llama_context_default_params();
    ctx_params.n_ctx = static_cast<uint32_t>(config.context_size);
    ctx_params.n_batch = 512;  // Reasonable default for batch processing
    ctx_params.n_ubatch = 512; // Physical batch size
    ctx_params.n_threads = -1; // Auto-detect thread count
    ctx_params.n_threads_batch = -1; // Auto-detect

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
    const auto* vocab = llama_model_get_vocab(model_);
    if (vocab == nullptr) {
        llama_free(ctx_);
        llama_model_free(model_);
        ctx_ = nullptr;
        model_ = nullptr;
        return tl::unexpected(Error{
            ErrorCode::BackendInitFailed,
            "Failed to get model vocabulary"
        });
    }
    vocab_size_ = llama_vocab_n_tokens(vocab);

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

    kv_cache_token_count_ = 0;
    return {};
}

Expected<std::vector<int>> LlamaBackend::tokenize(const std::string& text, bool add_bos) {
    if (model_ == nullptr) {
        return tl::unexpected(Error{
            ErrorCode::BackendInitFailed,
            "Backend not initialized - model is null"
        });
    }

    // Allocate buffer for tokens (use context size as upper bound)
    std::vector<llama_token> tokens(context_size_);

    // Tokenize the text
    const int n_tokens = llama_tokenize(
        vocab_,
        text.c_str(),
        text.length(),
        tokens.data(),
        tokens.size(),
        add_bos,      // add_special: whether to add BOS
        false         // parse_special: whether to parse special tokens
    );

    if (n_tokens < 0) {
        return tl::unexpected(Error{
            ErrorCode::TokenizationFailed,
            "Tokenization failed - output buffer too small or other error"
        });
    }

    // Resize to actual token count
    tokens.resize(n_tokens);

    // Convert to int vector
    std::vector<int> result(tokens.begin(), tokens.end());
    return result;
}

Expected<std::string> LlamaBackend::detokenize(const std::vector<int>& tokens) {
    if (model_ == nullptr) {
        return tl::unexpected(Error{
            ErrorCode::BackendInitFailed,
            "Backend not initialized - model is null"
        });
    }

    std::string result;
    result.reserve(tokens.size() * 4);  // Rough estimate: average 4 bytes per token

    // Buffer for token piece
    std::vector<char> piece_buffer(128);

    for (int token : tokens) {
        // Convert token to piece
        const int n_chars = llama_token_to_piece(
            vocab_,
            static_cast<llama_token>(token),
            piece_buffer.data(),
            piece_buffer.size(),
            0,     // lstrip: number of chars to strip from left
            false  // special: render special tokens
        );

        if (n_chars < 0) {
            // Buffer too small, resize and retry
            piece_buffer.resize(-n_chars);
            const int retry = llama_token_to_piece(
                vocab_,
                static_cast<llama_token>(token),
                piece_buffer.data(),
                piece_buffer.size(),
                0,
                false
            );
            if (retry < 0) {
                return tl::unexpected(Error{
                    ErrorCode::TokenizationFailed,
                    "Failed to detokenize token: " + std::to_string(token)
                });
            }
            result.append(piece_buffer.data(), retry);
        } else {
            result.append(piece_buffer.data(), n_chars);
        }
    }

    return result;
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

    // Check context size
    if (static_cast<int>(prompt_tokens.size()) > context_size_) {
        return tl::unexpected(Error{
            ErrorCode::ContextWindowExceeded,
            "Prompt tokens exceed context size",
            "prompt_size=" + std::to_string(prompt_tokens.size()) +
            " context_size=" + std::to_string(context_size_)
        });
    }

    // Clear KV cache before processing new prompt
    auto memory = llama_get_memory(ctx_);
    llama_memory_clear(memory, false);
    kv_cache_token_count_ = 0;

    // Process prompt in batches
    int n_batch = 512; // Must match context params n_batch (or query it)
    llama_batch batch = llama_batch_init(n_batch, 0, 1); // Max batch size

    for (size_t i = 0; i < prompt_tokens.size(); i += n_batch) {
        int n_tokens = std::min(n_batch, static_cast<int>(prompt_tokens.size() - i));
        
        // Prepare batch
        batch.n_tokens = n_tokens;
        for (int j = 0; j < n_tokens; ++j) {
             batch.token[j] = prompt_tokens[i + j];
             batch.pos[j] = i + j;
             batch.n_seq_id[j] = 1;
             batch.seq_id[j][0] = 0;
             batch.logits[j] = false;
        }

        // Last token of the prompt dictates logits generation
        if (i + n_tokens == prompt_tokens.size()) {
            batch.logits[n_tokens - 1] = true;
        }

        // Decode batch
        if (llama_decode(ctx_, batch) != 0) {
            llama_batch_free(batch);
            return tl::unexpected(Error{
                ErrorCode::InferenceFailed,
                "Failed to decode prompt batch"
            });
        }
    }
    
    llama_batch_free(batch);

    // Update KV cache count
    kv_cache_token_count_ = static_cast<int>(prompt_tokens.size());

    // Generation loop
    std::string generated_text;
    generated_text.reserve(max_tokens * 4);  // Estimate
    std::vector<llama_token> generated_tokens;
    generated_tokens.reserve(max_tokens);

    for (int i = 0; i < max_tokens; ++i) {
        // Sample next token
        llama_token new_token = llama_sampler_sample(sampler_, ctx_, -1);

        // Accept the token (updates sampler state)
        llama_sampler_accept(sampler_, new_token);

        // Check for EOS token
        if (llama_vocab_is_eog(llama_model_get_vocab(model_), new_token)) {
            break;
        }

        generated_tokens.push_back(new_token);

        // Convert token to text
        std::vector<char> piece_buffer(128);
        const int n_chars = llama_token_to_piece(
            vocab_,
            new_token,
            piece_buffer.data(),
            piece_buffer.size(),
            0,
            false
        );

        std::string token_text;
        if (n_chars < 0) {
            piece_buffer.resize(-n_chars);
            const int retry = llama_token_to_piece(
                vocab_,
                new_token,
                piece_buffer.data(),
                piece_buffer.size(),
                0,
                false
            );
            if (retry >= 0) {
                token_text.assign(piece_buffer.data(), retry);
            }
        } else {
            token_text.assign(piece_buffer.data(), n_chars);
        }

        generated_text += token_text;

        // Call streaming callback if provided
        if (on_token.has_value()) {
            (*on_token)(token_text);
        }

        // Check stop sequences
        if (check_stop_sequence(generated_text, stop_sequences)) {
            // Remove the stop sequence from the output
            for (const auto& stop_seq : stop_sequences) {
                if (generated_text.size() >= stop_seq.size() &&
                    generated_text.substr(generated_text.size() - stop_seq.size()) == stop_seq) {
                    generated_text.resize(generated_text.size() - stop_seq.size());
                    break;
                }
            }
            break;
        }

        // Prepare next batch (single token)
        auto next_batch = llama_batch_get_one(&new_token, 1);

        // Decode the new token
        if (llama_decode(ctx_, next_batch) != 0) {
            return tl::unexpected(Error{
                ErrorCode::InferenceFailed,
                "Failed to decode token at position " + std::to_string(i)
            });
        }

        kv_cache_token_count_++;
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
    }
}

bool LlamaBackend::supports_template(PromptTemplate tmpl) const {
    // For MVP, we don't auto-detect from GGUF metadata
    // This could be enhanced in the future by checking model metadata
    (void)tmpl;
    return false;
}

int LlamaBackend::get_context_size() const {
    return context_size_;
}

int LlamaBackend::get_vocab_size() const {
    return vocab_size_;
}

bool LlamaBackend::check_stop_sequence(
    const std::string& generated_text,
    const std::vector<std::string>& stop_sequences
) const {
    for (const auto& stop_seq : stop_sequences) {
        if (stop_seq.empty()) continue;

        // Check if generated text contains this stop sequence
        // We use find() instead of suffix check to be robust against
        // trailing whitespace or multiple tokens generated at once
        if (generated_text.find(stop_seq) != std::string::npos) {
            return true;
        }
    }
    return false;
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
