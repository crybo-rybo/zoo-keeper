/**
 * @file huggingface.hpp
 * @brief HuggingFace Hub client wrapping llama.cpp common download infrastructure.
 */

#pragma once

#include "zoo/core/types.hpp"
#include "zoo/hub/types.hpp"

#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace zoo::hub {

/**
 * @brief Cached model entry discovered from the llama.cpp download cache.
 */
struct CachedModelInfo {
    std::string user;      ///< HuggingFace user/org name.
    std::string model;     ///< Model name within the repository.
    std::string tag;       ///< Version tag (e.g. "Q4_K_M", "latest").
    size_t size_bytes = 0; ///< GGUF file size in bytes (may be 0).

    /// Returns "user/model" or "user/model:tag" if tag is not "latest".
    [[nodiscard]] std::string to_string() const {
        return user + "/" + model + (tag == "latest" ? "" : ":" + tag);
    }

    bool operator==(const CachedModelInfo& other) const = default;
};

/**
 * @brief Client for interacting with the HuggingFace Hub API.
 *
 * Wraps llama.cpp's `common` download infrastructure for HuggingFace resolution,
 * model downloading with ETag caching, resume support, and split-file handling.
 * Models are downloaded into the llama.cpp shared cache directory, so files
 * downloaded by any llama.cpp tool are immediately available.
 */
class HuggingFaceClient {
  public:
    /**
     * @brief Configuration for HuggingFace Hub API access.
     */
    struct Config {
        std::string api_base_url = "https://huggingface.co"; ///< HuggingFace API base URL.
        std::string token;        ///< Optional bearer token for gated model access.
        int timeout_seconds = 30; ///< HTTP request timeout.

        [[nodiscard]] Expected<void> validate() const {
            if (api_base_url.empty()) {
                return std::unexpected(
                    Error{ErrorCode::InvalidConfig, "HuggingFace API base URL cannot be empty"});
            }
            if (timeout_seconds <= 0) {
                return std::unexpected(Error{ErrorCode::InvalidConfig, "Timeout must be positive"});
            }
            return {};
        }
    };

    /**
     * @brief Creates a new HuggingFace client with the given configuration.
     */
    static Expected<std::unique_ptr<HuggingFaceClient>> create(Config config);

    /**
     * @brief Creates a new HuggingFace client with default configuration.
     */
    static Expected<std::unique_ptr<HuggingFaceClient>> create();

    ~HuggingFaceClient();
    HuggingFaceClient(const HuggingFaceClient&) = delete;
    HuggingFaceClient& operator=(const HuggingFaceClient&) = delete;
    HuggingFaceClient(HuggingFaceClient&&) noexcept;
    HuggingFaceClient& operator=(HuggingFaceClient&&) noexcept;

    /**
     * @brief Result of parsing a HuggingFace model identifier string.
     */
    struct ParsedIdentifier {
        std::string repo_id;                 ///< Repository ID (e.g. "TheBloke/Mistral-7B-GGUF").
        std::optional<std::string> filename; ///< Specific filename (from "::" syntax).
        std::optional<std::string> tag;      ///< Version tag (from ":" syntax, e.g. "Q4_K_M").
    };

    /**
     * @brief Parses a model identifier into its components.
     *
     * Accepts formats:
     * - "owner/repo::filename.gguf" — specific file in a repository
     * - "owner/repo:tag" — repository with a quantization/version tag (ollama-style)
     * - "owner/repo" — repository without a specific file (resolves to best GGUF)
     *
     * @param identifier The identifier string to parse.
     * @return Parsed components, or an error if the format is invalid.
     */
    static Expected<ParsedIdentifier> parse_identifier(std::string_view identifier);

    /**
     * @brief Resolves the GGUF file for a HuggingFace repository.
     *
     * Uses llama.cpp's Ollama-compatible HF API to resolve the best GGUF file
     * for the given repository and optional tag. Results are cached locally.
     *
     * @param repo_id_with_tag Repository identifier, optionally with ":tag".
     * @return Repository info with the resolved GGUF file, or an error.
     */
    Expected<HuggingFaceRepoInfo> list_gguf_files(const std::string& repo_id_with_tag);

    /**
     * @brief Resolves the download URL for a specific file in a repository.
     */
    Expected<std::string> resolve_download_url(const std::string& repo_id,
                                               const std::string& filename);

    /**
     * @brief Downloads a model from HuggingFace into the llama.cpp cache.
     *
     * Leverages llama.cpp's download infrastructure: ETag validation, resume,
     * multi-split GGUF support, and retry with exponential backoff.
     *
     * @param repo_id_with_tag Repository identifier, optionally with ":tag".
     * @return The local file path to the downloaded model, or an error.
     */
    Expected<std::string> download_model(const std::string& repo_id_with_tag);

    /**
     * @brief Downloads a single file from a URL to a local path.
     *
     * Uses llama.cpp's single-file downloader with ETag caching and resume.
     *
     * @param url Full download URL.
     * @param destination_path Local filesystem path for the downloaded file.
     * @return The final local file path on success, or an error.
     */
    Expected<std::string> download_file(const std::string& url,
                                        const std::string& destination_path);

    /**
     * @brief Lists models available in the llama.cpp download cache.
     *
     * Returns models downloaded by any llama.cpp tool (llama-cli, llama-server, etc.)
     * since they share the same cache directory.
     */
    static std::vector<CachedModelInfo> list_cached_models();

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    explicit HuggingFaceClient(std::unique_ptr<Impl> impl);
};

} // namespace zoo::hub
