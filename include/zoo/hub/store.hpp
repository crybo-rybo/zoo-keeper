/**
 * @file store.hpp
 * @brief Local model catalog with download integration and model loading helpers.
 */

#pragma once

#include "zoo/agent.hpp"
#include "zoo/core/model.hpp"
#include "zoo/core/types.hpp"
#include "zoo/hub/huggingface.hpp"
#include "zoo/hub/types.hpp"

#include <memory>
#include <string>
#include <vector>

namespace zoo::hub {

/**
 * @brief Manages a local catalog of downloaded GGUF models.
 *
 * `ModelStore` tracks model files on disk, supports alias-based lookup,
 * and integrates with `GgufInspector` for auto-configuration and with
 * `HuggingFaceClient` for downloading new models. The catalog is persisted
 * as a JSON file in the store directory.
 */
class ModelStore {
  public:
    /**
     * @brief Opens or creates a model store at the configured directory.
     *
     * If the store directory or catalog file does not exist, they are created.
     * If the catalog exists, it is loaded and validated.
     *
     * @param config Store configuration. Uses `~/.zoo-keeper/models/` by default.
     * @return An initialized ModelStore, or an error if the catalog is corrupted.
     */
    static Expected<std::unique_ptr<ModelStore>> open(ModelStoreConfig config = {});

    ~ModelStore();
    ModelStore(const ModelStore&) = delete;
    ModelStore& operator=(const ModelStore&) = delete;
    ModelStore(ModelStore&&) noexcept;
    ModelStore& operator=(ModelStore&&) noexcept;

    // --- Catalog operations ---

    /**
     * @brief Registers an existing local GGUF file in the catalog.
     *
     * The file is inspected automatically to populate metadata.
     *
     * @param file_path Absolute path to the GGUF file.
     * @param aliases Optional short names for the model.
     * @return The new catalog entry, or an error.
     */
    Expected<ModelEntry> add(const std::string& file_path, std::vector<std::string> aliases = {});

    /**
     * @brief Removes a model from the catalog.
     *
     * @param name_or_alias Model name, alias, or path to look up.
     * @param delete_file If true, also deletes the GGUF file from disk.
     */
    Expected<void> remove(const std::string& name_or_alias, bool delete_file = false);

    /**
     * @brief Adds an alias for an existing model in the catalog.
     */
    Expected<void> add_alias(const std::string& name_or_alias, const std::string& new_alias);

    /**
     * @brief Returns all models in the catalog.
     */
    [[nodiscard]] std::vector<ModelEntry> list() const;

    /**
     * @brief Finds a model by name, alias, or path.
     *
     * Resolution order: exact alias match, then name substring match, then path match.
     */
    [[nodiscard]] Expected<ModelEntry> find(const std::string& query) const;

    // --- Integration helpers ---

    /**
     * @brief Returns a ModelConfig auto-configured from stored metadata.
     */
    Expected<ModelConfig> model_config(const std::string& name_or_alias) const;

    /**
     * @brief Loads a core::Model directly from the store.
     */
    Expected<std::unique_ptr<core::Model>>
    load_model(const std::string& name_or_alias,
               const GenerationOptions& options = GenerationOptions{}) const;

    /**
     * @brief Creates an Agent directly from the store.
     */
    Expected<std::unique_ptr<Agent>>
    create_agent(const std::string& name_or_alias, const AgentConfig& agent_config = AgentConfig{},
                 const GenerationOptions& options = GenerationOptions{}) const;

    // --- Download + add ---

    /**
     * @brief Downloads a model from HuggingFace and adds it to the store.
     *
     * @param client An initialized HuggingFaceClient.
     * @param identifier HuggingFace identifier ("owner/repo::file.gguf" or "owner/repo").
     * @param on_progress Optional download progress callback.
     * @param aliases Optional short names for the model.
     * @return The new catalog entry, or an error.
     */
    Expected<ModelEntry> pull(HuggingFaceClient& client, const std::string& identifier,
                              DownloadProgressCallback on_progress = {},
                              std::vector<std::string> aliases = {});

    /**
     * @brief Returns the store configuration.
     */
    [[nodiscard]] const ModelStoreConfig& config() const noexcept;

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    explicit ModelStore(std::unique_ptr<Impl> impl);
};

} // namespace zoo::hub
