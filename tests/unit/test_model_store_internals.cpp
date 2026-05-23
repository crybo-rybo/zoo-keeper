/**
 * @file test_model_store_internals.cpp
 * @brief Unit tests for private ModelStore catalog mutation and import helpers.
 */

#include "hub/store_catalog_io.hpp"
#include "hub/store_internals.hpp"
#include "zoo/hub/types.hpp"

#include <gtest/gtest.h>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace {

class TempDir {
  public:
    TempDir() {
        const auto unique =
            std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
        path_ = std::filesystem::temp_directory_path() / ("zoo-store-internals-" + unique);
        std::filesystem::create_directories(path_);
    }

    ~TempDir() {
        std::error_code ec;
        std::filesystem::remove_all(path_, ec);
    }

    [[nodiscard]] const std::filesystem::path& path() const noexcept {
        return path_;
    }

  private:
    std::filesystem::path path_;
};

zoo::hub::ModelEntry make_entry(std::string id, std::string name, std::string file_path,
                                std::vector<std::string> aliases = {}) {
    zoo::hub::ModelEntry entry;
    entry.id = std::move(id);
    entry.file_path = std::move(file_path);
    entry.info.file_path = entry.file_path;
    entry.info.name = std::move(name);
    entry.aliases = std::move(aliases);
    entry.added_at = "2026-03-31T12:00:00Z";
    return entry;
}

#ifdef ZOO_PROJECT_SOURCE_DIR
std::filesystem::path fixture_vocab_model_path() {
    return std::filesystem::path(ZOO_PROJECT_SOURCE_DIR) / "tests/fixtures/ggml-vocab-gpt-2.gguf";
}
#endif

} // namespace

TEST(ModelStoreInternalsTest, MutateCatalogDoesNotPersistFailedMutation) {
    TempDir temp_dir;
    zoo::hub::ModelStoreConfig config;
    config.store_directory = temp_dir.path().string();
    zoo::hub::detail::CatalogRepository repository(config);

    std::vector<zoo::hub::ModelEntry> cached = {make_entry("keep-me", "keep-me", "/tmp/keep.gguf")};
    ASSERT_TRUE(repository.save(cached).has_value());

    auto result = zoo::hub::detail::mutate_catalog(
        repository, cached, [](std::vector<zoo::hub::ModelEntry>& current) -> zoo::Expected<void> {
            current.clear();
            return zoo::Expected<void>(std::unexpected(
                zoo::Error{zoo::ErrorCode::InvalidConfig, "simulated mutation failure"}));
        });
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, zoo::ErrorCode::InvalidConfig);

    auto loaded = repository.load();
    ASSERT_TRUE(loaded.has_value()) << loaded.error().to_string();
    ASSERT_EQ(loaded->size(), 1u);
    EXPECT_EQ((*loaded)[0].id, "keep-me");
}

TEST(ModelStoreInternalsTest, MutateCatalogPersistsSuccessfulMutation) {
    TempDir temp_dir;
    zoo::hub::ModelStoreConfig config;
    config.store_directory = temp_dir.path().string();
    zoo::hub::detail::CatalogRepository repository(config);

    std::vector<zoo::hub::ModelEntry> cached;
    auto result = zoo::hub::detail::mutate_catalog(
        repository, cached,
        [](std::vector<zoo::hub::ModelEntry>& current) -> zoo::Expected<zoo::hub::ModelEntry> {
            current.push_back(make_entry("added", "added-model", "/tmp/added.gguf"));
            return current.back();
        });
    ASSERT_TRUE(result.has_value()) << result.error().to_string();
    EXPECT_EQ(result->id, "added");
    ASSERT_EQ(cached.size(), 1u);

    auto loaded = repository.load();
    ASSERT_TRUE(loaded.has_value()) << loaded.error().to_string();
    ASSERT_EQ(loaded->size(), 1u);
    EXPECT_EQ((*loaded)[0].id, "added");
}

#ifdef ZOO_PROJECT_SOURCE_DIR
TEST(ModelStoreInternalsTest, ModelImporterRejectsDuplicateFilePath) {
    TempDir temp_dir;
    zoo::hub::ModelStoreConfig config;
    config.store_directory = temp_dir.path().string();
    zoo::hub::detail::CatalogRepository repository(config);

    const auto model_path = fixture_vocab_model_path();
    ASSERT_TRUE(std::filesystem::exists(model_path));

    std::vector<zoo::hub::ModelEntry> entries;
    auto first = zoo::hub::detail::ModelImporter::add_local_file(
        entries, repository, model_path.string(), std::vector<std::string>{"first"});
    ASSERT_TRUE(first.has_value()) << first.error().to_string();

    auto duplicate = zoo::hub::detail::ModelImporter::add_local_file(
        entries, repository, model_path.string(), std::vector<std::string>{"second"});
    ASSERT_FALSE(duplicate.has_value());
    EXPECT_EQ(duplicate.error().code, zoo::ErrorCode::ModelAlreadyExists);

    auto loaded = repository.load();
    ASSERT_TRUE(loaded.has_value()) << loaded.error().to_string();
    EXPECT_EQ(loaded->size(), 1u);
}
#endif
