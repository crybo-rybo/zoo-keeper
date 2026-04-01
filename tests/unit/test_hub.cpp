/**
 * @file test_hub.cpp
 * @brief Unit tests for hub layer: identifier parsing, auto-config, catalog JSON.
 */

#include "hub/path_utils.hpp"
#include "zoo/hub/huggingface.hpp"
#include "zoo/hub/inspector.hpp"
#include "zoo/hub/store.hpp"
#include "zoo/hub/types.hpp"

#include <gtest/gtest.h>
#include <llama.h>
#include <nlohmann/json.hpp>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <string>
#include <type_traits>

namespace {

std::filesystem::path project_source_dir() {
    return std::filesystem::current_path().parent_path();
}

std::filesystem::path vendored_fixture_model_path() {
    return project_source_dir() / "extern/llama.cpp/models/ggml-vocab-gpt-2.gguf";
}

class TempDir {
  public:
    TempDir() {
        const auto unique =
            std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
        path_ = std::filesystem::temp_directory_path() / ("zoo-hub-tests-" + unique);
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

void write_catalog(const std::filesystem::path& store_dir, const nlohmann::json& json) {
    std::ofstream out(store_dir / "catalog.json");
    ASSERT_TRUE(out.is_open());
    out << json.dump(2) << '\n';
}

void sentinel_log_callback(ggml_log_level, const char*, void*) {}

class ScopedLlamaLogger {
  public:
    ScopedLlamaLogger(ggml_log_callback callback, void* user_data) {
        llama_log_get(&original_callback_, &original_user_data_);
        llama_log_set(callback, user_data);
    }

    ~ScopedLlamaLogger() {
        llama_log_set(original_callback_, original_user_data_);
    }

  private:
    ggml_log_callback original_callback_ = nullptr;
    void* original_user_data_ = nullptr;
};

template <typename T>
concept HasApiBaseUrl = requires(T config) { config.api_base_url; };

template <typename T>
concept HasTimeoutSeconds = requires(T config) { config.timeout_seconds; };

template <typename T>
concept HasListGgufFiles =
    requires(T client, const std::string& identifier) { client.list_gguf_files(identifier); };

using PullSignature = zoo::Expected<zoo::hub::ModelEntry> (zoo::hub::ModelStore::*)(
    zoo::hub::HuggingFaceClient&, const std::string&, std::vector<std::string>);

static_assert(!HasApiBaseUrl<zoo::hub::HuggingFaceClient::Config>);
static_assert(!HasTimeoutSeconds<zoo::hub::HuggingFaceClient::Config>);
static_assert(!HasListGgufFiles<zoo::hub::HuggingFaceClient>);
static_assert(std::is_same_v<decltype(&zoo::hub::ModelStore::pull), PullSignature>);

} // namespace

// ---- HuggingFace identifier parsing ----

TEST(HuggingFaceParseTest, ParseRepoWithFile) {
    auto result = zoo::hub::HuggingFaceClient::parse_identifier(
        "TheBloke/Mistral-7B-GGUF::mistral-7b.Q4_K_M.gguf");
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->repo_id, "TheBloke/Mistral-7B-GGUF");
    ASSERT_TRUE(result->filename.has_value());
    EXPECT_EQ(*result->filename, "mistral-7b.Q4_K_M.gguf");
}

TEST(HuggingFaceParseTest, ParseRepoOnly) {
    auto result = zoo::hub::HuggingFaceClient::parse_identifier("TheBloke/Mistral-7B-GGUF");
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->repo_id, "TheBloke/Mistral-7B-GGUF");
    EXPECT_FALSE(result->filename.has_value());
    EXPECT_FALSE(result->tag.has_value());
}

TEST(HuggingFaceParseTest, ParseRepoWithTag) {
    auto result =
        zoo::hub::HuggingFaceClient::parse_identifier("bartowski/Llama-3.2-3B-GGUF:Q4_K_M");
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->repo_id, "bartowski/Llama-3.2-3B-GGUF");
    EXPECT_FALSE(result->filename.has_value());
    ASSERT_TRUE(result->tag.has_value());
    EXPECT_EQ(*result->tag, "Q4_K_M");
}

TEST(HuggingFaceParseTest, ParseRepoWithLatestTag) {
    // "latest" tag should be treated as no tag.
    auto result = zoo::hub::HuggingFaceClient::parse_identifier("owner/repo:latest");
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->repo_id, "owner/repo");
    EXPECT_FALSE(result->tag.has_value());
}

TEST(HuggingFaceParseTest, EmptyIdentifier) {
    auto result = zoo::hub::HuggingFaceClient::parse_identifier("");
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, zoo::ErrorCode::InvalidModelIdentifier);
}

TEST(HuggingFaceParseTest, MissingSlash) {
    auto result = zoo::hub::HuggingFaceClient::parse_identifier("just-a-name");
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, zoo::ErrorCode::InvalidModelIdentifier);
}

TEST(HuggingFaceParseTest, EmptyFilenameAfterSeparator) {
    auto result = zoo::hub::HuggingFaceClient::parse_identifier("owner/repo::");
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, zoo::ErrorCode::InvalidModelIdentifier);
}

TEST(HuggingFaceParseTest, MultipleSlashes) {
    auto result = zoo::hub::HuggingFaceClient::parse_identifier("a/b/c");
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, zoo::ErrorCode::InvalidModelIdentifier);
}

TEST(HuggingFaceParseTest, SlashAtStart) {
    auto result = zoo::hub::HuggingFaceClient::parse_identifier("/repo");
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, zoo::ErrorCode::InvalidModelIdentifier);
}

TEST(HuggingFaceParseTest, SlashAtEnd) {
    auto result = zoo::hub::HuggingFaceClient::parse_identifier("owner/");
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, zoo::ErrorCode::InvalidModelIdentifier);
}

// ---- Auto-configuration ----

TEST(AutoConfigTest, SetsModelPath) {
    zoo::hub::ModelInfo info;
    info.file_path = "/models/test.gguf";
    info.context_length = 4096;

    auto config = zoo::hub::GgufInspector::auto_configure(info);
    ASSERT_TRUE(config.has_value());
    EXPECT_EQ(config->model_path, "/models/test.gguf");
}

TEST(AutoConfigTest, CapsContextAt8192) {
    zoo::hub::ModelInfo info;
    info.file_path = "/models/test.gguf";
    info.context_length = 131072;

    auto config = zoo::hub::GgufInspector::auto_configure(info);
    ASSERT_TRUE(config.has_value());
    EXPECT_EQ(config->context_size, 8192);
}

TEST(AutoConfigTest, UsesTrainingContextWhenSmaller) {
    zoo::hub::ModelInfo info;
    info.file_path = "/models/test.gguf";
    info.context_length = 2048;

    auto config = zoo::hub::GgufInspector::auto_configure(info);
    ASSERT_TRUE(config.has_value());
    EXPECT_EQ(config->context_size, 2048);
}

TEST(AutoConfigTest, DefaultsWhenNoContextLength) {
    zoo::hub::ModelInfo info;
    info.file_path = "/models/test.gguf";
    info.context_length = 0;

    auto config = zoo::hub::GgufInspector::auto_configure(info);
    ASSERT_TRUE(config.has_value());
    EXPECT_EQ(config->context_size, 8192);
}

TEST(AutoConfigTest, OffloadsAllGpuLayers) {
    zoo::hub::ModelInfo info;
    info.file_path = "/models/test.gguf";

    auto config = zoo::hub::GgufInspector::auto_configure(info);
    ASSERT_TRUE(config.has_value());
    EXPECT_EQ(config->n_gpu_layers, -1);
}

TEST(AutoConfigTest, EnablesMmap) {
    zoo::hub::ModelInfo info;
    info.file_path = "/models/test.gguf";

    auto config = zoo::hub::GgufInspector::auto_configure(info);
    ASSERT_TRUE(config.has_value());
    EXPECT_TRUE(config->use_mmap);
    EXPECT_FALSE(config->use_mlock);
}

TEST(AutoConfigTest, EmptyPathReturnsError) {
    zoo::hub::ModelInfo info;
    info.file_path = "";

    auto config = zoo::hub::GgufInspector::auto_configure(info);
    ASSERT_FALSE(config.has_value());
    EXPECT_EQ(config.error().code, zoo::ErrorCode::InvalidModelPath);
}

// ---- ModelStoreConfig validation ----

TEST(ModelStoreConfigTest, ValidConfig) {
    zoo::hub::ModelStoreConfig config;
    config.store_directory = "/tmp/models";
    auto result = config.validate();
    EXPECT_TRUE(result.has_value());
}

TEST(ModelStoreConfigTest, EmptyDirectoryFails) {
    zoo::hub::ModelStoreConfig config;
    config.store_directory = "";
    auto result = config.validate();
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, zoo::ErrorCode::InvalidConfig);
}

TEST(ModelStoreConfigTest, EmptyCatalogFilenameFails) {
    zoo::hub::ModelStoreConfig config;
    config.store_directory = "/tmp";
    config.catalog_filename = "";
    auto result = config.validate();
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, zoo::ErrorCode::InvalidConfig);
}

// ---- HuggingFaceClient::Config validation ----

TEST(HuggingFaceConfigTest, ValidConfig) {
    zoo::hub::HuggingFaceClient::Config config;
    auto result = config.validate();
    EXPECT_TRUE(result.has_value());
}

TEST(HuggingFaceConfigTest, TokenOnlyConfigIsAccepted) {
    zoo::hub::HuggingFaceClient::Config config;
    config.token = "hf_test_token";
    auto result = config.validate();
    EXPECT_TRUE(result.has_value());
}

// ---- ModelInfo equality ----

TEST(ModelInfoTest, DefaultEquality) {
    zoo::hub::ModelInfo a;
    zoo::hub::ModelInfo b;
    EXPECT_EQ(a, b);
}

TEST(ModelInfoTest, DifferentNameNotEqual) {
    zoo::hub::ModelInfo a;
    a.name = "model-a";
    zoo::hub::ModelInfo b;
    b.name = "model-b";
    EXPECT_NE(a, b);
}

// ---- CachedModelInfo ----

TEST(CachedModelInfoTest, ToStringWithTag) {
    zoo::hub::CachedModelInfo info;
    info.user = "TheBloke";
    info.model = "Mistral-7B-GGUF";
    info.tag = "Q4_K_M";
    EXPECT_EQ(info.to_string(), "TheBloke/Mistral-7B-GGUF:Q4_K_M");
}

TEST(CachedModelInfoTest, ToStringLatestOmitsTag) {
    zoo::hub::CachedModelInfo info;
    info.user = "owner";
    info.model = "model";
    info.tag = "latest";
    EXPECT_EQ(info.to_string(), "owner/model");
}

// ---- Download destination helpers ----

TEST(HubPathTest, NamespacesDownloadDestinationsByRepository) {
    const auto first = zoo::hub::detail::build_download_destination("/tmp/zoo-cache",
                                                                    "owner-a/repo", "model.gguf");
    const auto second = zoo::hub::detail::build_download_destination("/tmp/zoo-cache",
                                                                     "owner-b/repo", "model.gguf");

    ASSERT_TRUE(first.has_value());
    ASSERT_TRUE(second.has_value());
    EXPECT_NE(*first, *second);
    EXPECT_EQ(*first, std::filesystem::path("/tmp/zoo-cache") / "owner-a" / "repo" / "model.gguf");
    EXPECT_EQ(*second, std::filesystem::path("/tmp/zoo-cache") / "owner-b" / "repo" / "model.gguf");
}

TEST(HubPathTest, PreservesNestedRepositoryFilePaths) {
    const auto destination = zoo::hub::detail::build_download_destination(
        "/tmp/zoo-store", "owner/repo", "subdir/model.Q4_K_M.gguf");

    ASSERT_TRUE(destination.has_value());
    EXPECT_EQ(*destination, std::filesystem::path("/tmp/zoo-store") / "owner" / "repo" / "subdir" /
                                "model.Q4_K_M.gguf");
}

// ---- ModelStore catalog persistence / validation ----

TEST(ModelStoreCatalogTest, OpenPreservesMetadataAndSourceFields) {
    TempDir temp_dir;

    write_catalog(
        temp_dir.path(),
        nlohmann::json{
            {"version", 1},
            {"models",
             nlohmann::json::array({nlohmann::json{
                 {"id", "model-1"},
                 {"file_path", "/tmp/model.gguf"},
                 {"info",
                  {{"file_path", "/tmp/model.gguf"},
                   {"name", "fixture-model"},
                   {"architecture", "llama"},
                   {"description", "7B Q4_K_M"},
                   {"parameter_count", 7},
                   {"file_size_bytes", 42},
                   {"embedding_dim", 64},
                   {"layer_count", 8},
                   {"context_length", 4096},
                   {"quantization", "Q4_K_M"},
                   {"metadata",
                    {{"general.name", "fixture-model"}, {"llama.context_length", "4096"}}}}},
                 {"aliases", nlohmann::json::array({"fixture"})},
                 {"source_url", "https://huggingface.co/owner/repo/resolve/main/model.gguf"},
                 {"huggingface_repo", "owner/repo"},
                 {"added_at", "2026-03-31T12:00:00Z"},
             }})},
        });

    zoo::hub::ModelStoreConfig config;
    config.store_directory = temp_dir.path().string();

    auto store = zoo::hub::ModelStore::open(config);
    ASSERT_TRUE(store.has_value()) << store.error().to_string();

    const auto entries = (*store)->list();
    ASSERT_EQ(entries.size(), 1u);
    ASSERT_EQ(entries[0].info.metadata.size(), 2u);
    EXPECT_EQ(entries[0].info.metadata.at("general.name"), "fixture-model");
    EXPECT_EQ(entries[0].source_url, "https://huggingface.co/owner/repo/resolve/main/model.gguf");
    EXPECT_EQ(entries[0].huggingface_repo, "owner/repo");
}

TEST(ModelStoreCatalogTest, OpenRejectsDuplicateAliasesInCatalog) {
    TempDir temp_dir;

    write_catalog(
        temp_dir.path(),
        nlohmann::json{
            {"version", 1},
            {"models",
             nlohmann::json::array({
                 nlohmann::json{{"id", "model-1"},
                                {"file_path", "/tmp/one.gguf"},
                                {"info", {{"file_path", "/tmp/one.gguf"}, {"name", "one"}}},
                                {"aliases", nlohmann::json::array({"shared"})},
                                {"added_at", "2026-03-31T12:00:00Z"}},
                 nlohmann::json{{"id", "model-2"},
                                {"file_path", "/tmp/two.gguf"},
                                {"info", {{"file_path", "/tmp/two.gguf"}, {"name", "two"}}},
                                {"aliases", nlohmann::json::array({"shared"})},
                                {"added_at", "2026-03-31T12:01:00Z"}},
             })},
        });

    zoo::hub::ModelStoreConfig config;
    config.store_directory = temp_dir.path().string();

    auto store = zoo::hub::ModelStore::open(config);
    ASSERT_FALSE(store.has_value());
    EXPECT_EQ(store.error().code, zoo::ErrorCode::StoreCorrupted);
}

TEST(ModelStoreCatalogTest, AddAliasRejectsEmptyAlias) {
    TempDir temp_dir;

    write_catalog(
        temp_dir.path(),
        nlohmann::json{
            {"version", 1},
            {"models", nlohmann::json::array({nlohmann::json{
                           {"id", "model-1"},
                           {"file_path", "/tmp/model.gguf"},
                           {"info", {{"file_path", "/tmp/model.gguf"}, {"name", "fixture-model"}}},
                           {"aliases", nlohmann::json::array({"fixture"})},
                           {"added_at", "2026-03-31T12:00:00Z"},
                       }})},
        });

    zoo::hub::ModelStoreConfig config;
    config.store_directory = temp_dir.path().string();

    auto store = zoo::hub::ModelStore::open(config);
    ASSERT_TRUE(store.has_value()) << store.error().to_string();

    auto result = (*store)->add_alias("fixture-model", "");
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, zoo::ErrorCode::InvalidConfig);
}

TEST(ModelStoreCatalogTest, AddRejectsDuplicateAndEmptyAliases) {
    const auto fixture_path = vendored_fixture_model_path();
    ASSERT_TRUE(std::filesystem::exists(fixture_path)) << fixture_path.string();

    TempDir temp_dir;
    const auto first_copy = temp_dir.path() / "first.gguf";
    const auto second_copy = temp_dir.path() / "second.gguf";
    std::filesystem::copy_file(fixture_path, first_copy);
    std::filesystem::copy_file(fixture_path, second_copy);

    zoo::hub::ModelStoreConfig config;
    config.store_directory = temp_dir.path().string();

    auto store = zoo::hub::ModelStore::open(config);
    ASSERT_TRUE(store.has_value()) << store.error().to_string();

    auto empty_alias = (*store)->add(first_copy.string(), {""});
    ASSERT_FALSE(empty_alias.has_value());
    EXPECT_EQ(empty_alias.error().code, zoo::ErrorCode::InvalidConfig);

    auto first = (*store)->add(first_copy.string(), {"fixture"});
    ASSERT_TRUE(first.has_value()) << first.error().to_string();

    auto duplicate_within_add = (*store)->add(second_copy.string(), {"fixture", "fixture"});
    ASSERT_FALSE(duplicate_within_add.has_value());
    EXPECT_EQ(duplicate_within_add.error().code, zoo::ErrorCode::InvalidConfig);
}

// ---- Inspector regression coverage ----

TEST(GgufInspectorTest, RestoresGlobalLoggerAfterInspect) {
    const auto model_path = vendored_fixture_model_path();
    ASSERT_TRUE(std::filesystem::exists(model_path)) << model_path.string();

    void* const sentinel_user_data = reinterpret_cast<void*>(static_cast<uintptr_t>(0x1234));
    ScopedLlamaLogger logger(sentinel_log_callback, sentinel_user_data);

    auto result = zoo::hub::GgufInspector::inspect(model_path.string());
    ASSERT_TRUE(result.has_value()) << result.error().to_string();

    ggml_log_callback current_callback = nullptr;
    void* current_user_data = nullptr;
    llama_log_get(&current_callback, &current_user_data);

    EXPECT_EQ(current_callback, sentinel_log_callback);
    EXPECT_EQ(current_user_data, sentinel_user_data);
}
