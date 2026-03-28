/**
 * @file test_hub.cpp
 * @brief Unit tests for hub layer: identifier parsing, auto-config, catalog JSON.
 */

#include "zoo/hub/huggingface.hpp"
#include "zoo/hub/inspector.hpp"
#include "zoo/hub/types.hpp"

#include <gtest/gtest.h>

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

TEST(HuggingFaceConfigTest, EmptyBaseUrlFails) {
    zoo::hub::HuggingFaceClient::Config config;
    config.api_base_url = "";
    auto result = config.validate();
    ASSERT_FALSE(result.has_value());
}

TEST(HuggingFaceConfigTest, NegativeTimeoutFails) {
    zoo::hub::HuggingFaceClient::Config config;
    config.timeout_seconds = -1;
    auto result = config.validate();
    ASSERT_FALSE(result.has_value());
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
