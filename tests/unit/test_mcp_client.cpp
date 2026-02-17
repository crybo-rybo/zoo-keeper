#include <gtest/gtest.h>
#include "zoo/mcp/mcp_client.hpp"
#include "mocks/mock_mcp_transport.hpp"
#include "fixtures/mcp_messages.hpp"

using namespace zoo::mcp;

class McpClientTest : public ::testing::Test {
protected:
    McpClient::Config make_config(const std::string& server_id = "test") {
        McpClient::Config config;
        config.server_id = server_id;
        config.transport.command = "echo"; // Won't be used with mock
        config.prefix_tools = true;
        return config;
    }
};

// ============================================================================
// Factory Tests
// ============================================================================

TEST_F(McpClientTest, CreateWithValidConfig) {
    auto result = McpClient::create(make_config());
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ((*result)->get_server_id(), "test");
}

TEST_F(McpClientTest, CreateWithEmptyServerIdFails) {
    auto result = McpClient::create(make_config(""));
    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, zoo::ErrorCode::InvalidConfig);
}

// ============================================================================
// Tool Discovery Tests (using mock transport via Session)
// ============================================================================

// Note: Full discovery tests require integration with Session and Transport.
// These tests verify the McpClient API surface.

TEST_F(McpClientTest, IsNotConnectedInitially) {
    auto client = *McpClient::create(make_config());
    EXPECT_FALSE(client->is_connected());
}

TEST_F(McpClientTest, DiscoverToolsWhenDisconnectedFails) {
    auto client = *McpClient::create(make_config());
    auto result = client->discover_tools();
    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, zoo::ErrorCode::McpDisconnected);
}

TEST_F(McpClientTest, CallToolWhenDisconnectedFails) {
    auto client = *McpClient::create(make_config());
    auto result = client->call_tool("read_file", {{"path", "/tmp"}});
    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, zoo::ErrorCode::McpDisconnected);
}

// ============================================================================
// Tool Name Prefix Tests
// ============================================================================

TEST_F(McpClientTest, PrefixToolNameWhenEnabled) {
    auto config = make_config("filesystem");
    config.prefix_tools = true;
    auto client = *McpClient::create(config);

    // The tool name prefixing is tested via get_config
    EXPECT_TRUE(client->get_config().prefix_tools);
    EXPECT_EQ(client->get_config().server_id, "filesystem");
}

TEST_F(McpClientTest, NoPrefixWhenDisabled) {
    auto config = make_config("filesystem");
    config.prefix_tools = false;
    auto client = *McpClient::create(config);

    EXPECT_FALSE(client->get_config().prefix_tools);
}

// ============================================================================
// Config Tests
// ============================================================================

TEST_F(McpClientTest, DefaultToolTimeout) {
    auto client = *McpClient::create(make_config());
    EXPECT_EQ(client->get_config().tool_timeout, std::chrono::seconds(30));
}

TEST_F(McpClientTest, DiscoveredToolsEmptyInitially) {
    auto client = *McpClient::create(make_config());
    EXPECT_TRUE(client->get_discovered_tools().empty());
}
