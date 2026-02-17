#include <gtest/gtest.h>
#include "zoo/mcp/protocol/session.hpp"
#include "mocks/mock_mcp_transport.hpp"
#include "fixtures/mcp_messages.hpp"

using namespace zoo::mcp;
using namespace zoo::mcp::protocol;

class SessionTest : public ::testing::Test {
protected:
    void SetUp() override {
        transport_ = std::make_shared<zoo::testing::MockMcpTransport>();
    }

    std::shared_ptr<zoo::testing::MockMcpTransport> transport_;
};

// ============================================================================
// State Machine Tests
// ============================================================================

TEST_F(SessionTest, InitialStateIsDisconnected) {
    Session session(transport_);
    EXPECT_EQ(session.get_state(), SessionState::Disconnected);
}

TEST_F(SessionTest, SuccessfulInitializationTransitionsToReady) {
    // Queue the initialize response to be auto-delivered
    transport_->enqueue_response(zoo::testing::mcp_fixtures::initialize_response(1));

    Session session(transport_);
    auto result = session.initialize();

    ASSERT_TRUE(result.has_value()) << result.error().to_string();
    EXPECT_EQ(session.get_state(), SessionState::Ready);
}

TEST_F(SessionTest, FailedConnectTransitionsBackToDisconnected) {
    transport_->should_fail_connect = true;

    Session session(transport_);
    auto result = session.initialize();

    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(session.get_state(), SessionState::Disconnected);
}

TEST_F(SessionTest, InitializeWhenNotDisconnectedFails) {
    transport_->enqueue_response(zoo::testing::mcp_fixtures::initialize_response(1));

    Session session(transport_);
    auto result1 = session.initialize();
    ASSERT_TRUE(result1.has_value());

    // Try to initialize again while Ready
    auto result2 = session.initialize();
    EXPECT_FALSE(result2.has_value());
    EXPECT_EQ(result2.error().code, zoo::ErrorCode::McpSessionFailed);
}

TEST_F(SessionTest, ShutdownTransitionsToDisconnected) {
    transport_->enqueue_response(zoo::testing::mcp_fixtures::initialize_response(1));

    Session session(transport_);
    (void)session.initialize();

    session.shutdown();
    EXPECT_EQ(session.get_state(), SessionState::Disconnected);
}

TEST_F(SessionTest, ShutdownWhenDisconnectedIsNoOp) {
    Session session(transport_);
    session.shutdown(); // Should not crash
    EXPECT_EQ(session.get_state(), SessionState::Disconnected);
}

// ============================================================================
// Capability Parsing Tests
// ============================================================================

TEST_F(SessionTest, ParsesServerCapabilities) {
    transport_->enqueue_response(zoo::testing::mcp_fixtures::initialize_response(1));

    Session session(transport_);
    (void)session.initialize();

    const auto& caps = session.get_server_capabilities();
    EXPECT_TRUE(caps.tools);
    EXPECT_TRUE(caps.tools_list_changed);
    EXPECT_TRUE(caps.resources);
    EXPECT_TRUE(caps.resources_subscribe);
    EXPECT_FALSE(caps.resources_list_changed);
    EXPECT_TRUE(caps.prompts);
    EXPECT_TRUE(caps.logging);
}

TEST_F(SessionTest, ParsesMinimalServerCapabilities) {
    transport_->enqueue_response(zoo::testing::mcp_fixtures::initialize_response_minimal(1));

    Session session(transport_);
    (void)session.initialize();

    const auto& caps = session.get_server_capabilities();
    EXPECT_FALSE(caps.tools);
    EXPECT_FALSE(caps.resources);
    EXPECT_FALSE(caps.prompts);
    EXPECT_FALSE(caps.logging);
}

TEST_F(SessionTest, ParsesServerInfo) {
    transport_->enqueue_response(zoo::testing::mcp_fixtures::initialize_response(1));

    Session session(transport_);
    (void)session.initialize();

    const auto& info = session.get_server_info();
    EXPECT_EQ(info.name, "test-server");
    EXPECT_EQ(info.version, "1.0.0");
}

// ============================================================================
// Handshake Message Tests
// ============================================================================

TEST_F(SessionTest, SendsInitializeRequestOnConnect) {
    transport_->enqueue_response(zoo::testing::mcp_fixtures::initialize_response(1));

    Session session(transport_);
    (void)session.initialize();

    // First message should be the initialize request
    ASSERT_FALSE(transport_->sent_messages.empty());

    auto j = nlohmann::json::parse(transport_->sent_messages[0]);
    EXPECT_EQ(j["method"], "initialize");
    EXPECT_EQ(j["params"]["protocolVersion"], "2024-11-05");
    EXPECT_TRUE(j["params"].contains("clientInfo"));
}

TEST_F(SessionTest, SendsInitializedNotificationAfterHandshake) {
    transport_->enqueue_response(zoo::testing::mcp_fixtures::initialize_response(1));

    Session session(transport_);
    (void)session.initialize();

    // Second message should be the initialized notification
    ASSERT_GE(transport_->sent_messages.size(), 2u);

    auto j = nlohmann::json::parse(transport_->sent_messages[1]);
    EXPECT_EQ(j["method"], "notifications/initialized");
    EXPECT_FALSE(j.contains("id")); // Notification has no ID
}

// ============================================================================
// Request/Response Tests
// ============================================================================

TEST_F(SessionTest, SendRequestAndReceiveResponse) {
    transport_->enqueue_response(zoo::testing::mcp_fixtures::initialize_response(1));

    Session session(transport_);
    (void)session.initialize();

    // Queue a response for the next request
    transport_->enqueue_response(zoo::testing::mcp_fixtures::tools_list_response(2));

    auto future = session.send_request("tools/list");
    auto result = future.get();

    ASSERT_TRUE(result.has_value()) << result.error().to_string();
    EXPECT_TRUE(result->contains("tools"));
}

TEST_F(SessionTest, SendNotificationSucceeds) {
    transport_->enqueue_response(zoo::testing::mcp_fixtures::initialize_response(1));

    Session session(transport_);
    (void)session.initialize();

    auto result = session.send_notification("notifications/cancelled", {{"requestId", 5}});
    EXPECT_TRUE(result.has_value());
}

TEST_F(SessionTest, SendRequestWithTransportFailureReturnsError) {
    transport_->enqueue_response(zoo::testing::mcp_fixtures::initialize_response(1));

    Session session(transport_);
    (void)session.initialize();

    transport_->should_fail_send = true;

    auto future = session.send_request("tools/list");
    auto result = future.get();

    EXPECT_FALSE(result.has_value());
}

// ============================================================================
// Error Response Handling Tests
// ============================================================================

TEST_F(SessionTest, InitializeWithServerErrorFails) {
    // Queue an error response
    nlohmann::json error_resp = {
        {"jsonrpc", "2.0"},
        {"error", {{"code", -32600}, {"message", "Server init failed"}}},
        {"id", 1}
    };
    transport_->enqueue_response(error_resp.dump());

    Session session(transport_);
    auto result = session.initialize();

    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(session.get_state(), SessionState::Disconnected);
}

// ============================================================================
// Destructor Tests
// ============================================================================

TEST_F(SessionTest, DestructorCleansUpReadySession) {
    transport_->enqueue_response(zoo::testing::mcp_fixtures::initialize_response(1));

    {
        Session session(transport_);
        (void)session.initialize();
        EXPECT_EQ(session.get_state(), SessionState::Ready);
    }
    // After destruction, transport should be disconnected
    EXPECT_FALSE(transport_->connected);
}
