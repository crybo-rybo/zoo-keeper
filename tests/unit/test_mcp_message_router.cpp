#include <gtest/gtest.h>
#include "zoo/mcp/protocol/message_router.hpp"
#include "zoo/mcp/protocol/json_rpc.hpp"
#include "fixtures/mcp_messages.hpp"

using namespace zoo::mcp;
using namespace zoo::mcp::protocol;
using zoo::ErrorCode;
using zoo::Expected;

class MessageRouterTest : public ::testing::Test {
protected:
    MessageRouter router_;
};

// ============================================================================
// Request/Response Correlation Tests
// ============================================================================

TEST_F(MessageRouterTest, CreatePendingRequestReturnsIncrementingIds) {
    auto [id1, future1] = router_.create_pending_request();
    auto [id2, future2] = router_.create_pending_request();

    EXPECT_NE(id1, id2);
    EXPECT_EQ(id2, id1 + 1);
}

TEST_F(MessageRouterTest, RouteResponseResolvesCorrectFuture) {
    auto [id, future] = router_.create_pending_request();

    zoo::mcp::JsonRpcResponse response;
    response.result = nlohmann::json{{"data", "test"}};
    response.id = id;

    router_.route_response(response);

    auto result = future.get();
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ((*result)["data"], "test");
}

TEST_F(MessageRouterTest, RouteResponseWithError) {
    auto [id, future] = router_.create_pending_request();

    zoo::mcp::JsonRpcResponse response;
    response.error = zoo::mcp::JsonRpcError{-32600, "Invalid Request", std::nullopt};
    response.id = id;

    router_.route_response(response);

    auto result = future.get();
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, ErrorCode::McpProtocolError);
}

TEST_F(MessageRouterTest, RouteResponseForUnknownIdIsIgnored) {
    auto [id, future] = router_.create_pending_request();

    zoo::mcp::JsonRpcResponse response;
    response.result = nlohmann::json{{"data", "test"}};
    response.id = id + 999; // Wrong ID

    // Should not crash or resolve the existing future
    router_.route_response(response);

    EXPECT_EQ(router_.pending_count(), 1u);
}

TEST_F(MessageRouterTest, RouteMultipleResponsesInOrder) {
    auto [id1, future1] = router_.create_pending_request();
    auto [id2, future2] = router_.create_pending_request();

    // Respond to id2 first, then id1
    zoo::mcp::JsonRpcResponse resp2;
    resp2.result = nlohmann::json{{"value", 2}};
    resp2.id = id2;
    router_.route_response(resp2);

    zoo::mcp::JsonRpcResponse resp1;
    resp1.result = nlohmann::json{{"value", 1}};
    resp1.id = id1;
    router_.route_response(resp1);

    auto result1 = future1.get();
    auto result2 = future2.get();

    ASSERT_TRUE(result1.has_value());
    ASSERT_TRUE(result2.has_value());
    EXPECT_EQ((*result1)["value"], 1);
    EXPECT_EQ((*result2)["value"], 2);
}

TEST_F(MessageRouterTest, RouteResponseWithEmptyResult) {
    auto [id, future] = router_.create_pending_request();

    zoo::mcp::JsonRpcResponse response;
    response.id = id;
    // No result set

    router_.route_response(response);

    auto result = future.get();
    ASSERT_TRUE(result.has_value());
}

// ============================================================================
// Notification Tests
// ============================================================================

TEST_F(MessageRouterTest, NotificationHandlerIsCalled) {
    std::string received_method;
    nlohmann::json received_params;

    router_.set_notification_handler([&](const std::string& method, const nlohmann::json& params) {
        received_method = method;
        received_params = params;
    });

    router_.route_message(zoo::testing::mcp_fixtures::valid_notification());

    EXPECT_EQ(received_method, "notifications/initialized");
}

TEST_F(MessageRouterTest, NotificationWithoutHandlerDoesNotCrash) {
    // No handler set — should not crash
    router_.route_message(zoo::testing::mcp_fixtures::valid_notification());
}

// ============================================================================
// Route Message (raw JSON) Tests
// ============================================================================

TEST_F(MessageRouterTest, RouteMessageRoutesResponse) {
    auto [id, future] = router_.create_pending_request();

    // Build a response matching the ID
    nlohmann::json resp = {
        {"jsonrpc", "2.0"},
        {"result", {{"status", "ok"}}},
        {"id", id}
    };

    router_.route_message(resp.dump());

    auto result = future.get();
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ((*result)["status"], "ok");
}

TEST_F(MessageRouterTest, RouteMessageIgnoresMalformedJson) {
    // Should not crash
    router_.route_message("not json at all");
    router_.route_message("{malformed}");

    EXPECT_EQ(router_.pending_count(), 0u);
}

// ============================================================================
// Cancel All Tests
// ============================================================================

TEST_F(MessageRouterTest, CancelAllResolvesAllPendingWithError) {
    auto [id1, future1] = router_.create_pending_request();
    auto [id2, future2] = router_.create_pending_request();
    auto [id3, future3] = router_.create_pending_request();

    EXPECT_EQ(router_.pending_count(), 3u);

    router_.cancel_all("Shutting down");

    EXPECT_EQ(router_.pending_count(), 0u);

    auto result1 = future1.get();
    auto result2 = future2.get();
    auto result3 = future3.get();

    EXPECT_FALSE(result1.has_value());
    EXPECT_FALSE(result2.has_value());
    EXPECT_FALSE(result3.has_value());

    EXPECT_EQ(result1.error().code, ErrorCode::McpDisconnected);
}

TEST_F(MessageRouterTest, CancelAllWithNoRequestsDoesNotCrash) {
    EXPECT_EQ(router_.pending_count(), 0u);
    router_.cancel_all();
    EXPECT_EQ(router_.pending_count(), 0u);
}

// ============================================================================
// Pending Count Tests
// ============================================================================

TEST_F(MessageRouterTest, PendingCountTracksRequests) {
    EXPECT_EQ(router_.pending_count(), 0u);

    auto [id1, f1] = router_.create_pending_request();
    EXPECT_EQ(router_.pending_count(), 1u);

    auto [id2, f2] = router_.create_pending_request();
    EXPECT_EQ(router_.pending_count(), 2u);

    // Resolve one
    zoo::mcp::JsonRpcResponse resp;
    resp.result = nlohmann::json{};
    resp.id = id1;
    router_.route_response(resp);

    EXPECT_EQ(router_.pending_count(), 1u);

    // Cancel remaining
    router_.cancel_all();
    EXPECT_EQ(router_.pending_count(), 0u);
}

// ============================================================================
// Default Timeout Tests
// ============================================================================

TEST_F(MessageRouterTest, DefaultTimeoutIsConfigurable) {
    MessageRouter router(std::chrono::seconds(60));
    EXPECT_EQ(router.default_timeout(), std::chrono::seconds(60));
}

TEST_F(MessageRouterTest, DefaultTimeoutIs30Seconds) {
    EXPECT_EQ(router_.default_timeout(), std::chrono::seconds(30));
}
