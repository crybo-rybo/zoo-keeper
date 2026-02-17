#include <gtest/gtest.h>
#include <nlohmann/json.hpp>
#include "zoo/mcp/protocol/json_rpc.hpp"
#include "fixtures/mcp_messages.hpp"

using namespace zoo::mcp;
using namespace zoo::mcp::protocol;

// ============================================================================
// Encoding Tests
// ============================================================================

TEST(JsonRpcEncodeTest, EncodeRequestWithIntId) {
    JsonRpcRequest req;
    req.method = "initialize";
    req.params = {{"protocolVersion", "2024-11-05"}};
    req.id = 1;

    auto encoded = JsonRpc::encode_request(req);
    auto j = nlohmann::json::parse(encoded);

    EXPECT_EQ(j["jsonrpc"], "2.0");
    EXPECT_EQ(j["method"], "initialize");
    EXPECT_EQ(j["params"]["protocolVersion"], "2024-11-05");
    EXPECT_EQ(j["id"], 1);
}

TEST(JsonRpcEncodeTest, EncodeRequestWithStringId) {
    JsonRpcRequest req;
    req.method = "tools/list";
    req.id = std::string("abc-123");

    auto encoded = JsonRpc::encode_request(req);
    auto j = nlohmann::json::parse(encoded);

    EXPECT_EQ(j["id"], "abc-123");
}

TEST(JsonRpcEncodeTest, EncodeNotification) {
    auto encoded = JsonRpc::encode_notification("notifications/initialized");
    auto j = nlohmann::json::parse(encoded);

    EXPECT_EQ(j["jsonrpc"], "2.0");
    EXPECT_EQ(j["method"], "notifications/initialized");
    EXPECT_FALSE(j.contains("id"));
}

TEST(JsonRpcEncodeTest, EncodeNotificationWithParams) {
    nlohmann::json params = {{"level", "info"}, {"message", "test"}};
    auto encoded = JsonRpc::encode_notification("notifications/message", params);
    auto j = nlohmann::json::parse(encoded);

    EXPECT_EQ(j["params"]["level"], "info");
}

TEST(JsonRpcEncodeTest, EncodeResponseWithResult) {
    JsonRpcResponse resp;
    resp.result = nlohmann::json{{"protocolVersion", "2024-11-05"}};
    resp.id = 1;

    auto encoded = JsonRpc::encode_response(resp);
    auto j = nlohmann::json::parse(encoded);

    EXPECT_EQ(j["jsonrpc"], "2.0");
    EXPECT_EQ(j["result"]["protocolVersion"], "2024-11-05");
    EXPECT_EQ(j["id"], 1);
    EXPECT_FALSE(j.contains("error"));
}

TEST(JsonRpcEncodeTest, EncodeResponseWithError) {
    JsonRpcResponse resp;
    resp.error = JsonRpcError{-32600, "Invalid Request", std::nullopt};
    resp.id = 1;

    auto encoded = JsonRpc::encode_response(resp);
    auto j = nlohmann::json::parse(encoded);

    EXPECT_EQ(j["error"]["code"], -32600);
    EXPECT_EQ(j["error"]["message"], "Invalid Request");
    EXPECT_FALSE(j.contains("result"));
}

TEST(JsonRpcEncodeTest, EncodeResponseWithErrorData) {
    JsonRpcResponse resp;
    resp.error = JsonRpcError{-32602, "Invalid params", nlohmann::json{{"details", "missing field"}}};
    resp.id = 2;

    auto encoded = JsonRpc::encode_response(resp);
    auto j = nlohmann::json::parse(encoded);

    EXPECT_EQ(j["error"]["data"]["details"], "missing field");
}

TEST(JsonRpcEncodeTest, EncodeRequestWithEmptyParams) {
    JsonRpcRequest req;
    req.method = "tools/list";
    req.id = 1;

    auto encoded = JsonRpc::encode_request(req);
    auto j = nlohmann::json::parse(encoded);

    // Empty params should not appear in output
    EXPECT_FALSE(j.contains("params"));
}

// ============================================================================
// Decoding Tests
// ============================================================================

TEST(JsonRpcDecodeTest, DecodeValidRequest) {
    auto result = JsonRpc::decode(zoo::testing::mcp_fixtures::valid_request());

    EXPECT_TRUE(result.is_request());
    EXPECT_FALSE(result.is_response());
    EXPECT_FALSE(result.is_error());
    EXPECT_FALSE(result.is_notification());

    EXPECT_EQ(result.request->method, "initialize");
    EXPECT_TRUE(result.request->id.has_value());
    EXPECT_EQ(std::get<int>(*result.request->id), 1);
}

TEST(JsonRpcDecodeTest, DecodeValidResponse) {
    auto result = JsonRpc::decode(zoo::testing::mcp_fixtures::valid_response());

    EXPECT_FALSE(result.is_request());
    EXPECT_TRUE(result.is_response());
    EXPECT_FALSE(result.is_error());

    EXPECT_TRUE(result.response->result.has_value());
    EXPECT_EQ((*result.response->result)["protocolVersion"], "2024-11-05");
}

TEST(JsonRpcDecodeTest, DecodeValidNotification) {
    auto result = JsonRpc::decode(zoo::testing::mcp_fixtures::valid_notification());

    EXPECT_TRUE(result.is_request());
    EXPECT_TRUE(result.is_notification());
    EXPECT_EQ(result.request->method, "notifications/initialized");
    EXPECT_FALSE(result.request->id.has_value());
}

TEST(JsonRpcDecodeTest, DecodeErrorResponse) {
    auto result = JsonRpc::decode(zoo::testing::mcp_fixtures::error_response());

    EXPECT_TRUE(result.is_response());
    EXPECT_TRUE(result.response->is_error());
    EXPECT_EQ(result.response->error->code, -32600);
    EXPECT_EQ(result.response->error->message, "Invalid Request");
}

TEST(JsonRpcDecodeTest, DecodeErrorResponseWithData) {
    auto result = JsonRpc::decode(zoo::testing::mcp_fixtures::error_response_with_data());

    EXPECT_TRUE(result.response->error->data.has_value());
    EXPECT_EQ((*result.response->error->data)["details"], "missing field");
}

TEST(JsonRpcDecodeTest, DecodeMalformedJson) {
    auto result = JsonRpc::decode(zoo::testing::mcp_fixtures::malformed_json());

    EXPECT_TRUE(result.is_error());
    EXPECT_FALSE(result.is_request());
    EXPECT_FALSE(result.is_response());
}

TEST(JsonRpcDecodeTest, DecodeMissingJsonrpc) {
    auto result = JsonRpc::decode(zoo::testing::mcp_fixtures::missing_jsonrpc());

    EXPECT_TRUE(result.is_error());
}

TEST(JsonRpcDecodeTest, DecodeWrongJsonrpcVersion) {
    auto result = JsonRpc::decode(zoo::testing::mcp_fixtures::wrong_jsonrpc_version());

    EXPECT_TRUE(result.is_error());
}

TEST(JsonRpcDecodeTest, DecodeEmptyObject) {
    auto result = JsonRpc::decode("{}");

    EXPECT_TRUE(result.is_error());
}

TEST(JsonRpcDecodeTest, DecodeNonObject) {
    auto result = JsonRpc::decode("[1,2,3]");

    EXPECT_TRUE(result.is_error());
}

// ============================================================================
// Round-trip Tests
// ============================================================================

TEST(JsonRpcRoundTripTest, RequestRoundTrip) {
    JsonRpcRequest orig;
    orig.method = "tools/call";
    orig.params = {{"name", "read_file"}, {"arguments", {{"path", "/tmp/test"}}}};
    orig.id = 42;

    auto encoded = JsonRpc::encode_request(orig);
    auto decoded = JsonRpc::decode(encoded);

    ASSERT_TRUE(decoded.is_request());
    EXPECT_EQ(decoded.request->method, orig.method);
    EXPECT_EQ(decoded.request->params, orig.params);
    EXPECT_EQ(std::get<int>(*decoded.request->id), 42);
}

TEST(JsonRpcRoundTripTest, ResponseRoundTrip) {
    JsonRpcResponse orig;
    orig.result = nlohmann::json{{"content", {{"type", "text"}, {"text", "hello"}}}};
    orig.id = 42;

    auto encoded = JsonRpc::encode_response(orig);
    auto decoded = JsonRpc::decode(encoded);

    ASSERT_TRUE(decoded.is_response());
    EXPECT_EQ(decoded.response->result, orig.result);
    EXPECT_EQ(std::get<int>(decoded.response->id), 42);
}

TEST(JsonRpcRoundTripTest, ErrorResponseRoundTrip) {
    JsonRpcResponse orig;
    orig.error = JsonRpcError{-32601, "Method not found", std::nullopt};
    orig.id = 5;

    auto encoded = JsonRpc::encode_response(orig);
    auto decoded = JsonRpc::decode(encoded);

    ASSERT_TRUE(decoded.is_response());
    ASSERT_TRUE(decoded.response->is_error());
    EXPECT_EQ(decoded.response->error->code, -32601);
    EXPECT_EQ(decoded.response->error->message, "Method not found");
}

TEST(JsonRpcRoundTripTest, NotificationRoundTrip) {
    auto encoded = JsonRpc::encode_notification("notifications/progress",
                                                 {{"token", "abc"}, {"progress", 50}});
    auto decoded = JsonRpc::decode(encoded);

    ASSERT_TRUE(decoded.is_notification());
    EXPECT_EQ(decoded.request->method, "notifications/progress");
    EXPECT_EQ(decoded.request->params["progress"], 50);
}

TEST(JsonRpcEncodeTest, EncodeRequestWithNoId) {
    JsonRpcRequest req;
    req.method = "notifications/cancelled";
    // No id set — this is a notification

    auto encoded = JsonRpc::encode_request(req);
    auto j = nlohmann::json::parse(encoded);

    EXPECT_FALSE(j.contains("id"));
}

TEST(JsonRpcDecodeTest, DecodeResponseWithNullResult) {
    std::string json = R"({"jsonrpc":"2.0","result":null,"id":1})";
    auto result = JsonRpc::decode(json);

    EXPECT_TRUE(result.is_response());
    EXPECT_TRUE(result.response->result.has_value());
    EXPECT_TRUE(result.response->result->is_null());
}

TEST(JsonRpcDecodeTest, DecodeStringId) {
    std::string json = R"({"jsonrpc":"2.0","method":"test","id":"string-id-42"})";
    auto result = JsonRpc::decode(json);

    ASSERT_TRUE(result.is_request());
    ASSERT_TRUE(result.request->id.has_value());
    EXPECT_EQ(std::get<std::string>(*result.request->id), "string-id-42");
}
