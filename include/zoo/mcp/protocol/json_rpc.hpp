#pragma once

#include "../types.hpp"
#include <nlohmann/json.hpp>
#include <string>

namespace zoo {
namespace mcp {
namespace protocol {

/**
 * @brief JSON-RPC 2.0 codec for encoding/decoding messages.
 *
 * Handles serialization of JsonRpcRequest/JsonRpcResponse to/from JSON strings.
 * All methods are static and stateless.
 */
class JsonRpc {
public:
    // ========================================================================
    // Encoding
    // ========================================================================

    static std::string encode_request(const JsonRpcRequest& request) {
        nlohmann::json j;
        j["jsonrpc"] = request.jsonrpc;
        j["method"] = request.method;

        if (!request.params.is_null() && !request.params.empty()) {
            j["params"] = request.params;
        }

        if (request.id.has_value()) {
            encode_id(j, "id", *request.id);
        }

        return j.dump();
    }

    static std::string encode_response(const JsonRpcResponse& response) {
        nlohmann::json j;
        j["jsonrpc"] = response.jsonrpc;

        if (response.result.has_value()) {
            j["result"] = *response.result;
        }

        if (response.error.has_value()) {
            nlohmann::json err;
            err["code"] = response.error->code;
            err["message"] = response.error->message;
            if (response.error->data.has_value()) {
                err["data"] = *response.error->data;
            }
            j["error"] = err;
        }

        encode_id(j, "id", response.id);

        return j.dump();
    }

    static std::string encode_notification(const std::string& method,
                                           const nlohmann::json& params = nlohmann::json::object()) {
        nlohmann::json j;
        j["jsonrpc"] = "2.0";
        j["method"] = method;
        if (!params.is_null() && !params.empty()) {
            j["params"] = params;
        }
        return j.dump();
    }

    // ========================================================================
    // Decoding
    // ========================================================================

    struct DecodeResult {
        std::optional<JsonRpcRequest> request;
        std::optional<JsonRpcResponse> response;
        std::optional<std::string> error_message;

        bool is_request() const { return request.has_value(); }
        bool is_response() const { return response.has_value(); }
        bool is_error() const { return error_message.has_value(); }
        bool is_notification() const { return is_request() && !request->id.has_value(); }
    };

    static DecodeResult decode(const std::string& input) {
        DecodeResult result;

        nlohmann::json j;
        try {
            j = nlohmann::json::parse(input);
        } catch (const nlohmann::json::exception& e) {
            result.error_message = std::string("JSON parse error: ") + e.what();
            return result;
        }

        if (!j.is_object()) {
            result.error_message = "JSON-RPC message must be an object";
            return result;
        }

        // Check for jsonrpc field
        if (!j.contains("jsonrpc") || j["jsonrpc"] != "2.0") {
            result.error_message = "Missing or invalid jsonrpc version (must be \"2.0\")";
            return result;
        }

        // Determine if this is a request/notification or a response
        if (j.contains("method")) {
            // Request or notification
            JsonRpcRequest req;
            req.jsonrpc = j["jsonrpc"];
            req.method = j["method"].get<std::string>();

            if (j.contains("params")) {
                req.params = j["params"];
            }

            if (j.contains("id")) {
                req.id = decode_id(j["id"]);
            }

            result.request = std::move(req);
        } else if (j.contains("result") || j.contains("error")) {
            // Response
            JsonRpcResponse resp;
            resp.jsonrpc = j["jsonrpc"];

            if (j.contains("result")) {
                resp.result = j["result"];
            }

            if (j.contains("error")) {
                auto& err = j["error"];
                JsonRpcError rpc_error;
                rpc_error.code = err.value("code", 0);
                rpc_error.message = err.value("message", "");
                if (err.contains("data")) {
                    rpc_error.data = err["data"];
                }
                resp.error = std::move(rpc_error);
            }

            if (j.contains("id")) {
                resp.id = decode_id(j["id"]);
            }

            result.response = std::move(resp);
        } else {
            result.error_message = "Invalid JSON-RPC message: missing method, result, or error";
        }

        return result;
    }

private:
    static void encode_id(nlohmann::json& j, const std::string& key, const RequestId& id) {
        std::visit([&](const auto& val) { j[key] = val; }, id);
    }

    static RequestId decode_id(const nlohmann::json& j) {
        if (j.is_number_integer()) {
            return j.get<int>();
        } else if (j.is_string()) {
            return j.get<std::string>();
        }
        return 0; // fallback
    }
};

} // namespace protocol
} // namespace mcp
} // namespace zoo
