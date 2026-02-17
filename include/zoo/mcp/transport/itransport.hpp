#pragma once

#include "../../types.hpp"
#include <functional>
#include <string>

namespace zoo {
namespace mcp {
namespace transport {

/**
 * @brief Abstract interface for MCP transports.
 *
 * A transport handles the physical communication with an MCP server.
 * Implementations handle stdin/stdout, SSE, or other transport mechanisms.
 *
 * Threading model:
 * - connect()/disconnect() called from the calling thread
 * - send() called from the inference thread (must be thread-safe)
 * - receive_callback invoked from the transport's read thread
 */
class ITransport {
public:
    using ReceiveCallback = std::function<void(const std::string&)>;
    using ErrorCallback = std::function<void(const std::string&)>;

    virtual ~ITransport() = default;

    virtual Expected<void> connect() = 0;
    virtual void disconnect() = 0;
    virtual bool is_connected() const = 0;

    virtual Expected<void> send(const std::string& message) = 0;

    virtual void set_receive_callback(ReceiveCallback callback) = 0;
    virtual void set_error_callback(ErrorCallback callback) = 0;
};

} // namespace transport
} // namespace mcp
} // namespace zoo
