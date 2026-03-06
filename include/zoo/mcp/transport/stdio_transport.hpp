#pragma once

#include "itransport.hpp"
#include "../../types.hpp"

// subprocess.h from llama.cpp vendor directory
#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4200)
#endif

#include "subprocess.h"

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

#include <atomic>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace zoo {
namespace mcp {
namespace transport {

/**
 * @brief MCP transport over subprocess stdin/stdout.
 *
 * Spawns a child process (e.g., an MCP server) and communicates via
 * newline-delimited JSON over stdin/stdout.
 *
 * Uses subprocess.h from llama.cpp vendor directory for cross-platform
 * process management.
 *
 * Threading model:
 * - A background read thread continuously reads stdout and invokes receive_callback
 * - send() is thread-safe (protected by mutex)
 * - connect()/disconnect() are not thread-safe (called from main thread only)
 */
class StdioTransport : public ITransport {
public:
    struct Config {
        std::string command;                                          ///< Command to execute (e.g., "npx", "python")
        std::vector<std::string> args;                                ///< Arguments (e.g., {"-y", "@modelcontextprotocol/server-filesystem", "/tmp"})
    };

    explicit StdioTransport(Config config)
        : config_(std::move(config)) {}

    ~StdioTransport() override {
        disconnect();
    }

    // Non-copyable, non-movable
    StdioTransport(const StdioTransport&) = delete;
    StdioTransport& operator=(const StdioTransport&) = delete;
    StdioTransport(StdioTransport&&) = delete;
    StdioTransport& operator=(StdioTransport&&) = delete;

    Expected<void> connect() override {
        if (connected_.load()) {
            return tl::unexpected(Error{ErrorCode::McpTransportFailed, "Already connected"});
        }

        // Build command line array for subprocess.
        // subprocess_create expects a null-terminated array of C strings.
        // Reserve upfront so that no reallocation occurs after the first
        // push_back — all c_str() pointers remain valid for the lifetime
        // of config_ (which is stored by value and never mutated here).
        std::vector<const char*> cmd_parts;
        cmd_parts.reserve(config_.args.size() + 2);  // command + args + nullptr
        cmd_parts.push_back(config_.command.c_str());
        for (const auto& arg : config_.args) {
            cmd_parts.push_back(arg.c_str());
        }
        cmd_parts.push_back(nullptr);

        int options = subprocess_option_inherit_environment |
                      subprocess_option_search_user_path;

        int result = subprocess_create(
            cmd_parts.data(),
            options,
            &process_
        );

        if (result != 0) {
            return tl::unexpected(Error{
                ErrorCode::McpTransportFailed,
                "Failed to spawn subprocess: " + config_.command
            });
        }

        // Cache FILE* handles at connect time so send() and disconnect()
        // operate on stable pointers protected by io_mutex_.
        stdin_fp_ = subprocess_stdin(&process_);

        connected_.store(true);

        // Start read thread; if thread creation fails, clean up the
        // subprocess so it is not leaked (connected_ would remain true
        // but no reader would be draining stdout).
        try {
            read_thread_ = std::thread([this]() {
                read_loop();
            });
        } catch (const std::system_error& e) {
            connected_.store(false);
            stdin_fp_ = nullptr;
            subprocess_destroy(&process_);
            return tl::unexpected(Error{
                ErrorCode::McpTransportFailed,
                std::string("Failed to start read thread: ") + e.what()
            });
        }

        return {};
    }

    void disconnect() override {
        if (!connected_.load()) {
            return;
        }

        // Signal the read loop to stop
        connected_.store(false);

        // Close stdin under the I/O lock to prevent send() from writing
        // to a closed handle (TOCTOU race).
        {
            std::lock_guard<std::mutex> lock(io_mutex_);
            if (stdin_fp_) {
                fclose(stdin_fp_);
                stdin_fp_ = nullptr;
            }
        }

        // Wait for read thread to finish (it will see EOF and break)
        if (read_thread_.joinable()) {
            read_thread_.join();
        }

        // Now safe to destroy subprocess — read thread is done
        subprocess_destroy(&process_);
    }

    bool is_connected() const override {
        return connected_.load();
    }

    Expected<void> send(const std::string& message) override {
        if (!connected_.load()) {
            return tl::unexpected(Error{ErrorCode::McpTransportFailed, "Not connected"});
        }

        std::string line = message + "\n";

        // Hold io_mutex_ while writing to prevent disconnect() from
        // closing stdin_fp_ between our check and the write.
        std::lock_guard<std::mutex> lock(io_mutex_);
        if (!stdin_fp_) {
            return tl::unexpected(Error{ErrorCode::McpTransportFailed, "Subprocess stdin not available"});
        }

        size_t written = fwrite(line.c_str(), 1, line.size(), stdin_fp_);
        fflush(stdin_fp_);

        if (written != line.size()) {
            return tl::unexpected(Error{ErrorCode::McpTransportFailed, "Failed to write to subprocess stdin"});
        }

        return {};
    }

    void set_receive_callback(ReceiveCallback callback) override {
        receive_callback_ = std::move(callback);
    }

    void set_error_callback(ErrorCallback callback) override {
        error_callback_ = std::move(callback);
    }

private:
    void read_loop() {
        FILE* stdout_fp = subprocess_stdout(&process_);
        if (!stdout_fp) {
            if (error_callback_) {
                error_callback_("Failed to get subprocess stdout");
            }
            return;
        }

        std::string buffer;
        char chunk[4096];

        while (connected_.load()) {
            if (!fgets(chunk, sizeof(chunk), stdout_fp)) {
                if (feof(stdout_fp) || ferror(stdout_fp)) {
                    break; // EOF or error
                }
                continue;
            }
            size_t bytes_read = strlen(chunk);

            buffer.append(chunk, bytes_read);

            // Process complete lines (newline-delimited JSON)
            size_t pos;
            while ((pos = buffer.find('\n')) != std::string::npos) {
                std::string line = buffer.substr(0, pos);
                buffer.erase(0, pos + 1);

                // Skip empty lines
                if (line.empty() || (line.size() == 1 && line[0] == '\r')) {
                    continue;
                }

                // Strip trailing \r if present
                if (!line.empty() && line.back() == '\r') {
                    line.pop_back();
                }

                if (receive_callback_) {
                    receive_callback_(line);
                }
            }
        }

        // If we exited the loop while still "connected", it means the process died
        if (connected_.load()) {
            connected_.store(false);
            if (error_callback_) {
                error_callback_("Subprocess exited unexpectedly");
            }
        }
    }

    Config config_;
    subprocess_s process_{};
    std::atomic<bool> connected_{false};
    std::thread read_thread_;
    std::mutex io_mutex_;         ///< Guards stdin_fp_ across send() and disconnect()
    FILE* stdin_fp_ = nullptr;    ///< Cached at connect(), nulled at disconnect()
    ReceiveCallback receive_callback_;
    ErrorCallback error_callback_;
};

} // namespace transport
} // namespace mcp
} // namespace zoo
