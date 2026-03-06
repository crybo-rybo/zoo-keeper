#include <gtest/gtest.h>
#include "zoo/engine/request_queue.hpp"
#include <thread>
#include <chrono>
#include <vector>

using namespace zoo;
using namespace zoo::engine;

class RequestQueueTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Fresh queue for each test
    }
};

// ============================================================================
// Basic Operations
// ============================================================================

TEST_F(RequestQueueTest, PushPop) {
    RequestQueue queue;

    auto msg = Message::user("Hello");
    Request req(msg);

    EXPECT_TRUE(queue.push(std::move(req)));
    EXPECT_EQ(queue.size(), 1);
    EXPECT_FALSE(queue.empty());

    auto popped = queue.pop();
    ASSERT_TRUE(popped.has_value());
    EXPECT_EQ(popped->message.content, "Hello");
    EXPECT_TRUE(queue.empty());
    EXPECT_EQ(queue.size(), 0);
}

TEST_F(RequestQueueTest, MultipleRequests) {
    RequestQueue queue;

    for (int i = 0; i < 5; ++i) {
        auto msg = Message::user("Message " + std::to_string(i));
        Request req(msg);
        EXPECT_TRUE(queue.push(std::move(req)));
    }

    EXPECT_EQ(queue.size(), 5);

    for (int i = 0; i < 5; ++i) {
        auto popped = queue.pop();
        ASSERT_TRUE(popped.has_value());
        EXPECT_EQ(popped->message.content, "Message " + std::to_string(i));
    }

    EXPECT_TRUE(queue.empty());
}

TEST_F(RequestQueueTest, EmptyQueue) {
    RequestQueue queue;
    EXPECT_TRUE(queue.empty());
    EXPECT_EQ(queue.size(), 0);
}

// ============================================================================
// Capacity Limits
// ============================================================================

TEST_F(RequestQueueTest, MaxSizeLimit) {
    RequestQueue queue(3);  // Max 3 items

    auto msg1 = Message::user("1");
    auto msg2 = Message::user("2");
    auto msg3 = Message::user("3");
    auto msg4 = Message::user("4");

    EXPECT_TRUE(queue.push(Request(msg1)));
    EXPECT_TRUE(queue.push(Request(msg2)));
    EXPECT_TRUE(queue.push(Request(msg3)));
    EXPECT_FALSE(queue.push(Request(msg4)));  // Should fail - queue full

    EXPECT_EQ(queue.size(), 3);
}

TEST_F(RequestQueueTest, UnlimitedQueue) {
    RequestQueue queue(0);  // Unlimited

    for (int i = 0; i < 100; ++i) {
        auto msg = Message::user("Message " + std::to_string(i));
        EXPECT_TRUE(queue.push(Request(msg)));
    }

    EXPECT_EQ(queue.size(), 100);
}

// ============================================================================
// Threading Tests
// ============================================================================

TEST_F(RequestQueueTest, ConcurrentPush) {
    RequestQueue queue;
    constexpr int num_threads = 4;
    constexpr int items_per_thread = 25;

    std::vector<std::thread> threads;

    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([&queue, t]() {
            for (int i = 0; i < items_per_thread; ++i) {
                auto msg = Message::user("Thread " + std::to_string(t) + " Message " + std::to_string(i));
                queue.push(Request(msg));
            }
        });
    }

    for (auto& thread : threads) {
        thread.join();
    }

    EXPECT_EQ(queue.size(), num_threads * items_per_thread);
}

TEST_F(RequestQueueTest, ProducerConsumer) {
    RequestQueue queue;
    constexpr int num_items = 100;
    std::atomic<int> consumed{0};

    // Producer thread
    std::thread producer([&queue]() {
        for (int i = 0; i < num_items; ++i) {
            auto msg = Message::user("Message " + std::to_string(i));
            queue.push(Request(msg));
            std::this_thread::sleep_for(std::chrono::microseconds(10));
        }
    });

    // Consumer thread
    std::thread consumer([&queue, &consumed]() {
        while (consumed < num_items) {
            auto req = queue.pop_for(std::chrono::milliseconds(100));
            if (req.has_value()) {
                consumed++;
            }
        }
    });

    producer.join();
    consumer.join();

    EXPECT_EQ(consumed, num_items);
    EXPECT_TRUE(queue.empty());
}

TEST_F(RequestQueueTest, BlockingPop) {
    RequestQueue queue;
    bool popped = false;

    // Start consumer thread that blocks
    std::thread consumer([&queue, &popped]() {
        auto req = queue.pop();
        if (req.has_value()) {
            popped = true;
        }
    });

    // Give consumer time to block
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    EXPECT_FALSE(popped);

    // Push item - should unblock consumer
    auto msg = Message::user("Unblock");
    queue.push(Request(msg));

    consumer.join();
    EXPECT_TRUE(popped);
}

// ============================================================================
// Timeout Tests
// ============================================================================

TEST_F(RequestQueueTest, PopTimeout) {
    RequestQueue queue;

    auto start = std::chrono::steady_clock::now();
    auto req = queue.pop_for(std::chrono::milliseconds(100));
    auto elapsed = std::chrono::steady_clock::now() - start;

    EXPECT_FALSE(req.has_value());
    EXPECT_GE(elapsed, std::chrono::milliseconds(100));
    EXPECT_LT(elapsed, std::chrono::milliseconds(500));  // Wide slack for loaded CI machines
}

TEST_F(RequestQueueTest, PopTimeoutSuccess) {
    RequestQueue queue;

    auto msg = Message::user("Available");
    queue.push(Request(msg));

    auto req = queue.pop_for(std::chrono::milliseconds(100));
    EXPECT_TRUE(req.has_value());
    EXPECT_EQ(req->message.content, "Available");
}

TEST_F(RequestQueueTest, PopTimeoutDelayed) {
    RequestQueue queue;

    // Push item after 50ms
    std::thread producer([&queue]() {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        auto msg = Message::user("Delayed");
        queue.push(Request(msg));
    });

    // Try to pop with 200ms timeout - should succeed
    auto req = queue.pop_for(std::chrono::milliseconds(200));
    EXPECT_TRUE(req.has_value());

    producer.join();
}

// ============================================================================
// Shutdown Tests
// ============================================================================

TEST_F(RequestQueueTest, ShutdownPreventsNewPush) {
    RequestQueue queue;
    queue.shutdown();

    auto msg = Message::user("After shutdown");
    EXPECT_FALSE(queue.push(Request(msg)));
    EXPECT_TRUE(queue.is_shutdown());
}

TEST_F(RequestQueueTest, ShutdownUnblocksWaitingPop) {
    RequestQueue queue;
    bool pop_returned = false;
    std::optional<Request> result;

    std::thread consumer([&queue, &pop_returned, &result]() {
        result = queue.pop();
        pop_returned = true;
    });

    // Give consumer time to block
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    EXPECT_FALSE(pop_returned);

    // Shutdown should unblock
    queue.shutdown();

    consumer.join();
    EXPECT_TRUE(pop_returned);
    EXPECT_FALSE(result.has_value());  // Should return nullopt on shutdown
}

TEST_F(RequestQueueTest, ShutdownWithPendingItems) {
    RequestQueue queue;

    // Add items before shutdown
    for (int i = 0; i < 3; ++i) {
        auto msg = Message::user("Message " + std::to_string(i));
        queue.push(Request(msg));
    }

    queue.shutdown();

    // Should still be able to pop existing items
    auto req1 = queue.pop();
    EXPECT_TRUE(req1.has_value());

    auto req2 = queue.pop();
    EXPECT_TRUE(req2.has_value());

    auto req3 = queue.pop();
    EXPECT_TRUE(req3.has_value());

    // Now queue is empty and shutdown - should return nullopt
    auto req4 = queue.pop();
    EXPECT_FALSE(req4.has_value());
}

TEST_F(RequestQueueTest, ShutdownMultipleConsumers) {
    RequestQueue queue;
    std::atomic<int> unblocked{0};

    std::vector<std::thread> consumers;
    for (int i = 0; i < 5; ++i) {
        consumers.emplace_back([&queue, &unblocked]() {
            auto req = queue.pop();
            if (!req.has_value()) {
                unblocked++;
            }
        });
    }

    // Give consumers time to block
    std::this_thread::sleep_for(std::chrono::milliseconds(50));

    // Shutdown should unblock all consumers
    queue.shutdown();

    for (auto& consumer : consumers) {
        consumer.join();
    }

    EXPECT_EQ(unblocked, 5);
}

// ============================================================================
// Clear Tests
// ============================================================================

TEST_F(RequestQueueTest, ClearQueue) {
    RequestQueue queue;

    for (int i = 0; i < 10; ++i) {
        auto msg = Message::user("Message " + std::to_string(i));
        queue.push(Request(msg));
    }

    EXPECT_EQ(queue.size(), 10);

    queue.clear();

    EXPECT_EQ(queue.size(), 0);
    EXPECT_TRUE(queue.empty());
}

TEST_F(RequestQueueTest, ClearEmptyQueue) {
    RequestQueue queue;
    queue.clear();  // Should not crash
    EXPECT_TRUE(queue.empty());
}

// ============================================================================
// Edge Cases
// ============================================================================

TEST_F(RequestQueueTest, CapacityAfterPop) {
    RequestQueue queue(2);  // Max 2 items

    auto msg1 = Message::user("1");
    auto msg2 = Message::user("2");
    auto msg3 = Message::user("3");

    EXPECT_TRUE(queue.push(Request(msg1)));
    EXPECT_TRUE(queue.push(Request(msg2)));
    EXPECT_FALSE(queue.push(Request(msg3)));  // Full

    // Pop one item — capacity should free up
    auto popped = queue.pop_for(std::chrono::milliseconds(10));
    ASSERT_TRUE(popped.has_value());
    EXPECT_EQ(popped->message.content, "1");

    // Now we can push again
    EXPECT_TRUE(queue.push(Request(msg3)));
    EXPECT_EQ(queue.size(), 2);
}

TEST_F(RequestQueueTest, RapidPushPop) {
    RequestQueue queue;

    for (int i = 0; i < 1000; ++i) {
        auto msg = Message::user("Rapid " + std::to_string(i));
        queue.push(Request(msg));
        auto popped = queue.pop();
        EXPECT_TRUE(popped.has_value());
    }

    EXPECT_TRUE(queue.empty());
}

TEST_F(RequestQueueTest, StressTest) {
    RequestQueue queue;
    constexpr int num_producers = 4;
    constexpr int num_consumers = 2;
    constexpr int items_per_producer = 50;
    std::atomic<int> consumed{0};

    std::vector<std::thread> threads;

    // Producers
    for (int p = 0; p < num_producers; ++p) {
        threads.emplace_back([&queue, p]() {
            for (int i = 0; i < items_per_producer; ++i) {
                auto msg = Message::user("P" + std::to_string(p) + " M" + std::to_string(i));
                queue.push(Request(msg));
            }
        });
    }

    // Consumers
    const int total_items = num_producers * items_per_producer;
    for (int c = 0; c < num_consumers; ++c) {
        threads.emplace_back([&queue, &consumed]() {
            while (consumed < total_items) {
                auto req = queue.pop_for(std::chrono::milliseconds(100));
                if (req.has_value()) {
                    consumed++;
                }
            }
        });
    }

    for (auto& thread : threads) {
        thread.join();
    }

    EXPECT_EQ(consumed, total_items);
}
