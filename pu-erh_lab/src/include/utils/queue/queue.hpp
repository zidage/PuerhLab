/*
 * @file        pu-erh_lab/src/include/queue/queue.hpp
 * @brief       Implementation of a non-blocking queue for various use cases
 * @author      Yurun Zi
 * @date        2025-03-20
 * @license     MIT
 *
 * @copyright   Copyright (c) 2025 Yurun Zi
 */

// Copyright (c) 2025 Yurun Zi
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include <queue>
#include <mutex>

#pragma once

namespace puerhlab {
/**
 * @brief A thread-safe non-blocking task queue used by a RawDecoder.
 */
template <typename T>
class NonBlockingQueue {
 public:
  std::queue<T> _request_queue;
  // Mutex used for non-blocking queue
  std::mutex _front_mtx;
  std::mutex _rear_mtx;
  std::condition_variable cv;

  explicit NonBlockingQueue() = default;

  /**
   * @brief A thread-safe wrapper for _request_queue push() method
   *
   * @param new_request the request to enqueue
   */
  void push(T new_request) {
    {
      std::lock_guard<std::mutex> lock(_rear_mtx);
      _request_queue.push(std::move(new_request));
    }
    cv.notify_one();
  }

  /**
   * @brief A thread-safe wrapper for pop() method
   *
   * @return std::shared_ptr<DecodeRequest>
   */
  T pop() {
    std::unique_lock<std::mutex> lock(_front_mtx);
    // Wait for the queue to be fill with at least one value
    cv.wait(lock, [this] { return !_request_queue.empty(); });

    auto handled_request = _request_queue.front();
    _request_queue.pop();
    return handled_request;
  }
 };
};
