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
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include <cstdint>
#include <mutex>
#include <queue>


#pragma once

namespace puerhlab {
/**
 * @brief A thread-safe non-blocking task queue used by a RawDecoder.
 */
template <typename T> class ConcurrentBlockingQueue {
public:
  std::uint32_t _max_size;
  std::uint32_t _low_threadshold;
  std::uint32_t _high_threadshold;
  std::queue<T> _queue;
  // Mutex used for non-blocking queue
  std::mutex mtx;
  std::condition_variable _producer_cv;
  std::condition_variable _consumer_cv;

  explicit ConcurrentBlockingQueue();

  explicit ConcurrentBlockingQueue(uint32_t max_size) : _max_size(max_size) {
    _low_threadshold = (uint32_t)(max_size * 0.6);
    _high_threadshold = (uint32_t)(max_size * 0.8);
  }

  /**
   * @brief A thread-safe wrapper for _request_queue push() method
   *
   * @param new_request the request to enqueue
   */
  void push(T new_request) {
    {
      std::unique_lock<std::mutex> lock(mtx);
      _producer_cv.wait(lock, [this]() {
        return _queue.size() < _high_threadshold;
      });
      _queue.push(std::move(new_request));
    }
    _consumer_cv.notify_one();
  }

  /**
   * @brief A thread-safe wrapper for pop() method
   *
   * @return the front-most element of the queue
   */
  T pop() {
    std::unique_lock<std::mutex> lock(mtx);
    // Wait for the queue to be fill with at least one value
    _consumer_cv.wait(lock, [this] { return !_queue.empty(); });

    auto handled_request = _queue.front();
    _queue.pop();

    if (_queue.size() <= _low_threadshold) {
      _producer_cv.notify_all();
    }

    return handled_request;
  }
};
}; // namespace puerhlab
