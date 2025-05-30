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
#include <optional>
#include <queue>

#pragma once

namespace puerhlab {
/**
 * @brief A thread-safe blocking task queue used by the threadpool.
 */
template <typename T>
class ConcurrentBlockingQueue {
 public:
  std::uint32_t           _max_size;
  bool                    _has_capacity_limit = true;
  std::queue<T>           _queue;
  // Mutex used for non-blocking queue
  std::mutex              mtx;
  std::condition_variable _producer_cv;
  std::condition_variable _consumer_cv;

  explicit ConcurrentBlockingQueue() { _has_capacity_limit = false; };

  explicit ConcurrentBlockingQueue(uint32_t max_size) : _max_size(max_size) {}

  /**
   * @brief A thread-safe wrapper for _request_queue push() method
   *
   * @param new_request the request to enqueue
   */
  void push(T new_request) {
    {
      std::unique_lock<std::mutex> lock(mtx);
      _queue.push(std::move(new_request));
    }
    _consumer_cv.notify_all();
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

    return handled_request;
  }
};

/**
 * @brief A thread-safe non-blocking ring buffer.
 */
template <typename T>
class LockFreeMPMCQueue {
 public:
  explicit LockFreeMPMCQueue(size_t capacity) : _capacity(capacity), _buffer(capacity) {
    for (size_t i = 0; i < capacity; ++i) {
      _buffer[i].sequence.store(i, std::memory_order_relaxed);
    }
    _head.store(0, std::memory_order_relaxed);
    _tail.store(0, std::memory_order_relaxed);
  }

  bool push(const T &item) {
    size_t pos = _tail.load(std::memory_order_relaxed);
    while (true) {
      Slot    &slot = _buffer[pos % _capacity];
      size_t   seq  = slot.sequence.load(std::memory_order_acquire);
      intptr_t diff = (intptr_t)seq - (intptr_t)pos;
      if (diff == 0) {
        if (_tail.compare_exchange_weak(pos, pos + 1, std::memory_order_relaxed)) {
          slot.data = item;
          slot.sequence.store(pos + 1, std::memory_order_release);
          return true;
        }
      } else if (diff < 0) {
        return false;  // queue is full
      } else {
        pos = _tail.load(std::memory_order_relaxed);  // retry
      }
    }
  }

  std::optional<T> pop() {
    size_t pos = _head.load(std::memory_order_relaxed);
    while (true) {
      Slot    &slot = _buffer[pos % _capacity];
      size_t   seq  = slot.sequence.load(std::memory_order_acquire);
      intptr_t diff = (intptr_t)seq - (intptr_t)(pos + 1);
      if (diff == 0) {
        if (_head.compare_exchange_weak(pos, pos + 1, std::memory_order_relaxed)) {
          T result = slot.data;
          slot.sequence.store(pos + _capacity, std::memory_order_release);
          return result;
        }
      } else if (diff < 0) {
        return std::nullopt;  // queue is empty
      } else {
        pos = _head.load(std::memory_order_relaxed);  // retry
      }
    }
  }

  bool empty() const { return _head.load(std::memory_order_acquire) == _tail.load(std::memory_order_acquire); }

 private:
  struct Slot {
    std::atomic<size_t> sequence;
    T                   data;
  };

  size_t              _capacity;
  std::vector<Slot>   _buffer;
  std::atomic<size_t> _head{0};
  std::atomic<size_t> _tail{0};
};

template <typename T>
class BlockingMPMCQueue {
 public:
  explicit BlockingMPMCQueue(size_t capacity) : _queue(capacity) {}

  void push(const T &item) {
    while (true) {
      if (_queue.push(item)) {
        {
          std::lock_guard<std::mutex> lock(_cv_mutex);
          _not_empty.notify_one();
        }
        return;
      }

      std::unique_lock<std::mutex> lock(_cv_mutex);
      _not_full.wait(lock);  // wait until space available
    }
  }

  T pop() {
    while (true) {
      auto item = _queue.pop();
      if (item.has_value()) {
        {
          std::lock_guard<std::mutex> lock(_cv_mutex);
          _not_full.notify_one();
        }
        return item.value();
      }

      std::unique_lock<std::mutex> lock(_cv_mutex);
      _not_empty.wait(lock);  // wait until item available
    }
  }

  bool empty() const { return _queue.empty(); }

 private:
  LockFreeMPMCQueue<T>    _queue;

  std::condition_variable _not_empty;
  std::condition_variable _not_full;
  std::mutex              _cv_mutex;
};
};  // namespace puerhlab
