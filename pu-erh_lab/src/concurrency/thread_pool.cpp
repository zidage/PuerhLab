/*
 * @file        pu-erh_lab/src/concurrency/thread_pool.cpp
 * @brief       A thread pool for parallel tasks
 * @author      ChatGPT
 * @date        2025-03-28
 * @license     MIT
 *
 * @copyright   Copyright (c) 2025 ChatGPT
 */

// Copyright (c) 2025 ChatGPT
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

#include "concurrency/thread_pool.hpp"

#include <algorithm>
#include <future>
#include <memory>
#include <mutex>
#include <vector>

namespace puerhlab {
ThreadPool::ThreadPool(size_t thread_count) : stop(false) {
  for (size_t i = 0; i < thread_count; ++i) {
    workers.emplace_back(&ThreadPool::WorkerThread, this);
  }
}

ThreadPool::~ThreadPool() {
  {
    std::unique_lock<std::mutex> lock_front(_front_mtx);
    std::unique_lock<std::mutex> lock_rear(_rear_mtx);
    stop = true;
  }
  condition.notify_all();
  for (std::thread &worker : workers) {
    worker.join();
  }
}

void ThreadPool::WorkerThread() {
  while (true) {
    std::function<void()> task;
    {
      std::unique_lock<std::mutex> lock(_front_mtx);
      condition.wait(lock, [this] { return stop || !tasks.empty(); });
      if (stop && tasks.empty()) return;
      task = std::move(tasks.front());
      tasks.pop();
    }
    task();
  }
}

template <typename Func, typename T>
auto ThreadPool::SubmitFile(std::ifstream &&file, file_path_t &&file_path, std::vector<T>& result, uint32_t id, Func func)
    -> std::future<void> {
  auto task = std::make_shared<std::packaged_task<void()>>(
      [&file, file_path, result, id, func] { func(std::move(file), std::move(file_path), result, id); });

  {
    std::unique_lock<std::mutex> lock(_rear_mtx);
    tasks.emplace([task]() { (*task)(); });
  }

  condition.notify_one();
  return task->get_future();
}
};  // namespace puerhlab