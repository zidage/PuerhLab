/*
 * @file        pu-erh_lab/src/include/concurrency/thread_pool.hpp
 * @brief       A thread pool for parallel tasks
 * @author      ChatGPT
 * @date        2025-03-19
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

#include <functional>
#include <future>
#include <queue>
#include <vector>


#include "type/type.hpp"

#pragma once

namespace puerhlab {
class ThreadPool {
public:
  ThreadPool(size_t thread_count);
  ~ThreadPool();

  template <typename Func, typename T>
  auto SubmitFile(std::vector<char> buffer, file_path_t file_path,
                  std::vector<T> &result, uint32_t id, Func func)
      -> std::future<void> {
    auto task = std::make_shared<std::packaged_task<void()>>(
        [buffer, file_path, &result, id, func] {
          func(buffer, file_path, result, id);
        });

    {
      std::unique_lock<std::mutex> lock(mtx);
      tasks.emplace([task]() { (*task)(); });
    }

    condition.notify_one();
    return task->get_future();
  }

private:
  std::queue<std::function<void()>> tasks;
  std::mutex mtx;
  std::condition_variable condition;
  std::vector<std::thread> workers;

  bool stop;

  void WorkerThread();
};
}; // namespace puerhlab
