//  Copyright 2025 Yurun Zi

//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at

//      http://www.apache.org/licenses/LICENSE-2.0

//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.

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
    std::unique_lock<std::mutex> lock(mtx);
    stop = true;
  }
  condition.notify_all();
  for (std::thread& worker : workers) {
    worker.join();
  }
}

void ThreadPool::Submit(std::function<void()> task) {
  {
    std::lock_guard<std::mutex> lock(mtx);
    tasks.push(task);
  }
  condition.notify_one();
}

void ThreadPool::WorkerThread() {
  while (true) {
    std::function<void()> task;
    {
      std::unique_lock<std::mutex> lock(mtx);
      condition.wait(lock, [this] { return stop || !tasks.empty(); });
      if (stop && tasks.empty()) return;
      task = std::move(tasks.front());
      tasks.pop();
    }
    task();
  }
}

};  // namespace puerhlab