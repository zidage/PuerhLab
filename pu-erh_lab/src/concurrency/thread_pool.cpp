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
ThreadPool::ThreadPool(size_t thread_count) : stop_(false) {
  for (size_t i = 0; i < thread_count; ++i) {
    workers_.emplace_back(&ThreadPool::WorkerThread, this);
  }
}

ThreadPool::~ThreadPool() {
  {
    std::unique_lock<std::mutex> lock(mtx_);
    stop_ = true;
  }
  condition_.notify_all();
  for (std::thread& worker : workers_) {
    worker.join();
  }
}

void ThreadPool::Submit(std::function<void()> task) {
  {
    std::lock_guard<std::mutex> lock(mtx_);
    tasks_.push(task);
  }
  condition_.notify_one();
}

void ThreadPool::WorkerThread() {
  while (true) {
    std::function<void()> task;
    {
      std::unique_lock<std::mutex> lock(mtx_);
      condition_.wait(lock, [this] { return stop_ || !tasks_.empty(); });
      if (stop_ && tasks_.empty()) return;
      task = std::move(tasks_.front());
      tasks_.pop();
    }
    task();
  }
}

};  // namespace puerhlab