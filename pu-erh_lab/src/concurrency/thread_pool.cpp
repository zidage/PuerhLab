//  Copyright 2025 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#include "concurrency/thread_pool.hpp"

#include <algorithm>
#include <future>
#include <memory>
#include <mutex>
#include <vector>

#ifdef HAVE_METAL
#include <puerhlab/metal/Metal.hpp>
#endif

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
#ifdef HAVE_METAL
    auto autorelease_pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
#endif
    task();
  }
}

};  // namespace puerhlab
