//  Copyright 2025 Yurun Zi
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.

// TODO: Change tasks to MPMS Queue to improve efficiency

#include <cstddef>
#include <cstdint>
#include <functional>
#include <future>
#include <memory>
#include <queue>
#include <type_traits>
#include <vector>

#include "type/type.hpp"
#include "utils/queue/queue.hpp"

#pragma once

namespace puerhlab {
class ThreadPool {
 public:
  ThreadPool(size_t thread_count);
  ~ThreadPool();

  void Submit(std::function<void()> task);

  template <typename F>
  void Submit(F&& task) {
    using TaskT = std::decay_t<F>;
    static_assert(std::is_copy_constructible_v<TaskT>,
                  "ThreadPool::Submit requires a copy-constructible task when using std::function."
                  " Wrap move-only state in std::shared_ptr or provide a copyable callable.");
    Submit(std::function<void()>(std::forward<F>(task)));
  }

 private:
  std::queue<std::function<void()>> tasks_;
  std::mutex                        mtx_;
  std::condition_variable           condition_;
  std::vector<std::thread>          workers_;

  bool                              stop_;

  void                              WorkerThread();
};
};  // namespace puerhlab
