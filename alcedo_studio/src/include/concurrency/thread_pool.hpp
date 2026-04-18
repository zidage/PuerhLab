//  Copyright 2025 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

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

namespace alcedo {
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
};  // namespace alcedo
