//  Copyright 2025 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#pragma once

#include <concepts>
#include <cstdint>
namespace puerhlab {
namespace IncrID {
template <typename IDType>
concept Incrementable = requires(IDType t) {
  { ++t } -> std::same_as<IDType&>;
};
template <Incrementable T>
class IDGenerator {
 private:
  T counter_;

 public:
  IDGenerator(T start_id) : counter_(start_id) {}
  auto GenerateID() -> T { return ++counter_; }
  auto GetCurrentID() const -> T { return counter_; }
  void SetStartID(T start_id) { counter_ = start_id; }
};
};  // namespace IncrID
};  // namespace puerhlab