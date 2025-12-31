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
  T _counter;

 public:
  IDGenerator(T start_id) : _counter(start_id) {}
  auto GenerateID() -> T { return ++_counter; }
  auto GetCurrentID() const -> T { return _counter; }
  void SetStartID(T start_id) { _counter = start_id; }
};
};  // namespace IncrID
};  // namespace puerhlab