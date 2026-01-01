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

#include "edit/operators/operator_factory.hpp"

#include <memory>

namespace puerhlab {
auto OperatorFactory::Instance() -> OperatorFactory& {
  static OperatorFactory instance;
  return instance;
}

void OperatorFactory::Register(const OperatorType& type, Creator creator) {
  creators_[type] = creator;
}

auto OperatorFactory::Create(const OperatorType& type, const nlohmann::json& params) const
    -> std::shared_ptr<IOperatorBase> {
  auto it = creators_.find(type);
  return (it != creators_.end()) ? (it->second)(params) : nullptr;
}
};  // namespace puerhlab