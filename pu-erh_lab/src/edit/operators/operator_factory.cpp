#include "edit/operators/operator_factory.hpp"

#include <memory>

namespace puerhlab {
auto OperatorFactory::Instance() -> OperatorFactory& {
  static OperatorFactory instance;
  return instance;
}

void OperatorFactory::Register(const OperatorType& type, Creator creator) {
  _creators[type] = creator;
}

auto OperatorFactory::Create(const OperatorType& type, const nlohmann::json& params) const
    -> std::shared_ptr<IOperatorBase> {
  auto it = _creators.find(type);
  return (it != _creators.end()) ? (it->second)(params) : nullptr;
}
};  // namespace puerhlab