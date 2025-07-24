#pragma once

#include <functional>
#include <memory>
#include <unordered_map>

#include "op_base.hpp"

namespace puerhlab {
enum class OperatorType : int {
  EXPOSURE,
  CONTRAST,
  TONE_REGION,
  CURVE,
  HLS,
  SATURATION,
  TINT,
  VIBRANCE,
  CST,
  CLARITY,
  SHARPEN,
  COLOR_WHEEL
};
class OperatorFactory {
 public:
  using Creator = std::function<std::shared_ptr<IOperatorBase>(const nlohmann::json&)>;

  static auto Instance() -> OperatorFactory&;

  void        Register(const OperatorType& type, Creator creator);

  auto        Create(const OperatorType& type, const nlohmann::json& params = {}) const
      -> std::shared_ptr<IOperatorBase>;

 private:
  std::unordered_map<OperatorType, Creator> _creators;
};
}  // namespace puerhlab
