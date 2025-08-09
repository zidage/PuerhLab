#pragma once

#include <functional>
#include <memory>
#include <unordered_map>

#include "op_base.hpp"

namespace puerhlab {
enum class OperatorType : int {
  EXPOSURE,
  CONTRAST,
  WHITE,
  BLACK,
  SHADOWS,
  HIGHLIGHTS,
  CURVE,
  HLS,
  SATURATION,
  TINT,
  VIBRANCE,
  CST,
  LMT,
  CLARITY,
  SHARPEN,
  COLOR_WHEEL,
  ACES_TONE_MAPPING
};
class OperatorFactory {
 public:
  using Creator = std::function<std::shared_ptr<IOperatorBase>(const nlohmann::json&)>;

  static auto Instance() -> OperatorFactory&;

  void        Register(const OperatorType& type, Creator creator);

  auto        Create(const OperatorType& type, const nlohmann::json& params = {}) const
      -> std::shared_ptr<IOperatorBase>;

  template <typename T>
  static Creator MakeCreator() {
    return [](const nlohmann::json& params) -> std::shared_ptr<IOperatorBase> {
      auto op = std::make_shared<T>(params);
      return std::static_pointer_cast<IOperatorBase>(op);  // 关键点
    };
  }

 private:
  std::unordered_map<OperatorType, Creator> _creators;
};
}  // namespace puerhlab
