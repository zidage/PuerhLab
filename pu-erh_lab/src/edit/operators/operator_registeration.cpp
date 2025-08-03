#include "edit/operators/operator_registeration.hpp"

#include "edit/operators/basic/contrast_op.hpp"
#include "edit/operators/basic/exposure_op.hpp"
#include "edit/operators/basic/tone_region_op.hpp"
#include "edit/operators/color/HLS_op.hpp"
#include "edit/operators/color/saturation_op.hpp"
#include "edit/operators/color/tint_op.hpp"
#include "edit/operators/color/vibrance_op.hpp"
#include "edit/operators/cst/cst_op.hpp"
#include "edit/operators/curve/curve_op.hpp"
#include "edit/operators/detail/clarity_op.hpp"
#include "edit/operators/detail/sharpen_op.hpp"
#include "edit/operators/operator_factory.hpp"
#include "edit/operators/wheel/color_wheel_op.hpp"

namespace puerhlab {
void RegisterAllOperators() {
  OperatorFactory::Instance().Register(OperatorType::CONTRAST, [](const nlohmann::json& params) {
    return std::make_shared<ContrastOp>(params);
  });

  OperatorFactory::Instance().Register(OperatorType::EXPOSURE, [](const nlohmann::json& params) {
    return std::make_shared<ExposureOp>(params);
  });

  OperatorFactory::Instance().Register(OperatorType::TONE_REGION, [](const nlohmann::json& params) {
    return std::make_shared<ToneRegionOp>(params);
  });

  OperatorFactory::Instance().Register(OperatorType::HLS, [](const nlohmann::json& params) {
    return std::make_shared<HLSOp>(params);
  });

  OperatorFactory::Instance().Register(OperatorType::SATURATION, [](const nlohmann::json& params) {
    return std::make_shared<SaturationOp>(params);
  });

  OperatorFactory::Instance().Register(OperatorType::TINT, [](const nlohmann::json& params) {
    return std::make_shared<TintOp>(params);
  });

  OperatorFactory::Instance().Register(OperatorType::VIBRANCE, [](const nlohmann::json& params) {
    return std::make_shared<VibranceOp>(params);
  });

  OperatorFactory::Instance().Register(OperatorType::CST, [](const nlohmann::json& params) {
    return std::make_shared<OCIO_ACES_Transform_Op>(params);
  });

  OperatorFactory::Instance().Register(OperatorType::CURVE, [](const nlohmann::json& params) {
    return std::make_shared<CurveOp>(params);
  });

  OperatorFactory::Instance().Register(OperatorType::CLARITY, [](const nlohmann::json& params) {
    return std::make_shared<ClarityOp>(params);
  });

  OperatorFactory::Instance().Register(OperatorType::SHARPEN, [](const nlohmann::json& params) {
    return std::make_shared<SharpenOp>(params);
  });

  OperatorFactory::Instance().Register(OperatorType::COLOR_WHEEL, [](const nlohmann::json& params) {
    return std::make_shared<ColorWheelOp>(params);
  });
}
};  // namespace puerhlab
