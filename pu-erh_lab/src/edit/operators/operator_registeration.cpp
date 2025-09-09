#include "edit/operators/operator_registeration.hpp"

#include "edit/operators/basic/auto_exposure_op.hpp"
#include "edit/operators/basic/black_op.hpp"
#include "edit/operators/basic/contrast_op.hpp"
#include "edit/operators/basic/exposure_op.hpp"
#include "edit/operators/basic/highlight_op.hpp"
#include "edit/operators/basic/shadow_op.hpp"
#include "edit/operators/basic/white_op.hpp"
#include "edit/operators/color/HLS_op.hpp"
#include "edit/operators/color/saturation_op.hpp"
#include "edit/operators/color/tint_op.hpp"
#include "edit/operators/color/vibrance_op.hpp"
#include "edit/operators/cst/cst_op.hpp"
#include "edit/operators/cst/lmt_op.hpp"
#include "edit/operators/curve/curve_op.hpp"
#include "edit/operators/detail/clarity_op.hpp"
#include "edit/operators/detail/sharpen_op.hpp"
#include "edit/operators/operator_factory.hpp"
#include "edit/operators/raw/raw_decode_op.hpp"
#include "edit/operators/tone_mapping/ACES_tone_mapping_op.hpp"
#include "edit/operators/wheel/color_wheel_op.hpp"

namespace puerhlab {
void RegisterAllOperators() {
  OperatorFactory::Instance().Register(OperatorType::RAW_DECODE, [](const nlohmann::json& params) {
    return std::make_shared<RawDecodeOp>(params);
  });

  OperatorFactory::Instance().Register(OperatorType::CONTRAST, [](const nlohmann::json& params) {
    return std::make_shared<ContrastOp>(params);
  });

  OperatorFactory::Instance().Register(OperatorType::EXPOSURE, [](const nlohmann::json& params) {
    return std::make_shared<ExposureOp>(params);
  });

  OperatorFactory::Instance().Register(OperatorType::WHITE, [](const nlohmann::json& params) {
    return std::make_shared<WhiteOp>(params);
  });

  OperatorFactory::Instance().Register(OperatorType::BLACK, [](const nlohmann::json& params) {
    return std::make_shared<BlackOp>(params);
  });

  OperatorFactory::Instance().Register(OperatorType::SHADOWS, [](const nlohmann::json& params) {
    return std::make_shared<ShadowsOp>(params);
  });

  OperatorFactory::Instance().Register(OperatorType::HIGHLIGHTS, [](const nlohmann::json& params) {
    return std::make_shared<HighlightsOp>(params);
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

  OperatorFactory::Instance().Register(OperatorType::LMT, [](const nlohmann::json& params) {
    return std::make_shared<OCIO_LMT_Transform_Op>(params);
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

  OperatorFactory::Instance().Register(
      OperatorType::ACES_TONE_MAPPING,
      [](const nlohmann::json& params) { return std::make_shared<ACESToneMappingOp>(params); });

  OperatorFactory::Instance().Register(
      OperatorType::AUTO_EXPOSURE,
      [](const nlohmann::json& params) { return std::make_shared<AutoExposureOp>(params); });
}
};  // namespace puerhlab
