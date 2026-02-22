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

#include "edit/operators/operator_registeration.hpp"

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
#include "edit/operators/cst/odt_op.hpp"
#include "edit/operators/curve/curve_op.hpp"
#include "edit/operators/detail/clarity_op.hpp"
#include "edit/operators/detail/sharpen_op.hpp"
#include "edit/operators/geometry/crop_rotate_op.hpp"
#include "edit/operators/geometry/resize_op.hpp"
#include "edit/operators/operator_factory.hpp"
#include "edit/operators/raw/raw_decode_op.hpp"
#include "edit/operators/wheel/color_wheel_op.hpp"
#include "edit/operators/geometry/lens_calib_op.hpp"

namespace puerhlab {
void RegisterAllOperators() {
  OperatorFactory::Instance().Register(OperatorType::RAW_DECODE, [](const nlohmann::json& params) {
    return std::make_shared<RawDecodeOp>(params);
  });

  OperatorFactory::Instance().Register(OperatorType::RESIZE, [](const nlohmann::json& params) {
    return std::make_shared<ResizeOp>(params);
  });

  OperatorFactory::Instance().Register(OperatorType::CROP_ROTATE, [](const nlohmann::json& params) {
    return std::make_shared<CropRotateOp>(params);
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

  OperatorFactory::Instance().Register(OperatorType::TO_WS, [](const nlohmann::json& params) {
    return std::make_shared<OCIO_ACES_Transform_Op>(params);
  });

  OperatorFactory::Instance().Register(OperatorType::TO_OUTPUT, [](const nlohmann::json& params) {
    return std::make_shared<OCIO_ACES_Transform_Op>(params);
  });

  OperatorFactory::Instance().Register(OperatorType::ODT, [](const nlohmann::json& params) {
    return std::make_shared<ACES_ODT_Op>(params);
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
      OperatorType::LENS_CALIBRATION,
      [](const nlohmann::json& params) { return std::make_shared<LensCalibOp>(params); });
}
};  // namespace puerhlab
