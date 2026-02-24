//  Copyright 2026 Yurun Zi
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

#include "edit/operators/basic/color_temp_op.hpp"

#include <iostream>
#include <opencv2/core.hpp>

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <limits>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "edit/operators/basic/camera_matrices.hpp"

namespace puerhlab {
namespace {
constexpr double kCalibrationLowCCT   = 2856.0;
constexpr double kCalibrationHighCCT  = 6504.0;
constexpr double kCustomCCTMin        = 2000.0;
constexpr double kCustomCCTMax        = 50000.0;
constexpr double kCustomTintMin       = -150.0;
constexpr double kCustomTintMax       = 150.0;
constexpr double kTintScale           = 3000.0;  // Adobe DNG SDK temperature/tint model.
constexpr double kDeterminantEpsilon  = 1e-10;
constexpr double kValueEpsilon        = 1e-10;
constexpr double kAsShotSolveEpsilon  = 1e-8;
constexpr int    kAsShotSolveMaxIter  = 16;
constexpr double kD50X                = 0.34567;
constexpr double kD50Y                = 0.35850;
constexpr double kD60X                = 0.32168;
constexpr double kD60Y                = 0.33767;
// ACEScg (AP1) white is D60. This is XYZ(D60) -> AP1 in column-vector form.
const cv::Matx33d kXyzD60ToAp1(1.6410233797, -0.3248032942, -0.2364246952, -0.6636628587,
                               1.6153315917, 0.0167563477, 0.0117218943, -0.0082844420,
                               0.9883948585);

struct CameraMatrixPair {
  cv::Matx33d cm1_ = cv::Matx33d::eye();
  cv::Matx33d cm2_ = cv::Matx33d::eye();
};

struct RobertsonLine {
  double r_;
  double u_;
  double v_;
  double t_;
};

static constexpr std::array<RobertsonLine, 31> kRobertsonLines = {
    RobertsonLine{0.0, 0.18006, 0.26352, -0.24341},
    RobertsonLine{10.0, 0.18066, 0.26589, -0.25479},
    RobertsonLine{20.0, 0.18133, 0.26846, -0.26876},
    RobertsonLine{30.0, 0.18208, 0.27119, -0.28539},
    RobertsonLine{40.0, 0.18293, 0.27407, -0.30470},
    RobertsonLine{50.0, 0.18388, 0.27709, -0.32675},
    RobertsonLine{60.0, 0.18494, 0.28021, -0.35156},
    RobertsonLine{70.0, 0.18611, 0.28342, -0.37915},
    RobertsonLine{80.0, 0.18740, 0.28668, -0.40955},
    RobertsonLine{90.0, 0.18880, 0.28997, -0.44278},
    RobertsonLine{100.0, 0.19032, 0.29326, -0.47888},
    RobertsonLine{125.0, 0.19462, 0.30141, -0.58204},
    RobertsonLine{150.0, 0.19962, 0.30921, -0.70471},
    RobertsonLine{175.0, 0.20525, 0.31647, -0.84901},
    RobertsonLine{200.0, 0.21142, 0.32312, -1.01820},
    RobertsonLine{225.0, 0.21807, 0.32909, -1.21680},
    RobertsonLine{250.0, 0.22511, 0.33439, -1.45120},
    RobertsonLine{275.0, 0.23247, 0.33904, -1.72980},
    RobertsonLine{300.0, 0.24010, 0.34308, -2.06370},
    RobertsonLine{325.0, 0.24792, 0.34655, -2.46810},
    RobertsonLine{350.0, 0.25591, 0.34951, -2.96410},
    RobertsonLine{375.0, 0.26400, 0.35200, -3.58140},
    RobertsonLine{400.0, 0.27218, 0.35407, -4.36330},
    RobertsonLine{425.0, 0.28039, 0.35577, -5.37620},
    RobertsonLine{450.0, 0.28863, 0.35714, -6.72620},
    RobertsonLine{475.0, 0.29685, 0.35823, -8.59550},
    RobertsonLine{500.0, 0.30505, 0.35907, -11.32400},
    RobertsonLine{525.0, 0.31320, 0.35968, -15.62800},
    RobertsonLine{550.0, 0.32129, 0.36011, -23.32500},
    RobertsonLine{575.0, 0.32931, 0.36038, -40.77000},
    RobertsonLine{600.0, 0.33724, 0.36051, -116.45000},
};

auto ClampFinite(double value, double min_value, double max_value) -> double {
  if (!std::isfinite(value)) {
    return min_value;
  }
  return std::clamp(value, min_value, max_value);
}

auto IsFiniteMatrix(const cv::Matx33d& m) -> bool {
  for (int r = 0; r < 3; ++r) {
    for (int c = 0; c < 3; ++c) {
      if (!std::isfinite(m(r, c))) {
        return false;
      }
    }
  }
  return true;
}

auto NormalizeCameraName(const std::string& input) -> std::string {
  std::string normalized;
  normalized.reserve(input.size());

  bool last_space = true;
  for (char ch : input) {
    const unsigned char uch = static_cast<unsigned char>(ch);
    if (std::isalnum(uch)) {
      normalized.push_back(static_cast<char>(std::tolower(uch)));
      last_space = false;
    } else if (!last_space) {
      normalized.push_back(' ');
      last_space = true;
    }
  }

  while (!normalized.empty() && normalized.back() == ' ') {
    normalized.pop_back();
  }
  return normalized;
}

auto MatrixFromArray9(const double m[9]) -> cv::Matx33d {
  return cv::Matx33d(m[0], m[1], m[2], m[3], m[4], m[5], m[6], m[7], m[8]);
}

auto CameraMatrixDatabaseIndex() -> const std::unordered_map<std::string, CameraMatrixPair>& {
  static const std::unordered_map<std::string, CameraMatrixPair> index = [] {
    std::unordered_map<std::string, CameraMatrixPair> out;
    const size_t                                       count =
        sizeof(all_camera_matrices) / sizeof(all_camera_matrices[0]);
    out.reserve(count);

    for (size_t i = 0; i < count; ++i) {
      const auto& item = all_camera_matrices[i];
      const auto  key  = NormalizeCameraName(item.camera_name_);
      if (key.empty() || out.contains(key)) {
        continue;
      }
      out.emplace(key, CameraMatrixPair{MatrixFromArray9(item.color_matrix_1_),
                                        MatrixFromArray9(item.color_matrix_2_)});
    }
    return out;
  }();
  return index;
}

auto CameraLookupCache() -> std::unordered_map<std::string, std::optional<CameraMatrixPair>>& {
  static std::unordered_map<std::string, std::optional<CameraMatrixPair>> cache;
  return cache;
}

auto CameraLookupCacheMutex() -> std::mutex& {
  static std::mutex mutex;
  return mutex;
}

auto LookupCameraMatrices(const std::string& camera_make, const std::string& camera_model,
                          CameraMatrixPair& out) -> bool {
  const auto full_key = NormalizeCameraName(camera_make + " " + camera_model);
  const auto model_key = NormalizeCameraName(camera_model);
  const auto cache_key = full_key + "|" + model_key;

  {
    std::lock_guard<std::mutex> lock(CameraLookupCacheMutex());
    auto                        it = CameraLookupCache().find(cache_key);
    if (it != CameraLookupCache().end()) {
      if (!it->second.has_value()) {
        return false;
      }
      out = *it->second;
      return true;
    }
  }

  const auto& db      = CameraMatrixDatabaseIndex();
  auto        find_db = [&](const std::string& key) -> const CameraMatrixPair* {
    if (key.empty()) {
      return nullptr;
    }
    auto it = db.find(key);
    if (it == db.end()) {
      return nullptr;
    }
    return &it->second;
  };

  const CameraMatrixPair* found = find_db(full_key);
  if (!found) {
    found = find_db(model_key);
  }
  if (!found && !model_key.empty()) {
    const auto make_key = NormalizeCameraName(camera_make);
    if (!make_key.empty() && model_key.starts_with(make_key + " ")) {
      found = find_db(model_key);
    }
  }

  {
    std::lock_guard<std::mutex> lock(CameraLookupCacheMutex());
    if (found) {
      CameraLookupCache().emplace(cache_key, *found);
      out = *found;
      return true;
    }
    CameraLookupCache().emplace(cache_key, std::nullopt);
  }

  return false;
}

auto HasValidCamXyz(const float m[9]) -> bool {
  double sum = 0.0;
  for (int i = 0; i < 9; ++i) {
    if (!std::isfinite(m[i])) {
      return false;
    }
    sum += std::abs(static_cast<double>(m[i]));
  }
  return sum > kValueEpsilon;
}

auto BuildFallbackXyzToCamera(const OperatorParams& params, cv::Matx33d& out) -> bool {
  if (!HasValidCamXyz(params.raw_cam_xyz_)) {
    return false;
  }

  const double g = std::max(static_cast<double>(params.raw_pre_mul_[1]), kValueEpsilon);
  const cv::Matx33d pre =
      cv::Matx33d::diag(cv::Vec3d(params.raw_pre_mul_[0] / g, 1.0, params.raw_pre_mul_[2] / g));
  const cv::Matx33d cam_xyz(params.raw_cam_xyz_[0], params.raw_cam_xyz_[1], params.raw_cam_xyz_[2],
                            params.raw_cam_xyz_[3], params.raw_cam_xyz_[4], params.raw_cam_xyz_[5],
                            params.raw_cam_xyz_[6], params.raw_cam_xyz_[7], params.raw_cam_xyz_[8]);
  out = pre * cam_xyz;
  return IsFiniteMatrix(out);
}

auto InterpolateColorMatrix(const cv::Matx33d& cm1, const cv::Matx33d& cm2, double cct)
    -> cv::Matx33d {
  if (!std::isfinite(cct)) {
    return cm2;
  }

  if (cct <= kCalibrationLowCCT) {
    return cm1;
  }
  if (cct >= kCalibrationHighCCT) {
    return cm2;
  }

  const double inv_t  = 1.0 / cct;
  const double inv_t1 = 1.0 / kCalibrationLowCCT;
  const double inv_t2 = 1.0 / kCalibrationHighCCT;
  const double denom  = inv_t2 - inv_t1;
  if (std::abs(denom) <= kValueEpsilon) {
    return cm2;
  }

  const double w = std::clamp((inv_t - inv_t1) / denom, 0.0, 1.0);
  return cm1 * (1.0 - w) + cm2 * w;
}

auto Invert3x3(const cv::Matx33d& m, cv::Matx33d& out) -> bool {
  const double det = cv::determinant(m);
  if (!std::isfinite(det) || std::abs(det) < kDeterminantEpsilon) {
    return false;
  }
  out = m.inv();
  return IsFiniteMatrix(out);
}

auto XYZToXY(const cv::Vec3d& xyz, cv::Vec2d& out_xy) -> bool {
  const double sum = xyz[0] + xyz[1] + xyz[2];
  if (!std::isfinite(sum) || std::abs(sum) <= kValueEpsilon) {
    return false;
  }
  const double x = xyz[0] / sum;
  const double y = xyz[1] / sum;
  if (!std::isfinite(x) || !std::isfinite(y) || y <= 0.0 || x <= 0.0 || (x + y) >= 1.0) {
    return false;
  }
  out_xy = cv::Vec2d(x, y);
  return true;
}

auto XYToXYZ(const cv::Vec2d& xy) -> cv::Vec3d {
  const double y = std::max(xy[1], kValueEpsilon);
  const double X = xy[0] / y;
  const double Y = 1.0;
  const double Z = (1.0 - xy[0] - xy[1]) / y;
  return cv::Vec3d(X, Y, Z);
}

auto XYToUV(const cv::Vec2d& xy) -> cv::Vec2d {
  const double den = (-xy[0] + 6.0 * xy[1] + 1.5);
  if (std::abs(den) <= kValueEpsilon) {
    return cv::Vec2d(0.0, 0.0);
  }
  return cv::Vec2d(2.0 * xy[0] / den, 3.0 * xy[1] / den);
}

auto UVToXY(const cv::Vec2d& uv) -> cv::Vec2d {
  const double den = (uv[0] - 4.0 * uv[1] + 2.0);
  if (std::abs(den) <= kValueEpsilon) {
    return cv::Vec2d(kD50X, kD50Y);
  }
  return cv::Vec2d(1.5 * uv[0] / den, uv[1] / den);
}

auto UVToTemperatureTint(const cv::Vec2d& uv, double& out_cct, double& out_tint) -> bool {
  double last_dt = 0.0;
  double last_du = 0.0;
  double last_dv = 0.0;

  for (size_t i = 1; i < kRobertsonLines.size(); ++i) {
    double du   = 1.0;
    double dv   = kRobertsonLines[i].t_;
    double len  = std::sqrt(1.0 + dv * dv);
    du /= len;
    dv /= len;

    const double uu = uv[0] - kRobertsonLines[i].u_;
    const double vv = uv[1] - kRobertsonLines[i].v_;
    double       dt = -uu * dv + vv * du;

    if (dt <= 0.0 || i == (kRobertsonLines.size() - 1)) {
      if (dt > 0.0) {
        dt = 0.0;
      }
      dt            = -dt;

      double blend  = 0.0;
      if (i > 1 && (dt + last_dt) > kValueEpsilon) {
        blend = dt / (last_dt + dt);
      }
      blend = std::clamp(blend, 0.0, 1.0);

      const double mired =
          kRobertsonLines[i - 1].r_ * blend + kRobertsonLines[i].r_ * (1.0 - blend);
      out_cct = (mired <= kValueEpsilon) ? kCustomCCTMax : (1e6 / mired);

      const double uu1 = uv[0] - (kRobertsonLines[i - 1].u_ * blend +
                                  kRobertsonLines[i].u_ * (1.0 - blend));
      const double vv1 = uv[1] - (kRobertsonLines[i - 1].v_ * blend +
                                  kRobertsonLines[i].v_ * (1.0 - blend));

      double du1 = du * blend + last_du * (1.0 - blend);
      double dv1 = dv * blend + last_dv * (1.0 - blend);
      len        = std::sqrt(du1 * du1 + dv1 * dv1);
      if (len <= kValueEpsilon) {
        out_tint = 0.0;
        return true;
      }
      du1 /= len;
      dv1 /= len;

      const double d_uv = -(uu1 * du1 + vv1 * dv1);
      out_tint          = d_uv * kTintScale;
      return std::isfinite(out_cct) && std::isfinite(out_tint);
    }

    last_dt = dt;
    last_du = du;
    last_dv = dv;
  }

  return false;
}

auto TemperatureTintToUV(double cct, double tint) -> cv::Vec2d {
  const double safe_cct = ClampFinite(cct, kCustomCCTMin, kCustomCCTMax);
  const double mired    = 1e6 / safe_cct;

  size_t       index    = 1;
  while (index < kRobertsonLines.size() && mired < kRobertsonLines[index].r_) {
    ++index;
  }
  if (index >= kRobertsonLines.size()) {
    index = kRobertsonLines.size() - 1;
  }
  if (index == 0) {
    index = 1;
  }

  const double denom = kRobertsonLines[index].r_ - kRobertsonLines[index - 1].r_;
  const double blend =
      (std::abs(denom) <= kValueEpsilon) ? 0.0 : (kRobertsonLines[index].r_ - mired) / denom;

  double u = kRobertsonLines[index - 1].u_ * blend + kRobertsonLines[index].u_ * (1.0 - blend);
  double v = kRobertsonLines[index - 1].v_ * blend + kRobertsonLines[index].v_ * (1.0 - blend);

  const double uu1 = 1.0;
  const double vv1 = kRobertsonLines[index - 1].t_;
  const double len1 = std::sqrt(1.0 + vv1 * vv1);

  const double uu2 = 1.0;
  const double vv2 = kRobertsonLines[index].t_;
  const double len2 = std::sqrt(1.0 + vv2 * vv2);

  const double uu3 = (uu1 / len1) * blend + (uu2 / len2) * (1.0 - blend);
  const double vv3 = (vv1 / len1) * blend + (vv2 / len2) * (1.0 - blend);
  const double duv = ClampFinite(tint, kCustomTintMin, kCustomTintMax) / kTintScale;

  u += uu3 * (-duv);
  v += vv3 * (-duv);
  return cv::Vec2d(u, v);
}

auto BuildBradfordCAT(const cv::Vec2d& src_xy, const cv::Vec2d& dst_xy) -> cv::Matx33d {
  static const cv::Matx33d kBradford(0.8951, 0.2664, -0.1614, -0.7502, 1.7135, 0.0367, 0.0389,
                                     -0.0685, 1.0296);
  static const cv::Matx33d kBradfordInv(0.9869929, -0.1470543, 0.1599627, 0.4323053, 0.5183603,
                                        0.0492912, -0.0085287, 0.0400428, 0.9684867);

  const cv::Vec3d src_xyz = XYToXYZ(src_xy);
  const cv::Vec3d dst_xyz = XYToXYZ(dst_xy);

  const cv::Vec3d src_lms = kBradford * src_xyz;
  const cv::Vec3d dst_lms = kBradford * dst_xyz;

  const cv::Matx33d diag = cv::Matx33d::diag(cv::Vec3d(
      dst_lms[0] / std::max(src_lms[0], kValueEpsilon), dst_lms[1] / std::max(src_lms[1], kValueEpsilon),
      dst_lms[2] / std::max(src_lms[2], kValueEpsilon)));
  return kBradfordInv * diag * kBradford;
}

void StoreMatrix(const cv::Matx33d& src, float dst[9]) {
  for (int r = 0; r < 3; ++r) {
    for (int c = 0; c < 3; ++c) {
      dst[r * 3 + c] = static_cast<float>(src(r, c));
    }
  }
}

auto ResolveColorMatrixEndpoints(const OperatorParams& params, cv::Matx33d& cm1, cv::Matx33d& cm2)
    -> bool {
  CameraMatrixPair pair;
  if (LookupCameraMatrices(params.raw_camera_make_, params.raw_camera_model_, pair)) {
    cm1 = pair.cm1_;
    cm2 = pair.cm2_;
    return IsFiniteMatrix(cm1) && IsFiniteMatrix(cm2);
  }

  cv::Matx33d fallback;
  if (!BuildFallbackXyzToCamera(params, fallback)) {
    return false;
  }
  cm1 = fallback;
  cm2 = fallback;
  return true;
}

auto SolveAsShotWhiteXY(const OperatorParams& params, const cv::Matx33d& cm1, const cv::Matx33d& cm2,
                        cv::Vec2d& out_xy, double& out_cct, double& out_tint) -> bool {
  const double r = std::max(static_cast<double>(params.raw_cam_mul_[0]), kValueEpsilon);
  const double g = std::max(static_cast<double>(params.raw_cam_mul_[1]), kValueEpsilon);
  const double b = std::max(static_cast<double>(params.raw_cam_mul_[2]), kValueEpsilon);
  const cv::Vec3d camera_neutral(g / r, 1.0, g / b);

  cv::Vec2d       xy(kD50X, kD50Y);
  for (int i = 0; i < kAsShotSolveMaxIter; ++i) {
    double       iter_cct  = 6500.0;
    double       iter_tint = 0.0;
    if (!UVToTemperatureTint(XYToUV(xy), iter_cct, iter_tint)) {
      break;
    }

    const cv::Matx33d xyz_to_camera = InterpolateColorMatrix(cm1, cm2, iter_cct);
    cv::Matx33d       camera_to_xyz;
    if (!Invert3x3(xyz_to_camera, camera_to_xyz)) {
      return false;
    }

    const cv::Vec3d white_xyz = camera_to_xyz * camera_neutral;
    cv::Vec2d       next_xy;
    if (!XYZToXY(white_xyz, next_xy)) {
      return false;
    }

    if (cv::norm(next_xy - xy) <= kAsShotSolveEpsilon) {
      xy = next_xy;
      break;
    }
    xy = next_xy;
  }

  if (!UVToTemperatureTint(XYToUV(xy), out_cct, out_tint)) {
    return false;
  }

  out_xy = xy;
  return true;
}
}  // namespace

ColorTempOp::ColorTempOp(const nlohmann::json& params) { SetParams(params); }

auto ColorTempOp::ParseMode(const std::string& mode) -> ColorTempMode {
  if (mode == "custom") {
    return ColorTempMode::CUSTOM;
  }
  if (mode == "as-shot" || mode == "as_shot") {
    return ColorTempMode::AS_SHOT;
  }
  return ColorTempMode::AS_SHOT;
}

auto ColorTempOp::ModeToString(ColorTempMode mode) -> std::string {
  switch (mode) {
    case ColorTempMode::CUSTOM:
      return "custom";
    case ColorTempMode::AS_SHOT:
    default:
      return "as_shot";
  }
}

void ColorTempOp::Apply(std::shared_ptr<ImageBuffer>) {
  throw std::runtime_error(
      "ColorTempOp: descriptor-only operator. Runtime matrices are resolved into global params.");
}

void ColorTempOp::ApplyGPU(std::shared_ptr<ImageBuffer>) {
  throw std::runtime_error(
      "ColorTempOp: descriptor-only operator. Runtime matrices are resolved into global params.");
}

auto ColorTempOp::GetParams() const -> nlohmann::json {
  nlohmann::json out;
  out[std::string(script_name_)] = {{"mode", ModeToString(mode_)},
                                    {"cct", custom_cct_},
                                    {"tint", custom_tint_},
                                    {"resolved_cct", resolved_cct_},
                                    {"resolved_tint", resolved_tint_}};
  return out;
}

void ColorTempOp::SetParams(const nlohmann::json& params) {
  if (!params.contains(script_name_)) {
    return;
  }

  const auto& j = params[script_name_];
  if (j.contains("mode") && j["mode"].is_string()) {
    mode_ = ParseMode(j["mode"].get<std::string>());
  }
  if (j.contains("cct")) {
    custom_cct_ = static_cast<float>(ClampFinite(j["cct"].get<double>(), kCustomCCTMin, kCustomCCTMax));
  }
  if (j.contains("tint")) {
    custom_tint_ = static_cast<float>(ClampFinite(j["tint"].get<double>(), kCustomTintMin, kCustomTintMax));
  }
  if (j.contains("resolved_cct")) {
    resolved_cct_ = j["resolved_cct"].get<float>();
  }
  if (j.contains("resolved_tint")) {
    resolved_tint_ = j["resolved_tint"].get<float>();
  }
}

void ColorTempOp::SetGlobalParams(OperatorParams& params) const {
  params.color_temp_mode_         = mode_;
  params.color_temp_custom_cct_   = custom_cct_;
  params.color_temp_custom_tint_  = custom_tint_;
  params.color_temp_resolved_cct_ = resolved_cct_;
  params.color_temp_resolved_tint_ = resolved_tint_;
  params.color_temp_runtime_dirty_ = true;
}

void ColorTempOp::EnableGlobalParams(OperatorParams& params, bool enable) {
  params.color_temp_enabled_      = enable;
  params.color_temp_runtime_dirty_ = true;
}

void ColorTempOp::ResolveRuntime(OperatorParams& params) const {
  if (!params.color_temp_enabled_) {
    params.color_temp_matrices_valid_ = false;
    return;
  }
  if (!params.raw_runtime_valid_) {
    params.color_temp_matrices_valid_ = false;
    return;
  }

  params.color_temp_mode_        = mode_;
  params.color_temp_custom_cct_  = custom_cct_;
  params.color_temp_custom_tint_ = custom_tint_;

  cv::Matx33d cm1;
  cv::Matx33d cm2;
  if (!ResolveColorMatrixEndpoints(params, cm1, cm2)) {
    params.color_temp_matrices_valid_ = false;
    return;
  }

  cv::Vec2d selected_xy(kD50X, kD50Y);
  double    selected_cct  = custom_cct_;
  double    selected_tint = custom_tint_;

  if (mode_ == ColorTempMode::AS_SHOT) {
    if (!SolveAsShotWhiteXY(params, cm1, cm2, selected_xy, selected_cct, selected_tint)) {
      params.color_temp_matrices_valid_ = false;
      return;
    }
  } else {
    selected_cct  = ClampFinite(custom_cct_, kCustomCCTMin, kCustomCCTMax);
    selected_tint = ClampFinite(custom_tint_, kCustomTintMin, kCustomTintMax);
    selected_xy   = UVToXY(TemperatureTintToUV(selected_cct, selected_tint));
  }

  const cv::Matx33d xyz_to_camera = InterpolateColorMatrix(cm1, cm2, selected_cct);
  cv::Matx33d       camera_to_xyz;
  if (!Invert3x3(xyz_to_camera, camera_to_xyz)) {
    std::cout << "ColorTempOp: Failed to invert XYZ to camera matrix.\n";
    params.color_temp_matrices_valid_ = false;
    return;
  }

  const cv::Matx33d cat_src_to_d50 = BuildBradfordCAT(selected_xy, cv::Vec2d(kD50X, kD50Y));
  const cv::Matx33d camera_to_xyz_d50 = cat_src_to_d50 * camera_to_xyz;

  const cv::Matx33d cat_d50_to_d60 = BuildBradfordCAT(cv::Vec2d(kD50X, kD50Y), cv::Vec2d(kD60X, kD60Y));
  const cv::Matx33d xyz_d50_to_ap1 = kXyzD60ToAp1 * cat_d50_to_d60;
  const cv::Matx33d camera_to_ap1  = xyz_d50_to_ap1 * camera_to_xyz_d50;

  StoreMatrix(camera_to_xyz, params.color_temp_cam_to_xyz_);
  StoreMatrix(camera_to_xyz_d50, params.color_temp_cam_to_xyz_d50_);
  StoreMatrix(xyz_d50_to_ap1, params.color_temp_xyz_d50_to_ap1_);
  StoreMatrix(camera_to_ap1, params.color_temp_cam_to_ap1_);

  resolved_cct_                         = static_cast<float>(selected_cct);
  resolved_tint_                        = static_cast<float>(selected_tint);
  params.color_temp_resolved_cct_       = resolved_cct_;
  params.color_temp_resolved_tint_      = resolved_tint_;
  params.color_temp_resolved_xy_[0]     = static_cast<float>(selected_xy[0]);
  params.color_temp_resolved_xy_[1]     = static_cast<float>(selected_xy[1]);
  params.color_temp_runtime_dirty_      = false;
  params.color_temp_matrices_valid_     = true;
}
}  // namespace puerhlab
