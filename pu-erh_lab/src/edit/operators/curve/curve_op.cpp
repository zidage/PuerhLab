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

#include "edit/operators/curve/curve_op.hpp"

#include <cstddef>
#include <opencv2/core.hpp>
#include <opencv2/core/types.hpp>

#include "json.hpp"

namespace puerhlab {

CurveOp::CurveOp(const std::vector<cv::Point2f>& control_points) { SetCtrlPts(control_points); }

CurveOp::CurveOp(const nlohmann::json& params) { SetParams(params); }

void CurveOp::ComputeTagents() {
  size_t             N = ctrl_pts_.size();
  std::vector<float> delta(N - 1);

  for (int i = 0; i < N - 1; ++i) {
    delta[i] = (ctrl_pts_[i + 1].y - ctrl_pts_[i].y) / h_[i];
  }

  m_[0]     = delta[0];
  m_[N - 1] = delta[N - 2];

  for (int i = 1; i < N - 1; ++i) {
    if (delta[i - 1] * delta[i] <= 0) {
      m_[i] = 0;
    } else {
      float w1 = 2 * h_[i] + h_[i - 1];
      float w2 = h_[i] + 2 * h_[i - 1];
      m_[i]    = (w1 + w2) > 0 ? (w1 + w2) / ((w1 / delta[i - 1]) + (w2 / delta[i])) : 0.0f;
    }
  }
}

auto CurveOp::EvaluateCurve(float x) const -> float {
  if (x <= ctrl_pts_.front().x) return ctrl_pts_.front().y;
  if (x >= ctrl_pts_.back().x) return ctrl_pts_.back().y;

  // Find segment
  int idx = 0;
  for (int i = 0; i < static_cast<int>(ctrl_pts_.size()) - 1; ++i) {
    if (x < ctrl_pts_[i + 1].x) {
      idx = i;
      break;
    }
  }

  float t   = (x - ctrl_pts_[idx].x) / h_[idx];

  // Hermite interpolation
  float h00 = (2 * t * t * t - 3 * t * t + 1);
  float h10 = (t * t * t - 2 * t * t + t);
  float h01 = (-2 * t * t * t + 3 * t * t);
  float h11 = (t * t * t - t * t);

  float y   = h00 * ctrl_pts_[idx].y + h10 * h_[idx] * m_[idx] + h01 * ctrl_pts_[idx + 1].y +
            h11 * h_[idx] * m_[idx + 1];

  return std::clamp(y, 0.0f, 1.0f);
}

void CurveOp::Apply(std::shared_ptr<ImageBuffer> input) {
  auto& img = input->GetCPUData();

  img.forEach<cv::Vec3f>([&](cv::Vec3f& pixel, const int*) {
    float lum     = 0.2126f * pixel[2] + 0.7152f * pixel[1] + 0.0722f * pixel[0];
    float new_lum = EvaluateCurve(lum);
    float ratio   = (lum > 1e-5f) ? new_lum / lum : 0.0f;
    pixel *= ratio;
  });
}


void CurveOp::SetCtrlPts(const std::vector<cv::Point2f>& control_points) {
  ctrl_pts_ = control_points;

  size_t N  = ctrl_pts_.size();
  h_.resize(N - 1);
  m_.resize(N);

  for (size_t i = 0; i < N - 1; ++i) {
    h_[i] = ctrl_pts_[i + 1].x - ctrl_pts_[i].x;
  }

  ComputeTagents();
}

auto CurveOp::GetParams() const -> nlohmann::json {
  nlohmann::json o;

  nlohmann::json inner;
  for (size_t i = 0; i < ctrl_pts_.size(); ++i) {
    nlohmann::json pt;
    pt["x"]                        = ctrl_pts_[i].x;
    pt["y"]                        = ctrl_pts_[i].y;
    inner[std::format("pts{}", i)] = pt;
  }

  inner["size"]   = ctrl_pts_.size();

  o[script_name_] = inner;

  return o;
}

void CurveOp::SetParams(const nlohmann::json& params) {
  if (!params.contains(script_name_)) {
    return;
  }

  nlohmann::json pts_json = params[script_name_].get<nlohmann::json>();
  if (!pts_json.contains("size")) {
    return;
  }

  size_t size = pts_json["size"].get<size_t>();
  ctrl_pts_.clear();
  h_.clear();
  m_.clear();
  for (size_t i = 0; i < size; ++i) {
    auto pt_name = std::format("pts{}", i);
    if (pts_json.contains(pt_name)) {
      nlohmann::json pt_json = pts_json[pt_name].get<nlohmann::json>();
      if (pt_json.contains("x") && pt_json.contains("y")) {
        cv::Point2f pt = {pt_json["x"], pt_json["y"]};
        ctrl_pts_.emplace_back(std::move(pt));
      }
    }
  }

  size_t N = ctrl_pts_.size();
  h_.resize(N - 1);
  m_.resize(N);

  for (size_t i = 0; i < N - 1; ++i) {
    h_[i] = ctrl_pts_[i + 1].x - ctrl_pts_[i].x;
  }

  ComputeTagents();
}

void CurveOp::SetGlobalParams(OperatorParams& params) const {
  params.curve_ctrl_pts_ = ctrl_pts_;
  params.curve_m_        = m_;
  params.curve_h_        = h_;
}
};  // namespace puerhlab