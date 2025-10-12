#include "edit/operators/curve/curve_op.hpp"

#include <cstddef>
#include <opencv2/core.hpp>
#include <opencv2/core/types.hpp>

#include "json.hpp"

namespace puerhlab {

CurveOp::CurveOp(const std::vector<cv::Point2f>& control_points) { SetCtrlPts(control_points); }

CurveOp::CurveOp(const nlohmann::json& params) { SetParams(params); }

void CurveOp::ComputeTagents() {
  size_t             N = _ctrl_pts.size();
  std::vector<float> delta(N - 1);

  for (int i = 0; i < N - 1; ++i) {
    delta[i] = (_ctrl_pts[i + 1].y - _ctrl_pts[i].y) / _h[i];
  }

  _m[0]     = delta[0];
  _m[N - 1] = delta[N - 2];

  for (int i = 1; i < N - 1; ++i) {
    if (delta[i - 1] * delta[i] <= 0) {
      _m[i] = 0;
    } else {
      float w1 = 2 * _h[i] + _h[i - 1];
      float w2 = _h[i] + 2 * _h[i - 1];
      _m[i]    = (w1 + w2) > 0 ? (w1 + w2) / ((w1 / delta[i - 1]) + (w2 / delta[i])) : 0.0f;
    }
  }
}

auto CurveOp::EvaluateCurve(float x) const -> float {
  if (x <= _ctrl_pts.front().x) return _ctrl_pts.front().y;
  if (x >= _ctrl_pts.back().x) return _ctrl_pts.back().y;

  // Find segment
  int idx = 0;
  for (int i = 0; i < static_cast<int>(_ctrl_pts.size()) - 1; ++i) {
    if (x < _ctrl_pts[i + 1].x) {
      idx = i;
      break;
    }
  }

  float t   = (x - _ctrl_pts[idx].x) / _h[idx];

  // Hermite interpolation
  float h00 = (2 * t * t * t - 3 * t * t + 1);
  float h10 = (t * t * t - 2 * t * t + t);
  float h01 = (-2 * t * t * t + 3 * t * t);
  float h11 = (t * t * t - t * t);

  float y   = h00 * _ctrl_pts[idx].y + h10 * _h[idx] * _m[idx] + h01 * _ctrl_pts[idx + 1].y +
            h11 * _h[idx] * _m[idx + 1];

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

auto CurveOp::ToKernel() const -> Kernel {
  return Kernel {
    ._type = Kernel::Type::Point,
    ._func = PointKernelFunc([this](Pixel& in) {
      float lum     = 0.2126f * in.r + 0.7152f * in.g + 0.0722f * in.b;
      float new_lum = EvaluateCurve(lum);
      float ratio   = (lum > 1e-5f) ? new_lum / lum : 0.0f;
      in.r *= ratio;
      in.g *= ratio;
      in.b *= ratio;
    })
  };
}

void CurveOp::SetCtrlPts(const std::vector<cv::Point2f>& control_points) {
  _ctrl_pts = control_points;

  size_t N  = _ctrl_pts.size();
  _h.resize(N - 1);
  _m.resize(N);

  for (size_t i = 0; i < N - 1; ++i) {
    _h[i] = _ctrl_pts[i + 1].x - _ctrl_pts[i].x;
  }

  ComputeTagents();
}

auto CurveOp::GetParams() const -> nlohmann::json {
  nlohmann::json o;

  nlohmann::json inner;
  for (size_t i = 0; i < _ctrl_pts.size(); ++i) {
    nlohmann::json pt;
    pt["x"]                        = _ctrl_pts[i].x;
    pt["y"]                        = _ctrl_pts[i].y;
    inner[std::format("pts{}", i)] = pt;
  }

  inner["size"]   = _ctrl_pts.size();

  o[_script_name] = inner;

  return o;
}

void CurveOp::SetParams(const nlohmann::json& params) {
  if (!params.contains(_script_name)) {
    return;
  }

  nlohmann::json pts_json = params[_script_name].get<nlohmann::json>();
  if (!pts_json.contains("size")) {
    return;
  }

  size_t size = pts_json["size"].get<size_t>();
  _ctrl_pts.clear();
  _h.clear();
  _m.clear();
  for (size_t i = 0; i < size; ++i) {
    auto pt_name = std::format("pts{}", i);
    if (pts_json.contains(pt_name)) {
      nlohmann::json pt_json = pts_json[pt_name].get<nlohmann::json>();
      if (pt_json.contains("x") && pt_json.contains("y")) {
        cv::Point2f pt = {pt_json["x"], pt_json["y"]};
        _ctrl_pts.emplace_back(std::move(pt));
      }
    }
  }

  size_t N = _ctrl_pts.size();
  _h.resize(N - 1);
  _m.resize(N);

  for (size_t i = 0; i < N - 1; ++i) {
    _h[i] = _ctrl_pts[i + 1].x - _ctrl_pts[i].x;
  }

  ComputeTagents();
}
};  // namespace puerhlab