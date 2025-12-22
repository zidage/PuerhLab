#include "edit/operators/detail/clarity_op.hpp"

#include <opencv2/core/hal/interface.h>

#include <memory>
#include <opencv2/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/opencv.hpp>
#include <vector>

#include "edit/operators/curve/curve_op.hpp"
#include "edit/operators/op_kernel.hpp"
#include "edit/operators/operator_factory.hpp"

namespace puerhlab {

float ClarityOp::_usm_radius = 5.0f;

ClarityOp::ClarityOp() : _clarity_offset(0) { _scale = 1.0f; }
ClarityOp::ClarityOp(float clarity_offset) : _clarity_offset(clarity_offset) {
  _scale = clarity_offset / 300.0f;
}

ClarityOp::ClarityOp(const nlohmann::json& params) { SetParams(params); }

void ClarityOp::CreateMidtoneMask(cv::Mat& input, cv::Mat& mask) const {
  cv::Mat luminosity_mask;
  cv::cvtColor(input, luminosity_mask, cv::COLOR_BGR2GRAY);

  // Apply a "U" shape curve
  luminosity_mask = luminosity_mask - 0.5f;
  luminosity_mask = luminosity_mask * 2.0f;
  cv::pow(luminosity_mask, 2.0, luminosity_mask);
  mask = 1.0f - luminosity_mask;

  // if (_blur_sigma > 0) {
  //   cv::GaussianBlur(mask, mask, cv::Size(), _blur_sigma, _blur_sigma);
  // }
}

void ClarityOp::Apply(std::shared_ptr<ImageBuffer> input) {
  // Adpated from
  // https://community.adobe.com/t5/photoshop-ecosystem-discussions/what-exactly-is-clarity/td-p/8957968
  cv::Mat& img = input->GetCPUData();

  cv::Mat  midtone_mask;
  CreateMidtoneMask(img, midtone_mask);

  cv::Mat blurred;
  // Use reflect padding to avoid brightness seams at tile boundaries
  cv::GaussianBlur(img, blurred, cv::Size(), _usm_radius, _usm_radius, cv::BORDER_REFLECT101);

  cv::Mat    high_pass  = img - blurred;

  const bool continuous = high_pass.isContinuous() && midtone_mask.isContinuous();
  const int  rows       = high_pass.rows;
  const int  cols       = high_pass.cols;

  if (continuous) {
    const int total    = rows * cols;
    auto*     hp_ptr   = high_pass.ptr<cv::Vec3f>();
    auto*     mask_ptr = midtone_mask.ptr<float>();
    for (int i = 0; i < total; ++i) {
      const float w = mask_ptr[i] * _scale;
      hp_ptr[i][0] *= w;
      hp_ptr[i][1] *= w;
      hp_ptr[i][2] *= w;
    }
  } else {
    for (int r = 0; r < rows; ++r) {
      auto*        hp_ptr = high_pass.ptr<cv::Vec3f>(r);
      const float* m      = midtone_mask.ptr<float>(r);
      for (int c = 0; c < cols; ++c) {
        const float w = m[c] * _scale;
        hp_ptr[c][0] *= w;
        hp_ptr[c][1] *= w;
        hp_ptr[c][2] *= w;
      }
    }
  }

  img += high_pass;
}


auto ClarityOp::GetParams() const -> nlohmann::json {
  nlohmann::json o;
  o[_script_name] = _clarity_offset;

  return o;
}

void ClarityOp::SetParams(const nlohmann::json& params) {
  if (params.contains(_script_name)) {
    _clarity_offset = params[_script_name];
  } else {
    _clarity_offset = 0.0f;
  }
  _scale = _clarity_offset / 300.0f;
}

void ClarityOp::SetGlobalParams(OperatorParams& params) const { params.clarity_offset = _scale; }
};  // namespace puerhlab