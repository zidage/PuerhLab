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
  cv::GaussianBlur(img, blurred, cv::Size(), _usm_radius, _usm_radius, cv::BORDER_REPLICATE);

  cv::Mat high_pass = img - blurred;

  cv::Mat mask_3channel;
  cv::cvtColor(midtone_mask, mask_3channel, cv::COLOR_GRAY2BGR);

  high_pass.forEach<cv::Vec3f>([&](cv::Vec3f& h, const int* pos) {
    const cv::Vec3f& m = mask_3channel.at<cv::Vec3f>(pos[0], pos[1]);
    h[0] *= m[0] * (_scale);
    h[1] *= m[1] * (_scale);
    h[2] *= m[2] * (_scale);
  });

  img += high_pass;
}

auto ClarityOp::ToKernel() const -> Kernel {
  // return Kernel{._type = Kernel::Type::Neighbor,
  //               ._func = NeighborKernelFunc([this](ImageAccessor& in) -> ImageAccessor {
  //                 cv::Mat& img = in._tile->tile_mat;

  //                 cv::Mat  midtone_mask;
  //                 CreateMidtoneMask(img, midtone_mask);

  //                 cv::Mat blurred;
  //                 cv::GaussianBlur(img, blurred, cv::Size(), _usm_radius, _usm_radius,
  //                                  cv::BORDER_REPLICATE);

  //                 cv::Mat high_pass = img - blurred;

  //                 cv::Mat mask_3channel;
  //                 cv::cvtColor(midtone_mask, mask_3channel, cv::COLOR_GRAY2BGR);

  //                 high_pass.forEach<cv::Vec3f>([&](cv::Vec3f& h, const int* pos) {
  //                   const cv::Vec3f& m = mask_3channel.at<cv::Vec3f>(pos[0], pos[1]);
  //                   h[0] *= m[0] * (_scale);
  //                   h[1] *= m[1] * (_scale);
  //                   h[2] *= m[2] * (_scale);
  //                 });

  //                 img += high_pass;
  //                 return in;
  //               })};
  throw std::runtime_error("ClarityOp::ToKernel not implemented yet.");
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
};  // namespace puerhlab