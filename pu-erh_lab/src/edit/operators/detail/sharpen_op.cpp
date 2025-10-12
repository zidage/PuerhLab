#include "edit/operators/detail/sharpen_op.hpp"

#include <opencv2/core.hpp>
#include <opencv2/core/base.hpp>
#include <opencv2/opencv.hpp>

#include "edit/operators/op_kernel.hpp"
#include "edit/operators/operator_factory.hpp"
#include "image/image_buffer.hpp"
#include "json.hpp"

namespace puerhlab {
static auto GetGaussianKernel1D(float sigma) -> std::vector<float> {
  int                radius = int(std::ceil(3 * sigma));
  int                size   = 2 * radius + 1;
  std::vector<float> kernel(size);

  float              sum = 0.0f;
  for (int i = -radius; i <= radius; ++i) {
    float val          = std::exp(-(i * i) / (2 * sigma * sigma));
    kernel[i + radius] = val;
    sum += val;
  }

  // normalize
  for (float& v : kernel) v /= sum;

  return kernel;
}

SharpenOp::SharpenOp(float offset, float radius, float threshold)
    : _offset(offset), _radius(radius), _threshold(threshold) {
  ComputeScale();
  _threshold /= 100.0f;
  _kernel = GetGaussianKernel1D(_radius);
}

SharpenOp::SharpenOp(const nlohmann::json& params) { SetParams(params); }

void SharpenOp::ComputeScale() { _scale = _offset / 100.0f; }

auto SharpenOp::GetParams() const -> nlohmann::json {
  nlohmann::json o;
  nlohmann::json inner;

  inner["offset"]    = _offset;
  inner["radius"]    = _radius;
  inner["threshold"] = _threshold;

  o[_script_name]    = inner;
  return o;
}

void SharpenOp::SetParams(const nlohmann::json& params) {
  nlohmann::json inner = params[_script_name].get<nlohmann::json>();
  if (inner.contains("offset")) {
    _offset = inner["offset"].get<float>();
  }
  if (inner.contains("radius")) {
    _radius = inner["radius"].get<float>();
  } else {
    _radius = 3.0f;
  }
  if (inner.contains("threshold")) {
    _threshold = inner["threshold"].get<float>();
    _threshold /= 100.0f;
  }
  ComputeScale();
  _kernel = GetGaussianKernel1D(_radius);
}

void SharpenOp::Apply(std::shared_ptr<ImageBuffer> input) {
  cv::Mat& img = input->GetCPUData();

  // Use USM to sharpen the image
  cv::Mat  blurred;
  cv::GaussianBlur(img, blurred, cv::Size(), _radius, _radius, cv::BORDER_REPLICATE);

  cv::Mat high_pass = img - blurred;
  if (_threshold > 0.0f) {
    cv::Mat high_pass_gray;
    cv::cvtColor(high_pass, high_pass_gray, cv::COLOR_BGR2GRAY);

    cv::Mat abs_high_pass_gray = cv::abs(high_pass_gray);

    cv::Mat mask;
    cv::threshold(abs_high_pass_gray, mask, _threshold, 1.0f, cv::THRESH_BINARY);

    cv::Mat mask_3channel;
    cv::cvtColor(mask, mask_3channel, cv::COLOR_GRAY2BGR);
    cv::multiply(high_pass, mask_3channel, high_pass);
  }

  cv::scaleAdd(high_pass, _scale, img, img);
  cv::threshold(img, img, 1.0f, 1.0f, cv::THRESH_TRUNC);
  cv::threshold(img, img, 0.0f, 0.0f, cv::THRESH_TOZERO);
}

auto SharpenOp::ToKernel() const -> Kernel {
  // The "size" of the kernel is essentially the size of the whole tile being processed.
  // Kernel USM_kernel {
  //   ._type = Kernel::Type::Neighbor,
  //   ._func = NeighborKernelFunc([this](const ImageAccessor& in) -> ImageAccessor {
  //     cv::Mat original = in._tile->tile_mat;
  //     cv::Mat blurred;
  //     // The "original" here contains the halo regions, so we can directly apply GaussianBlur on
  //     it.
  //     // Halo region will be trimmed when the tile is merged back to the full image.
  //     cv::GaussianBlur(original, blurred, cv::Size(5,5), _radius, _radius, cv::BORDER_REPLICATE);

  //     cv::Mat high_pass = original - blurred;

  //     if (_threshold > 0.0f) {
  //       cv::Mat high_pass_gray;
  //       cv::cvtColor(high_pass, high_pass_gray, cv::COLOR_BGR2GRAY);

  //       cv::Mat abs_high_pass_gray = cv::abs(high_pass_gray);

  //       cv::Mat mask;
  //       cv::threshold(abs_high_pass_gray, mask, _threshold, 1.0f, cv::THRESH_BINARY);

  //       cv::Mat mask_3channel;
  //       cv::cvtColor(mask, mask_3channel, cv::COLOR_GRAY2BGR);
  //       cv::multiply(high_pass, mask_3channel, high_pass);
  //     }

  //     cv::scaleAdd(high_pass, _scale, original, original);
  //     cv::threshold(original, original, 1.0f, 1.0f, cv::THRESH_TRUNC);
  //     cv::threshold(original, original, 0.0f, 0.0f, cv::THRESH_TOZERO);

  //     return in;
  //   })};
  // return USM_kernel;
  throw std::runtime_error("SharpenOp::ToKernel not implemented yet.");
}
};  // namespace puerhlab