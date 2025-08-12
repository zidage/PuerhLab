#include "edit/operators/cst/cv_cvt_op.hpp"

namespace puerhlab {
CVCvtColorOp::CVCvtColorOp(int code, std::optional<size_t> channel_index)
    : _code(code), _channel_index(channel_index) {}

CVCvtColorOp::CVCvtColorOp(const nlohmann::json& params) { SetParams(params); }

auto CVCvtColorOp::Apply(ImageBuffer& input) -> ImageBuffer {
  cv::Mat& img = input.GetCPUData();
  cv::UMat src;
  img.copyTo(src);

  if (img.empty()) {
    throw std::runtime_error("CVCvtColorOp: Input image is empty");
  }
  cv::cvtColor(src, src, _code);

  cv::Mat dst;
  src.copyTo(dst);

  if (_channel_index.has_value()) {
    std::vector<cv::Mat> channels;
    cv::split(dst, channels);
    return {std::move(channels.at(_channel_index.value()))};
  }
  return {std::move(dst)};
}
};  // namespace puerhlab