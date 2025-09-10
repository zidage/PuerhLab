#include "edit/operators/geometry/resize_op.hpp"

#include "image/image_buffer.hpp"

namespace puerhlab {
ResizeOp::ResizeOp(const nlohmann::json& params) { SetParams(params); }

auto ResizeOp::Apply(ImageBuffer& input) -> ImageBuffer {
  auto& img = input.GetCPUData();
  int   w   = img.cols;
  int   h   = img.rows;
  if (std::max(w, h) <= _maximum_edge) return {std::move(input)};

  float scale = static_cast<float>(_maximum_edge) / static_cast<float>(std::max(w, h));
  cv::resize(img, img, cv::Size(static_cast<int>(w * scale), static_cast<int>(h * scale)), 0, 0,
             cv::INTER_LANCZOS4);
  return {std::move(img)};
}

auto ResizeOp::GetParams() const -> nlohmann::json {
  nlohmann::json params;
  nlohmann::json inner;
  inner["maximum_edge"] = _maximum_edge;
  params[_script_name]  = inner;
  return params;
}

auto ResizeOp::SetParams(const nlohmann::json& params) -> void {
  if (params.contains(_script_name)) {
    auto inner = params.at(_script_name);
    if (inner.contains("maximum_edge")) {
      _maximum_edge = inner.at("maximum_edge").get<int>();
    } else {
      _maximum_edge = 4000;
    }
  } else {
    _maximum_edge = 4000;
  }
}
};  // namespace puerhlab