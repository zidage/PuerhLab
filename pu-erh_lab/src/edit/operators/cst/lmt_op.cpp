#include "edit/operators/cst/lmt_op.hpp"

#include "utils/string/convert.hpp"

namespace puerhlab {
OCIO_LMT_Transform_Op::OCIO_LMT_Transform_Op(std::filesystem::path& lmt_path)
    : _lmt_path(lmt_path) {
  config = OCIO::GetCurrentConfig();
}

OCIO_LMT_Transform_Op::OCIO_LMT_Transform_Op(const nlohmann::json& params) {
  config = OCIO::GetCurrentConfig();
  SetParams(params);
}

auto OCIO_LMT_Transform_Op::Apply(ImageBuffer& input) -> ImageBuffer {
  if (_lmt_path.empty()) {
    return {std::move(input)};
  }
  auto& img           = input.GetCPUData();

  auto  lmt_transform = OCIO::FileTransform::Create();
  auto  path_str      = _lmt_path.wstring();
  lmt_transform->setSrc(conv::ToBytes(path_str).c_str());
  lmt_transform->setInterpolation(OCIO::INTERP_BEST);
  lmt_transform->setDirection(OCIO::TRANSFORM_DIR_FORWARD);

  auto lmt_processor = config->getProcessor(lmt_transform);
  auto cpu           = lmt_processor->getDefaultCPUProcessor();

  cv::parallel_for_(cv::Range(0, img.rows), [&](const cv::Range& range) {
    for (int y = range.start; y < range.end; ++y) {
      cv::Vec3f* row = img.ptr<cv::Vec3f>(y);
      for (int x = 0; x < img.cols; ++x) {
        cpu->applyRGB(&row[x][0]);
      }
    }
  });

  return {std::move(img)};
}

auto OCIO_LMT_Transform_Op::GetParams() const -> nlohmann::json {
  nlohmann::json o;
  o[_script_name] = conv::ToBytes(_lmt_path.wstring());

  return o;
}

void OCIO_LMT_Transform_Op::SetParams(const nlohmann::json& params) {
  if (!params.contains(_script_name)) {
    // Empty path
    _lmt_path = std::filesystem::path();
  }
  _lmt_path = std::filesystem::path(conv::FromBytes(params[_script_name].get<std::string>()));
}
};  // namespace puerhlab