#include "edit/operators/cst/lmt_op.hpp"

#include "edit/operators/op_kernel.hpp"
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

void OCIO_LMT_Transform_Op::Apply(std::shared_ptr<ImageBuffer> input) {
  if (_lmt_path.empty()) {
    return;
  }
  auto& img           = input->GetCPUData();

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

  auto lmt_transform = OCIO::FileTransform::Create();
  auto path_str      = _lmt_path.wstring();
  lmt_transform->setSrc(conv::ToBytes(path_str).c_str());
  lmt_transform->setInterpolation(OCIO::INTERP_BEST);
  lmt_transform->setDirection(OCIO::TRANSFORM_DIR_FORWARD);

  auto lmt_processor = config->getProcessor(lmt_transform);
  auto cpu           = lmt_processor->getDefaultCPUProcessor();
  auto gpu           = lmt_processor->getDefaultGPUProcessor();
  cpu_processor      = cpu;
  gpu_processor      = gpu;
}

void OCIO_LMT_Transform_Op::SetGlobalParams(OperatorParams& params) const {
  params.cpu_lmt_processor = cpu_processor;
  params.gpu_lmt_processor = gpu_processor;
}
};  // namespace puerhlab