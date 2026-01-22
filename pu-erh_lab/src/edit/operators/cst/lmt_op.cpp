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

#include "edit/operators/cst/lmt_op.hpp"

#include "edit/operators/op_kernel.hpp"
#include "utils/string/convert.hpp"

namespace puerhlab {
OCIO_LMT_Transform_Op::OCIO_LMT_Transform_Op(std::filesystem::path& lmt_path)
    : lmt_path_(lmt_path) {
  config_ = OCIO::GetCurrentConfig();
}

OCIO_LMT_Transform_Op::OCIO_LMT_Transform_Op(const nlohmann::json& params) {
  config_ = OCIO::GetCurrentConfig();
  SetParams(params);
}

void OCIO_LMT_Transform_Op::Apply(std::shared_ptr<ImageBuffer> input) {
  if (lmt_path_.empty()) {
    return;
  }
  auto& img           = input->GetCPUData();

  auto  lmt_transform = OCIO::FileTransform::Create();
  auto  path_str      = lmt_path_.wstring();
  lmt_transform->setSrc(conv::ToBytes(path_str).c_str());
  lmt_transform->setInterpolation(OCIO::INTERP_BEST);
  lmt_transform->setDirection(OCIO::TRANSFORM_DIR_FORWARD);

  auto lmt_processor = config_->getProcessor(lmt_transform);
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
  o[script_name_] = conv::ToBytes(lmt_path_.wstring());

  return o;
}

void OCIO_LMT_Transform_Op::SetParams(const nlohmann::json& params) {
  if (!params.contains(script_name_)) {
    // Empty path
    lmt_path_ = std::filesystem::path();
  }
  lmt_path_ = std::filesystem::path(conv::FromBytes(params[script_name_].get<std::string>()));

  // auto lmt_transform = OCIO::FileTransform::Create();
  // auto path_str      = lmt_path_.wstring();
  // lmt_transform->setSrc(conv::ToBytes(path_str).c_str());
  // lmt_transform->setInterpolation(OCIO::INTERP_BEST);
  // lmt_transform->setDirection(OCIO::TRANSFORM_DIR_FORWARD);

  // auto lmt_processor = config_->getProcessor(lmt_transform);
  // auto cpu           = lmt_processor->getDefaultCPUProcessor();
  // auto gpu           = lmt_processor->getDefaultGPUProcessor();
  // cpu_processor_      = cpu;
  // gpu_processor_      = gpu;
}

void OCIO_LMT_Transform_Op::SetGlobalParams(OperatorParams& params) const {
  // params.cpu_lmt_processor_ = cpu_processor_;
  // params.gpu_lmt_processor_ = gpu_processor_;

  params.lmt_lut_path_      = lmt_path_;
  params.to_lmt_dirty_      = true;
}
};  // namespace puerhlab