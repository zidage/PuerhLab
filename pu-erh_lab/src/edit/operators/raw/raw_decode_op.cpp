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

#include "edit/operators/raw/raw_decode_op.hpp"

#include <opencv2/core/hal/interface.h>

#include <memory>

#include "decoders/processor/raw_processor.hpp"
#include "image/image_buffer.hpp"

namespace puerhlab {
RawDecodeOp::RawDecodeOp(const nlohmann::json& params) { SetParams(params); }

void RawDecodeOp::Apply(std::shared_ptr<ImageBuffer> input) {
  auto&                   buffer        = input->GetBuffer();

  std::unique_ptr<LibRaw> raw_processor = std::make_unique<LibRaw>();
  int                     ret = raw_processor->open_buffer((void*)buffer.data(), buffer.size());
  if (ret != LIBRAW_SUCCESS) {
    throw std::runtime_error("RawDecodeOp: Unable to read raw file using LibRAW");
  }

  raw_processor->imgdata.params.output_bps = 16;
  ImageBuffer output;
  latest_runtime_context_ = {};

  switch (backend_) {
    case RawProcessBackend::PUERH: {
      raw_processor->unpack();
      RawProcessor processor{params_, raw_processor->imgdata.rawdata, *raw_processor};

      output = processor.Process();
      latest_runtime_context_ = processor.GetRuntimeColorContext();
      raw_processor->recycle();
      break;
    }
    case RawProcessBackend::LIBRAW: {
      raw_processor->imgdata.params.output_color   = 1;
      raw_processor->imgdata.params.gamm[0]        = 1.0;  // Linear gamma
      raw_processor->imgdata.params.gamm[1]        = 1.0;
      raw_processor->imgdata.params.no_auto_bright = 0;  // Disable auto brightness
      raw_processor->imgdata.params.use_camera_wb  = 1;  // Discarded if user_wb is set for now
      raw_processor->imgdata.rawparams.use_dngsdk  = 1;

      raw_processor->unpack();
      raw_processor->dcraw_process();
      libraw_processed_image_t* img = raw_processor->dcraw_make_mem_image(&ret);
      if (ret != LIBRAW_SUCCESS) {
        throw std::runtime_error("RawDecodeOp: Unable to process raw file using LibRAW");
      }
      if (img->type != LIBRAW_IMAGE_BITMAP) {
        throw std::runtime_error("RawDecodeOp: Unsupported image type from LibRAW");
      }
      if (img->colors != 3) {
        throw std::runtime_error("RawDecodeOp: Only support 3-channel image from LibRAW");
      }
      cv::Mat result(img->height, img->width, CV_16UC3, img->data);
      result.convertTo(result, CV_32FC3, 1.0 / 65535.0);

      output = ImageBuffer(std::move(result));
      latest_runtime_context_                      = {};
      latest_runtime_context_.output_in_camera_space_ = false;
      latest_runtime_context_.camera_make_         = raw_processor->imgdata.idata.make;
      latest_runtime_context_.camera_model_        = raw_processor->imgdata.idata.model;
      raw_processor->dcraw_clear_mem(img);
      raw_processor->recycle();
      break;
    }
  }
  *input = std::move(output);
}

void RawDecodeOp::ApplyGPU(std::shared_ptr<ImageBuffer> input) {
  // GPU implementation not available yet.
  Apply(input);
}

auto RawDecodeOp::GetParams() const -> nlohmann::json {
  nlohmann::json params;
  nlohmann::json inner;

  inner["cuda"]                   = params_.cuda_;
  inner["highlights_reconstruct"] = params_.highlights_reconstruct_;
  inner["use_camera_wb"]          = params_.use_camera_wb_;
  inner["user_wb"]                = params_.user_wb_;
  inner["backend"]                = (backend_ == RawProcessBackend::PUERH) ? "puerh" : "libraw";
  inner["decode_res"]             = static_cast<int>(params_.decode_res_);

  params["raw"]                   = inner;
  return params;
}

void RawDecodeOp::SetParams(const nlohmann::json& params) {
  if (!params.is_object()) {
    throw std::runtime_error("RawDecodeOp: Params should be a json object");
  }

  nlohmann::json inner;
  if (params.contains("raw")) {
    inner = params["raw"];
  } else {
    return;
  }
  if (inner.contains("cuda")) params_.cuda_ = inner["cuda"].get<bool>();
  if (inner.contains("highlights_reconstruct"))
    params_.highlights_reconstruct_ = inner["highlights_reconstruct"].get<bool>();
  if (inner.contains("use_camera_wb")) params_.use_camera_wb_ = inner["use_camera_wb"].get<bool>();
  if (inner.contains("user_wb")) params_.user_wb_ = inner["user_wb"].get<uint32_t>();
  if (inner.contains("backend")) {
    std::string backend = inner["backend"].get<std::string>();
    if (backend == "puerh")
      backend_ = RawProcessBackend::PUERH;
    else if (backend == "libraw")
      backend_ = RawProcessBackend::LIBRAW;
    else
      throw std::runtime_error("RawDecodeOp: Unknown backend " + backend);
  }
  if (inner.contains("decode_res"))
    params_.decode_res_ = static_cast<DecodeRes>(inner["decode_res"].get<int>());
}

void RawDecodeOp::SetGlobalParams(OperatorParams& params) const {
  params.raw_runtime_valid_ =
      latest_runtime_context_.valid_;
  params.raw_decode_input_space_ = latest_runtime_context_.output_in_camera_space_
                                       ? RawDecodeInputSpace::CAMERA
                                       : RawDecodeInputSpace::AP0;

  for (int i = 0; i < 3; ++i) {
    params.raw_cam_mul_[i] = latest_runtime_context_.cam_mul_[i];
    params.raw_pre_mul_[i] = latest_runtime_context_.pre_mul_[i];
  }

  for (int i = 0; i < 9; ++i) {
    params.raw_cam_xyz_[i] = latest_runtime_context_.cam_xyz_[i];
    params.raw_rgb_cam_[i] = latest_runtime_context_.rgb_cam_[i];
  }

  params.raw_camera_make_ = latest_runtime_context_.camera_make_;
  params.raw_camera_model_ = latest_runtime_context_.camera_model_;
  params.color_temp_runtime_dirty_ = true;
}

void RawDecodeOp::EnableGlobalParams(OperatorParams&, bool) {
  // Still DO NOTHING
  // RawDecodeOp is not a streamable operator
}
};  // namespace puerhlab
