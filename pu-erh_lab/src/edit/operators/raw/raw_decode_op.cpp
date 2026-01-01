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

  switch (backend_) {
    case RawProcessBackend::PUERH: {
      raw_processor->unpack();
      RawProcessor processor{params_, raw_processor->imgdata.rawdata, *raw_processor};

      output = processor.Process();
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
      raw_processor->dcraw_clear_mem(img);
      raw_processor->recycle();
      break;
    }
  }
  *input = std::move(output);
}

auto RawDecodeOp::GetParams() const -> nlohmann::json {
  nlohmann::json params;
  nlohmann::json inner;

  inner["cuda"]                   = params_.cuda_;
  inner["highlights_reconstruct"] = params_.highlights_reconstruct_;
  inner["use_camera_wb"]          = params_.use_camera_wb_;
  inner["user_wb"]                = params_.user_wb_;
  inner["backend"]                = (backend_ == RawProcessBackend::PUERH) ? "puerh" : "libraw";

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
}

void RawDecodeOp::SetGlobalParams(OperatorParams&) const {
  throw std::runtime_error("RawDecodeOp does not support global parameters.");
}
};  // namespace puerhlab