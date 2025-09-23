#include "edit/operators/raw/raw_decode_op.hpp"

#include <opencv2/core/hal/interface.h>
#include <memory>

#include "decoders/processor/raw_processor.hpp"
#include "image/image_buffer.hpp"

namespace puerhlab {
RawDecodeOp::RawDecodeOp(const nlohmann::json& params) { SetParams(params); }

void RawDecodeOp::Apply(std::shared_ptr<ImageBuffer> input) {
  auto&  buffer = input->GetBuffer();

  std::unique_ptr<LibRaw> raw_processor = std::make_unique<LibRaw>();
  int    ret = raw_processor->open_buffer((void*)buffer.data(), buffer.size());
  if (ret != LIBRAW_SUCCESS) {
    throw std::runtime_error("RawDecodeOp: Unable to read raw file using LibRAW");
  }

  raw_processor->imgdata.params.output_bps = 16;
  ImageBuffer output;

  switch (_backend) {
    case RawProcessBackend::PUERH: {
      raw_processor->unpack();
      OpenCVRawProcessor processor{_params, raw_processor->imgdata.rawdata, *raw_processor};

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

  inner["cuda"]                   = _params._cuda;
  inner["highlights_reconstruct"] = _params._highlights_reconstruct;
  inner["use_camera_wb"]          = _params._use_camera_wb;
  inner["user_wb"]                = _params._user_wb;
  inner["backend"]                = (_backend == RawProcessBackend::PUERH) ? "puerh" : "libraw";

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
  if (inner.contains("cuda")) _params._cuda = inner["cuda"].get<bool>();
  if (inner.contains("highlights_reconstruct"))
    _params._highlights_reconstruct = inner["highlights_reconstruct"].get<bool>();
  if (inner.contains("use_camera_wb")) _params._use_camera_wb = inner["use_camera_wb"].get<bool>();
  if (inner.contains("user_wb")) _params._user_wb = inner["user_wb"].get<uint32_t>();
  if (inner.contains("backend")) {
    std::string backend = inner["backend"].get<std::string>();
    if (backend == "puerh")
      _backend = RawProcessBackend::PUERH;
    else if (backend == "libraw")
      _backend = RawProcessBackend::LIBRAW;
    else
      throw std::runtime_error("RawDecodeOp: Unknown backend " + backend);
  }
}
};  // namespace puerhlab