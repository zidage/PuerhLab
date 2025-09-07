#include "edit/operators/raw/raw_decode_op.hpp"

namespace puerhlab {
RawDecodeOp::RawDecodeOp(const nlohmann::json& params) { SetParams(params); }

auto RawDecodeOp::Apply(ImageBuffer& input) -> ImageBuffer { auto& buffer = input.GetBuffer(); }

auto RawDecodeOp::GetParams() const -> nlohmann::json {
  nlohmann::json params;
  params["cuda"]                   = _params._cuda;
  params["highlights_reconstruct"] = _params._highlights_reconstruct;
  params["use_camera_wb"]          = _params._use_camera_wb;
  params["user_wb"]                = _params._user_wb;
  params["backend"]                = (_backend == RawProcessBackend::PUERH) ? "puerh" : "libraw";
  return params;
}

void RawDecodeOp::SetParams(const nlohmann::json& params) {
  if (params.contains("cuda")) _params._cuda = params["cuda"].get<bool>();
  if (params.contains("highlights_reconstruct"))
    _params._highlights_reconstruct = params["highlights_reconstruct"].get<bool>();
  if (params.contains("use_camera_wb"))
    _params._use_camera_wb = params["use_camera_wb"].get<bool>();
  if (params.contains("user_wb")) _params._user_wb = params["user_wb"].get<uint32_t>();
}
};  // namespace puerhlab