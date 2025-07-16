#include "edit/operators/cst/cst_op.hpp"

#include <OpenColorIO/OpenColorIO.h>

#include <stdexcept>

#include "image/image_buffer.hpp"
#include "json.hpp"
#include "utils/string/convert.hpp"

namespace puerhlab {
OCIOTransformOp::OCIOTransformOp(const std::string& src, const std::string& dst)
    : _src_space(src), _dst_space(dst) {
  config = OCIO::GetCurrentConfig();
}

OCIOTransformOp::OCIOTransformOp(const std::string& src, const std::string& dst,
                                 const char* config_path)
    : _src_space(src), _dst_space(dst) {
  config = OCIO::Config::CreateFromFile(config_path);
}

auto OCIOTransformOp::Apply(ImageBuffer& input) -> ImageBuffer {
  auto                  processor = config->getProcessor(_src_space.c_str(), _dst_space.c_str());
  auto                  cpu       = processor->getDefaultCPUProcessor();

  auto&                 img       = input.GetCPUData();
  OCIO::PackedImageDesc desc(img.data, img.cols, img.rows, 3);

  cpu->apply(desc);
  return {std::move(img)};
}

auto OCIOTransformOp::GetParams() const -> nlohmann::json {
  nlohmann::json o;
  nlohmann::json inner;

  inner["src"]    = _src_space;
  inner["dest"]   = _dst_space;
  o[_script_name] = inner;

  return o;
}

void OCIOTransformOp::SetParams(const nlohmann::json& params) {
  if (!params.contains(_script_name)) {
    throw std::invalid_argument("CST Operator: Not a valid adjustments JSON");
  }
  nlohmann::json inner = params[_script_name].get<nlohmann::json>();
  if (!inner.contains("src") || !inner.contains("dst")) {
    throw std::invalid_argument("CST Operator: Not a valid adjustments JSON");
  }
  _src_space = inner["src"].get<std::string>();
  _dst_space = inner["dst"].get<std::string>();
}
};  // namespace puerhlab