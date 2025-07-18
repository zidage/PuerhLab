#include "edit/operators/cst/cst_op.hpp"

#include <OpenColorIO/OpenColorIO.h>
#include <OpenColorIO/OpenColorTransforms.h>
#include <OpenColorIO/OpenColorTypes.h>

#include <stdexcept>

#include "image/image_buffer.hpp"
#include "json.hpp"
#include "utils/string/convert.hpp"

namespace puerhlab {
OCIO_ACES_Transform_Op::OCIO_ACES_Transform_Op(const std::string& input, const std::string& output)
    : _input_transform(input), _output_transform(output) {
  config = OCIO::GetCurrentConfig();
}

OCIO_ACES_Transform_Op::OCIO_ACES_Transform_Op(const std::string& input, const std::string& output,
                                               const char* config_path)
    : _input_transform(input), _output_transform(output) {
  config = OCIO::Config::CreateFromFile(config_path);
}

auto OCIO_ACES_Transform_Op::Apply(ImageBuffer& input) -> ImageBuffer {
  auto& img = input.GetCPUData();
  if (!_input_transform.empty()) {
    auto                  idt = config->getProcessor(_input_transform.c_str(), "ACES - ACES2065-1");
    auto                  processor = idt->getDefaultCPUProcessor();

    OCIO::PackedImageDesc desc_idt(img.ptr<float>(0), img.cols, img.rows, 3);

    processor->apply(desc_idt);
  }
  if (!_output_transform.empty() && _output_transform.ends_with("Display")) {
    auto transform = OCIO::DisplayViewTransform::Create();
    transform->setSrc("ACES - ACES2065-1");
    transform->setDisplay(_output_transform.c_str());
    transform->setView("ACES 1.0 - SDR Video");

    auto                  odt       = config->getProcessor(transform);
    auto                  processor = odt->getDefaultCPUProcessor();
    OCIO::PackedImageDesc desc_odt(img.ptr<float>(0), img.cols, img.rows, 3);
    processor->apply(desc_odt);
  } else if (!_output_transform.empty()) {
    auto transform = OCIO::ColorSpaceTransform::Create();
    transform->setSrc("ACES - ACES2065-1");
    transform->setDst(_output_transform.c_str());

    auto                  csc       = config->getProcessor(transform);
    auto                  processor = csc->getDefaultCPUProcessor();
    OCIO::PackedImageDesc desc_odt(img.ptr<float>(0), img.cols, img.rows, 3);
    processor->apply(desc_odt);
  }
  return {std::move(img)};
}

auto OCIO_ACES_Transform_Op::GetParams() const -> nlohmann::json {
  nlohmann::json o;
  nlohmann::json inner;

  inner["src"]    = _input_transform;
  inner["dest"]   = _output_transform;
  o[_script_name] = inner;

  return o;
}

void OCIO_ACES_Transform_Op::SetParams(const nlohmann::json& params) {
  if (!params.contains(_script_name)) {
    throw std::invalid_argument("CST Operator: Not a valid adjustments JSON");
  }
  nlohmann::json inner = params[_script_name].get<nlohmann::json>();
  if (!inner.contains("src") || !inner.contains("dst")) {
    throw std::invalid_argument("CST Operator: Not a valid adjustments JSON");
  }
  _input_transform  = inner["src"].get<std::string>();
  _output_transform = inner["dst"].get<std::string>();
}

};  // namespace puerhlab