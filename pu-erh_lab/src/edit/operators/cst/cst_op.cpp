#include "edit/operators/cst/cst_op.hpp"

#include <OpenColorIO/OpenColorIO.h>
#include <OpenColorIO/OpenColorTransforms.h>
#include <OpenColorIO/OpenColorTypes.h>

#include <opencv2/core.hpp>
#include <opencv2/core/matx.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <stdexcept>
#include <string>
#include <vector>

#include "edit/operators/operator_factory.hpp"
#include "edit/operators/utils/functions.hpp"
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

OCIO_ACES_Transform_Op::OCIO_ACES_Transform_Op(std::filesystem::path& lmt_path)
    : _input_transform("ACES - ACEScct"), _output_transform("ACES - ACEScct"), _lmt_path(lmt_path) {
  config = OCIO::GetCurrentConfig();
}

OCIO_ACES_Transform_Op::OCIO_ACES_Transform_Op(const nlohmann::json& params) {
  config = OCIO::GetCurrentConfig();
  SetParams(params);
}

void OCIO_ACES_Transform_Op::Apply(std::shared_ptr<ImageBuffer> input) {
  auto& img = input->GetCPUData();

  if (!_input_transform.empty() && !_output_transform.empty()) {
    auto input_transform = OCIO::ColorSpaceTransform::Create();
    input_transform->setSrc(_input_transform.c_str());
    input_transform->setDst("ACES - ACES2065-1");
    auto                  idt = config->getProcessor(input_transform);
    auto                  cpu = idt->getDefaultCPUProcessor();
    OCIO::PackedImageDesc desc_idt(img.ptr<float>(0), img.cols, img.rows, 3);
    cpu->apply(desc_idt);

    auto output_transform = OCIO::LookTransform::Create();
    output_transform->setLooks("ACES 1.3 Reference Gamut Compression");
    output_transform->setSrc("ACES - ACES2065-1");
    output_transform->setDst(_output_transform.c_str());
    auto                  odt     = config->getProcessor(output_transform);
    auto                  odt_cpu = odt->getDefaultCPUProcessor();
    OCIO::PackedImageDesc desc_odt(img.ptr<float>(0), img.cols, img.rows, 3);
    odt_cpu->apply(desc_odt);
  } else if (!_input_transform.empty() && _output_transform.empty()) {
    auto transform = OCIO::ColorSpaceTransform::Create();
    // transform->setLooks("ACES 1.3 Reference Gamut Compression");
    transform->setSrc(_input_transform.c_str());
    transform->setDst("ACES - ACES2065-1");
    auto                  idt = config->getProcessor(transform);
    auto                  cpu = idt->getDefaultCPUProcessor();

    OCIO::PackedImageDesc desc(img.ptr<float>(0), img.cols, img.rows, 3);

    cpu->apply(desc);

  } else if (_input_transform.empty() && !_output_transform.empty() &&
             _output_transform.ends_with("Display")) {
    auto transform = OCIO::DisplayViewTransform::Create();
    transform->setSrc("ACES - ACES2065-1");
    transform->setDisplay(_output_transform.c_str());
    transform->setView("ACES 2.0 - SDR 100 nits (Rec.709)");

    auto                  odt = config->getProcessor(transform);
    auto                  cpu = odt->getDefaultCPUProcessor();

    OCIO::PackedImageDesc desc(img.ptr<float>(0), img.cols, img.rows, 3);

    cpu->apply(desc);
  } else if (_input_transform.empty() && !_output_transform.empty()) {
    auto transform = OCIO::LookTransform::Create();
    transform->setLooks("ACES 1.3 Reference Gamut Compression");
    transform->setSrc("ACES - ACES2065-1");
    transform->setDst(_output_transform.c_str());
    transform->setDirection(OCIO::TransformDirection::TRANSFORM_DIR_FORWARD);

    auto                  csc = config->getProcessor(transform);
    auto                  cpu = csc->getDefaultCPUProcessor();
    OCIO::PackedImageDesc desc(img.ptr<float>(0), img.cols, img.rows, 3);

    cpu->apply(desc);
  }
}

auto OCIO_ACES_Transform_Op::ToKernel() const -> Kernel {
  if (!_input_transform.empty() && !_output_transform.empty()) {
    auto input_transform = OCIO::ColorSpaceTransform::Create();
    input_transform->setSrc(_input_transform.c_str());
    input_transform->setDst("ACES - ACES2065-1");
    auto idt              = config->getProcessor(input_transform);
    auto cpu              = idt->getDefaultCPUProcessor();

    auto output_transform = OCIO::LookTransform::Create();
    output_transform->setLooks("ACES 1.3 Reference Gamut Compression");
    output_transform->setSrc("ACES - ACES2065-1");
    output_transform->setDst(_output_transform.c_str());
    auto odt     = config->getProcessor(output_transform);
    auto odt_cpu = odt->getDefaultCPUProcessor();

    return Kernel{._type = Kernel::Type::Point,
                  ._func = PointKernelFunc([cpu, odt_cpu](const Pixel& in) -> Pixel {
                    Pixel rgb = in;
                    cpu->applyRGB(&rgb.r);
                    odt_cpu->applyRGB(&rgb.r);
                    return rgb;
                  })};

  } else if (!_input_transform.empty() && _output_transform.empty()) {
    auto transform = OCIO::ColorSpaceTransform::Create();
    // transform->setLooks("ACES 1.3 Reference Gamut Compression");
    transform->setSrc(_input_transform.c_str());
    transform->setDst("ACES - ACES2065-1");
    auto idt = config->getProcessor(transform);
    auto cpu = idt->getDefaultCPUProcessor();

    return Kernel{._type = Kernel::Type::Point,
                  ._func = PointKernelFunc([cpu](const Pixel& in) -> Pixel {
                    Pixel rgb = in;
                    cpu->applyRGB(&rgb.r);
                    return rgb;
                  })};
  } else if (_input_transform.empty() && !_output_transform.empty() &&
             _output_transform.ends_with("Display")) {
    auto transform = OCIO::DisplayViewTransform::Create();
    transform->setSrc("ACES - ACES2065-1");
    transform->setDisplay(_output_transform.c_str());
    transform->setView("ACES 2.0 - SDR 100 nits (Rec.709)");

    auto odt = config->getProcessor(transform);
    auto cpu = odt->getDefaultCPUProcessor();

    return Kernel{._type = Kernel::Type::Point,
                  ._func = PointKernelFunc([cpu](const Pixel& in) -> Pixel {
                    Pixel rgb = in;
                    cpu->applyRGB(&rgb.r);
                    return rgb;
                  })};
  } else if (_input_transform.empty() && !_output_transform.empty()) {
    auto transform = OCIO::LookTransform::Create();
    transform->setLooks("ACES 1.3 Reference Gamut Compression");
    transform->setSrc("ACES - ACES2065-1");
    transform->setDst(_output_transform.c_str());
    transform->setDirection(OCIO::TransformDirection::TRANSFORM_DIR_FORWARD);

    auto csc = config->getProcessor(transform);
    auto cpu = csc->getDefaultCPUProcessor();
    return Kernel{._type = Kernel::Type::Point,
                  ._func = PointKernelFunc([cpu](const Pixel& in) -> Pixel {
                    Pixel rgb = in;
                    cpu->applyRGB(&rgb.r);
                    return rgb;
                  })};
  }
  throw std::runtime_error("OCIO_ACES_Transform_Op: No valid transform assigned to the operator");
}

auto OCIO_ACES_Transform_Op::ApplyLMT(ImageBuffer& input) -> ImageBuffer {
  if (!_lmt_path.has_value()) {
    throw std::runtime_error("OCIO_ACES_Transform_Op: No valid LMT look assigned to the operator");
  }
  auto& img           = input.GetCPUData();

  auto  lmt_transform = OCIO::FileTransform::Create();
  auto  path_str      = _lmt_path->wstring();
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

auto OCIO_ACES_Transform_Op::GetParams() const -> nlohmann::json {
  nlohmann::json o;
  nlohmann::json inner;

  inner["src"]       = _input_transform;
  inner["dest"]      = _output_transform;
  inner["limit"]     = _limit;
  inner["normalize"] = _normalize;

  if (_lmt_path.has_value()) {
    inner["lmt"] = _lmt_path->u8string();
  }
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
  if (inner.contains("limit")) {
    _limit = inner["limit"].get<bool>();
  }

  if (inner.contains("normalize")) {
    _normalize = inner["normalize"].get<bool>();
  }
}
};  // namespace puerhlab