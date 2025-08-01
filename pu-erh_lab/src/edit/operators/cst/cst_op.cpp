#include "edit/operators/cst/cst_op.hpp"

#include <OpenColorIO/OpenColorIO.h>
#include <OpenColorIO/OpenColorTransforms.h>
#include <OpenColorIO/OpenColorTypes.h>

#include <opencv2/core/matx.hpp>
#include <stdexcept>
#include <string>

#include "edit/operators/operator_factory.hpp"
#include "image/image_buffer.hpp"
#include "json.hpp"
#include "utils/string/convert.hpp"

namespace puerhlab {
OCIO_ACES_Transform_Op_Register::OCIO_ACES_Transform_Op_Register() {
  OperatorFactory::Instance().Register(OperatorType::CST, [](const nlohmann::json& params) {
    return std::make_shared<OCIO_ACES_Transform_Op>(params);
  });
}

static OCIO_ACES_Transform_Op_Register cst_reg;

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

auto OCIO_ACES_Transform_Op::Apply(ImageBuffer& input) -> ImageBuffer {
  auto& img = input.GetCPUData();
  if (!_input_transform.empty()) {
    auto transform = OCIO::LookTransform::Create();
    transform->setLooks("ACES 1.3 Reference Gamut Compression");
    transform->setSrc(_input_transform.c_str());
    transform->setDst("ACES - ACES2065-1");
    auto                  idt = config->getProcessor(transform);
    auto                  cpu = idt->getDefaultCPUProcessor();

    OCIO::PackedImageDesc desc_idt(img.ptr<float>(0), img.cols, img.rows, 3);

    cv::parallel_for_(cv::Range(0, img.rows), [&](const cv::Range& range) {
      for (int y = range.start; y < range.end; ++y) {
        cv::Vec3f* row = img.ptr<cv::Vec3f>(y);
        for (int x = 0; x < img.cols; ++x) {
          cpu->applyRGB(&row[x][0]);
        }
      }
    });
  }
  if (!_output_transform.empty() && _output_transform.ends_with("Display")) {
    auto transform = OCIO::DisplayViewTransform::Create();
    transform->setSrc("ACES - ACES2065-1");
    transform->setDisplay(_output_transform.c_str());
    transform->setView("ACES 2.0 - SDR 100 nits (Rec.709)");

    auto odt = config->getProcessor(transform);
    auto cpu = odt->getDefaultCPUProcessor();

    cv::parallel_for_(cv::Range(0, img.rows), [&](const cv::Range& range) {
      for (int y = range.start; y < range.end; ++y) {
        cv::Vec3f* row = img.ptr<cv::Vec3f>(y);
        for (int x = 0; x < img.cols; ++x) {
          cpu->applyRGB(&row[x][0]);
        }
      }
    });
  } else if (!_output_transform.empty()) {
    auto transform = OCIO::LookTransform::Create();
    transform->setLooks("ACES 1.3 Reference Gamut Compression");
    transform->setSrc("ACES - ACES2065-1");
    transform->setDst(_output_transform.c_str());
    transform->setDirection(OCIO::TransformDirection::TRANSFORM_DIR_FORWARD);

    auto csc = config->getProcessor(transform);
    auto cpu = csc->getDefaultCPUProcessor();
    cv::parallel_for_(cv::Range(0, img.rows), [&](const cv::Range& range) {
      for (int y = range.start; y < range.end; ++y) {
        cv::Vec3f* row = img.ptr<cv::Vec3f>(y);
        for (int x = 0; x < img.cols; ++x) {
          cpu->applyRGB(&row[x][0]);
        }
      }
    });
  }
  return {std::move(img)};
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

  inner["src"]  = _input_transform;
  inner["dest"] = _output_transform;

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
  if (inner.contains("lmt")) {
    _lmt_path = std::filesystem::path(inner["lmt"].get<std::u8string>());
  }
}

};  // namespace puerhlab