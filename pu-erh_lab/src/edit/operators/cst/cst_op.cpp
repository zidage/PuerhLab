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

auto OCIO_ACES_Transform_Op::Apply(ImageBuffer& input) -> ImageBuffer {
  auto& img = input.GetCPUData();

  if (!_input_transform.empty() && !_output_transform.empty()) {
    auto input_transform = OCIO::ColorSpaceTransform::Create();
    input_transform->setSrc(_input_transform.c_str());
    input_transform->setDst("ACES - ACES2065-1");
    auto                  idt = config->getProcessor(input_transform);
    auto                  cpu = idt->getDefaultCPUProcessor();
    OCIO::PackedImageDesc desc_idt(img.ptr<float>(0), img.cols, img.rows, 3);
    cpu->apply(desc_idt);

    if (_normalize) {
      cv::Mat resized;
      cv::resize(img, resized, cv::Size(512, 512));

      double maximum;
      cv::minMaxLoc(resized, nullptr, &maximum, nullptr, nullptr);

      // maximum *= 0.4;

      img.forEach<cv::Vec3f>([&](cv::Vec3f& pixel, const int*) {
        pixel[0] = pixel[0] / (float)maximum;
        pixel[1] = pixel[1] / (float)maximum;
        pixel[2] = pixel[2] / (float)maximum;
      });
      // cv::resize(img, resized, cv::Size(512, 512));
      // cv::cvtColor(resized, resized, cv::COLOR_RGB2BGR);
      // cv::imshow("Scaled", resized);
      // cv::waitKey(0);
    }

    if (_limit) {
      cv::Mat resized;
      cv::resize(img, resized, cv::Size(1024, 1024));

      cv::Mat hls, hls_small, brightness, saturation;
      cv::cvtColor(img, hls, cv::COLOR_RGB2HLS);
      cv::cvtColor(resized, hls_small, cv::COLOR_RGB2HLS);
      cv::extractChannel(hls_small, brightness, 1);
      cv::extractChannel(hls_small, saturation, 2);

      std::vector<float> brightness_values;
      std::vector<float> saturation_values;
      brightness_values.reserve(1024 * 1024);
      saturation_values.reserve(1024 * 1024);
      for (int y = 0; y < brightness.rows; ++y) {
        for (int x = 0; x < brightness.cols; ++x) {
          brightness_values.emplace_back(brightness.at<float>(y, x));
          saturation_values.emplace_back(saturation.at<float>(y, x));
        }
      }

      std::sort(brightness_values.begin(), brightness_values.end());
      std::sort(saturation_values.begin(), saturation_values.end());

      float L_soft  = brightness_values[static_cast<int>(brightness_values.size() * 0.7f)];
      float L_hard  = brightness_values[static_cast<int>(brightness_values.size() * 0.8f)];
      float S_soft  = saturation_values[static_cast<int>(saturation_values.size() * 0.7f)];
      float S_hard  = saturation_values[static_cast<int>(saturation_values.size() * 0.8f)];

      auto  sigmoid = [](float x, float steepness = 1.0f) {
        return 1.0f / (1.0f + std::exp(-steepness * x));
      };

      float target_hue =
          280.0f;  // Adjust this: e.g., 330.0f for warmer pink, 350.0f for cooler/magenta
      float hue_width = 80.0f;  // Adjust this: Smaller (e.g., 20.0f) for very precise targeting;
                                // larger (50.0f) if pink varies
      float hue_steepness = 0.01f;  // Adjust: Smaller for broader influence (gentle falloff);
                                    // larger (0.2f) for sharper drop

      hls.forEach<cv::Vec3f>([&](cv::Vec3f& pixel, const int*) {
        float L = pixel[1];
        float S = pixel[2];
        float H = pixel[0];
        float l_normalized =
            (L - L_soft) / (L_hard - L_soft + 1e-6f) * 2.0f - 1.0f;  // Center around 0 for sigmoid
        float l_factor = sigmoid(
            l_normalized * 5.0f);  // Steepness=5 for moderate curve; adjust higher for sharper

        float s_normalized = (S - S_soft) / (S_hard - S_soft + 1e-6f) * 2.0f - 1.0f;
        float s_factor     = sigmoid(s_normalized * 5.0f);

        float d            = std::min(std::abs(H - target_hue), 360.0f - std::abs(H - target_hue));
        float h_normalized = (hue_width - d) / hue_width;  // Apply sigmoid here too
        float h_weight     = sigmoid(
            h_normalized * (1.0f / hue_steepness));  // Lower steepness for broader hue influence

        float exposure_strength = l_factor * h_weight;
        if (exposure_strength > 0.0f) {
          float blend_factor = 1.0f - std::exp(-exposure_strength * 2.0f);
          S                  = S * (1.0f - blend_factor);
          // L                  = L + (1.0f - L) * blend_factor * 0.5f;
          // pixel[1]           = L;
          pixel[2]           = S;
        }
      });
      cv::cvtColor(hls, img, cv::COLOR_HLS2RGB);
    }

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