#include "decoders/processor/raw_processor.hpp"

#include <easy/profiler.h>
#include <libraw/libraw.h>  // Add this header for libraw_rawdata_t
#include <libraw/libraw_const.h>
#include <opencv2/core/hal/interface.h>

#include <algorithm>
#include <cstdint>
#include <opencv2/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/opencv.hpp>
#include <stdexcept>

#include "decoders/processor/cuda_operators.hpp"
#include "image/image_buffer.hpp"

namespace puerhlab {
OpenCVRawProcessor::OpenCVRawProcessor(const RawParams& params, const libraw_rawdata_t& rawdata,
                                       LibRaw& raw_processor)
    : _params(params), _raw_data(rawdata), _raw_processor(raw_processor) {}

void OpenCVRawProcessor::ApplyWhiteBalance() {
  const float* wb;
  const int*   user_wb_coeff;
  if (_params._use_camera_wb) {
    wb = _raw_data.color.cam_mul;
  } else {
    user_wb_coeff = _raw_data.color.WB_Coeffs[21];
    throw std::runtime_error("Raw Processor: Not Implemented");
  }

  auto                 maximum            = static_cast<uint16_t>(_raw_data.color.maximum);

  auto&                pre_debayer_buffer = _process_buffer;

  // Black level calculation
  // From
  // https://stackoverflow.com/questions/69526257/reading-raw-image-with-libraw-and-converting-to-dng-with-libtiff
  const auto           base_black_level   = static_cast<float>(_raw_data.color.black);
  std::array<float, 4> black_level        = {
      base_black_level + static_cast<float>(_raw_data.color.cblack[0]),
      base_black_level + static_cast<float>(_raw_data.color.cblack[1]),
      base_black_level + static_cast<float>(_raw_data.color.cblack[2]),
      base_black_level + static_cast<float>(_raw_data.color.cblack[3])};

  if (_raw_data.color.cblack[4] == 2 && _raw_data.color.cblack[5] == 2) {
    for (unsigned int x = 0; x < _raw_data.color.cblack[4]; ++x) {
      for (unsigned int y = 0; y < _raw_data.color.cblack[5]; ++y) {
        const auto index   = y * 2 + x;
        black_level[index] = _raw_data.color.cblack[6 + index];
      }
    }
  }

  if (_params._cuda) {
    auto& gpu_img = pre_debayer_buffer.GetGPUData();

    WhiteBalanceCorrection(gpu_img, black_level, wb, maximum, true, 0);

  } else {
    auto& cpu_img = pre_debayer_buffer.GetCPUData();
    // cpu_img -= black;
    //
    if (_raw_data.color.as_shot_wb_applied != 1) {
      cpu_img.forEach<uint16_t>([&](uint16_t& pixel, const int* pos) {
        int color_idx = _raw_processor.COLOR(pos[0], pos[1]);
        pixel         = static_cast<uint16_t>(
            std::max<int>(0, static_cast<int>(pixel) - black_level[color_idx]));
        float wb_mul = wb[color_idx] / wb[1];
        pixel        = static_cast<uint16_t>(pixel * wb_mul);
      });
    }
    // White level
    cpu_img = cpu_img * (65535.0f / maximum);
  }
}

void OpenCVRawProcessor::ApplyDebayer() {
  auto& pre_debayer_buffer = _process_buffer;
  if (_params._cuda) {
    auto& gpu_img = pre_debayer_buffer.GetGPUData();
    cv::cuda::cvtColor(gpu_img, gpu_img, cv::COLOR_BayerBG2RGB);
  } else {
    auto& img = pre_debayer_buffer.GetCPUData();
    cv::cvtColor(img, img, cv::COLOR_BayerBG2RGB);
  }
}

void OpenCVRawProcessor::ApplyColorSpaceTransform() {
  auto& debayer_buffer = _process_buffer;
  auto  color_coeffs   = _raw_data.color.rgb_cam;
  if (_params._cuda) {
    cv::Matx33f cam_rgb({color_coeffs[0][0], color_coeffs[0][1], color_coeffs[0][2],
                         color_coeffs[1][0], color_coeffs[1][1], color_coeffs[1][2],
                         color_coeffs[2][0], color_coeffs[2][1], color_coeffs[2][2]});
    // cam_rgb = cam_rgb.inv();
    // debayer_buffer.SyncToCPU();
    auto&       gpu_img = debayer_buffer.GetGPUData();
    gpu_img.convertTo(gpu_img, CV_32FC3, 1.0f / 65535.0f);

    cv::cuda::Stream stream;
    ApplyColorMatrix(gpu_img, gpu_img, cv::Mat(cam_rgb), stream);
    stream.waitForCompletion();
  } else {
    auto& img = debayer_buffer.GetCPUData();
    img.convertTo(img, CV_32FC3, 1.0f / 65535.0f);

    cv::Matx33f cam_rgb({color_coeffs[0][0], color_coeffs[0][1], color_coeffs[0][2],
                         color_coeffs[1][0], color_coeffs[1][1], color_coeffs[1][2],
                         color_coeffs[2][0], color_coeffs[2][1], color_coeffs[2][2]});
    // cam_rgb         = cam_rgb.inv();

    cv::transform(img, img, cv::Mat(cam_rgb));
  }
}

auto OpenCVRawProcessor::Process() -> ImageBuffer {
  auto    img_unpacked = _raw_data.raw_image;
  auto&   img_sizes    = _raw_data.sizes;

  cv::Mat unpacked_mat{img_sizes.raw_height, img_sizes.raw_width, CV_16UC1, img_unpacked};
  _process_buffer = {std::move(unpacked_mat)};

  EASY_BLOCK("Sync to GPU")
  if (_params._cuda) {
    _process_buffer.SyncToGPU();
  }
  EASY_END_BLOCK;

  EASY_BLOCK("White Balance")
  ApplyWhiteBalance();
  EASY_END_BLOCK;
  EASY_BLOCK("Debayer")
  ApplyDebayer();
  EASY_END_BLOCK;
  EASY_BLOCK("CST")
  ApplyColorSpaceTransform();
  EASY_END_BLOCK;

  EASY_BLOCK("Sync to CPU")
  if (_params._cuda) {
    _process_buffer.SyncToCPU();
    _process_buffer.ReleaseGPUData();
  }
  EASY_END_BLOCK;
  return {std::move(_process_buffer)};
}
}  // namespace puerhlab