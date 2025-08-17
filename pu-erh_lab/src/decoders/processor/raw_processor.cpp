#include "decoders/processor/raw_processor.hpp"

#include <easy/profiler.h>
#include <libraw/libraw.h>  // Add this header for libraw_rawdata_t
#include <libraw/libraw_const.h>
#include <opencv2/core/hal/interface.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <opencv2/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/opencv.hpp>
#include <stdexcept>

#include "decoders/processor/cuda_operators.hpp"
#include "decoders/processor/highlight_reconstruct.hpp"
#include "image/image_buffer.hpp"

namespace puerhlab {
OpenCVRawProcessor::OpenCVRawProcessor(const RawParams& params, const libraw_rawdata_t& rawdata,
                                       LibRaw& raw_processor)
    : _params(params), _raw_data(rawdata), _raw_processor(raw_processor) {}

std::vector<cv::Mat> OpenCVRawProcessor::ExtractBayerPlanes(const cv::Mat& bayer_image) {
  CV_Assert(bayer_image.type() == CV_16UC1);
  CV_Assert(bayer_image.rows % 2 == 0 && bayer_image.cols % 2 == 0);

  int                  height       = bayer_image.rows;
  int                  width        = bayer_image.cols;
  int                  plane_height = height / 2;
  int                  plane_width  = width / 2;

  std::vector<cv::Mat> planes(4);
  for (int i = 0; i < 4; ++i) {
    planes[i] = cv::Mat::zeros(plane_height, plane_width, CV_32F);
  }

  // 使用OpenCV的并行循环
  cv::parallel_for_(
      cv::Range(0, plane_height),
      [&](const cv::Range& range) {
        for (int py = range.start; py < range.end; ++py) {
          for (int px = 0; px < plane_width; ++px) {
            int      src_y              = py * 2;
            int      src_x              = px * 2;

            // 获取2x2 Bayer块的值
            uint16_t r_val              = bayer_image.at<uint16_t>(src_y, src_x);          // R
            uint16_t g1_val             = bayer_image.at<uint16_t>(src_y, src_x + 1);      // G1
            uint16_t g2_val             = bayer_image.at<uint16_t>(src_y + 1, src_x);      // G2
            uint16_t b_val              = bayer_image.at<uint16_t>(src_y + 1, src_x + 1);  // B

            // 转换并存储
            planes[0].at<float>(py, px) = r_val / 65535.0f;
            planes[1].at<float>(py, px) = g1_val / 65535.0f;
            planes[2].at<float>(py, px) = g2_val / 65535.0f;
            planes[3].at<float>(py, px) = b_val / 65535.0f;
          }
        }
      },
      cv::getNumThreads() * 4);

  return planes;
}

cv::Mat OpenCVRawProcessor::ReconstructBayerImage(const std::vector<cv::Mat>& planes) {
  CV_Assert(planes.size() == 4);

  int          plane_height = planes[0].rows;
  int          plane_width  = planes[0].cols;
  int          height       = plane_height * 2;
  int          width        = plane_width * 2;

  cv::Mat      result       = cv::Mat::zeros(height, width, CV_16UC1);
  uint16_t*    dst_ptr      = result.ptr<uint16_t>();

  const float* r_ptr        = planes[0].ptr<float>();
  const float* g1_ptr       = planes[1].ptr<float>();
  const float* g2_ptr       = planes[2].ptr<float>();
  const float* b_ptr        = planes[3].ptr<float>();

  for (int y = 0; y < height; y += 2) {
    for (int x = 0; x < width; x += 2) {
      int plane_idx                = (y / 2) * plane_width + (x / 2);

      // 从float转换回uint16，注意防止溢出
      dst_ptr[y * width + x]       = cv::saturate_cast<uint16_t>(r_ptr[plane_idx] * 65535.0f);
      dst_ptr[y * width + (x + 1)] = cv::saturate_cast<uint16_t>(g1_ptr[plane_idx] * 65535.0f);
      dst_ptr[(y + 1) * width + x] = cv::saturate_cast<uint16_t>(g2_ptr[plane_idx] * 65535.0f);
      dst_ptr[(y + 1) * width + (x + 1)] = cv::saturate_cast<uint16_t>(b_ptr[plane_idx] * 65535.0f);
    }
  }

  return result;
}

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
        float mask        = (color_idx == 0 || color_idx == 2) ? 1.0f : 0.0f;
        float wb_mul      = (wb[color_idx] / wb[1]) * mask + (1.0f - mask);
        float muled_pixel = pixel * wb_mul;
        muled_pixel       = fmaxf(0.0f, fminf(65535.0f, muled_pixel));
        pixel             = static_cast<uint16_t>(muled_pixel);
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
    cam_rgb = cam_rgb.inv();

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

  std::cout << _raw_processor.COLOR(0, 0) << " " << _raw_processor.COLOR(0, 1) << " "
            << _raw_processor.COLOR(1, 0) << " " << _raw_processor.COLOR(1, 1) << "\n";
  ApplyWhiteBalance();
  ApplyDebayer();
  // ApplyHighlightReconstruction();
  ApplyColorSpaceTransform();

  EASY_BLOCK("Sync to CPU")
  if (_params._cuda) {
    _process_buffer.SyncToCPU();
    _process_buffer.ReleaseGPUData();
  }
  EASY_END_BLOCK;
  return {std::move(_process_buffer)};
}
}  // namespace puerhlab