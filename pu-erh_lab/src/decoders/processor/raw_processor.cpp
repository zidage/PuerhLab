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

// TODO: Migrate to static pipeline architecture

#include "decoders/processor/raw_processor.hpp"

#include <libraw/libraw.h>  // Add this header for libraw_rawdata_t
#include <libraw/libraw_const.h>
#include <opencv2/core/hal/interface.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <opencv2/core.hpp>
#include <opencv2/core/base.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/utility.hpp>

#include "decoders/processor/operators/cpu/debayer_rcd.hpp"
#include "decoders/processor/operators/cpu/raw_proc_utils.hpp"
#include "type/type.hpp"

#ifdef HAVE_CUDA
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#endif

#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <stdexcept>

#include "decoders/processor/operators/cpu/color_space_conv.hpp"
#include "decoders/processor/operators/cpu/debayer_ahd.hpp"
#include "decoders/processor/operators/cpu/debayer_amaze.hpp"
#include "decoders/processor/operators/cpu/highlight_reconstruct.hpp"
#include "decoders/processor/operators/cpu/white_balance.hpp"

#ifdef HAVE_CUDA
#include "decoders/processor/operators/gpu/cuda_color_space_conv.hpp"
#include "decoders/processor/operators/gpu/cuda_debayer_ahd.hpp"
#include "decoders/processor/operators/gpu/cuda_white_balance.hpp"
#endif
#include "image/image_buffer.hpp"

namespace puerhlab {
namespace {
template <typename T>
auto DownsampleBayerRGGB2xTyped(const cv::Mat& src) -> cv::Mat {
  const int out_rows = src.rows / 2;
  const int out_cols = src.cols / 2;
  cv::Mat   dst(out_rows, out_cols, src.type());

  for (int y = 0; y < out_rows; ++y) {
    const T* row0 = src.ptr<T>(2 * y);
    const T* row1 = src.ptr<T>(2 * y + 1);
    T*       drow = dst.ptr<T>(y);
    for (int x = 0; x < out_cols; ++x) {
      const int sx = 2 * x;
      if ((y & 1) == 0) {
        // R G
        // G B
        drow[x] = (x & 1) == 0 ? row0[sx] : row0[sx + 1];
      } else {
        drow[x] = (x & 1) == 0 ? row1[sx] : row1[sx + 1];
      }
    }
  }

  return dst;
}

auto DownsampleBayerRGGB2x(const cv::Mat& src) -> cv::Mat {
  switch (src.type()) {
    case CV_32FC1:
      return DownsampleBayerRGGB2xTyped<float>(src);
    case CV_16UC1:
      return DownsampleBayerRGGB2xTyped<uint16_t>(src);
    default:
      throw std::runtime_error("RawProcessor: Unsupported Bayer type for downsample");
  }
}
}  // namespace

RawProcessor::RawProcessor(const RawParams& params, const libraw_rawdata_t& rawdata,
                           LibRaw& raw_processor)
    : params_(params), raw_data_(rawdata), raw_processor_(raw_processor) {}

void RawProcessor::SetDecodeRes() {
  // Adjust internal parameters based on decode resolution
  auto& cpu_data = process_buffer_.GetCPUData();
  switch (params_.decode_res_) {
    case DecodeRes::FULL:
      // No changes needed for full resolution
      break;
    case DecodeRes::HALF:
      // Downscale by factor of 2
      cpu_data = DownsampleBayerRGGB2x(cpu_data);
      break;
    case DecodeRes::QUARTER:
      // Downscale by factor of 4
      cpu_data = DownsampleBayerRGGB2x(DownsampleBayerRGGB2x(cpu_data));
      break;
    default:
      throw std::runtime_error("RawProcessor: Unknown decode resolution");
  }
}

void RawProcessor::ApplyLinearization() {
  auto& pre_debayer_buffer = process_buffer_;
#ifdef HAVE_CUDA
  if (params_.cuda_) {
    auto& gpu_img = pre_debayer_buffer.GetGPUData();
    CUDA::ToLinearRef(gpu_img, raw_processor_);
    return;
  }
#endif
  auto& cpu_img = pre_debayer_buffer.GetCPUData();
  CPU::ToLinearRef(cpu_img, raw_processor_);
}

void RawProcessor::ApplyDebayer() {
  auto& pre_debayer_buffer = process_buffer_;
#ifdef HAVE_CUDA
  if (params_.cuda_) {
    auto& gpu_img = pre_debayer_buffer.GetGPUData();
    CUDA::BayerRGGB2RGB_AHD(gpu_img);

    return;
  }
#endif
  auto& img = pre_debayer_buffer.GetCPUData();
  CPU::BayerRGGB2RGB_RCD(img);
  // Crop to valid area
  // cv::Rect crop_rect(_raw_data.sizes.raw_inset_crops[0].cleft,
  // _raw_data.sizes.raw_inset_crops[0].ctop,
  //                    _raw_data.sizes.raw_inset_crops[0].cwidth,
  //                    _raw_data.sizes.raw_inset_crops[0].cheight);
  if (params_.decode_res_ == DecodeRes::FULL) {
    cv::Rect crop_rect(raw_data_.sizes.left_margin, raw_data_.sizes.top_margin,
                       raw_data_.sizes.width, raw_data_.sizes.height);
    img = img(crop_rect);
  }
}

void RawProcessor::ApplyHighlightReconstruct() {
  if (params_.cuda_) {
    throw std::runtime_error("RawProcessor: Not implemented");
  } else {
    auto& img = process_buffer_.GetCPUData();
    if (!params_.highlights_reconstruct_) {
      // clamp to [0, 1]
      cv::threshold(img, img, 1.0f, 1.0f, cv::THRESH_TRUNC);
      cv::threshold(img, img, 0.0f, 0.0f, cv::THRESH_TOZERO);
      return;
    }
    CPU::HighlightReconstruct(img, raw_processor_);
  }
}

void RawProcessor::ConvertToWorkingSpace() {
  auto& debayer_buffer = process_buffer_;
  auto  color_coeffs   = raw_data_.color.rgb_cam;
#ifdef HAVE_CUDA
  if (params_.cuda_) {
    auto& gpu_img = debayer_buffer.GetGPUData();
    CUDA::ApplyColorMatrix(gpu_img, color_coeffs);
    return;
  }
#endif
  auto& img = debayer_buffer.GetCPUData();
  img.convertTo(img, CV_32FC3);
  auto pre_mul   = raw_data_.color.pre_mul;
  auto cam_mul   = raw_data_.color.cam_mul;
  auto wb_coeffs = raw_data_.color.WB_Coeffs;  // EXIF Lightsource Values
  auto cam_xyz   = raw_data_.color.cam_xyz;
  auto rgb_cam   = raw_data_.color.rgb_cam;
  if (!params_.use_camera_wb_) {
    // User specified white balance temperature
    auto user_temp_indices = CPU::GetWBIndicesForTemp(static_cast<float>(params_.user_wb_));
    CPU::ApplyColorMatrix(img, color_coeffs, pre_mul, cam_mul, wb_coeffs, user_temp_indices,
                          params_.user_wb_, cam_xyz);
    return;
  }
  CPU::ApplyColorMatrix(img, color_coeffs, pre_mul, cam_mul, cam_xyz);
}

auto RawProcessor::Process() -> ImageBuffer {
  auto    img_unpacked = raw_data_.raw_image;
  auto&   img_sizes    = raw_data_.sizes;

  cv::Mat unpacked_mat{img_sizes.raw_height, img_sizes.raw_width, CV_16UC1, img_unpacked};
  process_buffer_ = {std::move(unpacked_mat)};

#ifdef HAVE_CUDA

  if (params_.cuda_) {
    process_buffer_.SyncToGPU();
  }
#endif

  // std::cout << _raw_processor.COLOR(0, 0) << " " << _raw_processor.COLOR(0, 1) << " "
  //           << _raw_processor.COLOR(1, 0) << " " << _raw_processor.COLOR(1, 1) << "\n";
  CV_Assert(raw_processor_.COLOR(0, 0) == 0 && raw_processor_.COLOR(0, 1) == 1 &&
            raw_processor_.COLOR(1, 0) == 3 && raw_processor_.COLOR(1, 1) == 2);
  process_buffer_.GetCPUData().convertTo(process_buffer_.GetCPUData(), CV_32FC1, 1.0f / 65535.0f);

  SetDecodeRes();

  using clock = std::chrono::high_resolution_clock;
  auto start  = clock::now();
  ApplyLinearization();
  auto                                      linear_end      = clock::now();
  std::chrono::duration<double, std::milli> linear_duration = linear_end - start;
  std::cout << "Linearization took " << linear_duration.count() << " ms\n";

  ApplyHighlightReconstruct();
  auto                                      hl_end      = clock::now();
  std::chrono::duration<double, std::milli> hl_duration = hl_end - linear_end;
  std::cout << "Highlight Reconstruction took " << hl_duration.count() << " ms\n";

  ApplyDebayer();
  auto                                      debayer_end      = clock::now();
  std::chrono::duration<double, std::milli> debayer_duration = debayer_end - hl_end;
  std::cout << "Debayering took " << debayer_duration.count() << " ms\n";

  // Temporary fix for orientation
  switch (raw_data_.sizes.flip) {
    case 3:
      // 180 degree
      cv::rotate(process_buffer_.GetCPUData(), process_buffer_.GetCPUData(), cv::ROTATE_180);
      break;
    case 5:
      // Rotate 90 CCW
      cv::rotate(process_buffer_.GetCPUData(), process_buffer_.GetCPUData(),
                 cv::ROTATE_90_COUNTERCLOCKWISE);
      break;
    case 6:
      // Rotate 90 CW
      cv::rotate(process_buffer_.GetCPUData(), process_buffer_.GetCPUData(),
                 cv::ROTATE_90_CLOCKWISE);
      break;
    default:
      // Do nothing
      break;
  }

  ConvertToWorkingSpace();

  cv::Mat final_img = cv::Mat();
  final_img.create(process_buffer_.GetCPUData().rows, process_buffer_.GetCPUData().cols, CV_32FC4);
  cv::cvtColor(process_buffer_.GetCPUData(), final_img, cv::COLOR_RGB2RGBA);
  process_buffer_                                        = {std::move(final_img)};
  auto                                      cst_end      = clock::now();
  std::chrono::duration<double, std::milli> cst_duration = cst_end - debayer_end;
  std::cout << "Color Space Transformation took " << cst_duration.count() << " ms\n";
  std::cout << "Total processing took "
            << (linear_duration + hl_duration + debayer_duration + cst_duration).count() << " ms\n";
#ifdef HAVE_CUDA
  if (params_.cuda_) {
    process_buffer_.SyncToCPU();
    process_buffer_.ReleaseGPUData();
  }
#endif
  return {std::move(process_buffer_)};
}
}  // namespace puerhlab