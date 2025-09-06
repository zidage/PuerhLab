#include "decoders/processor/raw_processor.hpp"

#include <easy/profiler.h>
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
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <stdexcept>

#include "decoders/processor/operators/cpu/color_space_conv.hpp"
#include "decoders/processor/operators/cpu/debayer_ahd.hpp"
#include "decoders/processor/operators/cpu/debayer_amaze.hpp"
#include "decoders/processor/operators/cpu/highlight_reconstruct.hpp"
#include "decoders/processor/operators/cpu/white_balance.hpp"
#include "decoders/processor/operators/gpu/cuda_color_space_conv.hpp"
#include "decoders/processor/operators/gpu/cuda_debayer_ahd.hpp"
#include "decoders/processor/operators/gpu/cuda_white_balance.hpp"
#include "image/image_buffer.hpp"

namespace puerhlab {
OpenCVRawProcessor::OpenCVRawProcessor(const RawParams& params, const libraw_rawdata_t& rawdata,
                                       LibRaw& raw_processor)
    : _params(params), _raw_data(rawdata), _raw_processor(raw_processor) {}

void OpenCVRawProcessor::ApplyWhiteBalance() {
  auto& pre_debayer_buffer = _process_buffer;
  if (_params._cuda) {
    auto& gpu_img = pre_debayer_buffer.GetGPUData();

    CUDA::WhiteBalanceCorrection(gpu_img, _raw_processor);

  } else {
    auto& cpu_img = pre_debayer_buffer.GetCPUData();
    // cpu_img -= black;
    //
    CPU::WhiteBalanceCorrection(cpu_img, _raw_processor);
    // White level
    // cpu_img = cpu_img * (65535.0f / maximum);
  }
}

void OpenCVRawProcessor::ApplyDebayer() {
  auto& pre_debayer_buffer = _process_buffer;
  if (_params._cuda) {
    auto& gpu_img = pre_debayer_buffer.GetGPUData();
    CUDA::BayerRGGB2RGB_AHD(gpu_img);
    // Now gpu_img is CV_32FC3
    // cv::cuda::cvtColor(gpu_img, gpu_img, cv::COLOR_BayerBG2RGB);
  } else {
    auto& img = pre_debayer_buffer.GetCPUData();

    // auto maximum = _raw_data.color.linear_max;
    CPU::BayerRGGB2RGB_AMaZe(img);
    // cv::cvtColor(img, img, cv::COLOR_BayerBG2RGB);
    // img.convertTo(img, CV_32FC1, 1.0f / 65535.0f);
  }
}

void OpenCVRawProcessor::ApplyHighlightReconstruct() {
  if (_params._cuda) {
    throw std::runtime_error("OpenCVRawProcessor: Not implemented");
  } else {
    auto& img = _process_buffer.GetCPUData();
    img.convertTo(img, CV_32FC3);
    CPU::HighlightReconstruct(img, _raw_processor);
    // img.convertTo(img, CV_32FC1);
  }
}

void OpenCVRawProcessor::ApplyColorSpaceTransform() {
  auto& debayer_buffer = _process_buffer;
  auto  color_coeffs   = _raw_data.color.rgb_cam;
  if (_params._cuda) {
    auto& gpu_img = debayer_buffer.GetGPUData();

    CUDA::ApplyColorMatrix(gpu_img, color_coeffs);
  } else {
    auto& img = debayer_buffer.GetCPUData();
    img.convertTo(img, CV_32FC3);

    CPU::ApplyColorMatrix(img, color_coeffs);
  }
}

auto OpenCVRawProcessor::Process() -> ImageBuffer {
  auto  img_unpacked = _raw_data.raw_image;
  auto& img_sizes    = _raw_data.sizes;

  EASY_BLOCK("Unpacked Mat");
  cv::Mat unpacked_mat{img_sizes.raw_height, img_sizes.raw_width, CV_16UC1, img_unpacked};
  _process_buffer = {std::move(unpacked_mat)};
  EASY_END_BLOCK;

  EASY_BLOCK("Sync to GPU")
  if (_params._cuda) {
    _process_buffer.SyncToGPU();
  }
  EASY_END_BLOCK;

  // std::cout << _raw_processor.COLOR(0, 0) << " " << _raw_processor.COLOR(0, 1) << " "
  //           << _raw_processor.COLOR(1, 0) << " " << _raw_processor.COLOR(1, 1) << "\n";
  CV_Assert(_raw_processor.COLOR(0, 0) == 0 && _raw_processor.COLOR(0, 1) == 1 &&
            _raw_processor.COLOR(1, 0) == 3 && _raw_processor.COLOR(1, 1) == 2);
  EASY_BLOCK("White Balance Correction");
  ApplyWhiteBalance();
  EASY_END_BLOCK;
  EASY_BLOCK("AHD Debayer");
  ApplyDebayer();
  EASY_END_BLOCK;
  if (_params._highlights_reconstruct) ApplyHighlightReconstruct();
  EASY_BLOCK("CST");
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