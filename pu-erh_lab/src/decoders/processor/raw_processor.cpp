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

#include "decoders/processor/cpu_operators.hpp"
#include "decoders/processor/cuda_operators.hpp"
#include "decoders/processor/highlight_reconstruct.hpp"
#include "image/image_buffer.hpp"

namespace puerhlab {
OpenCVRawProcessor::OpenCVRawProcessor(const RawParams& params, const libraw_rawdata_t& rawdata,
                                       LibRaw& raw_processor)
    : _params(params), _raw_data(rawdata), _raw_processor(raw_processor) {}

void OpenCVRawProcessor::ApplyWhiteBalance() {
  const float*         wb                 = CPU::GetWBCoeff(_raw_data);
  auto&                pre_debayer_buffer = _process_buffer;

  // Black level calculation
  // From
  // https://stackoverflow.com/questions/69526257/reading-raw-image-with-libraw-and-converting-to-dng-with-libtiff
  std::array<float, 4> black_level        = CPU::CalculateBlackLevel(_raw_data);

  if (_params._cuda) {
    auto& gpu_img = pre_debayer_buffer.GetGPUData();

    auto  maximum = static_cast<uint16_t>(_raw_data.color.maximum);
    CUDA::WhiteBalanceCorrection(gpu_img, black_level, wb, maximum, true, 0);

  } else {
    auto& cpu_img = pre_debayer_buffer.GetCPUData();
    // cpu_img -= black;
    //
    CPU::WhiteBalanceCorrectionAndHighlightRestore(cpu_img, _raw_processor, black_level, wb);
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
    CPU::BayerRGGB2RGB_AHD(img, true, (_raw_data.color.maximum - _raw_data.color.black) / 65535.0f);
    // cv::cvtColor(img, img, cv::COLOR_BayerBG2RGB);
    // img.convertTo(img, CV_32FC1, 1.0f / 65535.0f);
  }
}

void OpenCVRawProcessor::ApplyColorSpaceTransform() {
  auto& debayer_buffer = _process_buffer;
  auto  color_coeffs   = _raw_data.color.rgb_cam;
  if (_params._cuda) {
    cv::Matx33f      cam_rgb({color_coeffs[0][0], color_coeffs[0][1], color_coeffs[0][2],
                              color_coeffs[1][0], color_coeffs[1][1], color_coeffs[1][2],
                              color_coeffs[2][0], color_coeffs[2][1], color_coeffs[2][2]});
    // cam_rgb = cam_rgb.inv();
    // debayer_buffer.SyncToCPU();
    auto&            gpu_img = debayer_buffer.GetGPUData();
    // gpu_img.convertTo(gpu_img, CV_32FC3);

    cv::cuda::Stream stream;
    CUDA::ApplyColorMatrix(gpu_img, gpu_img, cv::Mat(cam_rgb), stream);
    stream.waitForCompletion();
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
  // ApplyHighlightReconstruction();
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