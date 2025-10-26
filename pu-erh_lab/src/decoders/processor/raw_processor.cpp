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
OpenCVRawProcessor::OpenCVRawProcessor(const RawParams& params, const libraw_rawdata_t& rawdata,
                                       LibRaw& raw_processor)
    : _params(params), _raw_data(rawdata), _raw_processor(raw_processor) {}

void OpenCVRawProcessor::ApplyWhiteBalance() {
  auto& pre_debayer_buffer = _process_buffer;
#ifdef HAVE_CUDA
  if (_params._cuda) {
    auto& gpu_img = pre_debayer_buffer.GetGPUData();
    CUDA::WhiteBalanceCorrection(gpu_img, _raw_processor);
    return;
  }
#endif
  auto& cpu_img = pre_debayer_buffer.GetCPUData();
  CPU::WhiteBalanceCorrection(cpu_img, _raw_processor);
}

void OpenCVRawProcessor::ApplyDebayer() {
  auto& pre_debayer_buffer = _process_buffer;
#ifdef HAVE_CUDA
  if (_params._cuda) {
    auto& gpu_img = pre_debayer_buffer.GetGPUData();
    CUDA::BayerRGGB2RGB_AHD(gpu_img);
    return;
  }
#endif
  auto& img = pre_debayer_buffer.GetCPUData();
  CPU::BayerRGGB2RGB_AHD(img);
}

void OpenCVRawProcessor::ApplyHighlightReconstruct() {
  if (_params._cuda) {
    throw std::runtime_error("OpenCVRawProcessor: Not implemented");
  } else {
    auto& img = _process_buffer.GetCPUData();
    CPU::HighlightReconstruct(img, _raw_processor);
  }
}

void OpenCVRawProcessor::ApplyColorSpaceTransform() {
  auto& debayer_buffer = _process_buffer;
  auto  color_coeffs   = _raw_data.color.rgb_cam;
#ifdef HAVE_CUDA
  if (_params._cuda) {
    auto& gpu_img = debayer_buffer.GetGPUData();
    CUDA::ApplyColorMatrix(gpu_img, color_coeffs);
    return;
  }
#endif
  auto& img = debayer_buffer.GetCPUData();
  img.convertTo(img, CV_32FC3);
  CPU::ApplyColorMatrix(img, color_coeffs);
}

auto OpenCVRawProcessor::Process() -> ImageBuffer {
  auto    img_unpacked = _raw_data.raw_image;
  auto&   img_sizes    = _raw_data.sizes;

  cv::Mat unpacked_mat{img_sizes.raw_height, img_sizes.raw_width, CV_16UC1, img_unpacked};
  _process_buffer = {std::move(unpacked_mat)};

#ifdef HAVE_CUDA

  if (_params._cuda) {
    _process_buffer.SyncToGPU();
  }
#endif

  // std::cout << _raw_processor.COLOR(0, 0) << " " << _raw_processor.COLOR(0, 1) << " "
  //           << _raw_processor.COLOR(1, 0) << " " << _raw_processor.COLOR(1, 1) << "\n";
  CV_Assert(_raw_processor.COLOR(0, 0) == 0 && _raw_processor.COLOR(0, 1) == 1 &&
            _raw_processor.COLOR(1, 0) == 3 && _raw_processor.COLOR(1, 1) == 2);

  ApplyWhiteBalance();

  if (_params._highlights_reconstruct) ApplyHighlightReconstruct();

  ApplyDebayer();

  ApplyColorSpaceTransform();
#ifdef HAVE_CUDA
  if (_params._cuda) {
    _process_buffer.SyncToCPU();
    _process_buffer.ReleaseGPUData();
  }
#endif
  return {std::move(_process_buffer)};
}
}  // namespace puerhlab