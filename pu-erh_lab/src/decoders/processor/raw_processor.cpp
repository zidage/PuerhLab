// TODO: License
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
RawProcessor::RawProcessor(const RawParams& params, const libraw_rawdata_t& rawdata,
                           LibRaw& raw_processor)
    : _params(params), _raw_data(rawdata), _raw_processor(raw_processor) {}

void RawProcessor::ApplyLinearization() {
  auto& pre_debayer_buffer = _process_buffer;
#ifdef HAVE_CUDA
  if (_params._cuda) {
    auto& gpu_img = pre_debayer_buffer.GetGPUData();
    CUDA::ToLinearRef(gpu_img, _raw_processor);
    return;
  }
#endif
  auto& cpu_img = pre_debayer_buffer.GetCPUData();
  CPU::ToLinearRef(cpu_img, _raw_processor);
}

void RawProcessor::ApplyDebayer() {
  auto& pre_debayer_buffer = _process_buffer;
#ifdef HAVE_CUDA
  if (_params._cuda) {
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
  cv::Rect crop_rect(_raw_data.sizes.left_margin, _raw_data.sizes.top_margin, _raw_data.sizes.width,
                     _raw_data.sizes.height);
  img = img(crop_rect);
}

void RawProcessor::ApplyHighlightReconstruct() {
  if (_params._cuda) {
    throw std::runtime_error("RawProcessor: Not implemented");
  } else {
    auto& img = _process_buffer.GetCPUData();
    if (!_params._highlights_reconstruct) {
      // clamp to [0, 1]
      cv::threshold(img, img, 1.0f, 1.0f, cv::THRESH_TRUNC);
      cv::threshold(img, img, 0.0f, 0.0f, cv::THRESH_TOZERO);
      return;
    }
    CPU::HighlightReconstruct(img, _raw_processor);
  }
}

void RawProcessor::ConvertToWorkingSpace() {
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
  auto pre_mul   = _raw_data.color.pre_mul;
  auto cam_mul   = _raw_data.color.cam_mul;
  auto wb_coeffs = _raw_data.color.WB_Coeffs;  // EXIF Lightsource Values
  auto cam_xyz   = _raw_data.color.cam_xyz;
  if (!_params._use_camera_wb) {
    // User specified white balance temperature
    auto user_temp_indices = CPU::GetWBIndicesForTemp(static_cast<float>(_params._user_wb));
    CPU::ApplyColorMatrix(img, color_coeffs, pre_mul, cam_mul, wb_coeffs, user_temp_indices,
                          _params._user_wb, cam_xyz);
    return;
  }
  CPU::ApplyColorMatrix(img, color_coeffs, pre_mul, cam_mul, cam_xyz);
}

auto RawProcessor::Process() -> ImageBuffer {
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
  _process_buffer.GetCPUData().convertTo(_process_buffer.GetCPUData(), CV_32FC1, 1.0f / 65535.0f);

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
  switch (_raw_data.sizes.flip) {
    case 3:
      // 180 degree
      cv::rotate(_process_buffer.GetCPUData(), _process_buffer.GetCPUData(), cv::ROTATE_180);
      break;
    case 5:
      // Rotate 90 CCW
      cv::rotate(_process_buffer.GetCPUData(), _process_buffer.GetCPUData(), cv::ROTATE_90_COUNTERCLOCKWISE);
      break;
    case 6:
      // Rotate 90 CW
      cv::rotate(_process_buffer.GetCPUData(), _process_buffer.GetCPUData(), cv::ROTATE_90_CLOCKWISE);
      break;
    default:
      // Do nothing
      break;
  }

  ConvertToWorkingSpace();
  auto                                      cst_end      = clock::now();
  std::chrono::duration<double, std::milli> cst_duration = cst_end - debayer_end;
  std::cout << "Color Space Transformation took " << cst_duration.count() << " ms\n";
  std::cout << "Total processing took "
            << (linear_duration + hl_duration + debayer_duration + cst_duration).count() << " ms\n";
#ifdef HAVE_CUDA
  if (_params._cuda) {
    _process_buffer.SyncToCPU();
    _process_buffer.ReleaseGPUData();
  }
#endif
  return {std::move(_process_buffer)};
}
}  // namespace puerhlab