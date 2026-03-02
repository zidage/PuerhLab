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

#pragma once

#include <libraw/libraw.h>

#include <cstdint>
#include <string>
#include <opencv2/core/types.hpp>

#include "decoders/decoder_scheduler.hpp"
#include "decoders/processor/raw_color_context.hpp"
#include "image/image_buffer.hpp"
#include "operators/cpu/raw_proc_utils.hpp"
#include "type/type.hpp"

namespace puerhlab {

struct RawParams {
  bool     cuda_                   = false;
  bool     highlights_reconstruct_ = false;
  bool     use_camera_wb_          = true;
  uint32_t user_wb_ = 6500;  // If user wants to set a specific white balance temperature
  CPU::LightSourceType user_light_source_ =
      CPU::LightSourceType::UNKNOWN;  // If user wants to use a preset light source as the wb

  DecodeRes decode_res_ = DecodeRes::FULL;
};

class RawProcessor {
 private:
  ImageBuffer             process_buffer_;
  RawParams               params_;
  RawRuntimeColorContext  runtime_color_context_;

  const libraw_rawdata_t& raw_data_;
  LibRaw&                 raw_processor_;

  void                    SetDecodeRes();

  /**
   * @brief A procedure similar to DNG "Mapping Raw Values to Linear Reference Values" procedure.
   *
   * This method converts the raw sensor values to linear reference values. To support highlight
   * reconstruction, "as shot" white balance multipliers are applied here beforehand.
   */
  void                    ApplyLinearization();

  /**
   * @brief Apply highlight reconstruction if enabled.
   *
   * To make highlight reconstruction work properly, "as shot" white balance multipliers should be
   * applied before this step.
   */
  void                    ApplyHighlightReconstruct();

  /**
   * @brief Debayer the raw image according to the selected algorithm.
   *
   */
  void                    ApplyDebayer();

  void                    ApplyGeometricCorrections();

  /**
   * @brief Apply color space transformation from camera RGB to ACES2065-1.
   *
   * Currently, this procedure is not complete. LibRaw does not provide all the necesssary data to
   * perform a "more" accurate color space transformation according to the DNG specification.
   * In the future, we may consider converting all RAW files to DNG first using Adobe DNG Converter
   * to obtain the necessary data (ForwardMatrix, ColorMatrix1, ColorMatrix2, etc.).
   *
   * Due to the lack of lab measurement data, the current "color science" is solely based on the
   * color matrices provided by LibRaw, which are extracted from the camera profiles or the
   * corresponding DNG files.
   */
  void                    ConvertToWorkingSpace();

 public:
  RawProcessor() = delete;
  RawProcessor(const RawParams& params, const libraw_rawdata_t& rawdata, LibRaw& raw_processor,
               const RawRuntimeColorContext& pre_ctx);
  auto Process() -> ImageBuffer;
  auto GetRuntimeColorContext() const -> const RawRuntimeColorContext& { return runtime_color_context_; }
};
};  // namespace puerhlab
