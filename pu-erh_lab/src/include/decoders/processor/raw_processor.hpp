#pragma once

#include <libraw/libraw.h>

#include <cstdint>
#include <opencv2/core/types.hpp>

#include "image/image_buffer.hpp"
#include "operators/cpu/raw_proc_utils.hpp"
#include "type/type.hpp"

namespace puerhlab {
struct RawParams {
  bool     _cuda                   = false;
  bool     _highlights_reconstruct = false;
  bool     _use_camera_wb          = true;
  uint32_t _user_wb = 6500;  // If user wants to set a specific white balance temperature
  CPU::LightSourceType _user_light_source =
      CPU::LightSourceType::UNKNOWN;  // If user wants to use a preset light source as the wb
};

class RawProcessor {
 private:
  ImageBuffer             _process_buffer;
  RawParams               _params;

  const libraw_rawdata_t& _raw_data;
  LibRaw&                 _raw_processor;


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

  /**
   * @brief Apply color space transformation from camera RGB to ACES2065-1.
   *
   * Currently, this procedure is not complete. LibRaw does not provide all the necesssary data to
   * perform a "more" accurate color space transformation according to the DNG specification.
   * In the future, we may consider converting all RAW files to DNG first using Adobe DNG Converter
   * to obtain the necessary data (ForwardMatrix, ColorMatrix1, ColorMatrix2, etc.).
   *
   * Due to the lack of lab measurement data, the current "color science" is solely based on the
   * color matrices provided by LibRaw, which are extracted from the camera profiles or the corresponding DNG files. 
   */
  void                    ConvertToWorkingSpace();

 public:
  RawProcessor() = delete;
  RawProcessor(const RawParams& params, const libraw_rawdata_t& rawdata, LibRaw& raw_processor);
  auto Process() -> ImageBuffer;
};
};  // namespace puerhlab