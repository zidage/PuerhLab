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

  static constexpr float  _xyz_ap0[9] = {1.06349349f,     0.00641074032f, -0.0157827139f,
                                         -0.492064744f,   1.36820328f,    0.0913489163f,
                                         -0.00281119067f, 0.00463410327f, 0.916555226f};

  //   static constexpr float  _xyz_ap0[9] = {1.0498110175f,  0.0000000000f, -0.0000974845f,
  //                                          -0.4959030231f, 1.3733130458f, 0.0982400361f,
  //                                          0.0000000000f,  0.0000000000f, 0.9912520182f};

  void                    ApplyWhiteBalance();
  void                    ApplyDebayer();
  void                    ApplyHighlightReconstruct();
  void                    ApplyColorSpaceTransform();

 public:
  RawProcessor() = delete;
  RawProcessor(const RawParams& params, const libraw_rawdata_t& rawdata, LibRaw& raw_processor);
  auto Process() -> ImageBuffer;
};
};  // namespace puerhlab