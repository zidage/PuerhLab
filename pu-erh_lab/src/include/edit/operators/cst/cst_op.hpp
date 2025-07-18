#pragma once

#include <OpenColorIO/OpenColorIO.h>
#include <OpenColorIO/OpenColorTypes.h>

#include "edit/operators/op_base.hpp"
#include "image/image_buffer.hpp"

namespace puerhlab {
namespace OCIO = OCIO_NAMESPACE;
class OCIO_ACES_Transform_Op : public OperatorBase<OCIO_ACES_Transform_Op> {
 private:
  std::string            _input_transform;
  std::string            _output_transform;

  const char*            config_path;

  OCIO::ConstConfigRcPtr config;

 public:
  static constexpr std::string_view _canonical_name = "OCIO";
  static constexpr std::string_view _script_name    = "ocio";
  OCIO_ACES_Transform_Op()                          = delete;
  OCIO_ACES_Transform_Op(const std::string& input, const std::string& output);
  OCIO_ACES_Transform_Op(const std::string& input, const std::string& output,
                         const char* config_path);

  auto Apply(ImageBuffer& input) -> ImageBuffer override;
  auto GetParams() const -> nlohmann::json override;
  void SetParams(const nlohmann::json& params) override;
};

}  // namespace puerhlab