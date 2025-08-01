#pragma once

#include <OpenColorIO/OpenColorIO.h>
#include <OpenColorIO/OpenColorTypes.h>

#include <filesystem>
#include <optional>

#include "edit/operators/op_base.hpp"
#include "image/image_buffer.hpp"

namespace puerhlab {
namespace OCIO = OCIO_NAMESPACE;
struct OCIO_ACES_Transform_Op_Register {
  OCIO_ACES_Transform_Op_Register();
};

class OCIO_ACES_Transform_Op : public OperatorBase<OCIO_ACES_Transform_Op> {
 private:
  std::string                          _input_transform;
  std::string                          _output_transform;

  std::optional<std::filesystem::path> _lmt_path;

  OCIO::ConstConfigRcPtr               config;

 public:
  static constexpr PriorityLevel     _priority_level    = 0;
  // DO NOT USE THIS
  static constexpr PipelineStageName _affiliation_stage = PipelineStageName::Output_Transform;
  static constexpr std::string_view  _canonical_name    = "OCIO";
  static constexpr std::string_view  _script_name       = "ocio";
  OCIO_ACES_Transform_Op()                              = delete;
  OCIO_ACES_Transform_Op(const std::string& input, const std::string& output);
  OCIO_ACES_Transform_Op(const std::string& input, const std::string& output,
                         const char* config_path);
  OCIO_ACES_Transform_Op(std::filesystem::path& lmt_path);
  OCIO_ACES_Transform_Op(const nlohmann::json& params);

  auto Apply(ImageBuffer& input) -> ImageBuffer override;
  auto ApplyLMT(ImageBuffer& input) -> ImageBuffer;
  auto GetParams() const -> nlohmann::json override;
  void SetParams(const nlohmann::json& params) override;
};

}  // namespace puerhlab