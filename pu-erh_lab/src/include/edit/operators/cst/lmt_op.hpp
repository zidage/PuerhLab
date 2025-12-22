#pragma once

#include <OpenColorIO/OpenColorIO.h>
#include <OpenColorIO/OpenColorTypes.h>

#include <filesystem>
#include <optional>

#include "edit/operators/op_base.hpp"
#include "image/image_buffer.hpp"

namespace puerhlab {
namespace OCIO = OCIO_NAMESPACE;

/**
 * @brief An operator to apply ACES Look Modification Transform (LMT) using OpenColorIO
 *
 */
class OCIO_LMT_Transform_Op : public OperatorBase<OCIO_LMT_Transform_Op> {
 private:
  std::filesystem::path        _lmt_path;
  OCIO::ConstConfigRcPtr       config;

  OCIO::ConstCPUProcessorRcPtr cpu_processor;

 public:
  static constexpr PriorityLevel     _priority_level    = 3;
  // DO NOT USE THIS
  static constexpr PipelineStageName _affiliation_stage = PipelineStageName::Output_Transform;
  static constexpr std::string_view  _canonical_name    = "OCIO LMT";
  static constexpr std::string_view  _script_name       = "ocio_lmt";
  static constexpr OperatorType      _operator_type     = OperatorType::LMT;

  OCIO_LMT_Transform_Op()                               = delete;
  OCIO_LMT_Transform_Op(std::filesystem::path& lmt_path);
  OCIO_LMT_Transform_Op(const nlohmann::json& params);

  void Apply(std::shared_ptr<ImageBuffer> input) override;
  auto ToKernel() const -> Kernel override;
  auto GetParams() const -> nlohmann::json override;
  void SetParams(const nlohmann::json& params) override;

  void SetGlobalParams(OperatorParams& params) const override;
};

}  // namespace puerhlab