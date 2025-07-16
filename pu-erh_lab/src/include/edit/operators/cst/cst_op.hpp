#pragma once

#include <OpenColorIO/OpenColorIO.h>
#include <OpenColorIO/OpenColorTypes.h>

#include "edit/operators/op_base.hpp"
#include "image/image_buffer.hpp"

namespace puerhlab {
namespace OCIO = OCIO_NAMESPACE;
class OCIOTransformOp : public OperatorBase<OCIOTransformOp> {
 private:
  std::string            _src_space;
  std::string            _dst_space;

  const char*            config_path;

  OCIO::ConstConfigRcPtr config;

 public:
  static constexpr std::string_view _canonical_name = "Color Space Transform";
  static constexpr std::string_view _script_name    = "cst";
  OCIOTransformOp()                                 = delete;
  OCIOTransformOp(const std::string& src, const std::string& dst);
  OCIOTransformOp(const std::string& src, const std::string& dst, const char* config_path);

  auto Apply(ImageBuffer& input) -> ImageBuffer override;
  auto GetParams() const -> nlohmann::json override;
  void SetParams(const nlohmann::json& params) override;
};
}  // namespace puerhlab