#pragma once

#include <OpenColorIO/OpenColorIO.h>
#include <OpenColorIO/OpenColorTypes.h>

#include "edit/operators/op_base.hpp"
#include "image/image_buffer.hpp"

namespace puerhlab {
namespace OCIO = OCIO_NAMESPACE;
class ACES_IDT_Op : public OperatorBase<ACES_IDT_Op> {
 private:
  std::string            _src_space;
  std::string            _dst_space;

  const char*            config_path;

  OCIO::ConstConfigRcPtr config;

 public:
  static constexpr std::string_view _canonical_name = "Input Device Transform";
  static constexpr std::string_view _script_name    = "IDT";
  ACES_IDT_Op()                                     = delete;
  ACES_IDT_Op(const std::string& src, const std::string& dst);
  ACES_IDT_Op(const std::string& src, const std::string& dst, const char* config_path);

  auto Apply(ImageBuffer& input) -> ImageBuffer override;
  auto GetParams() const -> nlohmann::json override;
  void SetParams(const nlohmann::json& params) override;
};

class ACES_ODT_Op : public OperatorBase<ACES_ODT_Op> {
 private:
  std::string            _src_space;
  std::string            _dst_space;

  const char*            config_path;

  OCIO::ConstConfigRcPtr config;

 public:
  static constexpr std::string_view _canonical_name = "Output Device Transform";
  static constexpr std::string_view _script_name    = "ODT";
  ACES_ODT_Op()                                     = delete;
  ACES_ODT_Op(const std::string& src, const std::string& dst);
  ACES_ODT_Op(const std::string& src, const std::string& dst, const char* config_path);

  auto Apply(ImageBuffer& input) -> ImageBuffer override;
  auto GetParams() const -> nlohmann::json override;
  void SetParams(const nlohmann::json& params) override;
};
}  // namespace puerhlab