#pragma once

#include <memory>

#include "image/image_buffer.hpp"
#include "json.hpp"

namespace puerhlab {
template <typename Derived>
class OperatorBase {
 public:
  virtual auto Apply(ImageBuffer& input) -> ImageBuffer = 0;
  virtual auto GetCanonicalName() const -> std::string {
    return std::string(Derived::_canonical_name);
  }
  virtual auto GetScriptName() const -> std::string { return std::string(Derived::_script_name); }

  virtual auto GetParams() const -> nlohmann::json = 0;
  virtual void SetParams(const nlohmann::json&)    = 0;

  virtual ~OperatorBase()                          = default;
};
};  // namespace puerhlab
