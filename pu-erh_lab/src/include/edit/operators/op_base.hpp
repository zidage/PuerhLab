#pragma once

#include <memory>

#include "image/image_buffer.hpp"
#include "json.hpp"

namespace puerhlab {
class OperatorBase {
 public:
  virtual void Apply(const ImageBuffer& input, const ImageBuffer& output) const = 0;
  virtual auto Clone() const -> std::unique_ptr<OperatorBase>                   = 0;
  virtual auto name() const -> std::string                                      = 0;

  virtual auto GetParams() const -> nlohmann::json                              = 0;
  virtual void SetParams(const nlohmann::json&)                                 = 0;

  virtual ~OperatorBase()                                                       = default;
};
};  // namespace puerhlab
