#pragma once

#include <memory>

#include "image/image_buffer.hpp"
#include "json.hpp"

namespace puerhlab {
/**
 * @brief A base class for all operators
 *
 * @tparam Derived CRTP derived class
 */
template <typename Derived>
class OperatorBase {
 public:
  /**
   * @brief Apply the adjustment from the operator
   *
   * @param input
   * @return ImageBuffer
   */
  virtual auto Apply(ImageBuffer& input) -> ImageBuffer = 0;
  /**
   * @brief Get the canonical name of the operator (for display)
   *
   * @return std::string
   */
  virtual auto GetCanonicalName() const -> std::string {
    return std::string(Derived::_canonical_name);
  }
  /**
   * @brief Get the script name of the operator (for JSON serialization)
   *
   * @return std::string
   */
  virtual auto GetScriptName() const -> std::string { return std::string(Derived::_script_name); }

  /**
   * @brief Set the parameters of this operator from JSON
   *
   * @param params
   */
  virtual auto GetParams() const -> nlohmann::json = 0;
  /**
   * @brief Get JSON parameter for this operator
   *
   * @return nlohmann::json
   */
  virtual void SetParams(const nlohmann::json&)    = 0;

  virtual ~OperatorBase()                          = default;
};
};  // namespace puerhlab
