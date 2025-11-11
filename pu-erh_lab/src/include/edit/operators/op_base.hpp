#pragma once

#include <memory>

#include "image/image_buffer.hpp"
#include "json.hpp"
#include "op_kernel.hpp"
#include "type/type.hpp"

namespace puerhlab {
enum class PipelineStageName : int {
  Image_Loading       = 0,
  Geometry_Adjustment = 1,
  To_WorkingSpace     = 2,
  Basic_Adjustment    = 3,
  Color_Adjustment    = 4,
  Detail_Adjustment   = 5,
  Output_Transform    = 6,
  Stage_Count         = 7,
  Merged_Stage        = 8  // Special stage for merged streamable stages
};
class IOperatorBase {
 public:
  /**
   * @brief Apply the adjustment from the operator
   *
   * @param input
   * @return ImageBuffer
   */
  virtual void Apply(std::shared_ptr<ImageBuffer> input) = 0;
  /**
   * @brief Set the parameters of this operator from JSON
   *
   * @param params
   */
  virtual auto GetParams() const -> nlohmann::json       = 0;
  /**
   * @brief Get JSON parameter for this operator
   *
   * @return nlohmann::json
   */
  virtual void SetParams(const nlohmann::json&)          = 0;

  virtual auto GetScriptName() const -> std::string      = 0;

  virtual auto GetPriorityLevel() const -> PriorityLevel = 0;

  virtual auto GetStage() const -> PipelineStageName     = 0;

  virtual auto ToKernel() const -> Kernel                = 0;

  // virtual auto ToKernel_Vec() const -> Kernel            = 0;

  virtual ~IOperatorBase()                               = default;
};
/**
 * @brief A base class for all operators
 *
 * @tparam Derived CRTP derived class
 */
template <typename Derived>
class OperatorBase : public IOperatorBase {
 public:
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
  auto GetScriptName() const -> std::string override { return std::string(Derived::_script_name); }

  auto GetPriorityLevel() const -> PriorityLevel override { return Derived::_priority_level; }

  auto GetStage() const -> PipelineStageName override { return Derived::_affiliation_stage; }
};

struct OpStream {
  std::vector<std::shared_ptr<IOperatorBase>> _ops;

  bool                                        AddToStream(const std::shared_ptr<IOperatorBase>& op) {
    // Ensure all kernels in the stream are of the same type
    _ops.push_back(op);
    return true;
  }

  void Clear() { _ops.clear(); }
};
};  // namespace puerhlab
