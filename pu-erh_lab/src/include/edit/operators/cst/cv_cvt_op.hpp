#pragma once

#include <opencv2/core.hpp>
#include <opencv2/core/mat.hpp>

#include "edit/operators/op_base.hpp"

namespace puerhlab {

class CVCvtColorOp : public OperatorBase<CVCvtColorOp> {
 private:
  int                   _code;
  std::optional<size_t> _channel_index;

 public:
  static constexpr PriorityLevel     _priority_level    = 1;
  static constexpr PipelineStageName _affiliation_stage = PipelineStageName::Basic_Adjustment;
  static constexpr std::string_view  _canonical_name    = "CV CvtColor";
  static constexpr std::string_view  _script_name       = "CV_CvtColor";

  CVCvtColorOp()                                        = default;
  CVCvtColorOp(int code, std::optional<size_t> channel_index);
  CVCvtColorOp(const nlohmann::json& params);

  void Apply(std::shared_ptr<ImageBuffer> input) override;
  auto ToKernel() const -> Kernel override { throw std::runtime_error("CVCvtColorOp: ToKernel not implemented"); }
  auto GetParams() const -> nlohmann::json override;
  void SetParams(const nlohmann::json& params) override;
};
}  // namespace puerhlab