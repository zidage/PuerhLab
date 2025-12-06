#include "edit/operators/op_base.hpp"

namespace puerhlab {
struct ROI {
  int   x;
  int   y;
  float resize_factor;
};

class ResizeOp : public OperatorBase<ResizeOp> {
 private:
  int  _maximum_edge;

  bool enable_roi = false;
  ROI  roi;

 public:
  static constexpr PriorityLevel     _priority_level    = 1;
  static constexpr PipelineStageName _affiliation_stage = PipelineStageName::Image_Loading;
  static constexpr std::string_view  _canonical_name    = "Resize";
  static constexpr std::string_view  _script_name       = "resize";
  static constexpr OperatorType      _operator_type     = OperatorType::RESIZE;

  ResizeOp()                                            = default;
  ResizeOp(const nlohmann::json& params);

  void Apply(std::shared_ptr<ImageBuffer> input) override;
  auto ToKernel() const -> Kernel override;
  auto GetParams() const -> nlohmann::json override;
  void SetParams(const nlohmann::json& params) override;
};
};  // namespace puerhlab