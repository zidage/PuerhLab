#include "edit/operators/op_base.hpp"

namespace puerhlab {
class ResizeOp : public OperatorBase<ResizeOp> {
 private:
  int _maximum_edge;

 public:
  static constexpr PriorityLevel     _priority_level    = 1;
  static constexpr PipelineStageName _affiliation_stage = PipelineStageName::Image_Loading;
  static constexpr std::string_view  _canonical_name    = "Resize";
  static constexpr std::string_view  _script_name       = "resize";

  ResizeOp()                                            = default;
  ResizeOp(const nlohmann::json& params);

  void Apply(std::shared_ptr<ImageBuffer> input) override;
  auto GetParams() const -> nlohmann::json override;
  void SetParams(const nlohmann::json& params) override;
};
};  // namespace puerhlab