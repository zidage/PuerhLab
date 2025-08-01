#pragma once

#include <string>
#include <string_view>

#include "edit/operators/op_base.hpp"

namespace puerhlab {
struct ToneRegionOpRegister {
  ToneRegionOpRegister();
};
/**
 * @brief Tone regions: black level, white level, shadows and highlights
 *
 */
enum class ToneRegion { BLACK, WHITE, SHADOWS, HIGHLIGHTS };
class ToneRegionOp : public OperatorBase<ToneRegionOp> {
 private:
  /**
   * @brief A relative number for enhancing or dehancing a specific tone region
   *
   */
  float       _offset;
  /**
   * @brief An absolute number to represent the contrast after adjustment
   *
   */
  float       _scale;

  /**
   * @brief The tone region to adjust
   *
   */
  ToneRegion  _region;

  static auto RegionToString(ToneRegion region) -> std::string;
  /**
   * @brief
   *
   * @param region_str
   * @return ToneRegion
   */
  static auto StringToRegion(std::string& region_str) -> ToneRegion;
  auto        ComputeWeight(float luminance) const -> float;
  void        ComputeScale();

 public:
  static constexpr PriorityLevel     _priority_level    = 1;
  static constexpr PipelineStageName _affiliation_stage = PipelineStageName::Basic_Adjustment;
  static constexpr std::string_view  _canonical_name    = "ToneRegion";
  static constexpr std::string_view  _script_name       = "tone_region";
  ToneRegionOp()                                        = delete;
  ToneRegionOp(ToneRegion region);
  ToneRegionOp(float offset, ToneRegion region);
  ToneRegionOp(const nlohmann::json& params);

  auto Apply(ImageBuffer& input) -> ImageBuffer override;
  auto GetParams() const -> nlohmann::json override;
  void SetParams(const nlohmann::json& params) override;
};
}  // namespace puerhlab