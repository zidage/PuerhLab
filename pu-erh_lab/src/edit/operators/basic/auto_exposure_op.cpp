#include "edit/operators/basic/auto_exposure_op.hpp"

#include <opencv2/core.hpp>

namespace puerhlab {
AutoExposureOp::AutoExposureOp(const nlohmann::json& params) {}

void AutoExposureOp::Apply(std::shared_ptr<ImageBuffer> input) {
  cv::Mat& L      = input->GetCPUData();

  // float    v_low  = ComputePercentile(L, p_clip_low);
  float    v_ref  = ComputePercentile(L, p_ref);
  float    v_high = ComputePercentile(L, p_clip_high);

  if (v_ref <= 1e-6f) v_ref = 1e-6f;  // avoid division by zero

  float gain        = target_L / v_ref;
  gain              = std::clamp(gain, min_gain, max_gain);

  float high_thresh = v_high;

  L.forEach<float>([&](float& pixel, const int*) {
    float orig   = pixel;
    float scaled = orig * gain;

    if (preserve_highlights && scaled > high_thresh) {
      float fade_range = std::max(1.0f, 0.7f * high_thresh);
      float t          = (orig - high_thresh) / fade_range;
      t                = std::clamp(t, 0.0f, 1.0f);

      float local_gain = (1.0f - t) * gain + t * 1.0f;
      scaled           = orig * local_gain;
    }

    if (apply_shadow_toe && orig < 10.0f) {
      float toe_in = toe(scaled, toe_strength);
      pixel        = toe_in;
    } else {
      pixel = scaled;
    }

    if (soft_clip_strength > 0.0f) {
      pixel = SoftClip(pixel, soft_clip_start, soft_clip_strength);
    }
  });
}

auto AutoExposureOp::GetParams() const -> nlohmann::json { return {}; }

void AutoExposureOp::SetParams(const nlohmann::json& params) {}
};  // namespace puerhlab