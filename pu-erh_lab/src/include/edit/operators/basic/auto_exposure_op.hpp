#include <cmath>

#include "edit/operators/op_base.hpp"

namespace puerhlab {

class AutoExposureOp : public OperatorBase<AutoExposureOp> {
 private:
  int                               bins                = 16384;
  float                             p_ref               = 0.75f;
  float                             p_clip_low          = 0.01f;
  float                             p_clip_high         = 0.75f;
  float                             target_L            = 50.0f;
  float                             min_gain            = 0.5f;
  float                             max_gain            = 4.0f;
  float                             soft_clip_start     = 95.0f;
  float                             soft_clip_strength  = 0.75f;
  bool                              preserve_highlights = false;
  bool                              apply_shadow_toe    = true;
  float                             toe_strength        = 0.6f;

  std::optional<cv::Mat>            _hist;
  std::optional<std::vector<float>> _cdf;

  inline float                      ComputePercentile(const cv::Mat& L, float pct) {
    // int   histSize = bins;
    // float range[]  = {0.0f, 100.0f};
    // if (!_hist.has_value() || !_cdf.has_value()) {
    //   const float* histRange = {range};
    //   cv::Mat      hist;
    //   cv::calcHist(&L, 1, 0, cv::Mat(), hist, 1, &histSize, &histRange, true, false);
    //   hist /= L.total();

    //   // Step 2: CDF
    //   std::vector<float> cdf(histSize);
    //   cdf[0] = hist.at<float>(0);
    //   for (int i = 1; i < histSize; ++i) cdf[i] = cdf[i - 1] + hist.at<float>(i);

    //   _hist = hist.clone();
    //   _cdf  = std::move(cdf);
    // }
    // // Step 4: Threshold
    // auto& cdf       = _cdf.value();
    // float threshold = 0.0f;
    // for (int i = 0; i < histSize; ++i) {
    //   if (cdf[i] >= pct / 100.0f) {
    //     threshold = range[0] + (range[1] - range[0]) * (i / static_cast<float>(histSize - 1));
    //     break;
    //   }
    // }
    // return threshold;
    CV_Assert(L.type() == CV_32F);
    // build histogram
    float  minVal = 0.0f, maxVal = 100.0f;
    // If data contains values outside 0..100, expand range
    double mn, mx;
    cv::minMaxLoc(L, &mn, &mx);
    if (mn < minVal) minVal = (float)mn;
    if (mx > maxVal) maxVal = (float)mx;

    const int            B = std::max(256, bins);
    std::vector<int64_t> hist(B);
    float                range = maxVal - minVal;
    if (range <= 0.0f) return (float)mn;

    const float invRange = (float)(B - 1) / range;
    L.forEach<float>([&](float v, const int* /*idx*/) {
      int idx = (int)std::floor((v - minVal) * invRange + 0.5f);
      if (idx < 0) idx = 0;
      if (idx >= B) idx = B - 1;
      hist[idx]++;
    });
    // cumulative
    int64_t total = 0;
    for (auto h : hist) total += h;
    if (total == 0) return (float)mn;

    int64_t want = (int64_t)std::floor(pct * total + 0.5);
    int64_t acc  = 0;
    for (int i = 0; i < B; ++i) {
      acc += hist[i];
      if (acc >= want) {
        // linear interpolate within bin
        float binCenter = minVal + (i + 0.5f) * (range / B);
        return binCenter;
      }
    }
    return maxVal;
  }

  inline static float SoftClip(float x, float start, float strength) {
    if (x <= start) return x;
    // simple smooth approach: blend between linear and sqrt curve
    float excess = x - start;
    // strength: 0->hard, 1->very soft
    float s      = std::clamp(strength, 0.0f, 1.0f);
    // sqrt reduces slope; adjust amount by s
    float out =
        start + excess * (1.0f - s) + (std::sqrt(start + excess) - std::sqrt(start)) * (s * 2.0f);
    // ensure monotonic
    return out;
  }

  inline static float toe(float x, float strength) {
    if (strength <= 0.0f) return x;
    // map x in 0..T to a softer curve, here T ~ 0..10 L
    const float T = 10.0f;
    if (x <= 0.0f) return 0.0f;
    if (x >= T) return x;
    float t      = x / T;
    float s      = std::clamp(strength, 0.0f, 1.0f);
    float mapped = std::pow(t, 1.0f + (1.0f - s)) * T;  // stronger strength -> closer to linear
    return mapped;
  }

 public:
  static constexpr PriorityLevel     _priority_level    = 0;
  static constexpr PipelineStageName _affiliation_stage = PipelineStageName::Basic_Adjustment;
  static constexpr std::string_view  _canonical_name    = "Auto Exposure";
  static constexpr std::string_view  _script_name       = "auto_exposure";
  static constexpr OperatorType      _operator_type     = OperatorType::AUTO_EXPOSURE;

  AutoExposureOp()                                      = default;
  AutoExposureOp(const nlohmann::json& params);

  void Apply(std::shared_ptr<ImageBuffer> input) override;
  auto ToKernel() const -> Kernel override {
    throw std::runtime_error("AutoExposureOp does not support kernel processing.");
  }
  auto GetParams() const -> nlohmann::json override;
  void SetParams(const nlohmann::json& params) override;
};
};  // namespace puerhlab