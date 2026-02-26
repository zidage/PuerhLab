auto ColorTempSliderPosToCct(int pos) -> float {
  const float min_cct     = static_cast<float>(kColorTempCctMin);
  const float max_cct     = static_cast<float>(kColorTempCctMax);
  const int   clamped_pos = std::clamp(pos, kColorTempSliderUiMin, kColorTempSliderUiMax);

  float cct = min_cct;
  if (clamped_pos <= kColorTempSliderUiMid) {
    const float t = static_cast<float>(clamped_pos - kColorTempSliderUiMin) /
                    static_cast<float>(kColorTempSliderUiMid - kColorTempSliderUiMin);
    cct = min_cct + t * (kColorTempPivotCct - min_cct);
  } else {
    const float t = static_cast<float>(clamped_pos - kColorTempSliderUiMid) /
                    static_cast<float>(kColorTempSliderUiMax - kColorTempSliderUiMid);
    cct = kColorTempPivotCct + t * (max_cct - kColorTempPivotCct);
  }

  return std::clamp(cct, min_cct, max_cct);
}

auto ColorTempCctToSliderPos(float cct) -> int {
  const float min_cct     = static_cast<float>(kColorTempCctMin);
  const float max_cct     = static_cast<float>(kColorTempCctMax);
  const float clamped_cct = std::clamp(cct, min_cct, max_cct);

  float pos = static_cast<float>(kColorTempSliderUiMin);
  if (clamped_cct <= kColorTempPivotCct) {
    const float t = (clamped_cct - min_cct) / (kColorTempPivotCct - min_cct);
    pos           = static_cast<float>(kColorTempSliderUiMin) +
          t * static_cast<float>(kColorTempSliderUiMid - kColorTempSliderUiMin);
  } else {
    const float t = (clamped_cct - kColorTempPivotCct) / (max_cct - kColorTempPivotCct);
    pos           = static_cast<float>(kColorTempSliderUiMid) +
          t * static_cast<float>(kColorTempSliderUiMax - kColorTempSliderUiMid);
  }

  return std::clamp(static_cast<int>(std::lround(pos)), kColorTempSliderUiMin,
                    kColorTempSliderUiMax);
}
