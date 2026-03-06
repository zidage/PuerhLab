//  Copyright 2026 Yurun Zi
//
//  This file contains GPLv3-derived logic based on OpenDRT v1.1.0
//  by Jed Smith: https://github.com/jedypod/open-display-transform

#include "edit/operators/cst/open_drt_cpu.hpp"

#include <algorithm>
#include <bit>
#include <cstdint>
#include <mutex>
#include <stdexcept>
#include <unordered_map>

namespace puerhlab::odt_cpu {
namespace {

struct OpenDRTCacheKey {
  int           encoding_space_              = 0;
  int           encoding_etof_               = 0;
  int           look_preset_                 = 0;
  int           tonescale_preset_            = 0;
  int           creative_white_preset_       = 0;
  std::uint32_t peak_luminance_bits_         = 0;
  std::uint32_t creative_white_limit_bits_   = 0;
  std::uint32_t display_grey_luminance_bits_ = 0;
  std::uint32_t hdr_grey_boost_bits_         = 0;
  std::uint32_t hdr_purity_bits_             = 0;

  auto operator==(const OpenDRTCacheKey& other) const -> bool {
    return encoding_space_ == other.encoding_space_ &&
           encoding_etof_ == other.encoding_etof_ &&
           look_preset_ == other.look_preset_ &&
           tonescale_preset_ == other.tonescale_preset_ &&
           creative_white_preset_ == other.creative_white_preset_ &&
           peak_luminance_bits_ == other.peak_luminance_bits_ &&
           creative_white_limit_bits_ == other.creative_white_limit_bits_ &&
           display_grey_luminance_bits_ == other.display_grey_luminance_bits_ &&
           hdr_grey_boost_bits_ == other.hdr_grey_boost_bits_ &&
           hdr_purity_bits_ == other.hdr_purity_bits_;
  }
};

struct OpenDRTCacheKeyHash {
  auto operator()(const OpenDRTCacheKey& key) const noexcept -> std::size_t {
    std::size_t h = std::hash<int>{}(key.encoding_space_);
    h ^= std::hash<int>{}(key.encoding_etof_) + 0x9e3779b9 + (h << 6) + (h >> 2);
    h ^= std::hash<int>{}(key.look_preset_) + 0x9e3779b9 + (h << 6) + (h >> 2);
    h ^= std::hash<int>{}(key.tonescale_preset_) + 0x9e3779b9 + (h << 6) + (h >> 2);
    h ^= std::hash<int>{}(key.creative_white_preset_) + 0x9e3779b9 + (h << 6) + (h >> 2);
    h ^= std::hash<std::uint32_t>{}(key.peak_luminance_bits_) + 0x9e3779b9 + (h << 6) + (h >> 2);
    h ^= std::hash<std::uint32_t>{}(key.creative_white_limit_bits_) + 0x9e3779b9 + (h << 6) +
         (h >> 2);
    h ^= std::hash<std::uint32_t>{}(key.display_grey_luminance_bits_) + 0x9e3779b9 + (h << 6) +
         (h >> 2);
    h ^= std::hash<std::uint32_t>{}(key.hdr_grey_boost_bits_) + 0x9e3779b9 + (h << 6) + (h >> 2);
    h ^= std::hash<std::uint32_t>{}(key.hdr_purity_bits_) + 0x9e3779b9 + (h << 6) + (h >> 2);
    return h;
  }
};

auto RuntimeCache()
    -> std::unordered_map<OpenDRTCacheKey, ColorUtils::OpenDRTParams, OpenDRTCacheKeyHash>& {
  static std::unordered_map<OpenDRTCacheKey, ColorUtils::OpenDRTParams, OpenDRTCacheKeyHash> cache;
  return cache;
}

auto RuntimeCacheMutex() -> std::mutex& {
  static std::mutex cache_mutex;
  return cache_mutex;
}

auto BuildCacheKey(ColorUtils::ColorSpace encoding_space, ColorUtils::ETOF encoding_etof,
                   float peak_luminance,
                   const OpenDRTSettings& settings) -> OpenDRTCacheKey {
  return OpenDRTCacheKey{
      static_cast<int>(encoding_space),
      static_cast<int>(encoding_etof),
      static_cast<int>(settings.look_preset_),
      static_cast<int>(settings.tonescale_preset_),
      static_cast<int>(settings.creative_white_),
      std::bit_cast<std::uint32_t>(peak_luminance),
      std::bit_cast<std::uint32_t>(settings.creative_white_limit_),
      std::bit_cast<std::uint32_t>(settings.display_grey_luminance_),
      std::bit_cast<std::uint32_t>(settings.hdr_grey_boost_),
      std::bit_cast<std::uint32_t>(settings.hdr_purity_),
  };
}

auto Clamp01(float value) -> float { return std::clamp(value, 0.0f, 1.0f); }

auto CompressToeQuadratic(float x, float toe, bool inverse) -> float {
  if (toe == 0.0f) {
    return x;
  }
  if (!inverse) {
    return (x * x) / (x + toe);
  }
  return (x + std::sqrt(x * (4.0f * toe + x))) / 2.0f;
}

void ApplyLookPreset(OpenDRTLookPreset look, ColorUtils::OpenDRTParams* p);
void ApplyTonescalePreset(OpenDRTTonescalePreset tonescale, ColorUtils::OpenDRTParams* p);
void ApplyCreativeWhitePreset(OpenDRTCreativeWhitePreset preset, float creative_white_limit,
                              ColorUtils::OpenDRTParams* p);
void ResolveDisplayEncoding(ColorUtils::ColorSpace encoding_space, ColorUtils::ETOF encoding_etof,
                            ColorUtils::OpenDRTParams* p);
void FinalizePrecompute(float peak_luminance, float display_grey_luminance, float hdr_grey_boost,
                        float hdr_purity, ColorUtils::OpenDRTParams* p);

void ApplyLookPreset(OpenDRTLookPreset look, ColorUtils::OpenDRTParams* p) {
  switch (look) {
    case OpenDRTLookPreset::STANDARD:
      p->tn_con_ = 1.66f;
      p->tn_sh_ = 0.5f;
      p->tn_toe_ = 0.003f;
      p->tn_off_ = 0.005f;
      p->tn_hcon_enable_ = 0;
      p->tn_hcon_ = 0.0f;
      p->tn_hcon_pv_ = 1.0f;
      p->tn_hcon_st_ = 4.0f;
      p->tn_lcon_enable_ = 0;
      p->tn_lcon_ = 0.0f;
      p->tn_lcon_w_ = 0.5f;
      p->creative_white_ = 2;
      p->cwp_lm_ = 0.25f;
      p->rs_sa_ = 0.35f;
      p->rs_rw_ = 0.25f;
      p->rs_bw_ = 0.55f;
      p->pt_enable_ = 1;
      p->pt_lml_ = 0.25f;
      p->pt_lml_r_ = 0.5f;
      p->pt_lml_g_ = 0.0f;
      p->pt_lml_b_ = 0.1f;
      p->pt_lmh_ = 0.25f;
      p->pt_lmh_r_ = 0.5f;
      p->pt_lmh_b_ = 0.0f;
      p->ptl_enable_ = 1;
      p->ptl_c_ = 0.06f;
      p->ptl_m_ = 0.08f;
      p->ptl_y_ = 0.06f;
      p->ptm_enable_ = 1;
      p->ptm_low_ = 0.4f;
      p->ptm_low_rng_ = 0.25f;
      p->ptm_low_st_ = 0.5f;
      p->ptm_high_ = -0.8f;
      p->ptm_high_rng_ = 0.35f;
      p->ptm_high_st_ = 0.4f;
      p->brl_enable_ = 1;
      p->brl_ = 0.0f;
      p->brl_r_ = -2.5f;
      p->brl_g_ = -1.5f;
      p->brl_b_ = -1.5f;
      p->brl_rng_ = 0.5f;
      p->brl_st_ = 0.35f;
      p->brlp_enable_ = 1;
      p->brlp_ = -0.5f;
      p->brlp_r_ = -1.25f;
      p->brlp_g_ = -1.25f;
      p->brlp_b_ = -0.25f;
      p->hc_enable_ = 1;
      p->hc_r_ = 1.0f;
      p->hc_r_rng_ = 0.3f;
      p->hs_rgb_enable_ = 1;
      p->hs_r_ = 0.6f;
      p->hs_r_rng_ = 0.6f;
      p->hs_g_ = 0.35f;
      p->hs_g_rng_ = 1.0f;
      p->hs_b_ = 0.66f;
      p->hs_b_rng_ = 1.0f;
      p->hs_cmy_enable_ = 1;
      p->hs_c_ = 0.25f;
      p->hs_c_rng_ = 1.0f;
      p->hs_m_ = 0.0f;
      p->hs_m_rng_ = 1.0f;
      p->hs_y_ = 0.0f;
      p->hs_y_rng_ = 1.0f;
      return;
    case OpenDRTLookPreset::ARRIBA:
      p->tn_con_ = 1.05f;
      p->tn_sh_ = 0.5f;
      p->tn_toe_ = 0.1f;
      p->tn_off_ = 0.01f;
      p->tn_hcon_enable_ = 0;
      p->tn_hcon_ = 0.0f;
      p->tn_hcon_pv_ = 1.0f;
      p->tn_hcon_st_ = 4.0f;
      p->tn_lcon_enable_ = 1;
      p->tn_lcon_ = 1.5f;
      p->tn_lcon_w_ = 0.2f;
      p->creative_white_ = 2;
      p->cwp_lm_ = 0.25f;
      p->rs_sa_ = 0.35f;
      p->rs_rw_ = 0.25f;
      p->rs_bw_ = 0.55f;
      p->pt_enable_ = 1;
      p->pt_lml_ = 0.25f;
      p->pt_lml_r_ = 0.45f;
      p->pt_lml_g_ = 0.0f;
      p->pt_lml_b_ = 0.1f;
      p->pt_lmh_ = 0.25f;
      p->pt_lmh_r_ = 0.25f;
      p->pt_lmh_b_ = 0.0f;
      p->ptl_enable_ = 1;
      p->ptl_c_ = 0.06f;
      p->ptl_m_ = 0.08f;
      p->ptl_y_ = 0.06f;
      p->ptm_enable_ = 1;
      p->ptm_low_ = 1.0f;
      p->ptm_low_rng_ = 0.4f;
      p->ptm_low_st_ = 0.5f;
      p->ptm_high_ = -0.8f;
      p->ptm_high_rng_ = 0.66f;
      p->ptm_high_st_ = 0.6f;
      p->brl_enable_ = 1;
      p->brl_ = 0.0f;
      p->brl_r_ = -2.5f;
      p->brl_g_ = -1.5f;
      p->brl_b_ = -1.5f;
      p->brl_rng_ = 0.5f;
      p->brl_st_ = 0.35f;
      p->brlp_enable_ = 1;
      p->brlp_ = 0.0f;
      p->brlp_r_ = -1.7f;
      p->brlp_g_ = -2.0f;
      p->brlp_b_ = -0.5f;
      p->hc_enable_ = 1;
      p->hc_r_ = 1.0f;
      p->hc_r_rng_ = 0.3f;
      p->hs_rgb_enable_ = 1;
      p->hs_r_ = 0.6f;
      p->hs_r_rng_ = 0.8f;
      p->hs_g_ = 0.35f;
      p->hs_g_rng_ = 1.0f;
      p->hs_b_ = 0.66f;
      p->hs_b_rng_ = 1.0f;
      p->hs_cmy_enable_ = 1;
      p->hs_c_ = 0.15f;
      p->hs_c_rng_ = 1.0f;
      p->hs_m_ = 0.0f;
      p->hs_m_rng_ = 1.0f;
      p->hs_y_ = 0.0f;
      p->hs_y_rng_ = 1.0f;
      return;
    case OpenDRTLookPreset::SYLVAN:
      p->tn_con_ = 1.6f;
      p->tn_sh_ = 0.5f;
      p->tn_toe_ = 0.01f;
      p->tn_off_ = 0.01f;
      p->tn_hcon_enable_ = 0;
      p->tn_hcon_ = 0.0f;
      p->tn_hcon_pv_ = 1.0f;
      p->tn_hcon_st_ = 4.0f;
      p->tn_lcon_enable_ = 1;
      p->tn_lcon_ = 0.25f;
      p->tn_lcon_w_ = 0.75f;
      p->creative_white_ = 2;
      p->cwp_lm_ = 0.25f;
      p->rs_sa_ = 0.25f;
      p->rs_rw_ = 0.25f;
      p->rs_bw_ = 0.55f;
      p->pt_enable_ = 1;
      p->pt_lml_ = 0.15f;
      p->pt_lml_r_ = 0.5f;
      p->pt_lml_g_ = 0.15f;
      p->pt_lml_b_ = 0.1f;
      p->pt_lmh_ = 0.25f;
      p->pt_lmh_r_ = 0.15f;
      p->pt_lmh_b_ = 0.15f;
      p->ptl_enable_ = 1;
      p->ptl_c_ = 0.05f;
      p->ptl_m_ = 0.08f;
      p->ptl_y_ = 0.05f;
      p->ptm_enable_ = 1;
      p->ptm_low_ = 0.5f;
      p->ptm_low_rng_ = 0.5f;
      p->ptm_low_st_ = 0.5f;
      p->ptm_high_ = -0.8f;
      p->ptm_high_rng_ = 0.5f;
      p->ptm_high_st_ = 0.5f;
      p->brl_enable_ = 1;
      p->brl_ = -1.0f;
      p->brl_r_ = -2.0f;
      p->brl_g_ = -2.0f;
      p->brl_b_ = 0.0f;
      p->brl_rng_ = 0.25f;
      p->brl_st_ = 0.25f;
      p->brlp_enable_ = 1;
      p->brlp_ = -1.0f;
      p->brlp_r_ = -0.5f;
      p->brlp_g_ = -0.25f;
      p->brlp_b_ = -0.25f;
      p->hc_enable_ = 1;
      p->hc_r_ = 1.0f;
      p->hc_r_rng_ = 0.4f;
      p->hs_rgb_enable_ = 1;
      p->hs_r_ = 0.6f;
      p->hs_r_rng_ = 1.15f;
      p->hs_g_ = 0.8f;
      p->hs_g_rng_ = 1.25f;
      p->hs_b_ = 0.6f;
      p->hs_b_rng_ = 1.0f;
      p->hs_cmy_enable_ = 1;
      p->hs_c_ = 0.25f;
      p->hs_c_rng_ = 0.25f;
      p->hs_m_ = 0.25f;
      p->hs_m_rng_ = 0.5f;
      p->hs_y_ = 0.35f;
      p->hs_y_rng_ = 0.5f;
      return;
    case OpenDRTLookPreset::COLORFUL:
      p->tn_con_ = 1.5f;
      p->tn_sh_ = 0.5f;
      p->tn_toe_ = 0.003f;
      p->tn_off_ = 0.003f;
      p->tn_hcon_enable_ = 0;
      p->tn_hcon_ = 0.0f;
      p->tn_hcon_pv_ = 1.0f;
      p->tn_hcon_st_ = 4.0f;
      p->tn_lcon_enable_ = 1;
      p->tn_lcon_ = 0.4f;
      p->tn_lcon_w_ = 0.5f;
      p->creative_white_ = 2;
      p->cwp_lm_ = 0.25f;
      p->rs_sa_ = 0.35f;
      p->rs_rw_ = 0.25f;
      p->rs_bw_ = 0.55f;
      p->pt_enable_ = 1;
      p->pt_lml_ = 0.5f;
      p->pt_lml_r_ = 1.0f;
      p->pt_lml_g_ = 0.0f;
      p->pt_lml_b_ = 0.5f;
      p->pt_lmh_ = 0.15f;
      p->pt_lmh_r_ = 0.15f;
      p->pt_lmh_b_ = 0.15f;
      p->ptl_enable_ = 1;
      p->ptl_c_ = 0.05f;
      p->ptl_m_ = 0.06f;
      p->ptl_y_ = 0.05f;
      p->ptm_enable_ = 1;
      p->ptm_low_ = 0.8f;
      p->ptm_low_rng_ = 0.5f;
      p->ptm_low_st_ = 0.4f;
      p->ptm_high_ = -0.8f;
      p->ptm_high_rng_ = 0.4f;
      p->ptm_high_st_ = 0.4f;
      p->brl_enable_ = 1;
      p->brl_ = 0.0f;
      p->brl_r_ = -1.25f;
      p->brl_g_ = -1.25f;
      p->brl_b_ = -0.25f;
      p->brl_rng_ = 0.3f;
      p->brl_st_ = 0.5f;
      p->brlp_enable_ = 1;
      p->brlp_ = -0.5f;
      p->brlp_r_ = -1.25f;
      p->brlp_g_ = -1.25f;
      p->brlp_b_ = -0.5f;
      p->hc_enable_ = 1;
      p->hc_r_ = 1.0f;
      p->hc_r_rng_ = 0.4f;
      p->hs_rgb_enable_ = 1;
      p->hs_r_ = 0.5f;
      p->hs_r_rng_ = 0.8f;
      p->hs_g_ = 0.35f;
      p->hs_g_rng_ = 1.0f;
      p->hs_b_ = 0.5f;
      p->hs_b_rng_ = 1.0f;
      p->hs_cmy_enable_ = 1;
      p->hs_c_ = 0.25f;
      p->hs_c_rng_ = 1.0f;
      p->hs_m_ = 0.0f;
      p->hs_m_rng_ = 1.0f;
      p->hs_y_ = 0.25f;
      p->hs_y_rng_ = 1.0f;
      return;
    case OpenDRTLookPreset::AERY:
      p->tn_con_ = 1.15f;
      p->tn_sh_ = 0.5f;
      p->tn_toe_ = 0.04f;
      p->tn_off_ = 0.006f;
      p->tn_hcon_enable_ = 0;
      p->tn_hcon_ = 0.0f;
      p->tn_hcon_pv_ = 0.0f;
      p->tn_hcon_st_ = 0.5f;
      p->tn_lcon_enable_ = 1;
      p->tn_lcon_ = 0.5f;
      p->tn_lcon_w_ = 2.0f;
      p->creative_white_ = 1;
      p->cwp_lm_ = 0.25f;
      p->rs_sa_ = 0.25f;
      p->rs_rw_ = 0.2f;
      p->rs_bw_ = 0.5f;
      p->pt_enable_ = 1;
      p->pt_lml_ = 0.0f;
      p->pt_lml_r_ = 0.5f;
      p->pt_lml_g_ = 0.15f;
      p->pt_lml_b_ = 0.1f;
      p->pt_lmh_ = 0.0f;
      p->pt_lmh_r_ = 0.1f;
      p->pt_lmh_b_ = 0.0f;
      p->ptl_enable_ = 1;
      p->ptl_c_ = 0.05f;
      p->ptl_m_ = 0.08f;
      p->ptl_y_ = 0.05f;
      p->ptm_enable_ = 1;
      p->ptm_low_ = 0.8f;
      p->ptm_low_rng_ = 0.35f;
      p->ptm_low_st_ = 0.5f;
      p->ptm_high_ = -0.9f;
      p->ptm_high_rng_ = 0.5f;
      p->ptm_high_st_ = 0.3f;
      p->brl_enable_ = 1;
      p->brl_ = -3.0f;
      p->brl_r_ = 0.0f;
      p->brl_g_ = 0.0f;
      p->brl_b_ = 1.0f;
      p->brl_rng_ = 0.8f;
      p->brl_st_ = 0.15f;
      p->brlp_enable_ = 1;
      p->brlp_ = -1.0f;
      p->brlp_r_ = -1.0f;
      p->brlp_g_ = -1.0f;
      p->brlp_b_ = 0.0f;
      p->hc_enable_ = 1;
      p->hc_r_ = 0.5f;
      p->hc_r_rng_ = 0.25f;
      p->hs_rgb_enable_ = 1;
      p->hs_r_ = 0.6f;
      p->hs_r_rng_ = 1.0f;
      p->hs_g_ = 0.35f;
      p->hs_g_rng_ = 2.0f;
      p->hs_b_ = 0.5f;
      p->hs_b_rng_ = 1.5f;
      p->hs_cmy_enable_ = 1;
      p->hs_c_ = 0.35f;
      p->hs_c_rng_ = 1.0f;
      p->hs_m_ = 0.25f;
      p->hs_m_rng_ = 1.0f;
      p->hs_y_ = 0.35f;
      p->hs_y_rng_ = 0.5f;
      return;
    case OpenDRTLookPreset::DYSTOPIC:
      p->tn_con_ = 1.6f;
      p->tn_sh_ = 0.5f;
      p->tn_toe_ = 0.01f;
      p->tn_off_ = 0.008f;
      p->tn_hcon_enable_ = 1;
      p->tn_hcon_ = 0.25f;
      p->tn_hcon_pv_ = 0.0f;
      p->tn_hcon_st_ = 1.0f;
      p->tn_lcon_enable_ = 1;
      p->tn_lcon_ = 1.0f;
      p->tn_lcon_w_ = 0.75f;
      p->creative_white_ = 3;
      p->cwp_lm_ = 0.25f;
      p->rs_sa_ = 0.2f;
      p->rs_rw_ = 0.25f;
      p->rs_bw_ = 0.55f;
      p->pt_enable_ = 1;
      p->pt_lml_ = 0.15f;
      p->pt_lml_r_ = 0.0f;
      p->pt_lml_g_ = 0.0f;
      p->pt_lml_b_ = 0.0f;
      p->pt_lmh_ = 0.0f;
      p->pt_lmh_r_ = 0.0f;
      p->pt_lmh_b_ = 0.0f;
      p->ptl_enable_ = 1;
      p->ptl_c_ = 0.05f;
      p->ptl_m_ = 0.08f;
      p->ptl_y_ = 0.05f;
      p->ptm_enable_ = 1;
      p->ptm_low_ = 0.25f;
      p->ptm_low_rng_ = 0.25f;
      p->ptm_low_st_ = 0.8f;
      p->ptm_high_ = -0.8f;
      p->ptm_high_rng_ = 0.6f;
      p->ptm_high_st_ = 0.25f;
      p->brl_enable_ = 1;
      p->brl_ = -2.0f;
      p->brl_r_ = -2.0f;
      p->brl_g_ = -2.0f;
      p->brl_b_ = 0.0f;
      p->brl_rng_ = 0.35f;
      p->brl_st_ = 0.35f;
      p->brlp_enable_ = 1;
      p->brlp_ = 0.0f;
      p->brlp_r_ = -1.0f;
      p->brlp_g_ = -1.0f;
      p->brlp_b_ = -1.0f;
      p->hc_enable_ = 1;
      p->hc_r_ = 1.0f;
      p->hc_r_rng_ = 0.25f;
      p->hs_rgb_enable_ = 1;
      p->hs_r_ = 0.7f;
      p->hs_r_rng_ = 1.33f;
      p->hs_g_ = 1.0f;
      p->hs_g_rng_ = 2.0f;
      p->hs_b_ = 0.75f;
      p->hs_b_rng_ = 2.0f;
      p->hs_cmy_enable_ = 1;
      p->hs_c_ = 1.0f;
      p->hs_c_rng_ = 0.5f;
      p->hs_m_ = 1.0f;
      p->hs_m_rng_ = 1.0f;
      p->hs_y_ = 1.0f;
      p->hs_y_rng_ = 0.765f;
      return;
    case OpenDRTLookPreset::UMBRA:
      p->tn_con_ = 1.8f;
      p->tn_sh_ = 0.5f;
      p->tn_toe_ = 0.001f;
      p->tn_off_ = 0.015f;
      p->tn_hcon_enable_ = 0;
      p->tn_hcon_ = 0.0f;
      p->tn_hcon_pv_ = 1.0f;
      p->tn_hcon_st_ = 4.0f;
      p->tn_lcon_enable_ = 1;
      p->tn_lcon_ = 1.0f;
      p->tn_lcon_w_ = 1.0f;
      p->creative_white_ = 5;
      p->cwp_lm_ = 0.25f;
      p->rs_sa_ = 0.35f;
      p->rs_rw_ = 0.25f;
      p->rs_bw_ = 0.55f;
      p->pt_enable_ = 1;
      p->pt_lml_ = 0.0f;
      p->pt_lml_r_ = 0.5f;
      p->pt_lml_g_ = 0.0f;
      p->pt_lml_b_ = 0.15f;
      p->pt_lmh_ = 0.25f;
      p->pt_lmh_r_ = 0.25f;
      p->pt_lmh_b_ = 0.0f;
      p->ptl_enable_ = 1;
      p->ptl_c_ = 0.05f;
      p->ptl_m_ = 0.06f;
      p->ptl_y_ = 0.05f;
      p->ptm_enable_ = 1;
      p->ptm_low_ = 0.4f;
      p->ptm_low_rng_ = 0.35f;
      p->ptm_low_st_ = 0.66f;
      p->ptm_high_ = -0.6f;
      p->ptm_high_rng_ = 0.45f;
      p->ptm_high_st_ = 0.45f;
      p->brl_enable_ = 1;
      p->brl_ = -2.0f;
      p->brl_r_ = -4.5f;
      p->brl_g_ = -3.0f;
      p->brl_b_ = -4.0f;
      p->brl_rng_ = 0.35f;
      p->brl_st_ = 0.3f;
      p->brlp_enable_ = 1;
      p->brlp_ = 0.0f;
      p->brlp_r_ = -2.0f;
      p->brlp_g_ = -1.0f;
      p->brlp_b_ = -0.5f;
      p->hc_enable_ = 1;
      p->hc_r_ = 1.0f;
      p->hc_r_rng_ = 0.35f;
      p->hs_rgb_enable_ = 1;
      p->hs_r_ = 0.66f;
      p->hs_r_rng_ = 1.0f;
      p->hs_g_ = 0.5f;
      p->hs_g_rng_ = 2.0f;
      p->hs_b_ = 0.85f;
      p->hs_b_rng_ = 2.0f;
      p->hs_cmy_enable_ = 1;
      p->hs_c_ = 0.0f;
      p->hs_c_rng_ = 1.0f;
      p->hs_m_ = 0.25f;
      p->hs_m_rng_ = 1.0f;
      p->hs_y_ = 0.66f;
      p->hs_y_rng_ = 0.66f;
      return;
    default:
      return;
  }
}

void ApplyTonescalePreset(OpenDRTTonescalePreset tonescale, ColorUtils::OpenDRTParams* p) {
  switch (tonescale) {
    case OpenDRTTonescalePreset::USE_LOOK_PRESET:
      return;
    case OpenDRTTonescalePreset::LOW_CONTRAST:
      p->tn_con_ = 1.4f;
      p->tn_sh_ = 0.5f;
      p->tn_toe_ = 0.003f;
      p->tn_off_ = 0.005f;
      p->tn_hcon_enable_ = 0;
      p->tn_hcon_ = 0.0f;
      p->tn_hcon_pv_ = 1.0f;
      p->tn_hcon_st_ = 4.0f;
      p->tn_lcon_enable_ = 0;
      p->tn_lcon_ = 0.0f;
      p->tn_lcon_w_ = 0.5f;
      return;
    case OpenDRTTonescalePreset::MEDIUM_CONTRAST:
      p->tn_con_ = 1.66f;
      p->tn_sh_ = 0.5f;
      p->tn_toe_ = 0.003f;
      p->tn_off_ = 0.005f;
      p->tn_hcon_enable_ = 0;
      p->tn_hcon_ = 0.0f;
      p->tn_hcon_pv_ = 1.0f;
      p->tn_hcon_st_ = 4.0f;
      p->tn_lcon_enable_ = 0;
      p->tn_lcon_ = 0.0f;
      p->tn_lcon_w_ = 0.5f;
      return;
    case OpenDRTTonescalePreset::HIGH_CONTRAST:
      p->tn_con_ = 1.4f;
      p->tn_sh_ = 0.5f;
      p->tn_toe_ = 0.003f;
      p->tn_off_ = 0.005f;
      p->tn_hcon_enable_ = 0;
      p->tn_hcon_ = 0.0f;
      p->tn_hcon_pv_ = 1.0f;
      p->tn_hcon_st_ = 4.0f;
      p->tn_lcon_enable_ = 1;
      p->tn_lcon_ = 1.0f;
      p->tn_lcon_w_ = 0.5f;
      return;
    case OpenDRTTonescalePreset::ARRIBA_TONESCALE:
      p->tn_con_ = 1.05f;
      p->tn_sh_ = 0.5f;
      p->tn_toe_ = 0.1f;
      p->tn_off_ = 0.01f;
      p->tn_hcon_enable_ = 0;
      p->tn_hcon_ = 0.0f;
      p->tn_hcon_pv_ = 1.0f;
      p->tn_hcon_st_ = 4.0f;
      p->tn_lcon_enable_ = 1;
      p->tn_lcon_ = 1.5f;
      p->tn_lcon_w_ = 0.2f;
      return;
    case OpenDRTTonescalePreset::SYLVAN_TONESCALE:
      p->tn_con_ = 1.6f;
      p->tn_sh_ = 0.5f;
      p->tn_toe_ = 0.01f;
      p->tn_off_ = 0.01f;
      p->tn_hcon_enable_ = 0;
      p->tn_hcon_ = 0.0f;
      p->tn_hcon_pv_ = 1.0f;
      p->tn_hcon_st_ = 4.0f;
      p->tn_lcon_enable_ = 1;
      p->tn_lcon_ = 0.25f;
      p->tn_lcon_w_ = 0.75f;
      return;
    case OpenDRTTonescalePreset::COLORFUL_TONESCALE:
      p->tn_con_ = 1.5f;
      p->tn_sh_ = 0.5f;
      p->tn_toe_ = 0.003f;
      p->tn_off_ = 0.003f;
      p->tn_hcon_enable_ = 0;
      p->tn_hcon_ = 0.0f;
      p->tn_hcon_pv_ = 1.0f;
      p->tn_hcon_st_ = 4.0f;
      p->tn_lcon_enable_ = 1;
      p->tn_lcon_ = 0.4f;
      p->tn_lcon_w_ = 0.5f;
      return;
    case OpenDRTTonescalePreset::AERY_TONESCALE:
      p->tn_con_ = 1.15f;
      p->tn_sh_ = 0.5f;
      p->tn_toe_ = 0.04f;
      p->tn_off_ = 0.006f;
      p->tn_hcon_enable_ = 0;
      p->tn_hcon_ = 0.0f;
      p->tn_hcon_pv_ = 0.0f;
      p->tn_hcon_st_ = 0.5f;
      p->tn_lcon_enable_ = 1;
      p->tn_lcon_ = 0.5f;
      p->tn_lcon_w_ = 2.0f;
      return;
    case OpenDRTTonescalePreset::DYSTOPIC_TONESCALE:
      p->tn_con_ = 1.6f;
      p->tn_sh_ = 0.5f;
      p->tn_toe_ = 0.01f;
      p->tn_off_ = 0.008f;
      p->tn_hcon_enable_ = 1;
      p->tn_hcon_ = 0.25f;
      p->tn_hcon_pv_ = 0.0f;
      p->tn_hcon_st_ = 1.0f;
      p->tn_lcon_enable_ = 1;
      p->tn_lcon_ = 1.0f;
      p->tn_lcon_w_ = 0.75f;
      return;
    case OpenDRTTonescalePreset::UMBRA_TONESCALE:
      p->tn_con_ = 1.8f;
      p->tn_sh_ = 0.5f;
      p->tn_toe_ = 0.001f;
      p->tn_off_ = 0.015f;
      p->tn_hcon_enable_ = 0;
      p->tn_hcon_ = 0.0f;
      p->tn_hcon_pv_ = 1.0f;
      p->tn_hcon_st_ = 4.0f;
      p->tn_lcon_enable_ = 1;
      p->tn_lcon_ = 1.0f;
      p->tn_lcon_w_ = 1.0f;
      return;
    case OpenDRTTonescalePreset::ACES_1_X:
      p->tn_con_ = 1.0f;
      p->tn_sh_ = 0.35f;
      p->tn_toe_ = 0.02f;
      p->tn_off_ = 0.0f;
      p->tn_hcon_enable_ = 1;
      p->tn_hcon_ = 0.55f;
      p->tn_hcon_pv_ = 0.0f;
      p->tn_hcon_st_ = 2.0f;
      p->tn_lcon_enable_ = 1;
      p->tn_lcon_ = 1.13f;
      p->tn_lcon_w_ = 1.0f;
      return;
    case OpenDRTTonescalePreset::ACES_2_0:
      p->tn_con_ = 1.15f;
      p->tn_sh_ = 0.5f;
      p->tn_toe_ = 0.04f;
      p->tn_off_ = 0.0f;
      p->tn_hcon_enable_ = 0;
      p->tn_hcon_ = 1.0f;
      p->tn_hcon_pv_ = 1.0f;
      p->tn_hcon_st_ = 1.0f;
      p->tn_lcon_enable_ = 0;
      p->tn_lcon_ = 1.0f;
      p->tn_lcon_w_ = 0.6f;
      return;
    case OpenDRTTonescalePreset::MARVELOUS_TONESCAPE:
      p->tn_con_ = 1.5f;
      p->tn_sh_ = 0.5f;
      p->tn_toe_ = 0.003f;
      p->tn_off_ = 0.01f;
      p->tn_hcon_enable_ = 1;
      p->tn_hcon_ = 0.25f;
      p->tn_hcon_pv_ = 0.0f;
      p->tn_hcon_st_ = 4.0f;
      p->tn_lcon_enable_ = 1;
      p->tn_lcon_ = 1.0f;
      p->tn_lcon_w_ = 1.0f;
      return;
    case OpenDRTTonescalePreset::DAGRINCHI_TONEGROAN:
      p->tn_con_ = 1.2f;
      p->tn_sh_ = 0.5f;
      p->tn_toe_ = 0.02f;
      p->tn_off_ = 0.0f;
      p->tn_hcon_enable_ = 0;
      p->tn_hcon_ = 0.0f;
      p->tn_hcon_pv_ = 1.0f;
      p->tn_hcon_st_ = 1.0f;
      p->tn_lcon_enable_ = 0;
      p->tn_lcon_ = 0.0f;
      p->tn_lcon_w_ = 0.6f;
      return;
    default:
      return;
  }
}

void ApplyCreativeWhitePreset(OpenDRTCreativeWhitePreset preset, float creative_white_limit,
                              ColorUtils::OpenDRTParams* p) {
  switch (preset) {
    case OpenDRTCreativeWhitePreset::USE_LOOK_PRESET:
      return;
    case OpenDRTCreativeWhitePreset::D93:
      p->creative_white_ = 0;
      break;
    case OpenDRTCreativeWhitePreset::D75:
      p->creative_white_ = 1;
      break;
    case OpenDRTCreativeWhitePreset::D65:
      p->creative_white_ = 2;
      break;
    case OpenDRTCreativeWhitePreset::D60:
      p->creative_white_ = 3;
      break;
    case OpenDRTCreativeWhitePreset::D55:
      p->creative_white_ = 4;
      break;
    case OpenDRTCreativeWhitePreset::D50:
      p->creative_white_ = 5;
      break;
  }
  p->cwp_lm_ = Clamp01(creative_white_limit);
}

void ResolveDisplayEncoding(ColorUtils::ColorSpace encoding_space, ColorUtils::ETOF encoding_etof,
                            ColorUtils::OpenDRTParams* p) {
  if (encoding_space == ColorUtils::ColorSpace::REC709 &&
      encoding_etof == ColorUtils::ETOF::BT1886) {
    p->surround_ = 1;
    p->display_gamut_ = 0;
    p->display_eotf_ = 2;
    return;
  }
  if (encoding_space == ColorUtils::ColorSpace::REC709 &&
      encoding_etof == ColorUtils::ETOF::GAMMA_2_2) {
    p->surround_ = 2;
    p->display_gamut_ = 0;
    p->display_eotf_ = 1;
    return;
  }
  if (encoding_space == ColorUtils::ColorSpace::P3_D65 &&
      encoding_etof == ColorUtils::ETOF::GAMMA_2_2) {
    p->surround_ = 2;
    p->display_gamut_ = 1;
    p->display_eotf_ = 1;
    return;
  }
  if (encoding_space == ColorUtils::ColorSpace::P3_D60 &&
      encoding_etof == ColorUtils::ETOF::GAMMA_2_6) {
    p->surround_ = 0;
    p->display_gamut_ = 3;
    p->display_eotf_ = 3;
    return;
  }
  if (encoding_space == ColorUtils::ColorSpace::P3_DCI &&
      encoding_etof == ColorUtils::ETOF::GAMMA_2_6) {
    p->surround_ = 0;
    p->display_gamut_ = 4;
    p->display_eotf_ = 3;
    return;
  }
  if (encoding_space == ColorUtils::ColorSpace::XYZ &&
      encoding_etof == ColorUtils::ETOF::GAMMA_2_6) {
    p->surround_ = 0;
    p->display_gamut_ = 5;
    p->display_eotf_ = 3;
    return;
  }
  if (encoding_space == ColorUtils::ColorSpace::REC2020 &&
      encoding_etof == ColorUtils::ETOF::ST2084) {
    p->surround_ = 0;
    p->display_gamut_ = 2;
    p->display_eotf_ = 4;
    return;
  }
  if (encoding_space == ColorUtils::ColorSpace::REC2020 &&
      encoding_etof == ColorUtils::ETOF::HLG) {
    p->surround_ = 0;
    p->display_gamut_ = 2;
    p->display_eotf_ = 5;
    return;
  }
  if (encoding_space == ColorUtils::ColorSpace::P3_D65 &&
      encoding_etof == ColorUtils::ETOF::ST2084) {
    p->surround_ = 0;
    p->display_gamut_ = 1;
    p->display_eotf_ = 4;
    return;
  }

  throw std::runtime_error("ODT_Op: unsupported OpenDRT output combination for encoding_space=\"" +
                           ColorUtils::ColorSpaceToString(encoding_space) + "\" and encoding_etof=\"" +
                           ColorUtils::ETOFToString(encoding_etof) + "\".");
}

void FinalizePrecompute(float peak_luminance, float display_grey_luminance, float hdr_grey_boost,
                        float hdr_purity, ColorUtils::OpenDRTParams* p) {
  p->clamp_     = 1;
  p->ts_x1_     = std::pow(2.0f, 6.0f * p->tn_sh_ + 4.0f);
  p->ts_y1_     = peak_luminance / 100.0f;
  p->ts_x0_     = 0.18f + p->tn_off_;
  p->ts_y0_     = display_grey_luminance / 100.0f * (1.0f + hdr_grey_boost * std::log2(p->ts_y1_));
  p->ts_s0_     = CompressToeQuadratic(p->ts_y0_, p->tn_toe_, true);
  p->ts_p_      = p->tn_con_ / (1.0f + static_cast<float>(p->surround_) * 0.05f);
  p->ts_s10_    = p->ts_x0_ * (std::pow(p->ts_s0_, -1.0f / p->tn_con_) - 1.0f);
  p->ts_m1_     = p->ts_y1_ / std::pow(p->ts_x1_ / (p->ts_x1_ + p->ts_s10_), p->tn_con_);
  p->ts_m2_     = CompressToeQuadratic(p->ts_m1_, p->tn_toe_, true);
  p->ts_s_      = p->ts_x0_ * (std::pow(p->ts_s0_ / p->ts_m2_, -1.0f / p->tn_con_) - 1.0f);
  p->ts_dsc_    = (p->display_eotf_ == 4) ? 0.01f : (p->display_eotf_ == 5) ? 0.1f : 100.0f / peak_luminance;
  p->pt_cmp_Lf_ = hdr_purity * std::min(1.0f, (peak_luminance - 100.0f) / 900.0f);
  p->s_Lp100_   = p->ts_x0_ * (std::pow(display_grey_luminance / 100.0f, -1.0f / p->tn_con_) - 1.0f);
  p->ts_s1_     = p->ts_s_ * p->pt_cmp_Lf_ + p->s_Lp100_ * (1.0f - p->pt_cmp_Lf_);
}

}  // namespace

auto OpenDRTLookPresetFromString(std::string_view value) -> OpenDRTLookPreset {
  if (value == "arriba") return OpenDRTLookPreset::ARRIBA;
  if (value == "sylvan") return OpenDRTLookPreset::SYLVAN;
  if (value == "colorful") return OpenDRTLookPreset::COLORFUL;
  if (value == "aery") return OpenDRTLookPreset::AERY;
  if (value == "dystopic") return OpenDRTLookPreset::DYSTOPIC;
  if (value == "umbra") return OpenDRTLookPreset::UMBRA;
  return OpenDRTLookPreset::STANDARD;
}

auto OpenDRTTonescalePresetFromString(std::string_view value) -> OpenDRTTonescalePreset {
  if (value == "low_contrast") return OpenDRTTonescalePreset::LOW_CONTRAST;
  if (value == "medium_contrast") return OpenDRTTonescalePreset::MEDIUM_CONTRAST;
  if (value == "high_contrast") return OpenDRTTonescalePreset::HIGH_CONTRAST;
  if (value == "arriba_tonescale") return OpenDRTTonescalePreset::ARRIBA_TONESCALE;
  if (value == "sylvan_tonescale") return OpenDRTTonescalePreset::SYLVAN_TONESCALE;
  if (value == "colorful_tonescale") return OpenDRTTonescalePreset::COLORFUL_TONESCALE;
  if (value == "aery_tonescale") return OpenDRTTonescalePreset::AERY_TONESCALE;
  if (value == "dystopic_tonescale") return OpenDRTTonescalePreset::DYSTOPIC_TONESCALE;
  if (value == "umbra_tonescale") return OpenDRTTonescalePreset::UMBRA_TONESCALE;
  if (value == "aces_1_x") return OpenDRTTonescalePreset::ACES_1_X;
  if (value == "aces_2_0") return OpenDRTTonescalePreset::ACES_2_0;
  if (value == "marvelous_tonescape") return OpenDRTTonescalePreset::MARVELOUS_TONESCAPE;
  if (value == "dagrinchi_tonegroan") return OpenDRTTonescalePreset::DAGRINCHI_TONEGROAN;
  return OpenDRTTonescalePreset::USE_LOOK_PRESET;
}

auto OpenDRTCreativeWhitePresetFromString(std::string_view value) -> OpenDRTCreativeWhitePreset {
  if (value == "d93") return OpenDRTCreativeWhitePreset::D93;
  if (value == "d75") return OpenDRTCreativeWhitePreset::D75;
  if (value == "d65") return OpenDRTCreativeWhitePreset::D65;
  if (value == "d60") return OpenDRTCreativeWhitePreset::D60;
  if (value == "d55") return OpenDRTCreativeWhitePreset::D55;
  if (value == "d50") return OpenDRTCreativeWhitePreset::D50;
  return OpenDRTCreativeWhitePreset::USE_LOOK_PRESET;
}

auto OpenDRTLookPresetToString(OpenDRTLookPreset value) -> std::string {
  switch (value) {
    case OpenDRTLookPreset::STANDARD:
      return "standard";
    case OpenDRTLookPreset::ARRIBA:
      return "arriba";
    case OpenDRTLookPreset::SYLVAN:
      return "sylvan";
    case OpenDRTLookPreset::COLORFUL:
      return "colorful";
    case OpenDRTLookPreset::AERY:
      return "aery";
    case OpenDRTLookPreset::DYSTOPIC:
      return "dystopic";
    case OpenDRTLookPreset::UMBRA:
      return "umbra";
    default:
      return "standard";
  }
}

auto OpenDRTTonescalePresetToString(OpenDRTTonescalePreset value) -> std::string {
  switch (value) {
    case OpenDRTTonescalePreset::USE_LOOK_PRESET:
      return "use_look_preset";
    case OpenDRTTonescalePreset::LOW_CONTRAST:
      return "low_contrast";
    case OpenDRTTonescalePreset::MEDIUM_CONTRAST:
      return "medium_contrast";
    case OpenDRTTonescalePreset::HIGH_CONTRAST:
      return "high_contrast";
    case OpenDRTTonescalePreset::ARRIBA_TONESCALE:
      return "arriba_tonescale";
    case OpenDRTTonescalePreset::SYLVAN_TONESCALE:
      return "sylvan_tonescale";
    case OpenDRTTonescalePreset::COLORFUL_TONESCALE:
      return "colorful_tonescale";
    case OpenDRTTonescalePreset::AERY_TONESCALE:
      return "aery_tonescale";
    case OpenDRTTonescalePreset::DYSTOPIC_TONESCALE:
      return "dystopic_tonescale";
    case OpenDRTTonescalePreset::UMBRA_TONESCALE:
      return "umbra_tonescale";
    case OpenDRTTonescalePreset::ACES_1_X:
      return "aces_1_x";
    case OpenDRTTonescalePreset::ACES_2_0:
      return "aces_2_0";
    case OpenDRTTonescalePreset::MARVELOUS_TONESCAPE:
      return "marvelous_tonescape";
    case OpenDRTTonescalePreset::DAGRINCHI_TONEGROAN:
      return "dagrinchi_tonegroan";
    default:
      return "use_look_preset";
  }
}

auto OpenDRTCreativeWhitePresetToString(OpenDRTCreativeWhitePreset value) -> std::string {
  switch (value) {
    case OpenDRTCreativeWhitePreset::USE_LOOK_PRESET:
      return "use_look_preset";
    case OpenDRTCreativeWhitePreset::D93:
      return "d93";
    case OpenDRTCreativeWhitePreset::D75:
      return "d75";
    case OpenDRTCreativeWhitePreset::D65:
      return "d65";
    case OpenDRTCreativeWhitePreset::D60:
      return "d60";
    case OpenDRTCreativeWhitePreset::D55:
      return "d55";
    case OpenDRTCreativeWhitePreset::D50:
      return "d50";
    default:
      return "use_look_preset";
  }
}

auto ResolveOpenDRTRuntime(ColorUtils::ColorSpace encoding_space, ColorUtils::ETOF encoding_etof,
                           float peak_luminance,
                           const OpenDRTSettings& input_settings) -> ColorUtils::OpenDRTParams {
  OpenDRTSettings settings = input_settings;
  settings.creative_white_limit_ =
      std::clamp(settings.creative_white_limit_, 0.0f, 1.0f);
  settings.display_grey_luminance_ =
      std::clamp(settings.display_grey_luminance_, 3.0f, 25.0f);
  settings.hdr_grey_boost_ = std::clamp(settings.hdr_grey_boost_, 0.0f, 1.0f);
  settings.hdr_purity_     = std::clamp(settings.hdr_purity_, 0.0f, 1.0f);
  peak_luminance           = std::clamp(peak_luminance, 100.0f, 1000.0f);

  const OpenDRTCacheKey      cache_key =
      BuildCacheKey(encoding_space, encoding_etof, peak_luminance, settings);
  std::lock_guard<std::mutex> lock(RuntimeCacheMutex());
  auto&                       cache = RuntimeCache();
  const auto                  cached_it = cache.find(cache_key);
  if (cached_it != cache.end()) {
    return cached_it->second;
  }

  ColorUtils::OpenDRTParams runtime;
  ApplyLookPreset(settings.look_preset_, &runtime);
  ApplyTonescalePreset(settings.tonescale_preset_, &runtime);
  ApplyCreativeWhitePreset(settings.creative_white_, settings.creative_white_limit_, &runtime);
  ResolveDisplayEncoding(encoding_space, encoding_etof, &runtime);
  FinalizePrecompute(peak_luminance, settings.display_grey_luminance_, settings.hdr_grey_boost_,
                     settings.hdr_purity_, &runtime);

  cache.emplace(cache_key, runtime);
  return runtime;
}

auto ResolveOpenDRTDisplayLinearScale(const ColorUtils::OpenDRTParams& params) -> float {
  return (params.display_eotf_ == 4) ? 10000.0f : 1.0f;
}

}  // namespace puerhlab::odt_cpu
