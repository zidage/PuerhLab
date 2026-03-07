//  Copyright 2026 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//
//  This file also contains material subject to the upstream notices below.

//  This file contains GPLv3-derived logic based on OpenDRT v1.1.0
//  by Jed Smith: https://github.com/jedypod/open-display-transform

#pragma once

#include <string>
#include <string_view>

#include "edit/operators/utils/color_utils.hpp"

namespace puerhlab::odt_cpu {

enum class OpenDRTLookPreset : int {
  STANDARD = 0,
  ARRIBA,
  SYLVAN,
  COLORFUL,
  AERY,
  DYSTOPIC,
  UMBRA,
};

enum class OpenDRTTonescalePreset : int {
  USE_LOOK_PRESET = 0,
  LOW_CONTRAST,
  MEDIUM_CONTRAST,
  HIGH_CONTRAST,
  ARRIBA_TONESCALE,
  SYLVAN_TONESCALE,
  COLORFUL_TONESCALE,
  AERY_TONESCALE,
  DYSTOPIC_TONESCALE,
  UMBRA_TONESCALE,
  ACES_1_X,
  ACES_2_0,
  MARVELOUS_TONESCAPE,
  DAGRINCHI_TONEGROAN,
};

enum class OpenDRTCreativeWhitePreset : int {
  USE_LOOK_PRESET = 0,
  D93,
  D75,
  D65,
  D60,
  D55,
  D50,
};

struct OpenDRTSettings {
  OpenDRTLookPreset          look_preset_            = OpenDRTLookPreset::STANDARD;
  OpenDRTTonescalePreset     tonescale_preset_       = OpenDRTTonescalePreset::USE_LOOK_PRESET;
  OpenDRTCreativeWhitePreset creative_white_         = OpenDRTCreativeWhitePreset::USE_LOOK_PRESET;
  float                      creative_white_limit_   = 0.25f;
  float                      display_grey_luminance_ = 10.0f;
  float                      hdr_grey_boost_         = 0.13f;
  float                      hdr_purity_             = 0.5f;
};

auto OpenDRTLookPresetFromString(std::string_view value) -> OpenDRTLookPreset;
auto OpenDRTTonescalePresetFromString(std::string_view value) -> OpenDRTTonescalePreset;
auto OpenDRTCreativeWhitePresetFromString(std::string_view value) -> OpenDRTCreativeWhitePreset;

auto OpenDRTLookPresetToString(OpenDRTLookPreset value) -> std::string;
auto OpenDRTTonescalePresetToString(OpenDRTTonescalePreset value) -> std::string;
auto OpenDRTCreativeWhitePresetToString(OpenDRTCreativeWhitePreset value) -> std::string;

auto ResolveOpenDRTRuntime(ColorUtils::ColorSpace encoding_space, ColorUtils::EOTF encoding_eotf,
                           float peak_luminance,
                           const OpenDRTSettings& settings) -> ColorUtils::OpenDRTParams;

auto ResolveOpenDRTDisplayLinearScale(const ColorUtils::OpenDRTParams& params) -> float;

}  // namespace puerhlab::odt_cpu
