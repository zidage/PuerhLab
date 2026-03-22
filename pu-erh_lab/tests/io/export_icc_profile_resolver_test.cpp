//  Copyright 2026 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#include "io/image/export_icc_profile_resolver.hpp"

#include <gtest/gtest.h>

#include <array>

namespace puerhlab {
namespace {

using ColorSpace = ColorUtils::ColorSpace;
using EOTF = ColorUtils::EOTF;

TEST(ExportIccProfileResolverTests, SupportedProfilesResolveToBytes) {
  const std::array<ExportColorProfileConfig, 9> configs = {{
      {ColorSpace::REC709, EOTF::BT1886, 100.0f},
      {ColorSpace::REC709, EOTF::GAMMA_2_2, 100.0f},
      {ColorSpace::P3_D65, EOTF::GAMMA_2_2, 100.0f},
      {ColorSpace::P3_D65, EOTF::ST2084, 600.0f},
      {ColorSpace::P3_D60, EOTF::GAMMA_2_6, 100.0f},
      {ColorSpace::P3_DCI, EOTF::GAMMA_2_6, 100.0f},
      {ColorSpace::XYZ, EOTF::GAMMA_2_6, 100.0f},
      {ColorSpace::REC2020, EOTF::ST2084, 600.0f},
      {ColorSpace::REC2020, EOTF::HLG, 600.0f},
  }};

  for (const auto& config : configs) {
    const auto bytes = ExportIccProfileResolver::ResolveIccProfileBytes(config);
    EXPECT_FALSE(bytes.empty()) << ColorUtils::ColorSpaceToString(config.encoding_space) << "/"
                                << ColorUtils::EOTFToString(config.encoding_eotf);
  }
}

TEST(ExportIccProfileResolverTests, SupportedProfilesResolveToExistingFiles) {
  const std::array<ExportColorProfileConfig, 9> configs = {{
      {ColorSpace::REC709, EOTF::BT1886, 100.0f},
      {ColorSpace::REC709, EOTF::GAMMA_2_2, 100.0f},
      {ColorSpace::P3_D65, EOTF::GAMMA_2_2, 100.0f},
      {ColorSpace::P3_D65, EOTF::ST2084, 600.0f},
      {ColorSpace::P3_D60, EOTF::GAMMA_2_6, 100.0f},
      {ColorSpace::P3_DCI, EOTF::GAMMA_2_6, 100.0f},
      {ColorSpace::XYZ, EOTF::GAMMA_2_6, 100.0f},
      {ColorSpace::REC2020, EOTF::ST2084, 600.0f},
      {ColorSpace::REC2020, EOTF::HLG, 600.0f},
  }};

  for (const auto& config : configs) {
    const auto path = ExportIccProfileResolver::ResolveConfigProfilePath(config);
    ASSERT_TRUE(path.has_value()) << ColorUtils::ColorSpaceToString(config.encoding_space) << "/"
                                  << ColorUtils::EOTFToString(config.encoding_eotf);
    EXPECT_TRUE(std::filesystem::exists(*path));
    EXPECT_TRUE(std::filesystem::is_regular_file(*path));
  }
}

}  // namespace
}  // namespace puerhlab
