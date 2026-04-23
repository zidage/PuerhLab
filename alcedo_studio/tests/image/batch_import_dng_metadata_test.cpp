//  Copyright 2026 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#include <gtest/gtest.h>

#include <filesystem>

#include "image/image.hpp"
#include "image/metadata_extractor.hpp"

namespace alcedo {
namespace {

auto BatchImportDngSamplePath() -> std::filesystem::path {
  return std::filesystem::path(TEST_IMG_PATH) / "raw" / "batch_import" / "_DSC1306.dng";
}

TEST(BatchImportDngMetadataTest, ExtractsDisplayAndRuntimeMetadataForImportSample) {
  const auto sample_path = BatchImportDngSamplePath();
  if (!std::filesystem::exists(sample_path)) {
    GTEST_SKIP() << "Sample DNG not found: " << sample_path.string();
  }

  Image image(7, sample_path, ImageType::DNG);
  ASSERT_NO_THROW(MetadataExtractor::ExtractEXIF_ToImage(sample_path, image));
  ASSERT_TRUE(image.HasRawColorContext());

  const auto& display = image.exif_display_;
  const auto& ctx     = image.GetRawColorContext();

  EXPECT_FALSE(display.make_.empty());
  EXPECT_FALSE(display.model_.empty());
  EXPECT_GT(display.width_, 4000u);
  EXPECT_GT(display.height_, 3000u);
  EXPECT_GT(display.iso_, 0u);
  EXPECT_GT(display.focal_, 0.0f);
  EXPECT_GT(display.aperture_, 0.0f);

  EXPECT_TRUE(ctx.valid_);
  EXPECT_TRUE(ctx.output_in_camera_space_);
  EXPECT_FALSE(ctx.camera_make_.empty());
  EXPECT_FALSE(ctx.camera_model_.empty());
  EXPECT_TRUE(ctx.color_matrices_valid_);
  EXPECT_TRUE(ctx.as_shot_neutral_valid_);
  EXPECT_TRUE(ctx.calibration_illuminants_valid_);
}

}  // namespace
}  // namespace alcedo
