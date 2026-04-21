//  Copyright 2026 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#include <gtest/gtest.h>

#include <cmath>
#include <filesystem>

#include "edit/operators/basic/color_temp_op.hpp"
#include "image/image.hpp"
#include "image/metadata_extractor.hpp"

namespace alcedo {
namespace {

auto BadDngSamplePath() -> std::filesystem::path {
  return std::filesystem::path(TEST_IMG_PATH) / "raw" / "bad_dng" / "bad_color_dng.dng";
}

void ExpectMatrixNear(const double* actual, const double (&expected)[9], const double epsilon) {
  ASSERT_NE(actual, nullptr);
  for (int i = 0; i < 9; ++i) {
    EXPECT_NEAR(actual[i], expected[i], epsilon) << "matrix index " << i;
  }
}

TEST(MetadataExtractorTest, DngImportUsesEmbeddedColorMatrices) {
  const auto sample_path = BadDngSamplePath();
  if (!std::filesystem::exists(sample_path)) {
    GTEST_SKIP() << "Sample DNG not found: " << sample_path.string();
  }

  Image image(1, sample_path, ImageType::DNG);
  ASSERT_NO_THROW(MetadataExtractor::ExtractEXIF_ToImage(sample_path, image));
  ASSERT_TRUE(image.HasRawColorContext());

  const auto& ctx = image.GetRawColorContext();
  EXPECT_TRUE(ctx.valid_);
  EXPECT_TRUE(ctx.color_matrices_valid_);
  EXPECT_TRUE(ctx.forward_matrices_valid_);
  EXPECT_TRUE(ctx.as_shot_neutral_valid_);
  EXPECT_TRUE(ctx.calibration_illuminants_valid_);
  EXPECT_EQ(ctx.camera_make_, "Phase One");
  EXPECT_EQ(ctx.camera_model_, "IQ4 150MP");
  EXPECT_NEAR(ctx.as_shot_neutral_[0], 0.30864197, 1e-6);
  EXPECT_NEAR(ctx.as_shot_neutral_[1], 1.0, 1e-6);
  EXPECT_NEAR(ctx.as_shot_neutral_[2], 0.6578947305, 1e-6);
  EXPECT_NEAR(ctx.color_matrix_1_cct_, 5503.0, 1e-6);
  EXPECT_NEAR(ctx.color_matrix_2_cct_, 7504.0, 1e-6);

  static constexpr double kExpectedCm1[9] = {
      0.3100999889, 0.0, 0.0,
      0.0, 1.0, 0.0,
      0.0, 0.0, 0.8062999844,
  };
  static constexpr double kExpectedCm2[9] = {
      0.2423000034, 0.0, 0.0,
      0.0, 0.9901000261, 0.0,
      0.0, 0.0, 0.7989000081,
  };
  static constexpr double kExpectedFm[9] = {
      0.8295999765, 0.0538000013, 0.08089999812,
      0.3136999902, 0.8399000167, -0.1536000069,
      0.0132999993, -0.4061000046, 1.217900038,
  };

  ExpectMatrixNear(ctx.color_matrix_1_, kExpectedCm1, 1e-4);
  ExpectMatrixNear(ctx.color_matrix_2_, kExpectedCm2, 1e-4);
  ExpectMatrixNear(ctx.forward_matrix_1_, kExpectedFm, 1e-4);
  ExpectMatrixNear(ctx.forward_matrix_2_, kExpectedFm, 1e-4);
}

TEST(MetadataExtractorTest, RawColorContextSurvivesExifJsonRoundTrip) {
  const auto sample_path = BadDngSamplePath();
  if (!std::filesystem::exists(sample_path)) {
    GTEST_SKIP() << "Sample DNG not found: " << sample_path.string();
  }

  Image source(2, sample_path, ImageType::DNG);
  ASSERT_NO_THROW(MetadataExtractor::ExtractEXIF_ToImage(sample_path, source));
  ASSERT_TRUE(source.HasRawColorContext());

  const std::string persisted = source.ExifToJson();

  Image restored(3, sample_path, ImageType::DNG);
  ASSERT_NO_THROW(restored.JsonToExif(persisted));
  ASSERT_TRUE(restored.HasRawColorContext());

  const auto& original = source.GetRawColorContext();
  const auto& roundtrip = restored.GetRawColorContext();
  EXPECT_EQ(roundtrip.camera_make_, original.camera_make_);
  EXPECT_EQ(roundtrip.camera_model_, original.camera_model_);
  EXPECT_EQ(roundtrip.color_matrices_valid_, original.color_matrices_valid_);
  EXPECT_EQ(roundtrip.forward_matrices_valid_, original.forward_matrices_valid_);
  EXPECT_EQ(roundtrip.as_shot_neutral_valid_, original.as_shot_neutral_valid_);
  EXPECT_EQ(roundtrip.calibration_illuminants_valid_, original.calibration_illuminants_valid_);
  EXPECT_DOUBLE_EQ(roundtrip.color_matrix_1_cct_, original.color_matrix_1_cct_);
  EXPECT_DOUBLE_EQ(roundtrip.color_matrix_2_cct_, original.color_matrix_2_cct_);

  for (int i = 0; i < 9; ++i) {
    EXPECT_DOUBLE_EQ(roundtrip.color_matrix_1_[i], original.color_matrix_1_[i]);
    EXPECT_DOUBLE_EQ(roundtrip.color_matrix_2_[i], original.color_matrix_2_[i]);
    EXPECT_DOUBLE_EQ(roundtrip.forward_matrix_1_[i], original.forward_matrix_1_[i]);
    EXPECT_DOUBLE_EQ(roundtrip.forward_matrix_2_[i], original.forward_matrix_2_[i]);
  }
  for (int i = 0; i < 3; ++i) {
    EXPECT_DOUBLE_EQ(roundtrip.as_shot_neutral_[i], original.as_shot_neutral_[i]);
  }
}

TEST(MetadataExtractorTest, ColorTempOpSupportsDngWithoutCamXyzWhenDngMetadataIsPresent) {
  const auto sample_path = BadDngSamplePath();
  if (!std::filesystem::exists(sample_path)) {
    GTEST_SKIP() << "Sample DNG not found: " << sample_path.string();
  }

  Image image(4, sample_path, ImageType::DNG);
  ASSERT_NO_THROW(MetadataExtractor::ExtractEXIF_ToImage(sample_path, image));
  ASSERT_TRUE(image.HasRawColorContext());

  OperatorParams params;
  params.color_temp_enabled_ = true;
  params.PopulateRawMetadata(image.GetRawColorContext());

  for (float& value : params.raw_cam_xyz_) {
    value = 0.0f;
  }

  const nlohmann::json color_temp_params = {
      {"color_temp", {{"mode", "as_shot"}, {"cct", 6500.0}, {"tint", 0.0}}}};
  ColorTempOp op(color_temp_params);
  ASSERT_NO_THROW(op.SetGlobalParams(params));

  EXPECT_TRUE(params.color_temp_matrices_valid_);
  EXPECT_GT(params.color_temp_resolved_xy_[0], 0.0f);
  EXPECT_GT(params.color_temp_resolved_xy_[1], 0.0f);
  EXPECT_GE(params.color_temp_resolved_cct_, 2000.0f);
  EXPECT_LE(params.color_temp_resolved_cct_, 15000.0f);
}

TEST(MetadataExtractorTest, ColorTempOpUsesForwardMatrixForBadColorDngAsShot) {
  const auto sample_path = BadDngSamplePath();
  if (!std::filesystem::exists(sample_path)) {
    GTEST_SKIP() << "Sample DNG not found: " << sample_path.string();
  }

  Image image(5, sample_path, ImageType::DNG);
  ASSERT_NO_THROW(MetadataExtractor::ExtractEXIF_ToImage(sample_path, image));
  ASSERT_TRUE(image.HasRawColorContext());

  OperatorParams params;
  params.color_temp_enabled_ = true;
  params.PopulateRawMetadata(image.GetRawColorContext());

  const nlohmann::json color_temp_params = {
      {"color_temp", {{"mode", "as_shot"}, {"cct", 6500.0}, {"tint", 0.0}}}};
  ColorTempOp op(color_temp_params);
  ASSERT_NO_THROW(op.SetGlobalParams(params));

  EXPECT_TRUE(params.color_temp_matrices_valid_);
  EXPECT_GT(params.color_temp_resolved_cct_, 4000.0f);
  EXPECT_LT(params.color_temp_resolved_cct_, 6000.0f);
  EXPECT_GT(std::abs(params.color_temp_cam_to_xyz_d50_[1]), 0.01f);
  EXPECT_GT(std::abs(params.color_temp_cam_to_xyz_d50_[3]), 0.1f);
  EXPECT_GT(std::abs(params.color_temp_cam_to_xyz_d50_[7]), 0.1f);

  static constexpr float kExpectedCameraToXyzD50[9] = {
      2.6879040f, 0.0538000f, 0.1229680f,
      1.0163880f, 0.8399000f, -0.2334720f,
      0.0430920f, -0.4061000f, 1.8512081f,
  };

  for (int i = 0; i < 9; ++i) {
    EXPECT_NEAR(params.color_temp_cam_to_xyz_d50_[i], kExpectedCameraToXyzD50[i], 1e-3f)
        << "matrix index " << i;
  }
}

}  // namespace
}  // namespace alcedo
