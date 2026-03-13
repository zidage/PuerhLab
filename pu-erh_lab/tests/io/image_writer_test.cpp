//  Copyright 2026 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#include "io/image/image_writer.hpp"

#include <gtest/gtest.h>

#include <exiv2/exiv2.hpp>

#include <filesystem>
#include <memory>
#include <opencv2/imgcodecs.hpp>

#include "image/image_buffer.hpp"

namespace puerhlab {
namespace {

class ImageWriterTests : public ::testing::Test {
 protected:
  std::filesystem::path temp_dir_;

  void SetUp() override {
    temp_dir_ = std::filesystem::temp_directory_path() / "puerhlab_image_writer_test";
    std::filesystem::create_directories(temp_dir_);
    Exiv2::LogMsg::setLevel(Exiv2::LogMsg::Level::mute);
  }

  void TearDown() override {
    std::error_code ec;
    std::filesystem::remove_all(temp_dir_, ec);
  }
};

auto MakeColorProfile(ColorUtils::ColorSpace color_space, ColorUtils::EOTF eotf)
    -> ExportColorProfileConfig {
  return ExportColorProfileConfig{color_space, eotf, 600.0f};
}

}  // namespace

TEST_F(ImageWriterTests, UltraHdrTriggerMatchesHdrJpegCombinations) {
  ExportFormatOptions jpeg_options;
  jpeg_options.format_ = ImageFormatType::JPEG;

  EXPECT_TRUE(ImageWriter::ShouldWriteUltraHdr(
      jpeg_options, MakeColorProfile(ColorUtils::ColorSpace::REC2020, ColorUtils::EOTF::ST2084)));
  EXPECT_TRUE(ImageWriter::ShouldWriteUltraHdr(
      jpeg_options, MakeColorProfile(ColorUtils::ColorSpace::REC2020, ColorUtils::EOTF::HLG)));
  EXPECT_FALSE(ImageWriter::ShouldWriteUltraHdr(
      jpeg_options,
      MakeColorProfile(ColorUtils::ColorSpace::REC709, ColorUtils::EOTF::GAMMA_2_2)));

  ExportFormatOptions png_options;
  png_options.format_ = ImageFormatType::PNG;
  EXPECT_FALSE(ImageWriter::ShouldWriteUltraHdr(
      png_options, MakeColorProfile(ColorUtils::ColorSpace::REC2020, ColorUtils::EOTF::ST2084)));

  ExportFormatOptions tiff_options;
  tiff_options.format_ = ImageFormatType::TIFF;
  EXPECT_FALSE(ImageWriter::ShouldWriteUltraHdr(
      tiff_options, MakeColorProfile(ColorUtils::ColorSpace::REC2020, ColorUtils::EOTF::HLG)));

  ExportFormatOptions exr_options;
  exr_options.format_ = ImageFormatType::EXR;
  EXPECT_FALSE(ImageWriter::ShouldWriteUltraHdr(
      exr_options, MakeColorProfile(ColorUtils::ColorSpace::REC2020, ColorUtils::EOTF::ST2084)));
}

TEST_F(ImageWriterTests, LegacyJpegExportForcesUprightOrientation) {
  const auto src_path = temp_dir_ / "source.jpg";
  const auto dst_path = temp_dir_ / "exported.jpg";

  cv::Mat bgr(1, 2, CV_8UC3);
  bgr.at<cv::Vec3b>(0, 0) = cv::Vec3b(0, 0, 255);
  bgr.at<cv::Vec3b>(0, 1) = cv::Vec3b(0, 255, 0);
  ASSERT_TRUE(cv::imwrite(src_path.string(), bgr));

  {
    auto image = Exiv2::ImageFactory::open(src_path.string());
    ASSERT_TRUE(image != nullptr);
    image->readMetadata();
    Exiv2::ExifData exif_data = image->exifData();
    exif_data["Exif.Image.Orientation"] = static_cast<uint16_t>(6);
    image->setExifData(exif_data);
    image->writeMetadata();
  }

  cv::Mat rgba32f(1, 2, CV_32FC4);
  rgba32f.at<cv::Vec4f>(0, 0) = cv::Vec4f(1.0f, 0.0f, 0.0f, 1.0f);
  rgba32f.at<cv::Vec4f>(0, 1) = cv::Vec4f(0.0f, 1.0f, 0.0f, 1.0f);

  auto image_data = std::make_shared<ImageBuffer>(std::move(rgba32f));

  ExportFormatOptions options;
  options.format_ = ImageFormatType::JPEG;
  options.export_path_ = dst_path;

  ImageWriter::WriteImageToPath(
      src_path, image_data, options,
      ExportColorProfileConfig{ColorUtils::ColorSpace::REC709, ColorUtils::EOTF::GAMMA_2_2,
                               100.0f});

  ASSERT_TRUE(std::filesystem::exists(dst_path));

  auto output = Exiv2::ImageFactory::open(dst_path.string());
  ASSERT_TRUE(output != nullptr);
  output->readMetadata();

  const Exiv2::ExifData& exif_data = output->exifData();
  const auto orientation = exif_data.findKey(Exiv2::ExifKey("Exif.Image.Orientation"));
  if (orientation != exif_data.end()) {
    EXPECT_EQ(orientation->toString(), "1");
  }

  const cv::Mat decoded = cv::imread(dst_path.string(), cv::IMREAD_COLOR);
  ASSERT_FALSE(decoded.empty());
  EXPECT_EQ(decoded.cols, 2);
  EXPECT_EQ(decoded.rows, 1);
}

}  // namespace puerhlab
