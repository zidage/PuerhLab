//  Copyright 2026 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#include "io/image/image_writer.hpp"

#include <OpenImageIO/imageio.h>
#include <gtest/gtest.h>

#include <exiv2/exiv2.hpp>

#include <filesystem>
#include <fstream>
#include <memory>
#include <opencv2/imgcodecs.hpp>
#include <vector>

#if defined(PUERHLAB_HAS_ULTRAHDR)
#include <ultrahdr_api.h>
#endif

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

auto HasEmbeddedIccProfile(const std::filesystem::path& path) -> bool {
  OIIO_NAMESPACE_USING

  auto input = ImageInput::open(path.string());
  if (!input) {
    return false;
  }

  const auto& spec = input->spec();
  const auto* attr = spec.find_attribute("ICCProfile");
  const bool  has_icc = attr != nullptr;
  input->close();
  return has_icc;
}

void WriteTestJpeg(const std::filesystem::path& path, const std::vector<uint8_t>& rgb,
                   int width, int height) {
  OIIO_NAMESPACE_USING

  ImageSpec spec(width, height, 3, TypeDesc::UINT8);
  spec.channelnames = {"R", "G", "B"};
  std::unique_ptr<ImageOutput> output = ImageOutput::create(path.string());
  ASSERT_TRUE(output != nullptr);
  ASSERT_TRUE(output->open(path.string(), spec));
  ASSERT_TRUE(output->write_image(TypeDesc::UINT8, rgb.data()));
  ASSERT_TRUE(output->close());
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

  jpeg_options.hdr_export_mode_ = ExportFormatOptions::HDR_EXPORT_MODE::EMBEDDED_PROFILE_ONLY;
  EXPECT_FALSE(ImageWriter::ShouldWriteUltraHdr(
      jpeg_options, MakeColorProfile(ColorUtils::ColorSpace::REC2020, ColorUtils::EOTF::ST2084)));

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

  WriteTestJpeg(src_path, {255, 0, 0, 0, 255, 0}, 2, 1);

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

  {
    OIIO_NAMESPACE_USING
    auto decoded = ImageInput::open(dst_path.string());
    ASSERT_TRUE(decoded != nullptr);
    const auto& spec = decoded->spec();
    EXPECT_EQ(spec.width, 2);
    EXPECT_EQ(spec.height, 1);
    decoded->close();
  }
}

TEST_F(ImageWriterTests, EmbeddedHdrIccModeWritesRegularJpegWithProfile) {
  const auto src_path = temp_dir_ / "hdr_source.jpg";
  const auto dst_path = temp_dir_ / "embedded_hdr.jpg";

  WriteTestJpeg(src_path, {
                           144, 96, 48, 144, 96, 48,
                           144, 96, 48, 144, 96, 48,
                         }, 2, 2);

  cv::Mat rgba32f(2, 2, CV_32FC4, cv::Scalar(0.65f, 0.35f, 0.15f, 1.0f));
  auto    image_data = std::make_shared<ImageBuffer>(std::move(rgba32f));

  ExportFormatOptions options;
  options.format_ = ImageFormatType::JPEG;
  options.export_path_ = dst_path;
  options.hdr_export_mode_ = ExportFormatOptions::HDR_EXPORT_MODE::EMBEDDED_PROFILE_ONLY;

  const auto hdr_profile =
      MakeColorProfile(ColorUtils::ColorSpace::REC2020, ColorUtils::EOTF::ST2084);

  ImageWriter::WriteImageToPath(src_path, image_data, options, hdr_profile);

  ASSERT_TRUE(std::filesystem::exists(dst_path));
  EXPECT_TRUE(HasEmbeddedIccProfile(dst_path));

#if defined(PUERHLAB_HAS_ULTRAHDR)
  std::ifstream in(dst_path, std::ios::binary);
  ASSERT_TRUE(in.is_open());
  const auto bytes =
      std::vector<uint8_t>(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
  ASSERT_FALSE(bytes.empty());
  EXPECT_EQ(is_uhdr_image(const_cast<uint8_t*>(bytes.data()), static_cast<int>(bytes.size())), 0);
#endif
}

}  // namespace puerhlab
