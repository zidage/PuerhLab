//  Copyright 2026 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#ifdef HAVE_METAL

#include <gtest/gtest.h>
#include <opencv2/core.hpp>

#include <cmath>

#include "image/metal_image.hpp"
#include "metal/metal_utils/metal_convert_utils.hpp"

namespace puerhlab::metal {
namespace {

auto MetalAvailable() -> bool {
  auto* device = MTL::CreateSystemDefaultDevice();
  if (device == nullptr) {
    return false;
  }
  device->release();
  return true;
}

auto MakeRampU16C1() -> cv::Mat {
  cv::Mat image(2, 5, CV_16UC1);
  for (int y = 0; y < image.rows; ++y) {
    for (int x = 0; x < image.cols; ++x) {
      image.at<uint16_t>(y, x) = static_cast<uint16_t>((y * image.cols + x) * 1024U);
    }
  }
  return image;
}

auto MakeRampU16C4() -> cv::Mat {
  cv::Mat image(2, 3, CV_16UC4);
  for (int y = 0; y < image.rows; ++y) {
    for (int x = 0; x < image.cols; ++x) {
      image.at<cv::Vec<uint16_t, 4>>(y, x) =
          cv::Vec<uint16_t, 4>(x + y, x + y + 1, x + y + 2, 65535);
    }
  }
  return image;
}

}  // namespace

TEST(MetalImageTest, UploadDownloadRoundTripForIntegerFormats) {
  if (!MetalAvailable()) {
    GTEST_SKIP() << "Metal device is unavailable in this environment.";
  }
  MetalImage gray;
  gray.Upload(MakeRampU16C1());
  cv::Mat gray_roundtrip;
  gray.Download(gray_roundtrip);
  EXPECT_EQ(cv::countNonZero(gray_roundtrip != MakeRampU16C1()), 0);

  MetalImage rgba;
  rgba.Upload(MakeRampU16C4());
  cv::Mat rgba_roundtrip;
  rgba.Download(rgba_roundtrip);
  EXPECT_EQ(cv::countNonZero(rgba_roundtrip.reshape(1) != MakeRampU16C4().reshape(1)), 0);
}

TEST(MetalImageTest, CopyToPreservesContent) {
  if (!MetalAvailable()) {
    GTEST_SKIP() << "Metal device is unavailable in this environment.";
  }
  MetalImage source;
  source.Upload(MakeRampU16C4());

  MetalImage copied;
  source.CopyTo(copied);

  cv::Mat roundtrip;
  copied.Download(roundtrip);
  EXPECT_EQ(cv::countNonZero(roundtrip.reshape(1) != MakeRampU16C4().reshape(1)), 0);
}

TEST(MetalImageTest, ConvertToMatchesScaleAndShift) {
  if (!MetalAvailable()) {
    GTEST_SKIP() << "Metal device is unavailable in this environment.";
  }
  MetalImage source;
  source.Upload(MakeRampU16C1());

  MetalImage converted;
  source.ConvertTo(converted, PixelFormat::R32FLOAT, 1.0 / 65535.0, 0.25);

  cv::Mat roundtrip;
  converted.Download(roundtrip);
  ASSERT_EQ(roundtrip.type(), CV_32FC1);
  EXPECT_NEAR(roundtrip.at<float>(0, 0), 0.25f, 1e-6f);
  EXPECT_NEAR(roundtrip.at<float>(1, 4),
              static_cast<float>(9.0 * 1024.0 / 65535.0 + 0.25), 1e-5f);
}

TEST(MetalImageTest, RejectsUnsupportedThreeChannelFormats) {
  cv::Mat rgb(2, 2, CV_16UC3, cv::Scalar::all(1));
  MetalImage image;
  EXPECT_THROW(image.Upload(rgb), std::invalid_argument);
}

TEST(MetalUtilsTest, UtilityLayerPerformsConversion) {
  if (!MetalAvailable()) {
    GTEST_SKIP() << "Metal device is unavailable in this environment.";
  }
  MetalImage source;
  source.Upload(MakeRampU16C1());

  MetalImage converted = MetalImage::Create2D(static_cast<uint32_t>(source.Width()),
                                              static_cast<uint32_t>(source.Height()),
                                              PixelFormat::R32FLOAT);
  utils::ConvertTexture(source, converted, 1.0 / 65535.0, 0.0);

  cv::Mat roundtrip;
  converted.Download(roundtrip);
  ASSERT_EQ(roundtrip.type(), CV_32FC1);
  EXPECT_NEAR(roundtrip.at<float>(1, 4), static_cast<float>(9.0 * 1024.0 / 65535.0), 1e-5f);
}

}  // namespace puerhlab::metal

#endif
