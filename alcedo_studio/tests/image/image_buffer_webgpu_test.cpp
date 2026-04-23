//  Copyright 2026 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#ifdef HAVE_WEBGPU

#include <gtest/gtest.h>

#include <cmath>
#include <cstring>
#include <string>

#include "image/image_buffer.hpp"
#include "image/webgpu_context.hpp"

namespace alcedo {
namespace {

auto MakeRampU16C1() -> cv::Mat {
  cv::Mat image(3, 257, CV_16UC1);
  for (int y = 0; y < image.rows; ++y) {
    for (int x = 0; x < image.cols; ++x) {
      image.at<uint16_t>(y, x) = static_cast<uint16_t>(y * image.cols + x);
    }
  }
  return image;
}

auto MakeRampF32C4() -> cv::Mat {
  cv::Mat image(2, 5, CV_32FC4);
  for (int y = 0; y < image.rows; ++y) {
    for (int x = 0; x < image.cols; ++x) {
      image.at<cv::Vec4f>(y, x) =
          cv::Vec4f(static_cast<float>(x), static_cast<float>(y), 0.25f, 1.0f);
    }
  }
  return image;
}

auto WebGpuAvailable() -> bool {
  try {
    return webgpu::WebGpuContext::Instance().IsAvailable();
  } catch (...) {
    return false;
  }
}

auto WebGpuInitializationLog() -> const std::string& {
  static const std::string unavailable = "WebGPU context initialization threw before logging.";
  try {
    return webgpu::WebGpuContext::Instance().InitializationLog();
  } catch (...) {
    return unavailable;
  }
}

auto MatsEqual(const cv::Mat& lhs, const cv::Mat& rhs) -> bool {
  if (lhs.size() != rhs.size() || lhs.type() != rhs.type()) {
    return false;
  }
  for (int y = 0; y < lhs.rows; ++y) {
    const auto* left  = lhs.ptr<uint8_t>(y);
    const auto* right = rhs.ptr<uint8_t>(y);
    if (std::memcmp(left, right, static_cast<size_t>(lhs.cols) * lhs.elemSize()) != 0) {
      return false;
    }
  }
  return true;
}

}  // namespace

TEST(ImageBufferWebGpuTest, SyncRoundTripHandlesPaddedRows) {
  SCOPED_TRACE(WebGpuInitializationLog());
  if (!WebGpuAvailable()) {
    GTEST_SKIP() << "WebGPU device is unavailable in this environment.\n"
                 << WebGpuInitializationLog();
  }

  const auto  expected = MakeRampU16C1();
  ImageBuffer buffer{expected.clone()};

  buffer.SyncToGPU(GpuBackendKind::WebGPU);
  EXPECT_TRUE(buffer.gpu_data_valid_);
  EXPECT_EQ(buffer.GetGPUBackend(), GpuBackendKind::WebGPU);
  EXPECT_EQ(buffer.GetGPUWidth(), expected.cols);
  EXPECT_EQ(buffer.GetGPUHeight(), expected.rows);
  EXPECT_EQ(buffer.GetGPUType(), expected.type());

  ImageBuffer gpu_copy;
  gpu_copy.InitGPUData(buffer.GetGPUWidth(), buffer.GetGPUHeight(), buffer.GetGPUType(),
                       GpuBackendKind::WebGPU);
  buffer.CopyGPUDataTo(gpu_copy);
  gpu_copy.SyncToCPU();
  EXPECT_TRUE(MatsEqual(gpu_copy.GetCPUData(), expected));
}

TEST(ImageBufferWebGpuTest, ConvertGPUDataFallsBackThroughHostAndPreservesValues) {
  SCOPED_TRACE(WebGpuInitializationLog());
  if (!WebGpuAvailable()) {
    GTEST_SKIP() << "WebGPU device is unavailable in this environment.\n"
                 << WebGpuInitializationLog();
  }

  ImageBuffer buffer{MakeRampU16C1()};
  buffer.SyncToGPU(GpuBackendKind::WebGPU);
  buffer.ConvertGPUDataTo(CV_32FC1, 1.0 / 65535.0, 0.0);
  buffer.SyncToCPU();

  const auto& converted = buffer.GetCPUData();
  ASSERT_EQ(converted.type(), CV_32FC1);
  EXPECT_NEAR(converted.at<float>(0, 0), 0.0f, 1e-6f);
  EXPECT_NEAR(converted.at<float>(2, 256), static_cast<float>(770.0 / 65535.0), 1e-6f);
}

TEST(ImageBufferWebGpuTest, SupportsRgba32FloatRoundTrip) {
  SCOPED_TRACE(WebGpuInitializationLog());
  if (!WebGpuAvailable()) {
    GTEST_SKIP() << "WebGPU device is unavailable in this environment.\n"
                 << WebGpuInitializationLog();
  }

  const auto  expected = MakeRampF32C4();
  ImageBuffer buffer{expected.clone()};
  buffer.SyncToGPU(GpuBackendKind::WebGPU);
  buffer.ReleaseCPUData();
  buffer.SyncToCPU();

  const auto& actual = buffer.GetCPUData();
  ASSERT_EQ(actual.type(), expected.type());
  ASSERT_EQ(actual.size(), expected.size());
  for (int y = 0; y < actual.rows; ++y) {
    for (int x = 0; x < actual.cols; ++x) {
      const auto got = actual.at<cv::Vec4f>(y, x);
      const auto exp = expected.at<cv::Vec4f>(y, x);
      for (int c = 0; c < 4; ++c) {
        EXPECT_FLOAT_EQ(got[c], exp[c]);
      }
    }
  }
}

}  // namespace alcedo

#endif
