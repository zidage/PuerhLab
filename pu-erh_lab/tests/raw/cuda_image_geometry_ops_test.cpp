#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/imgproc.hpp>

#include "decoders/processor/operators/gpu/cuda_image_ops.hpp"
#include "decoders/processor/operators/gpu/cuda_rotate.hpp"
#include "edit/operators/geometry/cuda_geometry_ops.hpp"

namespace puerhlab {
namespace {

auto EnsureCudaDevice() -> bool {
  const int device_count = cv::cuda::getCudaEnabledDeviceCount();
  if (device_count <= 0) {
    return false;
  }
  cv::cuda::setDevice(0);
  return true;
}

auto MakeGradientMat(int rows, int cols, int type) -> cv::Mat {
  cv::Mat out(rows, cols, type);
  for (int y = 0; y < rows; ++y) {
    for (int x = 0; x < cols; ++x) {
      const float fx = static_cast<float>(x) / static_cast<float>(std::max(cols - 1, 1));
      const float fy = static_cast<float>(y) / static_cast<float>(std::max(rows - 1, 1));
      if (type == CV_32FC1) {
        out.at<float>(y, x) = 0.25f * fx + 0.75f * fy;
      } else if (type == CV_32FC3) {
        out.at<cv::Vec3f>(y, x) = cv::Vec3f(fx, fy, 0.5f * (fx + fy));
      } else if (type == CV_32FC4) {
        out.at<cv::Vec4f>(y, x) = cv::Vec4f(fx, fy, 0.5f * (fx + fy), 1.0f);
      }
    }
  }
  return out;
}

auto MeanAbsError(const cv::Mat& a, const cv::Mat& b) -> double {
  cv::Mat diff;
  cv::absdiff(a, b, diff);
  cv::Scalar m = cv::mean(diff);
  double total = 0.0;
  for (int c = 0; c < a.channels(); ++c) {
    total += m[c];
  }
  return total / static_cast<double>(a.channels());
}

}  // namespace

TEST(CudaImageOpsTest, MergeRgbAndRgbToRgbaMatchesCpuReference) {
  if (!EnsureCudaDevice()) {
    GTEST_SKIP() << "No CUDA device available.";
  }

  cv::Mat red(3, 4, CV_32FC1);
  cv::Mat green(3, 4, CV_32FC1);
  cv::Mat blue(3, 4, CV_32FC1);
  for (int y = 0; y < red.rows; ++y) {
    for (int x = 0; x < red.cols; ++x) {
      red.at<float>(y, x)   = static_cast<float>(x + y);
      green.at<float>(y, x) = static_cast<float>(x * 2 + y);
      blue.at<float>(y, x)  = static_cast<float>(x + y * 3);
    }
  }

  cv::cuda::GpuMat d_red(red);
  cv::cuda::GpuMat d_green(green);
  cv::cuda::GpuMat d_blue(blue);
  cv::cuda::GpuMat d_merged;
  CUDA::MergeRGB(d_red, d_green, d_blue, d_merged);

  cv::Mat merged_cpu;
  d_merged.download(merged_cpu);
  ASSERT_EQ(merged_cpu.type(), CV_32FC3);

  for (int y = 0; y < merged_cpu.rows; ++y) {
    for (int x = 0; x < merged_cpu.cols; ++x) {
      const cv::Vec3f v = merged_cpu.at<cv::Vec3f>(y, x);
      EXPECT_FLOAT_EQ(v[0], red.at<float>(y, x));
      EXPECT_FLOAT_EQ(v[1], green.at<float>(y, x));
      EXPECT_FLOAT_EQ(v[2], blue.at<float>(y, x));
    }
  }

  CUDA::RGBToRGBA(d_merged);
  cv::Mat rgba_cpu;
  d_merged.download(rgba_cpu);
  ASSERT_EQ(rgba_cpu.type(), CV_32FC4);
  for (int y = 0; y < rgba_cpu.rows; ++y) {
    for (int x = 0; x < rgba_cpu.cols; ++x) {
      const cv::Vec4f v = rgba_cpu.at<cv::Vec4f>(y, x);
      EXPECT_FLOAT_EQ(v[0], red.at<float>(y, x));
      EXPECT_FLOAT_EQ(v[1], green.at<float>(y, x));
      EXPECT_FLOAT_EQ(v[2], blue.at<float>(y, x));
      EXPECT_FLOAT_EQ(v[3], 1.0f);
    }
  }
}

TEST(CudaImageOpsTest, Rotate90And180MatchesCpuRotate) {
  if (!EnsureCudaDevice()) {
    GTEST_SKIP() << "No CUDA device available.";
  }

  const cv::Mat src = MakeGradientMat(5, 7, CV_32FC1);

  cv::cuda::GpuMat d_img(src);
  CUDA::Rotate90CW(d_img);
  cv::Mat cw_out;
  d_img.download(cw_out);
  cv::Mat cw_ref;
  cv::rotate(src, cw_ref, cv::ROTATE_90_CLOCKWISE);
  EXPECT_LT(MeanAbsError(cw_out, cw_ref), 1e-6);

  d_img.upload(src);
  CUDA::Rotate90CCW(d_img);
  cv::Mat ccw_out;
  d_img.download(ccw_out);
  cv::Mat ccw_ref;
  cv::rotate(src, ccw_ref, cv::ROTATE_90_COUNTERCLOCKWISE);
  EXPECT_LT(MeanAbsError(ccw_out, ccw_ref), 1e-6);

  d_img.upload(src);
  CUDA::Rotate180(d_img);
  cv::Mat r180_out;
  d_img.download(r180_out);
  cv::Mat r180_ref;
  cv::rotate(src, r180_ref, cv::ROTATE_180);
  EXPECT_LT(MeanAbsError(r180_out, r180_ref), 1e-6);
}

TEST(CudaGeometryOpsTest, ResizeAreaApproxVisuallyMatchesCpuInterArea) {
  if (!EnsureCudaDevice()) {
    GTEST_SKIP() << "No CUDA device available.";
  }

  const cv::Mat src = MakeGradientMat(96, 144, CV_32FC3);

  cv::cuda::GpuMat d_src(src);
  cv::cuda::GpuMat d_dst;
  CUDA::ResizeAreaApprox(d_src, d_dst, cv::Size(31, 19));

  cv::Mat gpu_out;
  d_dst.download(gpu_out);

  cv::Mat cpu_ref;
  cv::resize(src, cpu_ref, cv::Size(31, 19), 0.0, 0.0, cv::INTER_AREA);

  EXPECT_LT(MeanAbsError(gpu_out, cpu_ref), 2e-3);
}

TEST(CudaGeometryOpsTest, WarpAffineLinearVisuallyMatchesCpuWarpAffine) {
  if (!EnsureCudaDevice()) {
    GTEST_SKIP() << "No CUDA device available.";
  }

  const cv::Mat src = MakeGradientMat(80, 120, CV_32FC4);

  const cv::Mat matrix = (cv::Mat_<double>(2, 3) << 0.98, -0.12, 5.0, 0.12, 0.98, -3.0);
  const cv::Size out_size(96, 64);
  const cv::Scalar border(0.0, 0.0, 0.0, 1.0);

  cv::cuda::GpuMat d_src(src);
  cv::cuda::GpuMat d_dst;
  CUDA::WarpAffineLinear(d_src, d_dst, matrix, out_size, border);

  cv::Mat gpu_out;
  d_dst.download(gpu_out);

  cv::Mat cpu_ref;
  cv::warpAffine(src, cpu_ref, matrix, out_size, cv::INTER_LINEAR | cv::WARP_INVERSE_MAP,
                 cv::BORDER_CONSTANT, border);

  EXPECT_LT(MeanAbsError(gpu_out, cpu_ref), 3e-3);
}

}  // namespace puerhlab
