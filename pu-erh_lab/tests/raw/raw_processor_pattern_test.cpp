//  Copyright 2026 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#include <gtest/gtest.h>
#include <libraw/libraw.h>

#include <filesystem>
#include <fstream>
#include <limits>
#include <optional>
#include <string_view>
#include <utility>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>

#include "decoders/processor/raw_processor_internal.hpp"
#include "decoders/processor/raw_processor_pattern.hpp"
#include "edit/operators/raw/raw_decode_op.hpp"
#include "image/image_buffer.hpp"
#include "type/type.hpp"

namespace puerhlab {
namespace {

struct SampleSelection {
  std::filesystem::path path;
  BayerPattern2x2       pattern;
  libraw_image_sizes_t  sizes;
};

auto MakeXTransPattern() -> XTransPattern6x6 {
  static constexpr int kRawFc[36] = {
      1, 2, 1, 1, 0, 1,
      1, 0, 1, 2, 1, 2,
      0, 1, 0, 1, 2, 1,
      1, 2, 1, 1, 0, 1,
      1, 0, 1, 2, 1, 2,
      2, 1, 2, 0, 1, 0,
  };

  XTransPattern6x6 pattern = {};
  for (int i = 0; i < 36; ++i) {
    pattern.raw_fc[i] = kRawFc[i];
    pattern.rgb_fc[i] = FoldRawColorToRgb(kRawFc[i]);
  }
  return pattern;
}

auto ReadFileToBuffer(const std::filesystem::path& path) -> std::vector<uint8_t> {
  std::ifstream file(path, std::ios::binary);
  if (!file.is_open()) {
    return {};
  }

  file.seekg(0, std::ios::end);
  const std::streamsize size = file.tellg();
  file.seekg(0, std::ios::beg);
  if (size <= 0) {
    return {};
  }

  std::vector<uint8_t> buffer(static_cast<size_t>(size));
  if (!file.read(reinterpret_cast<char*>(buffer.data()), size)) {
    return {};
  }
  return buffer;
}

auto IsRawExtension(const std::filesystem::path& path) -> bool {
  const auto ext = path.extension().wstring();
  return ext == L".ARW" || ext == L".arw" || ext == L".NEF" || ext == L".nef" ||
         ext == L".DNG" || ext == L".dng" || ext == L".CR2" || ext == L".cr2" ||
         ext == L".RW2" || ext == L".rw2" || ext == L".ORF" || ext == L".orf";
}

auto CandidateNonRggbSamplePaths() -> std::vector<std::filesystem::path> {
  const auto root = std::filesystem::path(TEST_IMG_PATH) / "raw";
  return {
      root / "om1.dng",
      root / "camera" / "lumix" / "s1" / "P1000799.RW2",
      root / "camera" / "lumix" / "s5" / "P1000625.RW2",
      root / "camera" / "lumix" / "s9" / "P1000494.RW2",
      root / "camera" / "leica" / "m10" / "L1001108.dng",
      root / "camera" / "leica" / "q3" / "L1000222.DNG",
      root / "camera" / "ricoh" / "gr3" / "R0000948.DNG",
      root / "camera" / "hasselblad" / "x2d" / "B0004841.dng",
  };
}

auto TrySelectNonRggbSample(const std::filesystem::path& path) -> std::optional<SampleSelection> {
  LibRaw raw_processor;
  if (raw_processor.open_file(path.string().c_str()) != LIBRAW_SUCCESS) {
    return std::nullopt;
  }
  if (raw_processor.unpack() != LIBRAW_SUCCESS) {
    raw_processor.recycle();
    return std::nullopt;
  }

  const RawInputKind input_kind =
      ClassifyRawInput(raw_processor.imgdata.rawdata, raw_processor.imgdata.idata);
  if (input_kind != RawInputKind::BayerRaw || raw_processor.imgdata.idata.is_foveon != 0U ||
      raw_processor.imgdata.idata.filters == 0U || raw_processor.imgdata.idata.filters == 1U ||
      raw_processor.imgdata.idata.filters == 9U || raw_processor.is_fuji_rotated() != 0) {
    raw_processor.recycle();
    return std::nullopt;
  }

  const BayerPattern2x2 pattern = ReadLibRawBayerPattern(raw_processor);
  if (!IsClassic2x2Bayer(pattern) || IsRGGBPattern(pattern)) {
    raw_processor.recycle();
    return std::nullopt;
  }

  SampleSelection result = {
      .path    = path,
      .pattern = pattern,
      .sizes   = raw_processor.imgdata.sizes,
  };
  raw_processor.recycle();
  return result;
}

auto FindFirstNonRggbBayerSample() -> std::optional<SampleSelection> {
  for (const auto& candidate : CandidateNonRggbSamplePaths()) {
    if (!std::filesystem::exists(candidate) || !IsRawExtension(candidate)) {
      continue;
    }
    if (auto sample = TrySelectNonRggbSample(candidate)) {
      return sample;
    }
  }

  return std::nullopt;
}

auto ExpectedHalfDecodeSize(const libraw_image_sizes_t& sizes) -> cv::Size {
  int width  = static_cast<int>(sizes.raw_width) / 2;
  int height = static_cast<int>(sizes.raw_height) / 2;
  if (sizes.flip == 5 || sizes.flip == 6) {
    std::swap(width, height);
  }
  return {width, height};
}

auto LinearDngSamplePath() -> std::filesystem::path {
  return std::filesystem::path(TEST_IMG_PATH) / "raw" / "linear_dng" / "mfzoty.dng";
}

auto EnsureCudaDevice() -> bool {
  const int device_count = cv::cuda::getCudaEnabledDeviceCount();
  if (device_count <= 0) {
    return false;
  }
  cv::cuda::setDevice(0);
  return true;
}

auto CandidateLargeBayerSamplePaths() -> std::vector<std::filesystem::path> {
  const auto root = std::filesystem::path(TEST_IMG_PATH) / "raw";
  return {
      root / "camera" / "nikon" / "z8" / "DSC_1211.dng",
      root / "camera" / "sony" / "a7rv" / "DSC00064.ARW",
  };
}

auto TrySelectLargeBayerSample(const std::filesystem::path& path) -> std::optional<SampleSelection> {
  LibRaw raw_processor;
  if (raw_processor.open_file(path.string().c_str()) != LIBRAW_SUCCESS) {
    return std::nullopt;
  }
  if (raw_processor.unpack() != LIBRAW_SUCCESS) {
    raw_processor.recycle();
    return std::nullopt;
  }

  const RawInputKind input_kind =
      ClassifyRawInput(raw_processor.imgdata.rawdata, raw_processor.imgdata.idata);
  if (input_kind != RawInputKind::BayerRaw || raw_processor.imgdata.idata.is_foveon != 0U ||
      raw_processor.imgdata.idata.filters == 0U || raw_processor.imgdata.idata.filters == 1U ||
      raw_processor.is_fuji_rotated() != 0) {
    raw_processor.recycle();
    return std::nullopt;
  }

  const RawCfaPattern cfa_pattern = ReadLibRawCfaPattern(raw_processor);
  if (cfa_pattern.kind != RawCfaKind::Bayer2x2) {
    raw_processor.recycle();
    return std::nullopt;
  }

  const cv::Rect active_rect =
      detail::BuildDecodeCropRect(raw_processor.imgdata.sizes,
                                  cv::Size(static_cast<int>(raw_processor.imgdata.sizes.raw_width),
                                           static_cast<int>(raw_processor.imgdata.sizes.raw_height)),
                                  DecodeRes::FULL);
  const int long_edge = std::max(active_rect.width, active_rect.height);
  if (long_edge <= detail::kCudaTileThresholdLongEdge) {
    raw_processor.recycle();
    return std::nullopt;
  }

  SampleSelection result = {
      .path    = path,
      .pattern = cfa_pattern.bayer_pattern,
      .sizes   = raw_processor.imgdata.sizes,
  };
  raw_processor.recycle();
  return result;
}

auto FindLargeBayerSample() -> std::optional<SampleSelection> {
  for (const auto& candidate : CandidateLargeBayerSamplePaths()) {
    if (!std::filesystem::exists(candidate)) {
      continue;
    }
    if (auto sample = TrySelectLargeBayerSample(candidate)) {
      return sample;
    }
  }
  return std::nullopt;
}

auto MaxAbsDiff(const cv::Mat& lhs, const cv::Mat& rhs) -> float {
  if (lhs.size() != rhs.size() || lhs.type() != rhs.type()) {
    return std::numeric_limits<float>::infinity();
  }

  float max_diff = 0.0f;
  for (int y = 0; y < lhs.rows; ++y) {
    const cv::Vec4f* lhs_row = lhs.ptr<cv::Vec4f>(y);
    const cv::Vec4f* rhs_row = rhs.ptr<cv::Vec4f>(y);
    for (int x = 0; x < lhs.cols; ++x) {
      const cv::Vec4f delta = lhs_row[x] - rhs_row[x];
      max_diff = std::max(max_diff, std::abs(delta[0]));
      max_diff = std::max(max_diff, std::abs(delta[1]));
      max_diff = std::max(max_diff, std::abs(delta[2]));
      max_diff = std::max(max_diff, std::abs(delta[3]));
    }
  }
  return max_diff;
}

auto MaxAbsDiffNearTileSeams(const cv::Mat& lhs, const cv::Mat& rhs) -> float {
  if (lhs.size() != rhs.size() || lhs.type() != rhs.type()) {
    return std::numeric_limits<float>::infinity();
  }

  const int seam_stride = detail::kCudaTileInnerSize;
  float     max_diff    = 0.0f;
  for (int y = 0; y < lhs.rows; ++y) {
    const cv::Vec4f* lhs_row = lhs.ptr<cv::Vec4f>(y);
    const cv::Vec4f* rhs_row = rhs.ptr<cv::Vec4f>(y);
    const bool on_y_seam = y > 0 && (y % seam_stride == 0 || y % seam_stride == 1 ||
                                     y % seam_stride == seam_stride - 1);
    for (int x = 0; x < lhs.cols; ++x) {
      const bool on_x_seam = x > 0 && (x % seam_stride == 0 || x % seam_stride == 1 ||
                                       x % seam_stride == seam_stride - 1);
      if (!on_x_seam && !on_y_seam) {
        continue;
      }
      const cv::Vec4f delta = lhs_row[x] - rhs_row[x];
      max_diff = std::max(max_diff, std::abs(delta[0]));
      max_diff = std::max(max_diff, std::abs(delta[1]));
      max_diff = std::max(max_diff, std::abs(delta[2]));
      max_diff = std::max(max_diff, std::abs(delta[3]));
    }
  }
  return max_diff;
}

auto DecodeLargeRawWithMode(const std::filesystem::path& path, const detail::CudaExecutionMode mode,
                            const bool highlights_reconstruct) -> cv::Mat {
  std::vector<uint8_t> raw_bytes = ReadFileToBuffer(path);
  if (raw_bytes.empty()) {
    return {};
  }

  auto input = std::make_shared<ImageBuffer>(std::move(raw_bytes));
  nlohmann::json decode_params;
  decode_params["raw"] = {{"gpu_backend", "gpu"},
                          {"highlights_reconstruct", highlights_reconstruct},
                          {"use_camera_wb", true},
                          {"backend", "puerh"},
                          {"decode_res", static_cast<int>(DecodeRes::FULL)}};

  struct ModeOverrideGuard {
    ~ModeOverrideGuard() { detail::SetCudaExecutionModeOverrideForTesting(std::nullopt); }
  } guard;

  detail::SetCudaExecutionModeOverrideForTesting(mode);
  RawDecodeOp raw_decode_op(decode_params);
  raw_decode_op.Apply(input);

  cv::Mat decoded;
  input->GetCUDAImage().download(decoded);
  return decoded;
}

}  // namespace

TEST(RawProcessorPattern, DescribeAndValidateClassicPatterns) {
  const BayerPattern2x2 rggb = {{0, 1, 3, 2}, {0, 1, 1, 2}};
  const BayerPattern2x2 grbg = {{1, 0, 2, 3}, {1, 0, 2, 1}};
  const BayerPattern2x2 gbrg = {{1, 2, 0, 3}, {1, 2, 0, 1}};
  const BayerPattern2x2 bggr = {{2, 1, 3, 0}, {2, 1, 1, 0}};

  EXPECT_TRUE(IsClassic2x2Bayer(rggb));
  EXPECT_TRUE(IsClassic2x2Bayer(grbg));
  EXPECT_TRUE(IsClassic2x2Bayer(gbrg));
  EXPECT_TRUE(IsClassic2x2Bayer(bggr));

  EXPECT_TRUE(IsRGGBPattern(rggb));
  EXPECT_FALSE(IsRGGBPattern(grbg));
  EXPECT_FALSE(IsRGGBPattern(gbrg));
  EXPECT_FALSE(IsRGGBPattern(bggr));

  EXPECT_EQ(DescribeBayerPattern(rggb), "RGGB");
  EXPECT_EQ(DescribeBayerPattern(grbg), "GRBG");
  EXPECT_EQ(DescribeBayerPattern(gbrg), "GBRG");
  EXPECT_EQ(DescribeBayerPattern(bggr), "BGGR");
}

TEST(RawProcessorPattern, ClassifyRawInputLayouts) {
  libraw_rawdata_t raw_data = {};
  libraw_iparams_t idata    = {};

  raw_data.raw_image = reinterpret_cast<ushort*>(1);
  EXPECT_EQ(ClassifyRawInput(raw_data, idata), RawInputKind::BayerRaw);

  raw_data.raw_image    = nullptr;
  raw_data.color3_image = reinterpret_cast<ushort(*)[3]>(1);
  EXPECT_EQ(ClassifyRawInput(raw_data, idata), RawInputKind::DebayeredRgb);

  raw_data.color3_image = nullptr;
  raw_data.float3_image = reinterpret_cast<float(*)[3]>(1);
  EXPECT_EQ(ClassifyRawInput(raw_data, idata), RawInputKind::DebayeredRgb);

  raw_data.float3_image = nullptr;
  raw_data.color4_image = reinterpret_cast<ushort(*)[4]>(1);
  EXPECT_EQ(ClassifyRawInput(raw_data, idata), RawInputKind::Unsupported);

  idata.colors = 3;
  EXPECT_EQ(ClassifyRawInput(raw_data, idata), RawInputKind::DebayeredRgb);
}

TEST(RawProcessorPattern, LibRawRecognizesLinearDngAsThreeColorFullImage) {
  const std::filesystem::path sample_path = LinearDngSamplePath();
  if (!std::filesystem::exists(sample_path)) {
    GTEST_SKIP() << "Linear DNG sample not found: " << sample_path.string();
  }

  LibRaw raw_processor;
  ASSERT_EQ(raw_processor.open_file(sample_path.string().c_str()), LIBRAW_SUCCESS);
  ASSERT_EQ(raw_processor.unpack(), LIBRAW_SUCCESS);

  const auto& idata   = raw_processor.imgdata.idata;
  const auto& rawdata = raw_processor.imgdata.rawdata;

  EXPECT_NE(idata.dng_version, 0U);
  EXPECT_EQ(idata.filters, 0U);
  EXPECT_EQ(idata.colors, 3);
  EXPECT_EQ(rawdata.raw_image, nullptr);
  EXPECT_NE(rawdata.color4_image, nullptr);
  EXPECT_EQ(ClassifyRawInput(rawdata, idata), RawInputKind::DebayeredRgb);

  raw_processor.recycle();
}

TEST(RawProcessorPattern, ReadLibRawDetectsXTransPattern) {
  LibRaw raw_processor;
  raw_processor.imgdata.idata.filters = 9U;
  const XTransPattern6x6 expected = MakeXTransPattern();
  for (int row = 0; row < 6; ++row) {
    for (int col = 0; col < 6; ++col) {
      raw_processor.imgdata.idata.xtrans[row][col] =
          static_cast<char>(expected.raw_fc[row * 6 + col]);
    }
  }

  const RawCfaPattern pattern = ReadLibRawCfaPattern(raw_processor);
  EXPECT_EQ(pattern.kind, RawCfaKind::XTrans6x6);
  EXPECT_EQ(RawColorAt(pattern, 0, 0), expected.raw_fc[0]);
  EXPECT_EQ(RawColorAt(pattern, 4, 5), expected.raw_fc[4 * 6 + 5]);
  EXPECT_EQ(RgbColorAt(pattern, 5, 3), expected.rgb_fc[5 * 6 + 3]);
}

TEST(RawProcessorPattern, DownsampleKeepsBayerCellParity) {
  const BayerPattern2x2 pattern = {{1, 0, 2, 3}, {1, 0, 2, 1}};

  cv::Mat src(4, 4, CV_16UC1);
  for (int y = 0; y < src.rows; ++y) {
    for (int x = 0; x < src.cols; ++x) {
      src.at<uint16_t>(y, x) = static_cast<uint16_t>(10 * y + x);
    }
  }

  const cv::Mat downsampled = DownsampleBayer2x(src, pattern);
  ASSERT_EQ(downsampled.rows, 2);
  ASSERT_EQ(downsampled.cols, 2);
  EXPECT_EQ(downsampled.at<uint16_t>(0, 0), src.at<uint16_t>(0, 0));
  EXPECT_EQ(downsampled.at<uint16_t>(0, 1), src.at<uint16_t>(0, 3));
  EXPECT_EQ(downsampled.at<uint16_t>(1, 0), src.at<uint16_t>(3, 0));
  EXPECT_EQ(downsampled.at<uint16_t>(1, 1), src.at<uint16_t>(3, 3));
}

TEST(RawProcessorPattern, DownsampleXTransKeepsSampledPhase) {
  RawCfaPattern pattern = {};
  pattern.kind          = RawCfaKind::XTrans6x6;
  pattern.xtrans_pattern = MakeXTransPattern();

  cv::Mat src(12, 12, CV_16UC1);
  for (int y = 0; y < src.rows; ++y) {
    for (int x = 0; x < src.cols; ++x) {
      src.at<uint16_t>(y, x) = static_cast<uint16_t>(100 * y + x);
    }
  }

  const XTransPattern6x6 original_pattern = pattern.xtrans_pattern;
  const cv::Mat half = DownsampleRaw2x(src, pattern);

  ASSERT_EQ(half.rows, 6);
  ASSERT_EQ(half.cols, 6);
  EXPECT_EQ(half.at<uint16_t>(0, 0), src.at<uint16_t>(0, 0));
  EXPECT_EQ(half.at<uint16_t>(1, 2), src.at<uint16_t>(2, 4));
  EXPECT_EQ(half.at<uint16_t>(5, 5), src.at<uint16_t>(10, 10));
  EXPECT_EQ(RawColorAt(pattern, 0, 0), RawColorAt(original_pattern, 0, 0));
  EXPECT_EQ(RawColorAt(pattern, 1, 1), RawColorAt(original_pattern, 2, 2));
  EXPECT_EQ(RawColorAt(pattern, 4, 5), RawColorAt(original_pattern, 8, 10));
}

TEST(RawProcessorPattern, CpuBackendRejectsNonRggbBayer) {
  const auto sample = FindFirstNonRggbBayerSample();
  if (!sample.has_value()) {
    GTEST_SKIP() << "No non-RGGB classic Bayer sample found under TEST_IMG_PATH/raw.";
  }

  std::vector<uint8_t> raw_bytes = ReadFileToBuffer(sample->path);
  ASSERT_FALSE(raw_bytes.empty());

  auto input = std::make_shared<ImageBuffer>(std::move(raw_bytes));

  nlohmann::json decode_params;
  decode_params["raw"] = {{"gpu_backend", "cpu"},
                          {"highlights_reconstruct", false},
                          {"use_camera_wb", true},
                          {"backend", "puerh"},
                          {"decode_res", static_cast<int>(DecodeRes::FULL)}};

  RawDecodeOp raw_decode_op(decode_params);
  EXPECT_THROW(raw_decode_op.Apply(input), std::runtime_error);
}

TEST(RawProcessorPattern, CudaDecodeSupportsNonRggbClassicBayer) {
#ifndef HAVE_CUDA
  GTEST_SKIP() << "CUDA is not enabled in this build.";
#else
  const auto sample = FindFirstNonRggbBayerSample();
  if (!sample.has_value()) {
    GTEST_SKIP() << "No non-RGGB classic Bayer sample found under TEST_IMG_PATH/raw.";
  }

  std::vector<uint8_t> raw_bytes = ReadFileToBuffer(sample->path);
  ASSERT_FALSE(raw_bytes.empty());

  auto input = std::make_shared<ImageBuffer>(std::move(raw_bytes));

  nlohmann::json decode_params;
  decode_params["raw"] = {{"gpu_backend", "gpu"},
                          {"highlights_reconstruct", false},
                          {"use_camera_wb", true},
                          {"backend", "puerh"},
                          {"decode_res", static_cast<int>(DecodeRes::HALF)}};

  RawDecodeOp raw_decode_op(decode_params);
  ASSERT_NO_THROW(raw_decode_op.ApplyGPU(input));

  const cv::Size expected_size = ExpectedHalfDecodeSize(sample->sizes);
  EXPECT_EQ(input->GetGPUType(), CV_32FC4);
  EXPECT_EQ(input->GetGPUWidth(), expected_size.width);
  EXPECT_EQ(input->GetGPUHeight(), expected_size.height);

  OperatorParams params;
  raw_decode_op.SetGlobalParams(params);
  EXPECT_TRUE(params.raw_runtime_valid_);
  EXPECT_EQ(params.raw_decode_input_space_, RawDecodeInputSpace::CAMERA);
#endif
}

TEST(RawProcessorPattern, CpuDecodeSupportsLinearDngSampleAndSkipsHighlightRecovery) {
  const std::filesystem::path sample_path = LinearDngSamplePath();
  if (!std::filesystem::exists(sample_path)) {
    GTEST_SKIP() << "Linear DNG sample not found: " << sample_path.string();
  }

  std::vector<uint8_t> raw_bytes = ReadFileToBuffer(sample_path);
  ASSERT_FALSE(raw_bytes.empty());

  auto input = std::make_shared<ImageBuffer>(std::move(raw_bytes));

  nlohmann::json decode_params;
  decode_params["raw"] = {{"gpu_backend", "cpu"},
                          {"highlights_reconstruct", true},
                          {"use_camera_wb", true},
                          {"backend", "puerh"},
                          {"decode_res", static_cast<int>(DecodeRes::FULL)}};

  RawDecodeOp raw_decode_op(decode_params);
  ASSERT_NO_THROW(raw_decode_op.Apply(input));

  cv::Mat& decoded = input->GetCPUData();
  ASSERT_FALSE(decoded.empty());
  EXPECT_EQ(decoded.type(), CV_32FC4);
  EXPECT_EQ(decoded.cols, 6000);
  EXPECT_EQ(decoded.rows, 4000);

  const cv::Vec4f center = decoded.at<cv::Vec4f>(decoded.rows / 2, decoded.cols / 2);
  EXPECT_GE(center[0], 0.0f);
  EXPECT_GE(center[1], 0.0f);
  EXPECT_GE(center[2], 0.0f);
  EXPECT_FLOAT_EQ(center[3], 1.0f);

  OperatorParams params;
  raw_decode_op.SetGlobalParams(params);
  EXPECT_TRUE(params.raw_runtime_valid_);
  EXPECT_EQ(params.raw_decode_input_space_, RawDecodeInputSpace::CAMERA);
}

TEST(RawProcessorPattern, GpuDecodeSupportsLinearDngSample) {
#ifndef HAVE_CUDA
  GTEST_SKIP() << "CUDA is not enabled in this build.";
#else
  const std::filesystem::path sample_path = LinearDngSamplePath();
  if (!std::filesystem::exists(sample_path)) {
    GTEST_SKIP() << "Linear DNG sample not found: " << sample_path.string();
  }

  std::vector<uint8_t> raw_bytes = ReadFileToBuffer(sample_path);
  ASSERT_FALSE(raw_bytes.empty());

  auto input = std::make_shared<ImageBuffer>(std::move(raw_bytes));

  nlohmann::json decode_params;
  decode_params["raw"] = {{"gpu_backend", "gpu"},
                          {"highlights_reconstruct", true},
                          {"use_camera_wb", true},
                          {"backend", "puerh"},
                          {"decode_res", static_cast<int>(DecodeRes::FULL)}};

  RawDecodeOp raw_decode_op(decode_params);
  ASSERT_NO_THROW(raw_decode_op.Apply(input));

  EXPECT_EQ(input->GetGPUType(), CV_32FC4);
  EXPECT_EQ(input->GetGPUWidth(), 6000);
  EXPECT_EQ(input->GetGPUHeight(), 4000);

  OperatorParams params;
  raw_decode_op.SetGlobalParams(params);
  EXPECT_TRUE(params.raw_runtime_valid_);
  EXPECT_EQ(params.raw_decode_input_space_, RawDecodeInputSpace::CAMERA);
#endif
}

TEST(RawProcessorPattern, CudaExecutionModeUsesLongEdgeThresholdForLargeBayer) {
  RawParams params;
  params.gpu_backend_ = RawGpuBackend::GPU;

  RawCfaPattern bayer_pattern = {};
  bayer_pattern.kind          = RawCfaKind::Bayer2x2;
  EXPECT_EQ(detail::SelectCudaExecutionMode(params, bayer_pattern, cv::Rect(0, 0, 8500, 6000)),
            detail::CudaExecutionMode::FullFrame);
  EXPECT_EQ(detail::SelectCudaExecutionMode(params, bayer_pattern, cv::Rect(0, 0, 8501, 6000)),
            detail::CudaExecutionMode::Tiled);
  EXPECT_EQ(detail::SelectCudaExecutionMode(params, bayer_pattern, cv::Rect(0, 0, 6000, 4000)),
            detail::CudaExecutionMode::FullFrame);

  RawCfaPattern xtrans_pattern = {};
  xtrans_pattern.kind          = RawCfaKind::XTrans6x6;
  EXPECT_EQ(detail::SelectCudaExecutionMode(params, xtrans_pattern, cv::Rect(0, 0, 10240, 7680)),
            detail::CudaExecutionMode::FullFrame);
}

TEST(RawProcessorPattern, CudaLargeBayerForcedTiledMatchesFullFrameWithoutHighlight) {
#ifndef HAVE_CUDA
  GTEST_SKIP() << "CUDA is not enabled in this build.";
#else
  if (!EnsureCudaDevice()) {
    GTEST_SKIP() << "CUDA device is unavailable in this environment.";
  }

  const auto sample = FindLargeBayerSample();
  if (!sample.has_value()) {
    GTEST_SKIP() << "No Bayer RAW sample above the CUDA tile long-edge threshold was found.";
  }

  const cv::Mat full = DecodeLargeRawWithMode(sample->path, detail::CudaExecutionMode::FullFrame,
                                              false);
  const cv::Mat tiled =
      DecodeLargeRawWithMode(sample->path, detail::CudaExecutionMode::Tiled, false);
  ASSERT_FALSE(full.empty());
  ASSERT_FALSE(tiled.empty());
  EXPECT_EQ(full.size(), tiled.size());
  EXPECT_EQ(full.type(), tiled.type());
  EXPECT_LE(MaxAbsDiff(full, tiled), 1e-6f);
  EXPECT_LE(MaxAbsDiffNearTileSeams(full, tiled), 1e-6f);
#endif
}

TEST(RawProcessorPattern, CudaLargeBayerForcedTiledMatchesFullFrameWithHighlight) {
#ifndef HAVE_CUDA
  GTEST_SKIP() << "CUDA is not enabled in this build.";
#else
  if (!EnsureCudaDevice()) {
    GTEST_SKIP() << "CUDA device is unavailable in this environment.";
  }

  const auto sample = FindLargeBayerSample();
  if (!sample.has_value()) {
    GTEST_SKIP() << "No Bayer RAW sample above the CUDA tile long-edge threshold was found.";
  }

  const cv::Mat full = DecodeLargeRawWithMode(sample->path, detail::CudaExecutionMode::FullFrame,
                                              true);
  const cv::Mat tiled =
      DecodeLargeRawWithMode(sample->path, detail::CudaExecutionMode::Tiled, true);
  ASSERT_FALSE(full.empty());
  ASSERT_FALSE(tiled.empty());
  EXPECT_EQ(full.size(), tiled.size());
  EXPECT_EQ(full.type(), tiled.type());
  EXPECT_LE(MaxAbsDiff(full, tiled), 1e-6f);
  EXPECT_LE(MaxAbsDiffNearTileSeams(full, tiled), 1e-6f);
#endif
}

}  // namespace puerhlab
