//  Copyright 2026 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#include "io/image/ultra_hdr_writer.hpp"

#include <OpenImageIO/imageio.h>
#include <exiv2/exiv2.hpp>
#include <ultrahdr_api.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#if defined(__APPLE__)
#include <CoreFoundation/CoreFoundation.h>
#include <CoreGraphics/CGColorSpace.h>
#endif

namespace puerhlab {
namespace {
OIIO_NAMESPACE_USING

constexpr float kSdrWhiteNits = 203.0f;
constexpr float kHlgReferencePeakNits = 1000.0f;
constexpr float kPqReferencePeakNits = 10000.0f;

auto PathToUtf8(const std::filesystem::path& path) -> std::string {
  auto u8 = path.u8string();
  return std::string(reinterpret_cast<const char*>(u8.data()), u8.size());
}

void ThrowIfUhdrError(const uhdr_error_info_t& status, std::string_view operation) {
  if (status.error_code == UHDR_CODEC_OK) {
    return;
  }

  std::string message = "UltraHdrWriter: ";
  message += operation;
  if (status.has_detail && status.detail[0] != '\0') {
    message += ": ";
    message += status.detail;
  }
  throw std::runtime_error(message);
}

auto ResolveUhdrColorGamut(ColorUtils::ColorSpace color_space) -> uhdr_color_gamut_t {
  switch (color_space) {
    case ColorUtils::ColorSpace::REC709:
      return UHDR_CG_BT_709;
    case ColorUtils::ColorSpace::P3_D65:
    case ColorUtils::ColorSpace::P3_D60:
    case ColorUtils::ColorSpace::P3_DCI:
      return UHDR_CG_DISPLAY_P3;
    case ColorUtils::ColorSpace::REC2020:
      return UHDR_CG_BT_2100;
    default:
      throw std::runtime_error("UltraHdrWriter: unsupported HDR color space \"" +
                               ColorUtils::ColorSpaceToString(color_space) + "\".");
  }
}

auto MakeSdrBaseColorProfile(const ExportColorProfileConfig& color_profile)
    -> ExportColorProfileConfig {
  ExportColorProfileConfig base_profile = color_profile;
  base_profile.encoding_eotf = ColorUtils::EOTF::GAMMA_2_2;
  base_profile.peak_luminance = kSdrWhiteNits;
  return base_profile;
}

float DecodePq(float value) {
  constexpr float kPqM1 = 2610.0f / 16384.0f;
  constexpr float kPqM2 = 2523.0f / 4096.0f * 128.0f;
  constexpr float kPqC1 = 3424.0f / 4096.0f;
  constexpr float kPqC2 = 2413.0f / 4096.0f * 32.0f;
  constexpr float kPqC3 = 2392.0f / 4096.0f * 32.0f;

  const float clamped = std::clamp(value, 0.0f, 1.0f);
  const float powered = std::pow(clamped, 1.0f / kPqM2);
  const float numer   = std::max(powered - kPqC1, 0.0f);
  const float denom   = kPqC2 - kPqC3 * powered;
  if (denom <= 0.0f) {
    return 0.0f;
  }
  return std::pow(numer / denom, 1.0f / kPqM1);
}

float EncodeSrgb(float value) {
  const float clamped = std::clamp(value, 0.0f, 1.0f);
  if (clamped <= 0.0031308f) {
    return 12.92f * clamped;
  }
  return 1.055f * std::pow(clamped, 1.0f / 2.4f) - 0.055f;
}

float DecodeHlg(float value) {
  constexpr float kHlgA = 0.17883277f;
  constexpr float kHlgB = 0.28466892f;
  constexpr float kHlgC = 0.55991073f;
  constexpr float kOotfGamma = 1.2f;

  const float clamped = std::clamp(value, 0.0f, 1.0f);
  const float scene_linear = clamped <= 0.5f ? (clamped * clamped) / 3.0f
                                             : (std::exp((clamped - kHlgC) / kHlgA) + kHlgB) /
                                                   12.0f;
  return std::pow(scene_linear, kOotfGamma);
}

auto LinearScaleFor(ColorUtils::EOTF eotf) -> float {
  switch (eotf) {
    case ColorUtils::EOTF::ST2084:
      return kPqReferencePeakNits / kSdrWhiteNits;
    case ColorUtils::EOTF::HLG:
      return kHlgReferencePeakNits / kSdrWhiteNits;
    default:
      return 1.0f;
  }
}

auto ConvertHdrIntentToLinear(const cv::Mat& rgba32f, ColorUtils::EOTF eotf) -> cv::Mat {
  if (eotf == ColorUtils::EOTF::LINEAR) {
    return rgba32f.isContinuous() ? rgba32f : rgba32f.clone();
  }

  cv::Mat linear = rgba32f.clone();
  const float scale = LinearScaleFor(eotf);
  for (int y = 0; y < linear.rows; ++y) {
    auto* row = linear.ptr<cv::Vec4f>(y);
    for (int x = 0; x < linear.cols; ++x) {
      for (int c = 0; c < 3; ++c) {
        if (eotf == ColorUtils::EOTF::ST2084) {
          row[x][c] = DecodePq(row[x][c]) * scale;
        } else if (eotf == ColorUtils::EOTF::HLG) {
          row[x][c] = DecodeHlg(row[x][c]) * scale;
        }
      }
    }
  }
  return linear;
}

auto BuildSdrBaseRgb8(const cv::Mat& linear_rgba32f) -> cv::Mat {
  cv::Mat rgb8(linear_rgba32f.rows, linear_rgba32f.cols, CV_8UC3);
  for (int y = 0; y < linear_rgba32f.rows; ++y) {
    const auto* src_row = linear_rgba32f.ptr<cv::Vec4f>(y);
    auto*       dst_row = rgb8.ptr<cv::Vec3b>(y);
    for (int x = 0; x < linear_rgba32f.cols; ++x) {
      for (int c = 0; c < 3; ++c) {
        const float base_linear = std::clamp(src_row[x][c], 0.0f, 1.0f);
        dst_row[x][c] =
            static_cast<uint8_t>(std::lround(EncodeSrgb(base_linear) * 255.0f));
      }
    }
  }
  return rgb8;
}

void EraseExifKey(Exiv2::ExifData& exif_data, const char* key) {
  const auto it = exif_data.findKey(Exiv2::ExifKey(key));
  if (it != exif_data.end()) {
    exif_data.erase(it);
  }
}

void SanitizeExifData(Exiv2::ExifData& exif_data, int width, int height) {
  EraseExifKey(exif_data, "Exif.Image.Orientation");
  EraseExifKey(exif_data, "Exif.Photo.ColorSpace");
  EraseExifKey(exif_data, "Exif.Image.ColorSpace");
  EraseExifKey(exif_data, "Exif.Iop.InteroperabilityIndex");
  EraseExifKey(exif_data, "Exif.Iop.InteroperabilityVersion");

  exif_data["Exif.Image.ImageWidth"] = static_cast<uint32_t>(std::max(width, 1));
  exif_data["Exif.Image.ImageLength"] = static_cast<uint32_t>(std::max(height, 1));
  exif_data["Exif.Photo.PixelXDimension"] = static_cast<uint32_t>(std::max(width, 1));
  exif_data["Exif.Photo.PixelYDimension"] = static_cast<uint32_t>(std::max(height, 1));
}

#if defined(__APPLE__)
auto ResolveBaseExportColorSpace(const ExportColorProfileConfig& config) -> CFStringRef {
  switch (config.encoding_space) {
    case ColorUtils::ColorSpace::REC709:
      return kCGColorSpaceSRGB;
    case ColorUtils::ColorSpace::P3_D65:
    case ColorUtils::ColorSpace::P3_D60:
    case ColorUtils::ColorSpace::P3_DCI:
      return kCGColorSpaceDisplayP3;
    case ColorUtils::ColorSpace::REC2020:
      return kCGColorSpaceITUR_2020_sRGBGamma;
    case ColorUtils::ColorSpace::XYZ:
      return kCGColorSpaceGenericXYZ;
    default:
      return kCGColorSpaceSRGB;
  }
}

auto BuildICCProfileBytes(const ExportColorProfileConfig& config) -> std::vector<uint8_t> {
  std::vector<uint8_t> bytes;
  CFStringRef          color_name = ResolveBaseExportColorSpace(config);
  if (!color_name) {
    return bytes;
  }

  CGColorSpaceRef color_space = CGColorSpaceCreateWithName(color_name);
  if (!color_space) {
    return bytes;
  }

  CFDataRef icc_data = CGColorSpaceCopyICCData(color_space);
  CGColorSpaceRelease(color_space);
  if (!icc_data) {
    return bytes;
  }

  const auto size = static_cast<size_t>(CFDataGetLength(icc_data));
  bytes.resize(size);
  if (size > 0) {
    CFDataGetBytes(icc_data, CFRangeMake(0, static_cast<CFIndex>(size)), bytes.data());
  }
  CFRelease(icc_data);
  return bytes;
}
#else
auto BuildICCProfileBytes(const ExportColorProfileConfig&) -> std::vector<uint8_t> { return {}; }
#endif

void ApplyBaseJpegColorProfile(ImageSpec& spec, const ExportColorProfileConfig& color_profile) {
  const std::vector<uint8_t> icc_bytes = BuildICCProfileBytes(color_profile);
  if (icc_bytes.empty()) {
    return;
  }

  spec.attribute("oiio:ColorSpace",
                 ColorUtils::ColorSpaceToString(color_profile.encoding_space) + ":uhdr-base");
  spec.attribute("ICCProfile",
                 TypeDesc(TypeDesc::UINT8, TypeDesc::SCALAR, TypeDesc::NOSEMANTICS,
                          static_cast<int>(icc_bytes.size())),
                 icc_bytes.data());
}

void WriteBaseJpeg(const std::filesystem::path&    path,
                   const cv::Mat&                  rgb8,
                   int                             quality,
                   const ExportColorProfileConfig& color_profile) {
  const std::string dst = PathToUtf8(path);
  ImageSpec         spec(rgb8.cols, rgb8.rows, 3, TypeDesc::UINT8);
  spec.channelnames = {"R", "G", "B"};
  spec.attribute("CompressionQuality", quality);
  spec.attribute("Orientation", 1);
  ApplyBaseJpegColorProfile(spec, color_profile);

  std::unique_ptr<ImageOutput> out = ImageOutput::create(dst);
  if (!out) {
    throw std::runtime_error("UltraHdrWriter: failed to create base JPEG writer.");
  }
  if (!out->open(dst, spec)) {
    throw std::runtime_error("UltraHdrWriter: failed to open base JPEG path \"" + path.string() +
                             "\".");
  }
  if (!out->write_image(TypeDesc::UINT8, rgb8.data)) {
    const std::string error = out->geterror();
    out->close();
    throw std::runtime_error("UltraHdrWriter: failed to write base JPEG: " + error);
  }
  if (!out->close()) {
    throw std::runtime_error("UltraHdrWriter: failed to finalize base JPEG.");
  }
}

void AttachExifToJpeg(const std::filesystem::path& path, const std::vector<uint8_t>& exif_bytes) {
  if (exif_bytes.empty()) {
    return;
  }

  auto image = Exiv2::ImageFactory::open(PathToUtf8(path));
  if (!image) {
    throw std::runtime_error("UltraHdrWriter: failed to reopen base JPEG for EXIF injection.");
  }

  image->readMetadata();
  Exiv2::ExifData exif_data;
  Exiv2::ExifParser::decode(exif_data, exif_bytes.data(), exif_bytes.size());
  image->setExifData(exif_data);
  image->writeMetadata();
}

auto MakeTempBaseJpegPath() -> std::filesystem::path {
  const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
  const auto tid = std::hash<std::thread::id>{}(std::this_thread::get_id());
  return std::filesystem::temp_directory_path() /
         ("puerhlab_uhdr_base_" + std::to_string(stamp) + "_" + std::to_string(tid) + ".jpg");
}

class TempFileGuard {
 public:
  explicit TempFileGuard(std::filesystem::path path) : path_(std::move(path)) {}
  ~TempFileGuard() {
    std::error_code ec;
    std::filesystem::remove(path_, ec);
  }

  auto path() const -> const std::filesystem::path& { return path_; }

 private:
  std::filesystem::path path_;
};

auto ReadFileBytes(const std::filesystem::path& path) -> std::vector<uint8_t> {
  std::ifstream input(path, std::ios::binary);
  if (!input.is_open()) {
    throw std::runtime_error("UltraHdrWriter: failed to read temporary base JPEG \"" +
                             path.string() + "\".");
  }
  return std::vector<uint8_t>(std::istreambuf_iterator<char>(input), {});
}

auto BuildBaseJpegBytes(const cv::Mat&                  linear_rgba32f,
                        const ExportFormatOptions&      options,
                        const ExportColorProfileConfig& color_profile,
                        const std::vector<uint8_t>&     exif_bytes) -> std::vector<uint8_t> {
  const cv::Mat                   base_rgb8 = BuildSdrBaseRgb8(linear_rgba32f);
  const TempFileGuard             temp_file(MakeTempBaseJpegPath());
  const ExportColorProfileConfig  base_profile = MakeSdrBaseColorProfile(color_profile);
  WriteBaseJpeg(temp_file.path(), base_rgb8, options.quality_, base_profile);
  AttachExifToJpeg(temp_file.path(), exif_bytes);
  return ReadFileBytes(temp_file.path());
}

}  // namespace

auto UltraHdrWriter::BuildSanitizedExifData(const image_path_t& source_path, int width, int height)
    -> std::vector<uint8_t> {
  try {
    auto image = Exiv2::ImageFactory::open(PathToUtf8(source_path));
    if (!image) {
      return {};
    }

    image->readMetadata();
    Exiv2::ExifData exif_data = image->exifData();
    if (exif_data.empty()) {
      return {};
    }

    SanitizeExifData(exif_data, width, height);

    Exiv2::Blob blob;
    Exiv2::ByteOrder byte_order = image->byteOrder();
    if (byte_order == Exiv2::invalidByteOrder) {
      byte_order = Exiv2::littleEndian;
    }
    Exiv2::ExifParser::encode(blob, byte_order, exif_data);
    return std::vector<uint8_t>(blob.begin(), blob.end());
  } catch (...) {
    return {};
  }
}

void UltraHdrWriter::WriteImageToPath(const image_path_t&             src_path,
                                      const std::filesystem::path&    export_path,
                                      const cv::Mat&                  rgba32f,
                                      const ExportFormatOptions&      options,
                                      const ExportColorProfileConfig& color_profile) {
  if (rgba32f.empty() || rgba32f.type() != CV_32FC4) {
    throw std::runtime_error("UltraHdrWriter: expected non-empty CV_32FC4 image.");
  }

  cv::Mat linear_rgba32f = ConvertHdrIntentToLinear(rgba32f, color_profile.encoding_eotf);
  std::vector<uint8_t> exif_bytes =
      BuildSanitizedExifData(src_path, linear_rgba32f.cols, linear_rgba32f.rows);
  std::vector<uint8_t> base_jpeg_bytes =
      BuildBaseJpegBytes(linear_rgba32f, options, color_profile, exif_bytes);

  cv::Mat rgba16f;
  linear_rgba32f.convertTo(rgba16f, CV_16FC4);
  if (!rgba16f.isContinuous()) {
    rgba16f = rgba16f.clone();
  }

  uhdr_raw_image_t hdr_image = {};
  hdr_image.fmt = UHDR_IMG_FMT_64bppRGBAHalfFloat;
  hdr_image.cg = ResolveUhdrColorGamut(color_profile.encoding_space);
  hdr_image.ct = UHDR_CT_LINEAR;
  hdr_image.range = UHDR_CR_FULL_RANGE;
  hdr_image.w = static_cast<unsigned>(rgba16f.cols);
  hdr_image.h = static_cast<unsigned>(rgba16f.rows);
  hdr_image.planes[UHDR_PLANE_PACKED] = rgba16f.data;
  hdr_image.stride[UHDR_PLANE_PACKED] = static_cast<unsigned>(rgba16f.cols);

  uhdr_compressed_image_t sdr_image = {};
  sdr_image.data = base_jpeg_bytes.data();
  sdr_image.data_sz = base_jpeg_bytes.size();
  sdr_image.capacity = base_jpeg_bytes.size();
  sdr_image.cg = ResolveUhdrColorGamut(color_profile.encoding_space);
  sdr_image.ct = UHDR_CT_SRGB;
  sdr_image.range = UHDR_CR_FULL_RANGE;

  using EncoderPtr = std::unique_ptr<uhdr_codec_private_t, decltype(&uhdr_release_encoder)>;
  EncoderPtr encoder(uhdr_create_encoder(), &uhdr_release_encoder);
  if (!encoder) {
    throw std::runtime_error("UltraHdrWriter: failed to create encoder.");
  }

  ThrowIfUhdrError(uhdr_enc_set_raw_image(encoder.get(), &hdr_image, UHDR_HDR_IMG),
                   "uhdr_enc_set_raw_image");
  ThrowIfUhdrError(uhdr_enc_set_compressed_image(encoder.get(), &sdr_image, UHDR_SDR_IMG),
                   "uhdr_enc_set_compressed_image");
  ThrowIfUhdrError(uhdr_enc_set_output_format(encoder.get(), UHDR_CODEC_JPG),
                   "uhdr_enc_set_output_format");
  ThrowIfUhdrError(uhdr_enc_set_quality(encoder.get(), options.quality_, UHDR_GAIN_MAP_IMG),
                   "uhdr_enc_set_quality(gainmap)");
  ThrowIfUhdrError(uhdr_enc_set_using_multi_channel_gainmap(encoder.get(), 1),
                   "uhdr_enc_set_using_multi_channel_gainmap");
  ThrowIfUhdrError(uhdr_enc_set_preset(encoder.get(), UHDR_USAGE_BEST_QUALITY),
                   "uhdr_enc_set_preset");
  ThrowIfUhdrError(
      uhdr_enc_set_target_display_peak_brightness(
          encoder.get(), std::clamp(color_profile.peak_luminance, 203.0f, 10000.0f)),
      "uhdr_enc_set_target_display_peak_brightness");

  ThrowIfUhdrError(uhdr_encode(encoder.get()), "uhdr_encode");

  uhdr_compressed_image_t* encoded_stream = uhdr_get_encoded_stream(encoder.get());
  if (!encoded_stream || !encoded_stream->data || encoded_stream->data_sz == 0) {
    throw std::runtime_error("UltraHdrWriter: encoder produced an empty stream.");
  }

  std::ofstream output(export_path, std::ios::binary);
  if (!output.is_open()) {
    throw std::runtime_error("UltraHdrWriter: failed to open output file \"" +
                             export_path.string() + "\".");
  }
  output.write(static_cast<const char*>(encoded_stream->data),
               static_cast<std::streamsize>(encoded_stream->data_sz));
  if (!output.good()) {
    throw std::runtime_error("UltraHdrWriter: failed to write output file \"" +
                             export_path.string() + "\".");
  }
}

}  // namespace puerhlab
