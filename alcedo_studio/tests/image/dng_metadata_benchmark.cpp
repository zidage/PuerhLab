//  Copyright 2026 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#include <OpenImageIO/imageio.h>
#include <exiv2/exiv2.hpp>
#include <libraw/libraw.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {
OIIO_NAMESPACE_USING

struct MetadataSnapshot {
  std::string make;
  std::string model;
  std::string unique_camera_model;
  std::string lens_model;
  std::string lens_make;
  std::string date_time;
  float       aperture          = 0.0f;
  float       focal_length_mm   = 0.0f;
  float       focal_35mm_mm     = 0.0f;
  float       focus_distance_m  = 0.0f;
  double      exposure_seconds  = 0.0;
  uint64_t    iso               = 0;
  uint32_t    width             = 0;
  uint32_t    height            = 0;
  bool        has_color_matrix1 = false;
  bool        has_color_matrix2 = false;
  bool        has_forward_matrix1 = false;
  bool        has_forward_matrix2 = false;
  bool        has_as_shot_neutral = false;
  bool        has_cal_illuminant1 = false;
  bool        has_cal_illuminant2 = false;
  double      color_matrix1[9]  = {};
  double      color_matrix2[9]  = {};
  double      forward_matrix1[9] = {};
  double      forward_matrix2[9] = {};
  double      as_shot_neutral[3] = {};
  uint32_t    calibration_illuminant1 = 0;
  uint32_t    calibration_illuminant2 = 0;
};

auto TrimAscii(const std::string& value) -> std::string {
  std::string out = value;
  while (!out.empty() && (out.back() == '\0' || std::isspace(static_cast<unsigned char>(out.back())))) {
    out.pop_back();
  }
  size_t begin = 0;
  while (begin < out.size() &&
         (out[begin] == '\0' || std::isspace(static_cast<unsigned char>(out[begin])))) {
    ++begin;
  }
  if (begin > 0) {
    out.erase(0, begin);
  }
  return out;
}

auto TrimTrailingZeroPadded(const char* s, size_t max_len = 256) -> std::string {
  if (!s) return {};
  size_t len = std::min(std::strlen(s), max_len);
  while (len > 0 && (s[len - 1] == '\0' || std::isspace(static_cast<unsigned char>(s[len - 1])))) {
    --len;
  }
  return {s, len};
}

auto RationalToFloat(const Exiv2::Rational& value) -> float {
  if (value.second == 0) {
    return 0.0f;
  }
  return static_cast<float>(value.first) / static_cast<float>(value.second);
}

auto ParseNumericToken(const std::string& token, double& out_value) -> bool {
  if (token.empty()) {
    return false;
  }

  const size_t slash = token.find('/');
  if (slash != std::string::npos) {
    const std::string numerator_text   = token.substr(0, slash);
    const std::string denominator_text = token.substr(slash + 1);
    char*             numerator_end    = nullptr;
    char*             denominator_end  = nullptr;
    const double      numerator        = std::strtod(numerator_text.c_str(), &numerator_end);
    const double      denominator      = std::strtod(denominator_text.c_str(), &denominator_end);
    if (numerator_end == numerator_text.c_str() || denominator_end == denominator_text.c_str() ||
        !std::isfinite(numerator) || !std::isfinite(denominator) || std::abs(denominator) < 1e-12) {
      return false;
    }
    out_value = numerator / denominator;
    return std::isfinite(out_value);
  }

  char*        value_end = nullptr;
  const double value     = std::strtod(token.c_str(), &value_end);
  if (value_end == token.c_str() || !std::isfinite(value)) {
    return false;
  }
  out_value = value;
  return true;
}

auto ParseNumericList(std::string text) -> std::vector<double> {
  for (char& ch : text) {
    if (ch == ',' || ch == '[' || ch == ']' || ch == '(' || ch == ')' || ch == ';' ||
        ch == '{' || ch == '}' || ch == '"') {
      ch = ' ';
    }
  }

  std::istringstream  iss(text);
  std::string         token;
  std::vector<double> values;
  while (iss >> token) {
    double parsed = 0.0;
    if (ParseNumericToken(token, parsed)) {
      values.push_back(parsed);
    }
  }
  return values;
}

auto NormalizeKey(std::string_view key) -> std::string {
  std::string out;
  out.reserve(key.size());
  for (const char ch : key) {
    const unsigned char uch = static_cast<unsigned char>(ch);
    if (std::isalnum(uch)) {
      out.push_back(static_cast<char>(std::tolower(uch)));
    }
  }
  return out;
}

auto LeafKey(std::string_view key) -> std::string {
  const size_t pos = key.find_last_of(":.");
  return NormalizeKey(pos == std::string_view::npos ? key : key.substr(pos + 1));
}

auto PathToUtf8(const std::filesystem::path& path) -> std::string {
  const auto u8 = path.u8string();
  return std::string(reinterpret_cast<const char*>(u8.data()), u8.size());
}

auto ReadExifStringTag(const Exiv2::ExifData& exif_data, const char* key) -> std::string {
  const auto it = exif_data.findKey(Exiv2::ExifKey(key));
  if (it == exif_data.end()) {
    return {};
  }
  return TrimAscii(it->toString());
}

auto ReadExifNumericArrayTag(const Exiv2::ExifData& exif_data, const char* key, const int count,
                             double* values_out) -> bool {
  if (!values_out || count <= 0) {
    return false;
  }
  const auto it = exif_data.findKey(Exiv2::ExifKey(key));
  if (it == exif_data.end()) {
    return false;
  }
  const auto values = ParseNumericList(it->toString());
  if (static_cast<int>(values.size()) < count) {
    return false;
  }
  for (int i = 0; i < count; ++i) {
    values_out[i] = values[static_cast<size_t>(i)];
  }
  return true;
}

auto ReadExifUnsignedTag(const Exiv2::ExifData& exif_data, const char* key, uint32_t& out) -> bool {
  const auto it = exif_data.findKey(Exiv2::ExifKey(key));
  if (it == exif_data.end()) {
    return false;
  }
  try {
    out = static_cast<uint32_t>(it->toUint32());
    return true;
  } catch (...) {
    try {
      out = static_cast<uint32_t>(it->toInt64());
      return true;
    } catch (...) {
      return false;
    }
  }
}

auto ParseExposurePair(const std::string& text, double& out_seconds) -> bool {
  double value = 0.0;
  if (!ParseNumericToken(TrimAscii(text), value)) {
    return false;
  }
  out_seconds = value;
  return std::isfinite(out_seconds) && out_seconds > 0.0;
}

auto FindOiioParam(const ImageSpec& spec, const std::initializer_list<std::string_view>& keys)
    -> const ParamValue* {
  for (const auto& param : spec.extra_attribs) {
    const std::string full = NormalizeKey(param.name().string());
    const std::string leaf = LeafKey(param.name().string());
    for (const auto key : keys) {
      const std::string normalized = NormalizeKey(key);
      if (normalized == full || normalized == leaf) {
        return &param;
      }
    }
  }
  return nullptr;
}

auto ReadOiioStringTag(const ImageSpec& spec, const std::initializer_list<std::string_view>& keys)
    -> std::string {
  const ParamValue* param = FindOiioParam(spec, keys);
  if (!param) {
    return {};
  }
  return TrimAscii(param->get_string(0));
}

auto ReadOiioNumericArrayTag(const ImageSpec& spec, const std::initializer_list<std::string_view>& keys,
                             const int count, double* values_out) -> bool {
  if (!values_out || count <= 0) {
    return false;
  }
  const ParamValue* param = FindOiioParam(spec, keys);
  if (!param) {
    return false;
  }
  const auto values = ParseNumericList(param->get_string(0));
  if (static_cast<int>(values.size()) < count) {
    return false;
  }
  for (int i = 0; i < count; ++i) {
    values_out[i] = values[static_cast<size_t>(i)];
  }
  return true;
}

auto ReadOiioUnsignedTag(const ImageSpec& spec, const std::initializer_list<std::string_view>& keys,
                         uint32_t& out) -> bool {
  const ParamValue* param = FindOiioParam(spec, keys);
  if (!param) {
    return false;
  }
  const auto values = ParseNumericList(param->get_string(0));
  if (values.empty() || !std::isfinite(values.front()) || values.front() < 0.0) {
    return false;
  }
  out = static_cast<uint32_t>(std::llround(values.front()));
  return true;
}

void PopulateDngFieldsFromExif(const Exiv2::ExifData& exif_data, MetadataSnapshot& out) {
  out.unique_camera_model = ReadExifStringTag(exif_data, "Exif.Image.UniqueCameraModel");
  out.has_color_matrix1 =
      ReadExifNumericArrayTag(exif_data, "Exif.Image.ColorMatrix1", 9, out.color_matrix1);
  out.has_color_matrix2 =
      ReadExifNumericArrayTag(exif_data, "Exif.Image.ColorMatrix2", 9, out.color_matrix2);
  out.has_forward_matrix1 =
      ReadExifNumericArrayTag(exif_data, "Exif.Image.ForwardMatrix1", 9, out.forward_matrix1);
  out.has_forward_matrix2 =
      ReadExifNumericArrayTag(exif_data, "Exif.Image.ForwardMatrix2", 9, out.forward_matrix2);
  out.has_as_shot_neutral =
      ReadExifNumericArrayTag(exif_data, "Exif.Image.AsShotNeutral", 3, out.as_shot_neutral);
  out.has_cal_illuminant1 =
      ReadExifUnsignedTag(exif_data, "Exif.Image.CalibrationIlluminant1", out.calibration_illuminant1);
  out.has_cal_illuminant2 =
      ReadExifUnsignedTag(exif_data, "Exif.Image.CalibrationIlluminant2", out.calibration_illuminant2);
}

void PopulateCommonFieldsFromExif(const Exiv2::ExifData& exif_data, MetadataSnapshot& out) {
  out.make             = ReadExifStringTag(exif_data, "Exif.Image.Make");
  out.model            = ReadExifStringTag(exif_data, "Exif.Image.Model");
  out.lens_model       = ReadExifStringTag(exif_data, "Exif.Photo.LensModel");
  out.lens_make        = ReadExifStringTag(exif_data, "Exif.Photo.LensMake");
  out.date_time        = ReadExifStringTag(exif_data, "Exif.Photo.DateTimeOriginal");
  if (out.date_time.empty()) {
    out.date_time = ReadExifStringTag(exif_data, "Exif.Image.DateTime");
  }
  if (out.date_time.size() >= 10) {
    out.date_time[4] = '-';
    out.date_time[7] = '-';
  }

  const auto find_key = [&exif_data](const char* key) {
    return exif_data.findKey(Exiv2::ExifKey(key));
  };
  const auto end = exif_data.end();

  if (const auto it = find_key("Exif.Photo.FNumber"); it != end) {
    out.aperture = RationalToFloat(it->toRational());
  }
  if (const auto it = find_key("Exif.Photo.FocalLength"); it != end) {
    out.focal_length_mm = RationalToFloat(it->toRational());
  }
  if (const auto it = find_key("Exif.Photo.FocalLengthIn35mmFilm"); it != end) {
    out.focal_35mm_mm = static_cast<float>(it->toInt64());
  }
  if (const auto it = find_key("Exif.Photo.SubjectDistance"); it != end) {
    out.focus_distance_m = RationalToFloat(it->toRational());
  }
  if (const auto iso_it = find_key("Exif.Photo.ISOSpeedRatings"); iso_it != end) {
    out.iso = static_cast<uint64_t>(iso_it->toInt64());
  } else if (const auto iso_it = find_key("Exif.Photo.ISOSpeed"); iso_it != end) {
    out.iso = static_cast<uint64_t>(iso_it->toInt64());
  }
  if (const auto it = find_key("Exif.Photo.ExposureTime"); it != end) {
    const Exiv2::Rational value = it->toRational();
    if (value.second != 0) {
      out.exposure_seconds = static_cast<double>(value.first) / static_cast<double>(value.second);
    }
  }
  if (const auto width_it = find_key("Exif.Photo.PixelXDimension"); width_it != end) {
    out.width = width_it->toUint32();
  } else if (const auto width_it = find_key("Exif.Image.ImageWidth"); width_it != end) {
    out.width = width_it->toUint32();
  }
  if (const auto height_it = find_key("Exif.Photo.PixelYDimension"); height_it != end) {
    out.height = height_it->toUint32();
  } else if (const auto height_it = find_key("Exif.Image.ImageLength"); height_it != end) {
    out.height = height_it->toUint32();
  }
}

auto ExtractWithLibRawAndExif(const std::filesystem::path& path) -> MetadataSnapshot {
  MetadataSnapshot out;
  auto             raw = std::make_unique<LibRaw>();
#if defined(_WIN32)
  int ret = raw->open_file(path.wstring().c_str());
#else
  int ret = raw->open_file(path.string().c_str());
#endif
  if (ret != LIBRAW_SUCCESS) {
    throw std::runtime_error("LibRaw open_file failed");
  }
  ret = raw->unpack();
  if (ret != LIBRAW_SUCCESS) {
    raw->recycle();
    throw std::runtime_error("LibRaw unpack failed");
  }

  out.make            = TrimAscii(raw->imgdata.idata.make);
  out.model           = TrimAscii(raw->imgdata.idata.model);
  out.lens_make       = TrimTrailingZeroPadded(raw->imgdata.lens.LensMake);
  out.lens_model      = TrimTrailingZeroPadded(raw->imgdata.lens.Lens);
  out.aperture        = raw->imgdata.other.aperture;
  out.focal_length_mm = raw->imgdata.other.focal_len;
  out.focal_35mm_mm   =
      raw->imgdata.lens.FocalLengthIn35mmFormat > 0
          ? static_cast<float>(raw->imgdata.lens.FocalLengthIn35mmFormat)
          : 0.0f;
  out.iso             = static_cast<uint64_t>(raw->imgdata.other.iso_speed);
  out.width           = static_cast<uint32_t>(raw->imgdata.sizes.width);
  out.height          = static_cast<uint32_t>(raw->imgdata.sizes.height);
  out.exposure_seconds = raw->imgdata.other.shutter;
  raw->recycle();

  Exiv2::Image::UniquePtr image = Exiv2::ImageFactory::open(path.string());
  image->readMetadata();
  PopulateDngFieldsFromExif(image->exifData(), out);
  return out;
}

auto ExtractWithLibRawOpenAndExif(const std::filesystem::path& path) -> MetadataSnapshot {
  MetadataSnapshot out;
  auto             raw = std::make_unique<LibRaw>();
#if defined(_WIN32)
  int ret = raw->open_file(path.wstring().c_str());
#else
  int ret = raw->open_file(path.string().c_str());
#endif
  if (ret != LIBRAW_SUCCESS) {
    throw std::runtime_error("LibRaw open_file failed");
  }

  out.make            = TrimAscii(raw->imgdata.idata.make);
  out.model           = TrimAscii(raw->imgdata.idata.model);
  out.lens_make       = TrimTrailingZeroPadded(raw->imgdata.lens.LensMake);
  out.lens_model      = TrimTrailingZeroPadded(raw->imgdata.lens.Lens);
  out.aperture        = raw->imgdata.other.aperture;
  out.focal_length_mm = raw->imgdata.other.focal_len;
  out.focal_35mm_mm   =
      raw->imgdata.lens.FocalLengthIn35mmFormat > 0
          ? static_cast<float>(raw->imgdata.lens.FocalLengthIn35mmFormat)
          : 0.0f;
  out.iso             = static_cast<uint64_t>(raw->imgdata.other.iso_speed);
  out.width           = static_cast<uint32_t>(raw->imgdata.sizes.width);
  out.height          = static_cast<uint32_t>(raw->imgdata.sizes.height);
  out.exposure_seconds = raw->imgdata.other.shutter;
  raw->recycle();

  Exiv2::Image::UniquePtr image = Exiv2::ImageFactory::open(path.string());
  image->readMetadata();
  PopulateCommonFieldsFromExif(image->exifData(), out);
  PopulateDngFieldsFromExif(image->exifData(), out);
  return out;
}

auto ExtractWithExifOnly(const std::filesystem::path& path) -> MetadataSnapshot {
  MetadataSnapshot      out;
  Exiv2::Image::UniquePtr image = Exiv2::ImageFactory::open(path.string());
  image->readMetadata();
  PopulateCommonFieldsFromExif(image->exifData(), out);
  PopulateDngFieldsFromExif(image->exifData(), out);
  return out;
}

auto ExtractWithOiioOnly(const std::filesystem::path& path) -> MetadataSnapshot {
  MetadataSnapshot out;
  auto             input = ImageInput::open(PathToUtf8(path));
  if (!input) {
    throw std::runtime_error("OIIO open failed: " + geterror());
  }

  const ImageSpec& spec = input->spec();
  out.make              = ReadOiioStringTag(spec, {"Exif:Make", "tiff:Make"});
  out.model             = ReadOiioStringTag(spec, {"Exif:Model", "tiff:Model"});
  out.unique_camera_model =
      ReadOiioStringTag(spec, {"Exif:UniqueCameraModel", "dng:UniqueCameraModel", "UniqueCameraModel"});
  out.lens_model        = ReadOiioStringTag(spec, {"Exif:LensModel", "LensModel"});
  out.lens_make         = ReadOiioStringTag(spec, {"Exif:LensMake", "LensMake"});
  out.date_time         = ReadOiioStringTag(spec, {"Exif:DateTimeOriginal", "DateTimeOriginal", "DateTime"});

  if (out.date_time.size() >= 10) {
    out.date_time[4] = '-';
    out.date_time[7] = '-';
  }

  double aperture = 0.0;
  if (ReadOiioNumericArrayTag(spec, {"Exif:FNumber", "FNumber"}, 1, &aperture)) {
    out.aperture = static_cast<float>(aperture);
  }
  double focal = 0.0;
  if (ReadOiioNumericArrayTag(spec, {"Exif:FocalLength", "FocalLength"}, 1, &focal)) {
    out.focal_length_mm = static_cast<float>(focal);
  }
  double focal35 = 0.0;
  if (ReadOiioNumericArrayTag(spec, {"Exif:FocalLengthIn35mmFilm", "FocalLengthIn35mmFilm"}, 1,
                              &focal35)) {
    out.focal_35mm_mm = static_cast<float>(focal35);
  }
  double focus_distance = 0.0;
  if (ReadOiioNumericArrayTag(spec, {"Exif:SubjectDistance", "SubjectDistance"}, 1,
                              &focus_distance)) {
    out.focus_distance_m = static_cast<float>(focus_distance);
  }
  uint32_t iso = 0;
  if (ReadOiioUnsignedTag(spec, {"Exif:ISOSpeedRatings", "Exif:ISOSpeed", "ISOSpeedRatings", "ISOSpeed"},
                          iso)) {
    out.iso = iso;
  }
  const std::string exposure_text =
      ReadOiioStringTag(spec, {"Exif:ExposureTime", "ExposureTime", "ShutterSpeedValue"});
  ParseExposurePair(exposure_text, out.exposure_seconds);

  out.width  = static_cast<uint32_t>(std::max(spec.width, 0));
  out.height = static_cast<uint32_t>(std::max(spec.height, 0));

  out.has_color_matrix1 = ReadOiioNumericArrayTag(
      spec, {"Exif:ColorMatrix1", "dng:ColorMatrix1", "ColorMatrix1"}, 9, out.color_matrix1);
  out.has_color_matrix2 = ReadOiioNumericArrayTag(
      spec, {"Exif:ColorMatrix2", "dng:ColorMatrix2", "ColorMatrix2"}, 9, out.color_matrix2);
  out.has_forward_matrix1 = ReadOiioNumericArrayTag(
      spec, {"Exif:ForwardMatrix1", "dng:ForwardMatrix1", "ForwardMatrix1"}, 9, out.forward_matrix1);
  out.has_forward_matrix2 = ReadOiioNumericArrayTag(
      spec, {"Exif:ForwardMatrix2", "dng:ForwardMatrix2", "ForwardMatrix2"}, 9, out.forward_matrix2);
  out.has_as_shot_neutral = ReadOiioNumericArrayTag(
      spec, {"Exif:AsShotNeutral", "dng:AsShotNeutral", "AsShotNeutral"}, 3, out.as_shot_neutral);
  out.has_cal_illuminant1 = ReadOiioUnsignedTag(
      spec, {"Exif:CalibrationIlluminant1", "dng:CalibrationIlluminant1", "CalibrationIlluminant1"},
      out.calibration_illuminant1);
  out.has_cal_illuminant2 = ReadOiioUnsignedTag(
      spec, {"Exif:CalibrationIlluminant2", "dng:CalibrationIlluminant2", "CalibrationIlluminant2"},
      out.calibration_illuminant2);

  return out;
}

auto ExtractWithExifAndOiioSize(const std::filesystem::path& path) -> MetadataSnapshot {
  MetadataSnapshot out = ExtractWithExifOnly(path);
  auto             input = ImageInput::open(PathToUtf8(path));
  if (!input) {
    throw std::runtime_error("OIIO open failed: " + geterror());
  }

  const ImageSpec& spec = input->spec();
  const int        width =
      spec.full_width > 0 ? spec.full_width : spec.width;
  const int        height =
      spec.full_height > 0 ? spec.full_height : spec.height;
  if (width > 0) {
    out.width = static_cast<uint32_t>(width);
  }
  if (height > 0) {
    out.height = static_cast<uint32_t>(height);
  }
  return out;
}

template <typename Fn>
auto BenchmarkMethod(const std::string& label, const int iterations, Fn&& fn, MetadataSnapshot& snapshot)
    -> double {
  using Clock = std::chrono::steady_clock;
  const auto start = Clock::now();
  for (int i = 0; i < iterations; ++i) {
    snapshot = fn();
  }
  const auto elapsed_ms =
      std::chrono::duration<double, std::milli>(Clock::now() - start).count();
  std::cout << std::left << std::setw(20) << label << " total=" << std::setw(10) << elapsed_ms
            << " avg=" << (elapsed_ms / static_cast<double>(iterations)) << " ms\n";
  return elapsed_ms;
}

void PrintSnapshotSummary(const std::string& label, const MetadataSnapshot& snapshot) {
  std::cout << "\n[" << label << "]\n";
  std::cout << "  make/model: " << snapshot.make << " / " << snapshot.model << '\n';
  std::cout << "  unique model: " << snapshot.unique_camera_model << '\n';
  std::cout << "  lens: " << snapshot.lens_make << " / " << snapshot.lens_model << '\n';
  std::cout << "  aperture/focal/f35: " << snapshot.aperture << " / " << snapshot.focal_length_mm
            << " / " << snapshot.focal_35mm_mm << '\n';
  std::cout << "  iso/exposure: " << snapshot.iso << " / " << snapshot.exposure_seconds << '\n';
  std::cout << "  size: " << snapshot.width << "x" << snapshot.height << '\n';
  std::cout << "  DNG tags: cm1=" << snapshot.has_color_matrix1
            << " cm2=" << snapshot.has_color_matrix2
            << " fm1=" << snapshot.has_forward_matrix1
            << " fm2=" << snapshot.has_forward_matrix2
            << " neutral=" << snapshot.has_as_shot_neutral
            << " cal1=" << snapshot.has_cal_illuminant1
            << " cal2=" << snapshot.has_cal_illuminant2 << '\n';
}

}  // namespace

int main() {
  const std::filesystem::path sample =
      std::filesystem::path(TEST_IMG_PATH) / "raw" / "batch_import" / "_DSC1306.dng";
  if (!std::filesystem::exists(sample)) {
    std::cerr << "Sample not found: " << sample.string() << '\n';
    return 1;
  }

  constexpr int kIterations = 10;

  try {
    MetadataSnapshot libraw_snapshot;
    MetadataSnapshot libraw_open_snapshot;
    MetadataSnapshot exif_snapshot;
    MetadataSnapshot exif_oiio_snapshot;
    MetadataSnapshot oiio_snapshot;

    std::cout << "Benchmark sample: " << sample.string() << '\n';
    std::cout << "Iterations: " << kIterations << "\n\n";

    BenchmarkMethod("libraw+exiv2", kIterations,
                    [&]() { return ExtractWithLibRawAndExif(sample); }, libraw_snapshot);
    BenchmarkMethod("libraw-open+exiv2", kIterations,
                    [&]() { return ExtractWithLibRawOpenAndExif(sample); }, libraw_open_snapshot);
    BenchmarkMethod("exiv2-only", kIterations,
                    [&]() { return ExtractWithExifOnly(sample); }, exif_snapshot);
    BenchmarkMethod("exiv2+oiio-size", kIterations,
                    [&]() { return ExtractWithExifAndOiioSize(sample); }, exif_oiio_snapshot);
    BenchmarkMethod("oiio-only", kIterations,
                    [&]() { return ExtractWithOiioOnly(sample); }, oiio_snapshot);

    PrintSnapshotSummary("libraw+exiv2", libraw_snapshot);
    PrintSnapshotSummary("libraw-open+exiv2", libraw_open_snapshot);
    PrintSnapshotSummary("exiv2-only", exif_snapshot);
    PrintSnapshotSummary("exiv2+oiio-size", exif_oiio_snapshot);
    PrintSnapshotSummary("oiio-only", oiio_snapshot);
  } catch (const std::exception& e) {
    std::cerr << "Benchmark failed: " << e.what() << '\n';
    return 1;
  }

  return 0;
}
