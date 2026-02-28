//  Copyright 2026 Yurun Zi
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.

#include "image/metadata_extractor.hpp"

#include <libraw/libraw.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>

#include "type/supported_file_type.hpp"

namespace puerhlab {
namespace {
auto RationalToFloat(const Exiv2::Rational& value) -> float {
  if (value.second == 0) {
    return 0.0f;
  }
  return static_cast<float>(value.first) / static_cast<float>(value.second);
}

auto IsFinitePositive(float value) -> bool {
  return std::isfinite(value) && value > 0.0f;
}

auto TrimTrailingZeroPaddedLocal(const char* s, size_t max_len = 256) -> std::string {
  if (!s) return {};
  size_t len = std::min(std::strlen(s), max_len);
  while (len > 0 && (s[len - 1] == '\0' || std::isspace(static_cast<unsigned char>(s[len - 1])))) {
    --len;
  }
  return {s, len};
}

auto TrimAsciiLocal(const std::string& value) -> std::string {
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

auto ContainsCaseInsensitiveLocal(const std::string& text, const std::string& pattern) -> bool {
  if (text.empty() || pattern.empty()) return false;
  auto to_lower = [](unsigned char c) { return static_cast<char>(std::tolower(c)); };
  std::string lower_text(text.size(), '\0');
  std::string lower_pattern(pattern.size(), '\0');
  std::transform(text.begin(), text.end(), lower_text.begin(), to_lower);
  std::transform(pattern.begin(), pattern.end(), lower_pattern.begin(), to_lower);
  return lower_text.find(lower_pattern) != std::string::npos;
}

auto IsNikonCameraLocal(const std::string& make, const std::string& model) -> bool {
  return ContainsCaseInsensitiveLocal(make, "nikon") || ContainsCaseInsensitiveLocal(model, "nikon");
}

auto ResolveCropFactorHintLocal(float focal_mm, float focal_35mm_mm) -> float {
  if (!IsFinitePositive(focal_mm) || !IsFinitePositive(focal_35mm_mm)) return 0.0f;
  return focal_35mm_mm / focal_mm;
}

/// Populate a RawRuntimeColorContext directly from libraw's open-but-not-processed state.
/// Only requires open_file / unpack to have been called so that imgdata.rawdata.color,
/// imgdata.idata, imgdata.other, imgdata.lens are populated.
void PopulateMetadataRuntimeContext(LibRaw& raw_processor, RawRuntimeColorContext& ctx) {
  const auto& color = raw_processor.imgdata.rawdata.color;
  for (int i = 0; i < 3; ++i) {
    ctx.cam_mul_[i] = color.cam_mul[i];
    ctx.pre_mul_[i] = color.pre_mul[i];
  }
  for (int r = 0; r < 3; ++r) {
    for (int c = 0; c < 3; ++c) {
      ctx.cam_xyz_[r * 3 + c] = color.cam_xyz[r][c];
      ctx.rgb_cam_[r * 3 + c] = color.rgb_cam[r][c];
    }
  }

  ctx.camera_make_  = TrimAsciiLocal(raw_processor.imgdata.idata.make);
  ctx.camera_model_ = TrimAsciiLocal(raw_processor.imgdata.idata.model);

  ctx.lens_make_  = TrimTrailingZeroPaddedLocal(raw_processor.imgdata.lens.LensMake);
  ctx.lens_model_ = TrimTrailingZeroPaddedLocal(raw_processor.imgdata.lens.Lens);
  if (ctx.lens_model_.empty()) {
    ctx.lens_model_ = TrimTrailingZeroPaddedLocal(raw_processor.imgdata.lens.makernotes.Lens);
  }

  if (IsNikonCameraLocal(ctx.camera_make_, ctx.camera_model_)) {
    if (ctx.lens_make_.empty()) {
      ctx.lens_make_ = "Nikon";
    }
    // Nikon lens ID resolution: use makernotes LensID if direct name is not found
    if (ctx.lens_model_.empty()) {
      // Fallback: construct lens model from Nikon makernotes
      const auto& nikon = raw_processor.imgdata.lens.nikon;
      const bool has_nikon_signature = (nikon.LensIDNumber != 0 || nikon.LensType != 0 ||
                                        nikon.MCUVersion != 0 || nikon.LensFStops != 0 ||
                                        IsFinitePositive(nikon.EffectiveMaxAp));
      if (has_nikon_signature) {
        char model_buf[192] = {};
        std::snprintf(model_buf, sizeof(model_buf),
                      "Nikon LensID %u (type=0x%02X mcu=%u fStops=%u effMaxAp=%.2f)",
                      static_cast<unsigned>(nikon.LensIDNumber),
                      static_cast<unsigned>(nikon.LensType),
                      static_cast<unsigned>(nikon.MCUVersion),
                      static_cast<unsigned>(nikon.LensFStops),
                      static_cast<double>(nikon.EffectiveMaxAp));
        ctx.lens_model_ = model_buf;
        const auto& lens = raw_processor.imgdata.lens;
        if (IsFinitePositive(lens.MinFocal) && IsFinitePositive(lens.MaxFocal)) {
          char focal_buf[64] = {};
          if (std::fabs(lens.MinFocal - lens.MaxFocal) < 1e-4f) {
            std::snprintf(focal_buf, sizeof(focal_buf), " %.1fmm",
                          static_cast<double>(lens.MinFocal));
          } else {
            std::snprintf(focal_buf, sizeof(focal_buf), " %.1f-%.1fmm",
                          static_cast<double>(lens.MinFocal),
                          static_cast<double>(lens.MaxFocal));
          }
          ctx.lens_model_ += focal_buf;
        }
      }
    }
  }

  ctx.focal_length_mm_ = raw_processor.imgdata.other.focal_len;
  if (!IsFinitePositive(ctx.focal_length_mm_)) {
    ctx.focal_length_mm_ = raw_processor.imgdata.lens.makernotes.CurFocal;
  }
  ctx.aperture_f_number_ = raw_processor.imgdata.other.aperture;
  if (!IsFinitePositive(ctx.aperture_f_number_)) {
    ctx.aperture_f_number_ = raw_processor.imgdata.lens.makernotes.CurAp;
  }
  ctx.focus_distance_m_ = 0.0f;
  if (std::isfinite(raw_processor.imgdata.lens.makernotes.FocusRangeIndex) &&
      raw_processor.imgdata.lens.makernotes.FocusRangeIndex > 0.0f) {
    ctx.focus_distance_m_ = raw_processor.imgdata.lens.makernotes.FocusRangeIndex;
  }

  ctx.focal_35mm_mm_ = 0.0f;
  if (raw_processor.imgdata.lens.FocalLengthIn35mmFormat > 0) {
    ctx.focal_35mm_mm_ = static_cast<float>(raw_processor.imgdata.lens.FocalLengthIn35mmFormat);
  } else if (raw_processor.imgdata.lens.makernotes.FocalLengthIn35mmFormat > 0) {
    ctx.focal_35mm_mm_ =
        static_cast<float>(raw_processor.imgdata.lens.makernotes.FocalLengthIn35mmFormat);
  }
  ctx.crop_factor_hint_ = ResolveCropFactorHintLocal(ctx.focal_length_mm_, ctx.focal_35mm_mm_);

  ctx.lens_metadata_valid_ = !ctx.lens_model_.empty() && std::isfinite(ctx.focal_length_mm_) &&
                             ctx.focal_length_mm_ > 0.0f;
  // Mark as valid for downstream consumers (not an actual decode, just metadata)
  ctx.valid_                  = true;
  ctx.output_in_camera_space_ = true;  // no processing done, color data is camera-space
}

/// Populate ExifDisplayMetaData from a RawRuntimeColorContext + libraw other/sizes data.
void PopulateDisplayMetadataFromLibRaw(LibRaw& raw_processor, const RawRuntimeColorContext& ctx,
                                       ExifDisplayMetaData& display) {
  display.make_          = ctx.camera_make_;
  display.model_         = ctx.camera_model_;
  display.lens_          = ctx.lens_model_;
  display.lens_make_     = ctx.lens_make_;
  display.focal_         = ctx.focal_length_mm_;
  display.focal_35mm_    = ctx.focal_35mm_mm_;
  display.aperture_      = ctx.aperture_f_number_;
  display.focus_distance_m_ = ctx.focus_distance_m_;

  display.iso_ = static_cast<uint64_t>(raw_processor.imgdata.other.iso_speed);

  const float shutter_sec = raw_processor.imgdata.other.shutter;
  if (std::isfinite(shutter_sec) && shutter_sec > 0.0f) {
    if (shutter_sec >= 1.0f) {
      display.shutter_speed_ = {static_cast<int>(shutter_sec), 1};
    } else {
      display.shutter_speed_ = {1, static_cast<int>(1.0f / shutter_sec + 0.5f)};
    }
  }

  display.width_  = static_cast<uint32_t>(raw_processor.imgdata.sizes.width);
  display.height_ = static_cast<uint32_t>(raw_processor.imgdata.sizes.height);

  // Timestamp
  const time_t ts = raw_processor.imgdata.other.timestamp;
  if (ts > 0) {
    struct tm t {};
#if defined(_WIN32)
    gmtime_s(&t, &ts);
#else
    gmtime_r(&ts, &t);
#endif
    char buf[64] = {};
    std::strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", &t);
    display.date_time_str_ = buf;
  }
}

/// Set of raw file extensions (lowercase).
static const std::unordered_set<std::string> kRawExtensions = {
    ".arw", ".cr2", ".cr3", ".nef", ".dng", ".raw", ".raf", ".3fr", ".rw2"};

auto IsRawExtension(const std::filesystem::path& path) -> bool {
  std::string ext = path.extension().string();
  std::transform(ext.begin(), ext.end(), ext.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return kRawExtensions.count(ext) > 0;
}
}  // namespace

static void GetDisplayMetadataFromExif(Exiv2::ExifData&     exif_data,
                                       ExifDisplayMetaData& display_metadata) {
  if (exif_data.empty()) {
    return;
  }
  if (exif_data.findKey(Exiv2::ExifKey("Exif.Image.Make")) != exif_data.end()) {
    display_metadata.make_ = exif_data["Exif.Image.Make"].toString();
  }
  if (exif_data.findKey(Exiv2::ExifKey("Exif.Image.Model")) != exif_data.end()) {
    display_metadata.model_ = exif_data["Exif.Image.Model"].toString();
  }
  if (exif_data.findKey(Exiv2::ExifKey("Exif.Photo.LensModel")) != exif_data.end()) {
    display_metadata.lens_ = exif_data["Exif.Photo.LensModel"].toString();
  }
  if (exif_data.findKey(Exiv2::ExifKey("Exif.Photo.LensMake")) != exif_data.end()) {
    display_metadata.lens_make_ = exif_data["Exif.Photo.LensMake"].toString();
  }
  if (exif_data.findKey(Exiv2::ExifKey("Exif.Photo.FNumber")) != exif_data.end()) {
    display_metadata.aperture_ = RationalToFloat(exif_data["Exif.Photo.FNumber"].toRational());
  }
  if (exif_data.findKey(Exiv2::ExifKey("Exif.Photo.FocalLength")) != exif_data.end()) {
    display_metadata.focal_ = RationalToFloat(exif_data["Exif.Photo.FocalLength"].toRational());
  }
  if (exif_data.findKey(Exiv2::ExifKey("Exif.Photo.FocalLengthIn35mmFilm")) != exif_data.end()) {
    display_metadata.focal_35mm_ =
        static_cast<float>(exif_data["Exif.Photo.FocalLengthIn35mmFilm"].toInt64());
  }
  if (exif_data.findKey(Exiv2::ExifKey("Exif.Photo.SubjectDistance")) != exif_data.end()) {
    display_metadata.focus_distance_m_ =
        RationalToFloat(exif_data["Exif.Photo.SubjectDistance"].toRational());
  }
  if (exif_data.findKey(Exiv2::ExifKey("Exif.Photo.ISOSpeedRatings")) != exif_data.end()) {
    display_metadata.iso_ = exif_data["Exif.Photo.ISOSpeedRatings"].toInt64();
  }
  if (exif_data.findKey(Exiv2::ExifKey("Exif.Image.ShutterSpeedValue")) != exif_data.end()) {
    display_metadata.shutter_speed_ = exif_data["Exif.Image.ShutterSpeedValue"].toRational();
  }
  if (exif_data.findKey(Exiv2::ExifKey("Exif.Image.ImageLength")) != exif_data.end()) {
    display_metadata.height_ = exif_data["Exif.Image.ImageLength"].toUint32();
  }
  if (exif_data.findKey(Exiv2::ExifKey("Exif.Image.ImageWidth")) != exif_data.end()) {
    display_metadata.width_ = exif_data["Exif.Image.ImageWidth"].toUint32();
  }
  if (exif_data.findKey(Exiv2::ExifKey("Exif.Image.DateTime")) != exif_data.end()) {
    display_metadata.date_time_str_ = exif_data["Exif.Image.DateTime"].toString();
    if (display_metadata.date_time_str_.size() >= 10) {
      display_metadata.date_time_str_[4] =
          '-';  // Change from "YYYY:MM:DD HH:MM:SS" to "YYYY-MM-DD HH:MM:SS"
      display_metadata.date_time_str_[7] = '-';
    }
  }
}

auto MetadataExtractor::ExtractEXIF(const image_path_t& image_path) -> Exiv2::Image::UniquePtr {
  Exiv2::Image::UniquePtr image = Exiv2::ImageFactory::open(image_path.string());
  image->readMetadata();
  return image;
}

auto MetadataExtractor::ExtractEXIFFromBuffer(const uint8_t* buffer, size_t size)
    -> Exiv2::Image::UniquePtr {
  if (!buffer || size == 0) {
    throw std::runtime_error("MetadataExtractor: empty buffer");
  }
  Exiv2::Image::UniquePtr image =
      Exiv2::ImageFactory::open(reinterpret_cast<const Exiv2::byte*>(buffer), size);
  image->readMetadata();
  return image;
}

auto MetadataExtractor::EXIFToDisplayMetaData(const Exiv2::Image::UniquePtr& exif_data)
    -> ExifDisplayMetaData {
  ExifDisplayMetaData display_metadata;
  if (exif_data->exifData().empty()) {
    return display_metadata;
  }
  GetDisplayMetadataFromExif(exif_data->exifData(), display_metadata);
  return display_metadata;
}

auto MetadataExtractor::BufferToDisplayMetaData(const uint8_t* buffer, size_t size)
    -> ExifDisplayMetaData {
  ExifDisplayMetaData display_metadata;
  try {
    auto exif_data = ExtractEXIFFromBuffer(buffer, size);
    if (!exif_data || exif_data->exifData().empty()) {
      return display_metadata;
    }
    GetDisplayMetadataFromExif(exif_data->exifData(), display_metadata);
  } catch (...) {
    return ExifDisplayMetaData{};
  }
  return display_metadata;
}


auto MetadataExtractor::EXIFToJSON(const Exiv2::Image::UniquePtr& exif_data) -> nlohmann::json {
  // The full EXIF is too large, we only convert the display-friendly metadata to JSON
  nlohmann::json exif_json;
  auto           display_metadata = EXIFToDisplayMetaData(exif_data);
  exif_json                       = display_metadata.ToJson();
  return exif_json;
}

void MetadataExtractor::ExtractEXIF_ToImage(const image_path_t& image_path, Image& image) {
  // For raw files, prefer the libraw-based extraction path which also provides
  // RawRuntimeColorContext for pipeline operators.
  if (IsRawExtension(image_path)) {
    if (ExtractRawMetadata_ToImage(image_path, image)) {
      return;
    }
    // Fall through to Exiv2 if libraw extraction fails
  }

  auto exif_data = ExtractEXIF(image_path);
  if (exif_data->exifData().empty()) {
    return;
  }
  ExifDisplayMetaData display_metadata;
  GetDisplayMetadataFromExif(exif_data->exifData(), display_metadata);
  image.SetExifDisplayMetaData(std::move(display_metadata));
}

auto MetadataExtractor::ExtractRawMetadata_ToImage(const image_path_t& image_path, Image& image)
    -> bool {
  LibRaw raw_processor;

#if defined(_WIN32)
  int ret = raw_processor.open_file(image_path.wstring().c_str());
#else
  int ret = raw_processor.open_file(image_path.string().c_str());
#endif
  if (ret != LIBRAW_SUCCESS) {
    std::cerr << "MetadataExtractor: libraw open_file failed for '"
              << image_path.string() << "' (error " << ret << ")" << std::endl;
    return false;
  }

  ret = raw_processor.unpack();
  if (ret != LIBRAW_SUCCESS) {
    std::cerr << "MetadataExtractor: libraw unpack failed for '"
              << image_path.string() << "' (error " << ret << ")" << std::endl;
    raw_processor.recycle();
    return false;
  }

  RawRuntimeColorContext ctx{};
  PopulateMetadataRuntimeContext(raw_processor, ctx);

  ExifDisplayMetaData display{};
  PopulateDisplayMetadataFromLibRaw(raw_processor, ctx, display);

  raw_processor.recycle();

  image.SetExifDisplayMetaData(std::move(display));
  image.SetRawColorContext(std::move(ctx));
  return true;
}
}  // namespace puerhlab
