//  Copyright 2025 Yurun Zi
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

#include "io/image/image_writer.hpp"

#include <OpenImageIO/imageio.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <memory>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace puerhlab {
namespace {
OIIO_NAMESPACE_USING

auto PathToUtf8(const std::filesystem::path& path) -> std::string {
  auto u8 = path.u8string();
  return std::string(reinterpret_cast<const char*>(u8.data()), u8.size());
}

auto ShouldResize(const ExportFormatOptions& options) -> bool {
  return options.resize_enabled_ && options.max_length_side_ > 0;
}

auto ResizeRGBA32F(const cv::Mat& rgba32f, const ExportFormatOptions& options) -> cv::Mat {
  if (!ShouldResize(options)) return rgba32f;

  const int src_w = rgba32f.cols;
  const int src_h = rgba32f.rows;
  const int max_s = options.max_length_side_;
  if (src_w <= 0 || src_h <= 0 || max_s <= 0) return rgba32f;

  const int cur_max = std::max(src_w, src_h);
  if (cur_max <= max_s) return rgba32f;

  const double scale = static_cast<double>(max_s) / static_cast<double>(cur_max);
  const int    dst_w = std::max(1, static_cast<int>(std::lround(src_w * scale)));
  const int    dst_h = std::max(1, static_cast<int>(std::lround(src_h * scale)));

  cv::Mat      resized;
  cv::resize(rgba32f, resized, cv::Size(dst_w, dst_h), 0.0, 0.0, cv::INTER_AREA);
  return resized;
}

auto FormatSupportsAlpha(ImageFormatType fmt) -> bool {
  switch (fmt) {
    case ImageFormatType::PNG:
    case ImageFormatType::TIFF:
    case ImageFormatType::WEBP:
    case ImageFormatType::EXR:
      return true;
    default:
      return false;
  }
}

auto MakeOIIOBuffer(const cv::Mat& rgba32f, const ExportFormatOptions& options,
                    TypeDesc& out_spec_format, TypeDesc& out_input_format, int& out_channels)
    -> cv::Mat {
  const ImageFormatType fmt        = options.format_;
  const bool            want_alpha = FormatSupportsAlpha(fmt) && rgba32f.channels() == 4;

  if (fmt == ImageFormatType::JPEG || fmt == ImageFormatType::BMP) {
    out_channels = 3;
  } else {
    out_channels = want_alpha ? 4 : 3;
  }

  cv::Mat rgb_or_rgba;
  if (out_channels == 3) {
    cv::cvtColor(rgba32f, rgb_or_rgba, cv::COLOR_RGBA2RGB);
  } else {
    rgb_or_rgba = rgba32f;
  }

  if (fmt == ImageFormatType::EXR) {
    if (options.bit_depth_ == ExportFormatOptions::BIT_DEPTH::BIT_32) {
      out_spec_format  = TypeDesc::FLOAT;
      out_input_format = TypeDesc::FLOAT;
      return (rgb_or_rgba.type() == CV_32FC(out_channels) && rgb_or_rgba.isContinuous())
                 ? rgb_or_rgba
                 : rgb_or_rgba.clone();
    }

    out_spec_format  = TypeDesc::HALF;
    out_input_format = TypeDesc::FLOAT;
    return (rgb_or_rgba.type() == CV_32FC(out_channels) && rgb_or_rgba.isContinuous())
               ? rgb_or_rgba
               : rgb_or_rgba.clone();
  }

  if (fmt == ImageFormatType::JPEG || fmt == ImageFormatType::WEBP || fmt == ImageFormatType::BMP) {
    out_spec_format  = TypeDesc::UINT8;
    out_input_format = TypeDesc::UINT8;
    cv::Mat u8;
    rgb_or_rgba.convertTo(u8, CV_MAKETYPE(CV_8U, out_channels), 255.0);
    return u8.isContinuous() ? u8 : u8.clone();
  }

  if (fmt == ImageFormatType::PNG || fmt == ImageFormatType::TIFF) {
    switch (options.bit_depth_) {
      case ExportFormatOptions::BIT_DEPTH::BIT_8: {
        out_spec_format  = TypeDesc::UINT8;
        out_input_format = TypeDesc::UINT8;
        cv::Mat u8;
        rgb_or_rgba.convertTo(u8, CV_MAKETYPE(CV_8U, out_channels), 255.0);
        return u8.isContinuous() ? u8 : u8.clone();
      }
      case ExportFormatOptions::BIT_DEPTH::BIT_16: {
        out_spec_format  = TypeDesc::UINT16;
        out_input_format = TypeDesc::UINT16;
        cv::Mat u16;
        rgb_or_rgba.convertTo(u16, CV_MAKETYPE(CV_16U, out_channels), 65535.0);
        return u16.isContinuous() ? u16 : u16.clone();
      }
      case ExportFormatOptions::BIT_DEPTH::BIT_32: {
        out_spec_format  = TypeDesc::FLOAT;
        out_input_format = TypeDesc::FLOAT;
        return (rgb_or_rgba.type() == CV_32FC(out_channels) && rgb_or_rgba.isContinuous())
                   ? rgb_or_rgba
                   : rgb_or_rgba.clone();
      }
      default:
        break;
    }
  }

  out_spec_format  = TypeDesc::UINT8;
  out_input_format = TypeDesc::UINT8;
  out_channels     = 3;
  cv::Mat rgb;
  cv::cvtColor(rgba32f, rgb, cv::COLOR_RGBA2RGB);
  cv::Mat u8;
  rgb.convertTo(u8, CV_8UC3, 255.0);
  return u8.isContinuous() ? u8 : u8.clone();
}

auto ApplyOIIOFormatOptions(ImageSpec& spec, const ExportFormatOptions& options) -> void {
  switch (options.format_) {
    case ImageFormatType::JPEG:
    case ImageFormatType::WEBP:
      spec.attribute("CompressionQuality", options.quality_);
      break;
    case ImageFormatType::PNG:
      spec.attribute("CompressionLevel", options.compression_level_);
      spec.attribute("png:compressionLevel", options.compression_level_);
      break;
    case ImageFormatType::TIFF: {
      const int tiff_compress = static_cast<int>(options.tiff_compress_);
      spec.attribute("tiff:compression", tiff_compress);
      std::string compress_str = "none";
      if (options.tiff_compress_ == ExportFormatOptions::TIFF_COMPRESS::LZW) {
        compress_str = "lzw";
      } else if (options.tiff_compress_ == ExportFormatOptions::TIFF_COMPRESS::ZIP) {
        compress_str = "zip";
      }
      spec.attribute("compression", compress_str);
      spec.attribute("Compression", compress_str);
      break;
    }
    default:
      break;
  }
}

auto ForceUprightOrientation(ImageSpec& spec) -> void {
  // The pipeline already applies any required orientation to pixel data.
  // If we preserve the source Orientation metadata, some viewers will rotate again.
  //
  // OpenImageIO uses the standard "Orientation" metadata key (1 = normal/upright).
  spec.extra_attribs.remove("Exif:Orientation", TypeDesc::UNKNOWN, false);
  spec.extra_attribs.remove("EXIF:Orientation", TypeDesc::UNKNOWN, false);
  spec.extra_attribs.remove("exif:Orientation", TypeDesc::UNKNOWN, false);
  spec.extra_attribs.remove("tiff:Orientation", TypeDesc::UNKNOWN, false);
  spec.extra_attribs.remove("TIFF:Orientation", TypeDesc::UNKNOWN, false);
  spec.extra_attribs.remove("Orientation", TypeDesc::UNKNOWN, false);
  spec.attribute("Orientation", 1);
}

auto TryWriteWithOpenImageIO(const image_path_t& src_path, const std::filesystem::path& export_path,
                             const cv::Mat& rgba32f, const ExportFormatOptions& options,
                             std::string& out_error) -> bool {
  const std::string dst          = PathToUtf8(export_path);

  TypeDesc          spec_format  = TypeDesc::UINT8;
  TypeDesc          input_format = TypeDesc::UINT8;
  int               channels     = 0;
  cv::Mat           pixels = MakeOIIOBuffer(rgba32f, options, spec_format, input_format, channels);

  ImageSpec         outspec(pixels.cols, pixels.rows, channels, spec_format);
  if (channels == 3) outspec.channelnames = {"R", "G", "B"};
  if (channels == 4) outspec.channelnames = {"R", "G", "B", "A"};

  // Best-effort metadata copy (EXIF/IPTC/XMP/etc.) from source image.
  try {
    const std::string src = PathToUtf8(src_path);
    if (auto in = ImageInput::open(src)) {
      outspec.extra_attribs = in->spec().extra_attribs;
      in->close();
    }
  } catch (const std::exception&) {
    // Best effort: ignore metadata failures.
  }

  ForceUprightOrientation(outspec);
  ApplyOIIOFormatOptions(outspec, options);

  // OIIO v3 exports ImageOutput::create(string_view, ...) (and a UTF-16 helper).
  // Avoid the deprecated create(std::string, std::string) overload, which can
  // lead to unresolved externals on MSVC when linking against the DLL.
  std::unique_ptr<ImageOutput> out = ImageOutput::create(dst);
  if (!out) {
    out_error = "OpenImageIO: failed to create ImageOutput";
    return false;
  }

  if (!out->open(dst, outspec)) {
    out_error = "OpenImageIO: failed to open output: " + out->geterror();
    return false;
  }

  const stride_t xstride = static_cast<stride_t>(pixels.elemSize());
  const stride_t ystride = static_cast<stride_t>(pixels.step);
  if (!out->write_image(input_format, pixels.data, xstride, ystride, AutoStride)) {
    out_error = "OpenImageIO: failed to write image: " + out->geterror();
    out->close();
    return false;
  }

  out->close();
  return true;
}

auto TryWriteWithOpenCV(const std::filesystem::path& export_path, const cv::Mat& rgba32f,
                        const ExportFormatOptions& options, std::string& out_error) -> bool {
  const std::string dst        = export_path.string();

  const bool        want_alpha = FormatSupportsAlpha(options.format_);
  const int         channels   = want_alpha ? 4 : 3;

  cv::Mat           bgr_or_bgra;
  if (channels == 3) {
    cv::cvtColor(rgba32f, bgr_or_bgra, cv::COLOR_RGBA2BGR);
  } else {
    cv::cvtColor(rgba32f, bgr_or_bgra, cv::COLOR_RGBA2BGRA);
  }

  cv::Mat encoded;
  if (options.format_ == ImageFormatType::EXR ||
      (options.format_ == ImageFormatType::TIFF &&
       options.bit_depth_ == ExportFormatOptions::BIT_DEPTH::BIT_32)) {
    encoded = bgr_or_bgra;
  } else if (options.format_ == ImageFormatType::PNG || options.format_ == ImageFormatType::TIFF) {
    if (options.bit_depth_ == ExportFormatOptions::BIT_DEPTH::BIT_16) {
      bgr_or_bgra.convertTo(encoded, CV_MAKETYPE(CV_16U, channels), 65535.0);
    } else {
      bgr_or_bgra.convertTo(encoded, CV_MAKETYPE(CV_8U, channels), 255.0);
    }
  } else {
    bgr_or_bgra.convertTo(encoded, CV_MAKETYPE(CV_8U, channels), 255.0);
  }

  std::vector<int> params;
  switch (options.format_) {
    case ImageFormatType::JPEG:
      params = {cv::IMWRITE_JPEG_QUALITY, options.quality_};
      break;
    case ImageFormatType::WEBP:
      params = {cv::IMWRITE_WEBP_QUALITY, options.quality_};
      break;
    case ImageFormatType::PNG:
      params = {cv::IMWRITE_PNG_COMPRESSION, options.compression_level_};
      break;
    case ImageFormatType::TIFF:
      params = {cv::IMWRITE_TIFF_COMPRESSION, static_cast<int>(options.tiff_compress_)};
      break;
    case ImageFormatType::EXR:
      params = {cv::IMWRITE_EXR_TYPE, (options.bit_depth_ == ExportFormatOptions::BIT_DEPTH::BIT_32)
                                          ? cv::IMWRITE_EXR_TYPE_FLOAT
                                          : cv::IMWRITE_EXR_TYPE_HALF};
      break;
    default:
      break;
  }

  try {
    if (!cv::imwrite(dst, encoded, params)) {
      out_error = "OpenCV: imwrite returned false";
      return false;
    }
    return true;
  } catch (const cv::Exception& e) {
    out_error = std::string("OpenCV: ") + e.what();
    return false;
  }
}
}  // namespace

void ImageWriter::WriteImageToPath(const image_path_t&          src_path,
                                   std::shared_ptr<ImageBuffer> image_data,
                                   ExportFormatOptions          options) {
  if (!image_data) {
    throw std::runtime_error("ImageWriter: image_data is null");
  }
  if (options.export_path_.empty()) {
    throw std::runtime_error("ImageWriter: export_path is empty");
  }

  const auto export_path = options.export_path_;
  if (export_path.has_parent_path()) {
    std::filesystem::create_directories(export_path.parent_path());
  }

  if (!image_data->cpu_data_valid_) {
    if (image_data->gpu_data_valid_) {
      image_data->SyncToCPU();
    } else {
      throw std::runtime_error("ImageWriter: image_data has no valid CPU/GPU data");
    }
  }

  // Use GetCPUData() to acquire the actual image data. Expected: CV_32FC4 in [0,1].
  const cv::Mat& src_rgba32f = image_data->GetCPUData();
  if (src_rgba32f.empty()) {
    throw std::runtime_error("ImageWriter: CPU image data is empty");
  }
  if (src_rgba32f.type() != CV_32FC4) {
    throw std::runtime_error("ImageWriter: expected image data type CV_32FC4");
  }

  cv::Mat     working = ResizeRGBA32F(src_rgba32f.clone(), options);

  std::string oiio_err;
  try {
    if (TryWriteWithOpenImageIO(src_path, export_path, working, options, oiio_err)) {
      return;
    }
  } catch (const std::exception& e) {
    oiio_err = e.what();
  }

  std::string cv_err;
  if (TryWriteWithOpenCV(export_path, working, options, cv_err)) {
    return;
  }

  throw std::runtime_error("ImageWriter: export failed. OIIO: " + oiio_err +
                           " | OpenCV: " + cv_err);
}
};  // namespace puerhlab
