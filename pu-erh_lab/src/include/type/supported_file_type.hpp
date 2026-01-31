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

#pragma once

#include <filesystem>
#include <string>
#include <unordered_set>

namespace fs = std::filesystem;

namespace puerhlab {
// Image format types (should used in import and export modules)
enum class ImageFormatType : uint8_t {
  JPEG,
  PNG,
  TIFF,
  WEBP,
  DNG,
  ARW,
  RAW,
  CR2,
  CR3,
  NEF,
  BMP,
  RAF,
  _3FR,
  RW2,
  EXR
};

struct ExportFormatOptions {
  enum class TIFF_COMPRESS : uint8_t { NONE = 1, LZW = 5, ZIP = 8 };

  enum class BIT_DEPTH : uint8_t { BIT_8 = 8, BIT_16 = 16, BIT_32 = 32 };

  std::filesystem::path export_path_;

  ImageFormatType       format_            = ImageFormatType::JPEG;
  bool                  resize_enabled_    = false;
  int                   max_length_side_   = 0;  // 0 means no resizing

  int                   quality_           = 95;                   // For JPEG/WEBP
  BIT_DEPTH             bit_depth_         = BIT_DEPTH::BIT_16;    // For TIFF/PNG
  int                   compression_level_ = 5;                    // For PNG
  TIFF_COMPRESS         tiff_compress_     = TIFF_COMPRESS::NONE;  // For TIFF
};

static const std::unordered_set<std::wstring> supported_extensions = {
    L".jpg", L".jpeg", L".png", L".raw",  L".cr2", L".nef", L".tiff", L".bmp", L".dng",
    L".arw", L".cr3",  L".JPG", L".JPEG", L".PNG", L".RAW", L".CR2",  L".NEF", L".TIFF",
    L".BMP", L".DNG",  L".ARW", L".CR3",  L".RAF", L".3FR", L".RW2", L".3fr"};

inline bool is_supported_file(const fs::path& path) {
  if (!fs::is_regular_file(path)) return false;

  std::wstring ext = path.extension().wstring();
  return supported_extensions.count(ext) > 0;
}
};  // namespace puerhlab
