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
static const std::unordered_set<std::wstring> supported_extensions = {
    L".jpg", L".jpeg", L".png", L".raw",  L".cr2", L".nef", L".tiff", L".bmp", L".dng",
    L".arw", L".cr3",  L".JPG", L".JPEG", L".PNG", L".RAW", L".CR2",  L".NEF", L".TIFF",
    L".BMP", L".DNG",  L".ARW", L".CR3",  L".RAF", L".3FR", L".RW2",  L".3FR", L".3fr"};

inline bool is_supported_file(const fs::path& path) {
  if (!fs::is_regular_file(path)) return false;

  std::wstring ext = path.extension().wstring();
  return supported_extensions.count(ext) > 0;
}
};  // namespace puerhlab
