/*
 * @file        pu-erh_lab/src/include/type/supported_file_type.hpp
 * @brief       Collection of supported file types.
 * @author      ChatGPT
 * @date        2025-04-09
 * @license     MIT
 *
 * @copyright   Copyright (c) 2025 Yurun Zi
 */

// Copyright (c) 2025 ChatGPT
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#pragma once

#include <filesystem>
#include <string>
#include <unordered_set>

namespace fs = std::filesystem;

namespace puerhlab {
static const std::unordered_set<std::wstring> supported_extensions = {
    L".jpg", L".jpeg", L".png", L".raw", L".cr2",  L".nef", L".tiff", L".bmp",
    L".dng", L".arw",  L".cr3", L".JPG", L".JPEG", L".PNG", L".RAW",  L".CR2",
    L".NEF", L".TIFF", L".BMP", L".DNG", L".ARW",  L".CR3", L".RAF",  L".3FR"};

inline bool is_supported_file(const fs::path& path) {
  if (!fs::is_regular_file(path)) return false;

  std::wstring ext = path.extension().wstring();
  return supported_extensions.count(ext) > 0;
}
};  // namespace puerhlab
