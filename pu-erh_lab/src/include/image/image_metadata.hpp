/*
 * @file        pu-erh_lab/src/include/image/metadata.hpp
 * @brief       A exif-like metadata format used along with a image object
 * @author      Yurun Zi
 * @date        2025-03-25
 * @license     MIT
 *
 * @copyright   Copyright (c) 2025 Yurun Zi
 */

// Copyright (c) 2025 Yurun Zi
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include <libraw/libraw_const.h>
#include <libraw/libraw_types.h>

#include <string>

namespace puerhlab {
/**
 * @brief A exif-like metadata format used along with a image object. This
 * will include the data from the corresponding
 *  libraw_iparams_t
 *  libraw_image_sizes_t
 *  libraw_lensinfo_t
 *  libraw_imgother_t
 */
struct ImageMetadata {
  // libraw_iparams_t
  std::string make;
  std::string model;
  std::string actual_make;
  std::string actual_model;
  std::string software;
  unsigned    dng_version;  // for .dng file only
  int         colors;

  // libraw_image_size_t
  unsigned short height;
  unsigned short width;

  // libraw_lensinfo_t
  // libraw_makernotes_lens_t
  std::string             lens;
  std::string             lens_make;
  LibRaw_lens_focal_types focal_type;
  float                   cur_aperture;
  float                   cur_focal;
  bool                    has_attachment;  // for adapter etc.
};
};  // namespace puerhlab