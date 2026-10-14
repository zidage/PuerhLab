//  Copyright 2026 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#pragma once

#include <cstdint>
#include <filesystem>
#include <vector>

namespace alcedo {

/// One entry of a TIFF/EXIF IFD as stored in the file: tag number, TIFF field type (1..12), the
/// number of values, and the value bytes in little-endian byte order.
struct ExifIfdTag {
  uint16_t             tag_   = 0;
  uint16_t             type_  = 0;
  uint32_t             count_ = 0;
  std::vector<uint8_t> value_;
};

/// Size in bytes of one value of TIFF field type @p type, or 0 for a type outside 1..12.
auto ExifTiffTypeSize(uint16_t type) -> uint32_t;

/**
 * @brief Read the entries of the GPS IFD (EXIF tag 0x8825) of an image file.
 *
 * Reads any container that Exiv2 reads (TIFF-based RAW files, JPEG, DNG, CR3, HEIF, ...). The
 * entries are returned sorted by tag number with their file types and values unchanged.
 *
 * @return The GPS entries; empty when the file has no GPS IFD or Exiv2 does not know the container.
 * @throws std::runtime_error when the file cannot be read or its metadata is invalid.
 */
auto ReadExifGpsTags(const std::filesystem::path& image_path) -> std::vector<ExifIfdTag>;

/**
 * @brief Add a GPS IFD with @p gps_tags to a little-endian classic TIFF file.
 *
 * OpenImageIO writes no GPS IFD into TIFF files. This appends the GPS IFD and a copy of IFD0
 * with the GPS IFD pointer (tag 0x8825) at the end of the file, then points the TIFF header to
 * the new IFD0. The pixel data and the other IFDs stay unchanged. Does nothing when @p gps_tags
 * is empty.
 *
 * @throws std::runtime_error when the file is not a little-endian classic TIFF or cannot be
 *         read or written.
 */
void WriteTiffGpsIfd(const std::filesystem::path&   tiff_path,
                     const std::vector<ExifIfdTag>& gps_tags);

}  // namespace alcedo
