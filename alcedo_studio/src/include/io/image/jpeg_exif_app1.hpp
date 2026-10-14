//  Copyright 2026 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#pragma once

#include <filesystem>
#include <optional>
#include <vector>

#include "image/metadata.hpp"
#include "io/image/exif_gps_tags.hpp"

namespace alcedo {

struct JpegExifApp1Options {
  std::optional<ExifDisplayMetaData> metadata_;
  int                                width_            = 0;
  int                                height_           = 0;
  bool                               include_software_ = true;
  /// Entries of the GPS IFD, written unchanged. Empty writes no GPS IFD.
  std::vector<ExifIfdTag>            gps_tags_;
};

/// Build a JPEG APP1 Exif payload (`Exif\0\0` + TIFF IFDs) without Exiv2.
/// Always returns a non-empty payload when `include_software_` is true.
auto BuildJpegExifApp1Payload(const JpegExifApp1Options& options) -> std::vector<uint8_t>;

/// Build a TIFF block (little-endian TIFF header, IFD0 with only the GPS IFD pointer, and the
/// GPS IFD) for decoders that take EXIF data, such as OpenImageIO `decode_exif`. Empty when
/// @p gps_tags is empty.
auto BuildExifGpsTiffBlock(const std::vector<ExifIfdTag>& gps_tags) -> std::vector<uint8_t>;

/// Replace or insert the APP1 Exif segment in an existing JPEG file.
auto ReplaceJpegExifApp1Segment(const std::filesystem::path& path,
                                const std::vector<uint8_t>&  exif_payload) -> bool;

/// Build and write APP1 Exif for an exported JPEG (SDR or Ultra HDR).
auto ApplyJpegExifApp1Metadata(const std::filesystem::path& path,
                               const JpegExifApp1Options&   options) -> bool;

}  // namespace alcedo
