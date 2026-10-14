//  Copyright 2026 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#include "io/image/exif_gps_tags.hpp"

#include <algorithm>
#include <array>
#include <exiv2/exiv2.hpp>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>

namespace alcedo {
namespace {

constexpr uint16_t kGpsIfdTag    = 0x8825;
constexpr uint16_t kTiffLongType = 4;

void AppendU16LE(std::vector<uint8_t>& out, uint16_t value) {
  out.push_back(static_cast<uint8_t>(value & 0xFF));
  out.push_back(static_cast<uint8_t>((value >> 8) & 0xFF));
}

void AppendU32LE(std::vector<uint8_t>& out, uint32_t value) {
  for (int shift = 0; shift < 32; shift += 8) {
    out.push_back(static_cast<uint8_t>((value >> shift) & 0xFF));
  }
}

auto ReadU16LE(const uint8_t* bytes) -> uint16_t {
  return static_cast<uint16_t>(bytes[0] | (static_cast<uint16_t>(bytes[1]) << 8));
}

auto ReadU32LE(const uint8_t* bytes) -> uint32_t {
  return static_cast<uint32_t>(bytes[0]) | (static_cast<uint32_t>(bytes[1]) << 8) |
         (static_cast<uint32_t>(bytes[2]) << 16) | (static_cast<uint32_t>(bytes[3]) << 24);
}

/// One 12-byte IFD entry: tag, type, count, and the value or the offset of the value.
struct TiffIfdEntryBytes {
  uint16_t                tag_ = 0;
  std::array<uint8_t, 12> bytes_{};
};

auto MakeLongEntryBytes(uint16_t tag, uint32_t value) -> TiffIfdEntryBytes {
  std::vector<uint8_t> bytes;
  AppendU16LE(bytes, tag);
  AppendU16LE(bytes, kTiffLongType);
  AppendU32LE(bytes, 1);
  AppendU32LE(bytes, value);
  TiffIfdEntryBytes entry{.tag_ = tag};
  std::copy(bytes.begin(), bytes.end(), entry.bytes_.begin());
  return entry;
}

/// Encode an IFD of @p tags that starts at file offset @p ifd_offset, with value data after it.
auto EncodeGpsIfdAt(const std::vector<ExifIfdTag>& tags, uint32_t ifd_offset)
    -> std::vector<uint8_t> {
  std::vector<uint8_t> ifd;
  std::vector<uint8_t> values;
  const uint32_t       values_offset =
      ifd_offset + 2u + static_cast<uint32_t>(tags.size()) * 12u + 4u;
  AppendU16LE(ifd, static_cast<uint16_t>(tags.size()));
  for (const auto& tag : tags) {
    AppendU16LE(ifd, tag.tag_);
    AppendU16LE(ifd, tag.type_);
    AppendU32LE(ifd, tag.count_);
    if (tag.value_.size() <= 4) {
      std::vector<uint8_t> inline_value = tag.value_;
      inline_value.resize(4, 0);
      ifd.insert(ifd.end(), inline_value.begin(), inline_value.end());
    } else {
      if (values.size() % 2 != 0) values.push_back(0);
      AppendU32LE(ifd, values_offset + static_cast<uint32_t>(values.size()));
      values.insert(values.end(), tag.value_.begin(), tag.value_.end());
    }
  }
  AppendU32LE(ifd, 0);
  ifd.insert(ifd.end(), values.begin(), values.end());
  return ifd;
}

}  // namespace

auto ExifTiffTypeSize(uint16_t type) -> uint32_t {
  switch (type) {
    case 1:   // BYTE
    case 2:   // ASCII
    case 6:   // SBYTE
    case 7:   // UNDEFINED
      return 1;
    case 3:   // SHORT
    case 8:   // SSHORT
      return 2;
    case 4:   // LONG
    case 9:   // SLONG
    case 11:  // FLOAT
      return 4;
    case 5:   // RATIONAL
    case 10:  // SRATIONAL
    case 12:  // DOUBLE
      return 8;
    default:
      return 0;
  }
}

auto ReadExifGpsTags(const std::filesystem::path& image_path) -> std::vector<ExifIfdTag> {
  // Read the bytes through std::filesystem so that paths outside the ANSI code page open on
  // Windows; Exiv2 opens file names as narrow strings.
  std::ifstream input(image_path, std::ios::binary);
  if (!input.is_open()) {
    throw std::runtime_error("cannot open the source file to read its GPS metadata");
  }
  const std::vector<char> bytes(std::istreambuf_iterator<char>(input), {});
  if (!input.good() && !input.eof()) {
    throw std::runtime_error("cannot read the source file to read its GPS metadata");
  }

  std::vector<ExifIfdTag> tags;
  try {
    auto image = Exiv2::ImageFactory::open(reinterpret_cast<const Exiv2::byte*>(bytes.data()),
                                           bytes.size());
    image->readMetadata();
    for (const auto& datum : image->exifData()) {
      if (datum.ifdId() != Exiv2::IfdId::gpsId) {
        continue;
      }
      const auto     type      = static_cast<uint16_t>(datum.typeId());
      const uint32_t type_size = ExifTiffTypeSize(type);
      const size_t   size      = datum.size();
      if (type_size == 0 || size == 0 || size % type_size != 0) {
        continue;
      }
      ExifIfdTag tag;
      tag.tag_  = datum.tag();
      tag.type_ = type;
      tag.value_.resize(size);
      datum.copy(tag.value_.data(), Exiv2::littleEndian);
      tag.count_ = static_cast<uint32_t>(size / type_size);
      tags.push_back(std::move(tag));
    }
  } catch (const Exiv2::Error& error) {
    if (error.code() == Exiv2::ErrorCode::kerFileContainsUnknownImageType) {
      return {};
    }
    throw std::runtime_error(std::string("cannot read the GPS metadata of the source file: ") +
                             error.what());
  }
  std::sort(tags.begin(), tags.end(),
            [](const ExifIfdTag& a, const ExifIfdTag& b) { return a.tag_ < b.tag_; });
  return tags;
}

void WriteTiffGpsIfd(const std::filesystem::path&   tiff_path,
                     const std::vector<ExifIfdTag>& gps_tags) {
  std::vector<ExifIfdTag> tags;
  for (const auto& tag : gps_tags) {
    const uint32_t type_size = ExifTiffTypeSize(tag.type_);
    if (type_size != 0 && tag.count_ > 0 &&
        static_cast<uint64_t>(tag.count_) * type_size == tag.value_.size()) {
      tags.push_back(tag);
    }
  }
  if (tags.empty()) {
    return;
  }
  std::sort(tags.begin(), tags.end(),
            [](const ExifIfdTag& a, const ExifIfdTag& b) { return a.tag_ < b.tag_; });
  std::fstream file(tiff_path, std::ios::binary | std::ios::in | std::ios::out);
  if (!file.is_open()) {
    throw std::runtime_error("cannot open the exported TIFF file to write its GPS metadata");
  }
  const auto read_exact = [&file](std::streamoff offset, uint8_t* out, size_t size) {
    file.seekg(offset);
    file.read(reinterpret_cast<char*>(out), static_cast<std::streamsize>(size));
    if (!file) {
      throw std::runtime_error("the exported TIFF file ends inside its first IFD");
    }
  };

  std::array<uint8_t, 8> header{};
  read_exact(0, header.data(), header.size());
  if (header[0] != 'I' || header[1] != 'I' || ReadU16LE(header.data() + 2) != 42) {
    throw std::runtime_error(
        "the exported file is not a little-endian classic TIFF; its GPS metadata was not written");
  }
  const uint32_t ifd0_offset = ReadU32LE(header.data() + 4);

  std::array<uint8_t, 2> count_bytes{};
  read_exact(ifd0_offset, count_bytes.data(), count_bytes.size());
  const uint16_t       entry_count = ReadU16LE(count_bytes.data());
  std::vector<uint8_t> ifd0(static_cast<size_t>(entry_count) * 12u + 4u);
  read_exact(static_cast<std::streamoff>(ifd0_offset) + 2, ifd0.data(), ifd0.size());
  const uint32_t next_ifd_offset = ReadU32LE(ifd0.data() + static_cast<size_t>(entry_count) * 12u);

  file.seekp(0, std::ios::end);
  const uint64_t       end_offset = static_cast<uint64_t>(file.tellp());
  std::vector<uint8_t> appended;
  if (end_offset % 2 != 0) appended.push_back(0);
  const uint64_t gps_ifd_offset = end_offset + appended.size();
  const auto     gps_ifd        = EncodeGpsIfdAt(tags, static_cast<uint32_t>(gps_ifd_offset));
  appended.insert(appended.end(), gps_ifd.begin(), gps_ifd.end());
  if (appended.size() % 2 != 0) appended.push_back(0);

  // The new IFD0: the old entries without a previous GPS pointer, plus the new GPS pointer.
  std::vector<TiffIfdEntryBytes> entries;
  entries.reserve(entry_count + 1u);
  for (uint16_t i = 0; i < entry_count; ++i) {
    TiffIfdEntryBytes entry;
    std::copy_n(ifd0.begin() + static_cast<std::ptrdiff_t>(i) * 12, 12, entry.bytes_.begin());
    entry.tag_ = ReadU16LE(entry.bytes_.data());
    if (entry.tag_ != kGpsIfdTag) entries.push_back(entry);
  }
  entries.push_back(MakeLongEntryBytes(kGpsIfdTag, static_cast<uint32_t>(gps_ifd_offset)));
  std::stable_sort(entries.begin(), entries.end(),
                   [](const TiffIfdEntryBytes& a, const TiffIfdEntryBytes& b) {
                     return a.tag_ < b.tag_;
                   });
  const uint64_t new_ifd0_offset = end_offset + appended.size();
  AppendU16LE(appended, static_cast<uint16_t>(entries.size()));
  for (const auto& entry : entries) {
    appended.insert(appended.end(), entry.bytes_.begin(), entry.bytes_.end());
  }
  AppendU32LE(appended, next_ifd_offset);

  if (end_offset + appended.size() > 0xFFFFFFFFull) {
    throw std::runtime_error("the exported TIFF file is too large to add its GPS metadata");
  }
  file.seekp(static_cast<std::streamoff>(end_offset));
  file.write(reinterpret_cast<const char*>(appended.data()),
             static_cast<std::streamsize>(appended.size()));
  std::vector<uint8_t> header_offset;
  AppendU32LE(header_offset, static_cast<uint32_t>(new_ifd0_offset));
  file.seekp(4);
  file.write(reinterpret_cast<const char*>(header_offset.data()), 4);
  file.flush();
  if (!file) {
    throw std::runtime_error("cannot write the GPS metadata into the exported TIFF file");
  }
}

}  // namespace alcedo
