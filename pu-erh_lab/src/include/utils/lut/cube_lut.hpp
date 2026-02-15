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

#include <array>
#include <cerrno>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <istream>
#include <iterator>
#include <limits>
#include <string>
#include <string_view>
#include <vector>

// Minimal .cube (IRIDAS/Resolve) LUT parser tuned for OCIO bake output.
// The goal is to hand CUDA a contiguous float buffer plus domain metadata.
// Supported directives: TITLE (ignored), DOMAIN_MIN, DOMAIN_MAX, LUT_1D_SIZE, LUT_3D_SIZE.
// Data lines must be RGB triples. Order: optional 1D shaper data first, then 3D data.

namespace puerhlab {
struct CubeLut {
  std::vector<float> lut1d_;   // size: size1d * 3
  std::vector<float> lut3d_;   // size: edge3d^3 * 3 (RGB)
  int                size1d_ = 0;
  int                edge3d_ = 0;
  std::array<float, 3> domain_min_{0.f, 0.f, 0.f};
  std::array<float, 3> domain_max_{1.f, 1.f, 1.f};
  std::array<float, 3> domain_scale_{1.f, 1.f, 1.f};  // precomputed 1/(max-min) per channel

  [[nodiscard]] auto Has1D() const -> bool { return size1d_ > 0 && !lut1d_.empty(); }
  [[nodiscard]] auto Has3D() const -> bool { return edge3d_ > 0 && !lut3d_.empty(); }
};

namespace detail {
inline auto IsSpace(char c) -> bool {
  return std::isspace(static_cast<unsigned char>(c)) != 0;
}

inline auto Trim(std::string_view sv) -> std::string_view {
  size_t b = 0;
  while (b < sv.size() && IsSpace(sv[b])) ++b;
  size_t e = sv.size();
  while (e > b && IsSpace(sv[e - 1])) --e;
  return sv.substr(b, e - b);
}

inline auto StartsWithToken(std::string_view sv, std::string_view token) -> bool {
  if (sv.size() < token.size() || sv.compare(0, token.size(), token) != 0) {
    return false;
  }
  if (sv.size() == token.size()) {
    return true;
  }
  const char c = sv[token.size()];
  return IsSpace(c);
}

inline auto SkipWsAndCommas(const char* p, const char* end) -> const char* {
  while (p < end) {
    const char c = *p;
    if (c == ',' || IsSpace(c)) {
      ++p;
      continue;
    }
    break;
  }
  return p;
}

inline auto ParseFloatToken(const char*& p, const char* end, float& out) -> bool {
  p = SkipWsAndCommas(p, end);
  if (p >= end || *p == '#') {
    return false;
  }

  errno         = 0;
  char* next    = nullptr;
  const float v = std::strtof(p, &next);
  if (next == p || errno == ERANGE) {
    return false;
  }

  out = v;
  p   = next;
  return true;
}

inline auto ParseIntToken(const char*& p, const char* end, int& out) -> bool {
  p = SkipWsAndCommas(p, end);
  if (p >= end || *p == '#') {
    return false;
  }

  errno       = 0;
  char* next  = nullptr;
  const long v = std::strtol(p, &next, 10);
  if (next == p || errno == ERANGE || v < std::numeric_limits<int>::min() ||
      v > std::numeric_limits<int>::max()) {
    return false;
  }

  out = static_cast<int>(v);
  p   = next;
  return true;
}

inline auto ParseTriplet(std::string_view line, std::array<float, 3>& out) -> bool {
  const char* p   = line.data();
  const char* end = p + line.size();
  return ParseFloatToken(p, end, out[0]) && ParseFloatToken(p, end, out[1]) &&
         ParseFloatToken(p, end, out[2]);
}

inline auto ParseInt(std::string_view line, int& out) -> bool {
  const char* p   = line.data();
  const char* end = p + line.size();
  return ParseIntToken(p, end, out);
}

inline void ComputeDomainScale(CubeLut& lut) {
  for (int i = 0; i < 3; ++i) {
    const float span = lut.domain_max_[i] - lut.domain_min_[i];
    lut.domain_scale_[i] = (span != 0.f) ? (1.f / span) : 0.f;
  }
}

inline auto ParseCubeContentFast(std::string_view content, CubeLut& lut, std::string* err = nullptr)
    -> bool {
  lut = CubeLut{};  // reset

  constexpr std::string_view kTitle     = "TITLE";
  constexpr std::string_view kDomainMin = "DOMAIN_MIN";
  constexpr std::string_view kDomainMax = "DOMAIN_MAX";
  constexpr std::string_view kLut1DSize = "LUT_1D_SIZE";
  constexpr std::string_view kLut3DSize = "LUT_3D_SIZE";

  size_t target_1d = 0;
  size_t target_3d = 0;

  const char* p    = content.data();
  const char* end  = p + content.size();
  int         line_no = 1;

  while (p < end) {
    const char* line_start = p;
    const auto  remaining  = static_cast<size_t>(end - p);
    const auto  nl_raw     = std::memchr(p, '\n', remaining);
    const char* line_end   = (nl_raw != nullptr) ? static_cast<const char*>(nl_raw) : end;

    if (line_end > line_start && line_end[-1] == '\r') {
      --line_end;
    }

    std::string_view line(line_start, static_cast<size_t>(line_end - line_start));
    line = Trim(line);

    if (!line.empty() && line.front() != '#') {
      if (StartsWithToken(line, kTitle)) {
        // TITLE payload is informational only.
      } else if (StartsWithToken(line, kDomainMin)) {
        const auto payload = Trim(line.substr(kDomainMin.size()));
        if (!ParseTriplet(payload, lut.domain_min_)) {
          if (err) *err = "Malformed DOMAIN_MIN at line " + std::to_string(line_no);
          return false;
        }
      } else if (StartsWithToken(line, kDomainMax)) {
        const auto payload = Trim(line.substr(kDomainMax.size()));
        if (!ParseTriplet(payload, lut.domain_max_)) {
          if (err) *err = "Malformed DOMAIN_MAX at line " + std::to_string(line_no);
          return false;
        }
      } else if (StartsWithToken(line, kLut1DSize)) {
        const auto payload = Trim(line.substr(kLut1DSize.size()));
        if (!ParseInt(payload, lut.size1d_) || lut.size1d_ <= 0) {
          if (err) *err = "Malformed LUT_1D_SIZE at line " + std::to_string(line_no);
          return false;
        }
        target_1d = static_cast<size_t>(lut.size1d_) * 3;
        lut.lut1d_.reserve(target_1d);
      } else if (StartsWithToken(line, kLut3DSize)) {
        const auto payload = Trim(line.substr(kLut3DSize.size()));
        if (!ParseInt(payload, lut.edge3d_) || lut.edge3d_ <= 1) {
          if (err) *err = "Malformed LUT_3D_SIZE at line " + std::to_string(line_no);
          return false;
        }
        target_3d = static_cast<size_t>(lut.edge3d_) * lut.edge3d_ * lut.edge3d_ * 3;
        lut.lut3d_.reserve(target_3d);
      } else {
        // Data line (RGB triple)
        std::array<float, 3> rgb{};
        if (!ParseTriplet(line, rgb)) {
          if (err) *err = "Failed to parse RGB values at line " + std::to_string(line_no);
          return false;
        }

        if (lut.lut1d_.size() < target_1d) {
          lut.lut1d_.push_back(rgb[0]);
          lut.lut1d_.push_back(rgb[1]);
          lut.lut1d_.push_back(rgb[2]);
        } else if (lut.lut3d_.size() < target_3d) {
          lut.lut3d_.push_back(rgb[0]);
          lut.lut3d_.push_back(rgb[1]);
          lut.lut3d_.push_back(rgb[2]);
        } else {
          if (err) *err = "Unexpected extra data at line " + std::to_string(line_no);
          return false;
        }
      }
    }

    if (nl_raw == nullptr) {
      break;
    }
    p = static_cast<const char*>(nl_raw) + 1;
    ++line_no;
  }

  // Validate counts
  const size_t want_1d = (lut.size1d_ > 0) ? static_cast<size_t>(lut.size1d_) * 3 : 0;
  const size_t want_3d = (lut.edge3d_ > 0)
                             ? static_cast<size_t>(lut.edge3d_) * lut.edge3d_ * lut.edge3d_ * 3
                             : 0;

  if (lut.lut1d_.size() != want_1d) {
    if (err) {
      *err = "1D LUT entries mismatch: got " + std::to_string(lut.lut1d_.size()) +
             ", expected " + std::to_string(want_1d);
    }
    return false;
  }

  if (lut.lut3d_.size() != want_3d) {
    if (err) {
      *err = "3D LUT entries mismatch: got " + std::to_string(lut.lut3d_.size()) +
             ", expected " + std::to_string(want_3d);
    }
    return false;
  }

  ComputeDomainScale(lut);
  return true;
}
}  // namespace detail

inline auto ParseCubeStream(std::istream& is, CubeLut& lut, std::string* err = nullptr) -> bool {
  std::string content((std::istreambuf_iterator<char>(is)), std::istreambuf_iterator<char>());
  return detail::ParseCubeContentFast(content, lut, err);
}

inline auto ParseCubeFile(const std::filesystem::path& path, CubeLut& lut, std::string* err = nullptr)
    -> bool {
  std::ifstream ifs(path, std::ios::binary | std::ios::ate);
  if (!ifs.is_open()) {
    if (err) *err = "Failed to open file: " + path.string();
    return false;
  }

  const std::streamoff file_size = ifs.tellg();
  if (file_size < 0) {
    if (err) *err = "Failed to get file size: " + path.string();
    return false;
  }

  std::string content(static_cast<size_t>(file_size), '\0');
  ifs.seekg(0, std::ios::beg);
  if (file_size > 0 && !ifs.read(content.data(), file_size)) {
    if (err) *err = "Failed to read file: " + path.string();
    return false;
  }

  return detail::ParseCubeContentFast(content, lut, err);
}

inline auto ParseCubeString(std::string_view content, CubeLut& lut, std::string* err = nullptr)
    -> bool {
  return detail::ParseCubeContentFast(content, lut, err);
}

}  // namespace puerhlab
