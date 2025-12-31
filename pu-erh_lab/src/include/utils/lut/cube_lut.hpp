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
#include <cctype>
#include <filesystem>
#include <fstream>
#include <istream>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

// Minimal .cube (IRIDAS/Resolve) LUT parser tuned for OCIO bake output.
// The goal is to hand CUDA a contiguous float buffer plus domain metadata.
// Supported directives: TITLE (ignored), DOMAIN_MIN, DOMAIN_MAX, LUT_1D_SIZE, LUT_3D_SIZE.
// Data lines must be RGB triples. Order: optional 1D shaper data first, then 3D data.

namespace puerhlab {
struct CubeLut {
  std::vector<float> lut1d;   // size: size1d * 3
  std::vector<float> lut3d;   // size: edge3d^3 * 3 (RGB)
  int                size1d = 0;
  int                edge3d = 0;
  std::array<float, 3> domain_min{0.f, 0.f, 0.f};
  std::array<float, 3> domain_max{1.f, 1.f, 1.f};
  std::array<float, 3> domain_scale{1.f, 1.f, 1.f};  // precomputed 1/(max-min) per channel

  [[nodiscard]] auto Has1D() const -> bool { return size1d > 0 && !lut1d.empty(); }
  [[nodiscard]] auto Has3D() const -> bool { return edge3d > 0 && !lut3d.empty(); }
};

namespace detail {
inline auto Trim(std::string_view sv) -> std::string_view {
  size_t b = 0;
  while (b < sv.size() && std::isspace(static_cast<unsigned char>(sv[b])) != 0) ++b;
  size_t e = sv.size();
  while (e > b && std::isspace(static_cast<unsigned char>(sv[e - 1])) != 0) --e;
  return sv.substr(b, e - b);
}

inline auto StartsWith(std::string_view sv, std::string_view prefix) -> bool {
  return sv.size() >= prefix.size() && sv.compare(0, prefix.size(), prefix) == 0;
}

inline auto ParseInts(std::string_view line, int& out) -> bool {
  std::istringstream iss((std::string(line)));
  return (iss >> out) ? true : false;
}

inline auto ParseTriplet(std::string_view line, std::array<float, 3>& out) -> bool {
  std::istringstream iss((std::string(line)));
  return (iss >> out[0] >> out[1] >> out[2]) ? true : false;
}

inline auto ParseRgb(std::string_view line, std::array<float, 3>& out, int line_no, std::string* err)
    -> bool {
  if (!ParseTriplet(line, out)) {
    if (err) *err = "Failed to parse RGB values at line " + std::to_string(line_no);
    return false;
  }
  return true;
}

inline void ComputeDomainScale(CubeLut& lut) {
  for (int i = 0; i < 3; ++i) {
    const float span = lut.domain_max[i] - lut.domain_min[i];
    lut.domain_scale[i] = (span != 0.f) ? (1.f / span) : 0.f;
  }
}
}  // namespace detail

inline auto ParseCubeStream(std::istream& is, CubeLut& lut, std::string* err = nullptr) -> bool {
  lut = CubeLut{};  // reset

  int line_no = 0;
  while (is.good()) {
    std::string line_raw;
    std::getline(is, line_raw);
    ++line_no;

    auto line = detail::Trim(line_raw);
    if (line.empty()) continue;
    if (line[0] == '#') continue;

    if (detail::StartsWith(line, "TITLE")) {
      continue;  // ignore title
    }
    if (detail::StartsWith(line, "DOMAIN_MIN")) {
      auto payload = detail::Trim(line.substr(std::string("DOMAIN_MIN").size()));
      if (!detail::ParseTriplet(payload, lut.domain_min)) {
        if (err) *err = "Malformed DOMAIN_MIN at line " + std::to_string(line_no);
        return false;
      }
      continue;
    }
    if (detail::StartsWith(line, "DOMAIN_MAX")) {
      auto payload = detail::Trim(line.substr(std::string("DOMAIN_MAX").size()));
      if (!detail::ParseTriplet(payload, lut.domain_max)) {
        if (err) *err = "Malformed DOMAIN_MAX at line " + std::to_string(line_no);
        return false;
      }
      continue;
    }
    if (detail::StartsWith(line, "LUT_1D_SIZE")) {
      auto payload = detail::Trim(line.substr(std::string("LUT_1D_SIZE").size()));
      if (!detail::ParseInts(payload, lut.size1d) || lut.size1d <= 0) {
        if (err) *err = "Malformed LUT_1D_SIZE at line " + std::to_string(line_no);
        return false;
      }
      lut.lut1d.reserve(static_cast<size_t>(lut.size1d) * 3);
      continue;
    }
    if (detail::StartsWith(line, "LUT_3D_SIZE")) {
      auto payload = detail::Trim(line.substr(std::string("LUT_3D_SIZE").size()));
      if (!detail::ParseInts(payload, lut.edge3d) || lut.edge3d <= 1) {
        if (err) *err = "Malformed LUT_3D_SIZE at line " + std::to_string(line_no);
        return false;
      }
      const auto total = static_cast<size_t>(lut.edge3d) * lut.edge3d * lut.edge3d * 3;
      lut.lut3d.reserve(total);
      continue;
    }

    // Data line (RGB triple)
    std::array<float, 3> rgb{};
    if (!detail::ParseRgb(line, rgb, line_no, err)) return false;

    const size_t target_1d = (lut.size1d > 0) ? static_cast<size_t>(lut.size1d) * 3 : 0;
    if (lut.lut1d.size() < target_1d) {
      lut.lut1d.insert(lut.lut1d.end(), rgb.begin(), rgb.end());
      continue;
    }

    const size_t target_3d = (lut.edge3d > 0)
                                 ? static_cast<size_t>(lut.edge3d) * lut.edge3d * lut.edge3d * 3
                                 : 0;
    if (lut.lut3d.size() < target_3d) {
      lut.lut3d.insert(lut.lut3d.end(), rgb.begin(), rgb.end());
      continue;
    }

    if (err) *err = "Unexpected extra data at line " + std::to_string(line_no);
    return false;
  }

  // Validate counts
  const size_t want_1d = (lut.size1d > 0) ? static_cast<size_t>(lut.size1d) * 3 : 0;
  const size_t want_3d = (lut.edge3d > 0) ? static_cast<size_t>(lut.edge3d) * lut.edge3d * lut.edge3d * 3
                                          : 0;

  if (lut.lut1d.size() != want_1d) {
    if (err) {
      *err = "1D LUT entries mismatch: got " + std::to_string(lut.lut1d.size()) +
             ", expected " + std::to_string(want_1d);
    }
    return false;
  }

  if (lut.lut3d.size() != want_3d) {
    if (err) {
      *err = "3D LUT entries mismatch: got " + std::to_string(lut.lut3d.size()) +
             ", expected " + std::to_string(want_3d);
    }
    return false;
  }

  detail::ComputeDomainScale(lut);
  return true;
}

inline auto ParseCubeFile(const std::filesystem::path& path, CubeLut& lut, std::string* err = nullptr)
    -> bool {
  std::ifstream ifs(path);
  if (!ifs.is_open()) {
    if (err) *err = "Failed to open file: " + path.string();
    return false;
  }
  return ParseCubeStream(ifs, lut, err);
}

inline auto ParseCubeString(std::string_view content, CubeLut& lut, std::string* err = nullptr)
    -> bool {
  std::string        owned(content);
  std::istringstream iss{owned};
  return ParseCubeStream(static_cast<std::istream&>(iss), lut, err);
}

}  // namespace puerhlab
