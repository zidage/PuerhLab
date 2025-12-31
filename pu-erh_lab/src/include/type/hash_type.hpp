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

#include <xxhash.h>

#include <array>
#include <cstdint>
#include <functional>
#include <iomanip>
#include <sstream>
#include <string>

namespace puerhlab {
class Hash128 {
 public:
  Hash128() : _h{0, 0} {}
  explicit Hash128(const XXH128_hash_t& h) : _h(h) {}

  Hash128(uint64_t low, uint64_t high) : _h{low, high} {}

  uint64_t                low64() const { return _h.low64; }
  uint64_t                high64() const { return _h.high64; }

  std::array<uint8_t, 16> ToBytes() const {
    std::array<uint8_t, 16> arr{};
    uint64_t                lo = _h.low64;
    uint64_t                hi = _h.high64;
    for (int i = 0; i < 8; i++) {
      arr[i]     = static_cast<uint8_t>(lo & 0xFF);
      arr[i + 8] = static_cast<uint8_t>(hi & 0xFF);
      lo >>= 8;
      hi >>= 8;
    }
    return arr;
  }

  std::string ToString() const {
    std::ostringstream oss;
    oss << std::hex << std::setfill('0');
    oss << std::setw(16) << _h.high64;
    oss << std::setw(16) << _h.low64;
    return oss.str();
  }

  static Hash128 FromString(const std::string& str) {
    if (str.length() != 32) {
      throw std::invalid_argument("Hash128::FromString: Invalid string length");
    }
    uint64_t high = std::stoull(str.substr(0, 16), nullptr, 16);
    uint64_t low  = std::stoull(str.substr(16, 16), nullptr, 16);
    return Hash128(XXH128_hash_t{low, high});
  }

  std::size_t ToSizeT() const noexcept { return static_cast<std::size_t>(_h.low64 ^ _h.high64); }

  bool        operator==(const Hash128& other) const noexcept {
    return _h.low64 == other._h.low64 && _h.high64 == other._h.high64;
  }
  bool operator!=(const Hash128& other) const noexcept { return !(*this == other); }
  bool operator<(const Hash128& other) const noexcept {
    return (_h.high64 < other._h.high64) ||
           (_h.high64 == other._h.high64 && _h.low64 < other._h.low64);
  }

  const XXH128_hash_t& Raw() const { return _h; }

  static Hash128       Compute(const void* data, size_t length, uint64_t seed = 0) {
    XXH128_hash_t h = XXH3_128bits_withSeed(data, length, seed);
    return Hash128(h);
  }

  static Hash128 Blend(const Hash128& h1, const Hash128& h2) {
    XXH128_hash_t mix;
    mix.low64  = h1.low64() ^ (h2.high64() + 0x9e3779b97f4a7c15ULL);
    mix.high64 = h1.high64() ^ (h2.low64() + 0x85ebca77c2b2ae63ULL);
    return Hash128(mix);
  }

 private:
  XXH128_hash_t _h;
};
};  // namespace puerhlab

namespace std {
template <>
struct hash<puerhlab::Hash128> {
  std::size_t operator()(const puerhlab::Hash128& h) const noexcept {
    // Better hash combining, from Boost
    auto h1 = std::hash<uint64_t>{}(h.low64());
    auto h2 = std::hash<uint64_t>{}(h.high64());
    return h1 ^ (h2 + 0x9e3779b97f4a7c15ULL + (h1 << 6) + (h1 >> 2));
  }
};
}  // namespace std