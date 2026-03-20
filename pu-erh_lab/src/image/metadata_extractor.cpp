//  Copyright 2026 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#include "image/metadata_extractor.hpp"

#include <libraw/libraw.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "decoders/libraw_unpack_guard.hpp"
#include "edit/operators/basic/camera_matrices.hpp"
#include "json.hpp"
#include "type/supported_file_type.hpp"

namespace puerhlab {
namespace {
auto RationalToFloat(const Exiv2::Rational& value) -> float {
  if (value.second == 0) {
    return 0.0f;
  }
  return static_cast<float>(value.first) / static_cast<float>(value.second);
}

auto IsFinitePositive(float value) -> bool {
  return std::isfinite(value) && value > 0.0f;
}

auto TrimTrailingZeroPadded(const char* s, size_t max_len = 256) -> std::string {
  if (!s) return {};
  size_t len = std::min(std::strlen(s), max_len);
  while (len > 0 && (s[len - 1] == '\0' || std::isspace(static_cast<unsigned char>(s[len - 1])))) {
    --len;
  }
  return {s, len};
}

auto TrimAscii(const std::string& value) -> std::string {
  std::string out = value;
  while (!out.empty() && (out.back() == '\0' || std::isspace(static_cast<unsigned char>(out.back())))) {
    out.pop_back();
  }
  size_t begin = 0;
  while (begin < out.size() &&
         (out[begin] == '\0' || std::isspace(static_cast<unsigned char>(out[begin])))) {
    ++begin;
  }
  if (begin > 0) {
    out.erase(0, begin);
  }
  return out;
}

auto ContainsCaseInsensitive(const std::string& text, const std::string& pattern) -> bool {
  if (text.empty() || pattern.empty()) return false;
  auto to_lower = [](unsigned char c) { return static_cast<char>(std::tolower(c)); };
  std::string lower_text(text.size(), '\0');
  std::string lower_pattern(pattern.size(), '\0');
  std::transform(text.begin(), text.end(), lower_text.begin(), to_lower);
  std::transform(pattern.begin(), pattern.end(), lower_pattern.begin(), to_lower);
  return lower_text.find(lower_pattern) != std::string::npos;
}

auto IsNikonCamera(const std::string& make, const std::string& model) -> bool {
  return ContainsCaseInsensitive(make, "nikon") || ContainsCaseInsensitive(model, "nikon");
}

auto ResolveCropFactorHint(float focal_mm, float focal_35mm_mm) -> float {
  if (!IsFinitePositive(focal_mm) || !IsFinitePositive(focal_35mm_mm)) return 0.0f;
  return focal_35mm_mm / focal_mm;
}

// ---------------------------------------------------------------------------
//  Nikon lens ID lookup (moved from raw_processor.cpp)
// ---------------------------------------------------------------------------
struct NikonLensIdLookup {
  std::unordered_map<std::string, std::string> hex_id_map;
  std::unordered_map<uint64_t, std::string>    numeric_id_map;
  bool                                         valid = false;
};

auto NormalizeHexIdKey(const std::string& key) -> std::string {
  std::istringstream iss(key);
  std::string        token;
  std::string        out;
  bool               first = true;
  while (iss >> token) {
    if (token.size() == 1) {
      token = "0" + token;
    }
    if (token.size() > 2) {
      return {};
    }
    for (char& c : token) {
      c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
    }
    if (!first) {
      out.push_back(' ');
    }
    out += token;
    first = false;
  }
  return out;
}

auto UInt64ToHexIdKey(uint64_t value, bool little_endian) -> std::string {
  char buffer[3 * 8] = {};
  int  offset = 0;
  for (int i = 0; i < 8; ++i) {
    const int  index = little_endian ? i : (7 - i);
    const auto byte  = static_cast<unsigned>((value >> (index * 8)) & 0xFFULL);
    std::snprintf(buffer + offset, sizeof(buffer) - static_cast<size_t>(offset),
                  (i == 0) ? "%02X" : " %02X", byte);
    offset += (i == 0) ? 2 : 3;
  }
  return std::string(buffer);
}

auto LoadNikonLensIdLookup() -> NikonLensIdLookup {
  NikonLensIdLookup db;
  std::vector<std::filesystem::path> candidates;
#ifdef CONFIG_PATH
  candidates.emplace_back(std::filesystem::path(CONFIG_PATH) / "nikon_lens" / "id_map.json");
#endif
  candidates.emplace_back(std::filesystem::path("src/config/nikon_lens/id_map.json"));
  candidates.emplace_back(std::filesystem::path("pu-erh_lab/src/config/nikon_lens/id_map.json"));

  for (const auto& path : candidates) {
    std::error_code ec;
    if (!std::filesystem::exists(path, ec) || ec) {
      continue;
    }

    try {
      std::ifstream ifs(path, std::ios::binary);
      if (!ifs.is_open()) {
        continue;
      }
      nlohmann::json payload;
      ifs >> payload;

      if (payload.contains("hex_id_map") && payload["hex_id_map"].is_object()) {
        for (auto it = payload["hex_id_map"].begin(); it != payload["hex_id_map"].end(); ++it) {
          if (!it.value().is_string()) {
            continue;
          }
          const std::string key = NormalizeHexIdKey(it.key());
          if (key.empty()) {
            continue;
          }
          db.hex_id_map[key] = it.value().get<std::string>();
        }
      }

      if (payload.contains("numeric_id_map") && payload["numeric_id_map"].is_object()) {
        for (auto it = payload["numeric_id_map"].begin(); it != payload["numeric_id_map"].end();
             ++it) {
          if (!it.value().is_string()) {
            continue;
          }
          try {
            const uint64_t numeric_key = std::stoull(it.key());
            db.numeric_id_map[numeric_key] = it.value().get<std::string>();
          } catch (...) {
            continue;
          }
        }
      }

      db.valid = !db.hex_id_map.empty() || !db.numeric_id_map.empty();
      if (db.valid) {
        return db;
      }
    } catch (...) {
      continue;
    }
  }

  return db;
}

auto GetNikonLensIdLookup() -> const NikonLensIdLookup& {
  static const NikonLensIdLookup db = LoadNikonLensIdLookup();
  return db;
}

auto LookupNikonLensModelById(const libraw_lensinfo_t& lens) -> std::string {
  const auto& db = GetNikonLensIdLookup();
  if (!db.valid) {
    return {};
  }

  const uint64_t numeric_candidates[] = {
      static_cast<uint64_t>(lens.nikon.LensIDNumber),
      static_cast<uint64_t>(lens.makernotes.LensID),
  };
  for (const uint64_t id : numeric_candidates) {
    if (id == 0) {
      continue;
    }
    const auto it = db.numeric_id_map.find(id);
    if (it != db.numeric_id_map.end() && !it->second.empty()) {
      return it->second;
    }
  }

  const uint64_t hex_candidates[] = {
      static_cast<uint64_t>(lens.makernotes.LensID),
  };
  for (const uint64_t id : hex_candidates) {
    if (id == 0) {
      continue;
    }
    const std::string key_be = UInt64ToHexIdKey(id, false);
    auto it = db.hex_id_map.find(key_be);
    if (it != db.hex_id_map.end() && !it->second.empty()) {
      return it->second;
    }

    const std::string key_le = UInt64ToHexIdKey(id, true);
    it = db.hex_id_map.find(key_le);
    if (it != db.hex_id_map.end() && !it->second.empty()) {
      return it->second;
    }
  }

  return {};
}

auto ResolveNikonLensModel(const libraw_lensinfo_t& lens) -> std::string {
  std::string candidate = TrimTrailingZeroPadded(lens.makernotes.Lens);
  if (!candidate.empty()) {
    return candidate;
  }

  std::string mapped = LookupNikonLensModelById(lens);
  if (!mapped.empty()) {
    return mapped;
  }

  const auto& nikon = lens.nikon;
  const bool has_nikon_signature = (nikon.LensIDNumber != 0 || nikon.LensType != 0 ||
                                    nikon.MCUVersion != 0 || nikon.LensFStops != 0 ||
                                    IsFinitePositive(nikon.EffectiveMaxAp));
  if (!has_nikon_signature) {
    return {};
  }

  char model_buf[192] = {};
  std::snprintf(model_buf, sizeof(model_buf),
                "Nikon LensID %u (type=0x%02X mcu=%u fStops=%u effMaxAp=%.2f)",
                static_cast<unsigned>(nikon.LensIDNumber),
                static_cast<unsigned>(nikon.LensType),
                static_cast<unsigned>(nikon.MCUVersion),
                static_cast<unsigned>(nikon.LensFStops),
                static_cast<double>(nikon.EffectiveMaxAp));

  std::string model = model_buf;
  if (IsFinitePositive(lens.MinFocal) && IsFinitePositive(lens.MaxFocal)) {
    char focal_buf[64] = {};
    if (std::fabs(lens.MinFocal - lens.MaxFocal) < 1e-4f) {
      std::snprintf(focal_buf, sizeof(focal_buf), " %.1fmm", static_cast<double>(lens.MinFocal));
    } else {
      std::snprintf(focal_buf, sizeof(focal_buf), " %.1f-%.1fmm",
                    static_cast<double>(lens.MinFocal), static_cast<double>(lens.MaxFocal));
    }
    model += focal_buf;
  }
  return model;
}

// ---------------------------------------------------------------------------
//  Adobe DNG camera colour-matrix database lookup
//  (ported from color_temp_op.cpp – executed once at import time)
// ---------------------------------------------------------------------------

auto NormalizeCameraName(const std::string& input) -> std::string {
  std::string normalized;
  normalized.reserve(input.size() + input.size() / 4);

  bool last_space = true;
  bool last_alpha = false;
  bool last_digit = false;
  for (char ch : input) {
    const unsigned char uch = static_cast<unsigned char>(ch);
    if (std::isalnum(uch)) {
      const bool is_alpha = std::isalpha(uch) != 0;
      const bool is_digit = std::isdigit(uch) != 0;
      // Insert space at letter↔digit boundary so "Z5" → "z 5", "5D" → "5 d".
      if (!last_space && ((last_alpha && is_digit) || (last_digit && is_alpha))) {
        normalized.push_back(' ');
      }
      normalized.push_back(static_cast<char>(std::tolower(uch)));
      last_space = false;
      last_alpha = is_alpha;
      last_digit = is_digit;
    } else if (!last_space) {
      normalized.push_back(' ');
      last_space = true;
      last_alpha = false;
      last_digit = false;
    }
  }

  while (!normalized.empty() && normalized.back() == ' ') {
    normalized.pop_back();
  }
  return normalized;
}

auto CompactNormalizedCameraName(std::string normalized) -> std::string {
  normalized.erase(std::remove(normalized.begin(), normalized.end(), ' '), normalized.end());
  return normalized;
}

auto CompactCameraName(const std::string& input) -> std::string {
  return CompactNormalizedCameraName(NormalizeCameraName(input));
}

auto TokenizeCameraNameLoose(const std::string& input) -> std::vector<std::string> {
  std::vector<std::string> tokens;
  std::string              token;
  token.reserve(input.size());

  for (char ch : input) {
    const unsigned char uch = static_cast<unsigned char>(ch);
    if (std::isalnum(uch)) {
      token.push_back(static_cast<char>(std::tolower(uch)));
    } else if (!token.empty()) {
      tokens.push_back(std::move(token));
      token.clear();
    }
  }

  if (!token.empty()) {
    tokens.push_back(std::move(token));
  }
  return tokens;
}

auto IsMeaningfulCameraMatchToken(const std::string& token) -> bool {
  if (token.size() >= 2) {
    return true;
  }
  return !token.empty() && std::isdigit(static_cast<unsigned char>(token.front())) != 0;
}

auto UniqueMeaningfulCameraTokens(const std::vector<std::string>& tokens) -> std::vector<std::string> {
  std::vector<std::string> out;
  out.reserve(tokens.size());
  for (const auto& token : tokens) {
    if (!IsMeaningfulCameraMatchToken(token)) {
      continue;
    }
    if (std::find(out.begin(), out.end(), token) == out.end()) {
      out.push_back(token);
    }
  }
  return out;
}

auto ContainsCameraToken(const std::vector<std::string>& tokens, const std::string& token) -> bool {
  return std::find(tokens.begin(), tokens.end(), token) != tokens.end();
}

auto ContainsCameraTokenFragment(const std::vector<std::string>& tokens, const std::string& fragment)
    -> bool {
  if (fragment.empty()) {
    return false;
  }
  return std::any_of(tokens.begin(), tokens.end(), [&](const std::string& token) {
    return token.find(fragment) != std::string::npos;
  });
}

auto ResolveCameraColorMatrixAlias(const std::string& camera_make,
                                   const std::string& camera_model) -> std::string {
  const auto make_tokens  = TokenizeCameraNameLoose(camera_make);
  const auto model_tokens = TokenizeCameraNameLoose(camera_model);

  const bool is_hasselblad =
      ContainsCameraToken(make_tokens, "hasselblad") ||
      ContainsCameraToken(model_tokens, "hasselblad");
  if (!is_hasselblad) {
    return {};
  }

  const bool has_x2d  = ContainsCameraToken(model_tokens, "x2d") ||
                       ContainsCameraTokenFragment(model_tokens, "x2d");
  const bool has_100c = ContainsCameraToken(model_tokens, "100c") ||
                        ContainsCameraTokenFragment(model_tokens, "100c");
  if (!has_x2d || !has_100c) {
    return {};
  }

  if (ContainsCameraToken(model_tokens, "ii") ||
      ContainsCameraTokenFragment(model_tokens, "x2dii")) {
    return NormalizeCameraName("Hasselblad 100-22-Coated6");
  }
  return NormalizeCameraName("Hasselblad 100-20-Coated6");
}

struct CameraColorMatrixEntry {
  std::string name_;
  std::string compact_name_;
  std::vector<std::string> match_tokens_;
  double      cm1_[9];
  double      cm2_[9];
};

auto CameraColorMatrixDatabaseSorted() -> const std::vector<CameraColorMatrixEntry>& {
  static const std::vector<CameraColorMatrixEntry> index = [] {
    std::vector<CameraColorMatrixEntry> out;
    const size_t count = sizeof(all_camera_matrices) / sizeof(all_camera_matrices[0]);
    out.reserve(count);

    for (size_t i = 0; i < count; ++i) {
      const auto& item = all_camera_matrices[i];
      auto        key  = NormalizeCameraName(item.camera_name_);
      if (key.empty()) {
        continue;
      }
      CameraColorMatrixEntry entry;
      entry.name_         = std::move(key);
      entry.compact_name_ = CompactNormalizedCameraName(entry.name_);
      entry.match_tokens_ = UniqueMeaningfulCameraTokens(TokenizeCameraNameLoose(item.camera_name_));
      std::memcpy(entry.cm1_, item.color_matrix_1_, sizeof(entry.cm1_));
      std::memcpy(entry.cm2_, item.color_matrix_2_, sizeof(entry.cm2_));
      out.push_back(std::move(entry));
    }

    // Normalization can change ordering (e.g. "7D" -> "7 d"), so sort before binary search.
    std::sort(out.begin(), out.end(),
              [](const CameraColorMatrixEntry& a, const CameraColorMatrixEntry& b) {
                return a.name_ < b.name_;
              });

    // Deduplicate normalized keys while keeping the first matrix entry.
    out.erase(std::unique(out.begin(), out.end(),
                          [](const CameraColorMatrixEntry& a, const CameraColorMatrixEntry& b) {
                            return a.name_ == b.name_;
                          }),
              out.end());
    return out;
  }();
  return index;
}

auto FindInSortedColorMatrixDB(const std::vector<CameraColorMatrixEntry>& db,
                               const std::string& key) -> const CameraColorMatrixEntry* {
  if (key.empty()) {
    return nullptr;
  }
  auto it = std::lower_bound(
      db.begin(), db.end(), key,
      [](const CameraColorMatrixEntry& entry, const std::string& k) {
        return entry.name_ < k;
      });
  if (it != db.end() && it->name_ == key) {
    return &(*it);
  }
  return nullptr;
}

struct CameraMatrixCandidateScore {
  int total_score  = 0;
  int model_signal = 0;
};

auto ScoreApproximateColorMatrixCandidate(const CameraColorMatrixEntry& entry,
                                          const std::string& make_key,
                                          const std::string& make_compact,
                                          const std::string& full_compact,
                                          const std::string& model_compact,
                                          const std::vector<std::string>& model_tokens)
    -> CameraMatrixCandidateScore {
  CameraMatrixCandidateScore score;

  if (!make_key.empty()) {
    if (entry.name_ == make_key || entry.name_.starts_with(make_key + " ")) {
      score.total_score += 1200;
    } else if (!make_compact.empty() &&
               entry.compact_name_.find(make_compact) != std::string::npos) {
      score.total_score += 700;
    }
  }

  if (!full_compact.empty()) {
    if (entry.compact_name_ == full_compact) {
      score.model_signal += 900;
    } else if (full_compact.size() >= 6 &&
               (entry.compact_name_.find(full_compact) != std::string::npos ||
                full_compact.find(entry.compact_name_) != std::string::npos)) {
      score.model_signal += 260;
    }
  }

  if (!model_compact.empty()) {
    if (entry.compact_name_ == model_compact ||
        entry.compact_name_.ends_with(model_compact)) {
      score.model_signal += 700;
    } else if (model_compact.size() >= 4 &&
               (entry.compact_name_.find(model_compact) != std::string::npos ||
                model_compact.find(entry.compact_name_) != std::string::npos)) {
      score.model_signal += 260;
    }
  }

  for (const auto& token : model_tokens) {
    if (ContainsCameraToken(entry.match_tokens_, token)) {
      score.model_signal += 120;
      continue;
    }
    if (token.size() < 3) {
      continue;
    }
    const bool partial_match = std::any_of(
        entry.match_tokens_.begin(), entry.match_tokens_.end(),
        [&](const std::string& candidate_token) {
          return candidate_token.find(token) != std::string::npos ||
                 token.find(candidate_token) != std::string::npos;
        });
    if (partial_match) {
      score.model_signal += 40;
    }
  }

  score.total_score += score.model_signal;
  score.total_score -=
      static_cast<int>(std::abs(static_cast<int>(entry.match_tokens_.size()) -
                                static_cast<int>(model_tokens.size()))) *
      15;
  return score;
}

auto FindApproximateColorMatrixMatch(const std::vector<CameraColorMatrixEntry>& db,
                                     const std::string& camera_make,
                                     const std::string& camera_model)
    -> const CameraColorMatrixEntry* {
  const auto make_key     = NormalizeCameraName(camera_make);
  const auto make_compact = CompactNormalizedCameraName(make_key);
  const auto full_compact = CompactCameraName(camera_make + " " + camera_model);
  const auto model_compact = CompactCameraName(camera_model);
  auto       model_tokens  = UniqueMeaningfulCameraTokens(TokenizeCameraNameLoose(camera_model));
  if (model_tokens.empty()) {
    model_tokens = UniqueMeaningfulCameraTokens(TokenizeCameraNameLoose(camera_make + " " +
                                                                        camera_model));
  }
  if (model_tokens.empty() && model_compact.empty()) {
    return nullptr;
  }

  const CameraColorMatrixEntry* best          = nullptr;
  int                           best_score    = -1;
  int                           runner_up     = -1;
  for (const auto& entry : db) {
    const auto score = ScoreApproximateColorMatrixCandidate(
        entry, make_key, make_compact, full_compact, model_compact, model_tokens);
    if (score.model_signal < 120 || score.total_score < 650) {
      continue;
    }
    if (score.total_score > best_score) {
      runner_up  = best_score;
      best_score = score.total_score;
      best       = &entry;
    } else if (score.total_score > runner_up) {
      runner_up = score.total_score;
    }
  }

  if (!best) {
    return nullptr;
  }
  if (runner_up >= 0 && best_score < runner_up + 140) {
    return nullptr;
  }
  return best;
}

/// Look up Adobe DNG colour matrices for a camera make/model pair and store
/// the result directly into the provided double[9] arrays.
auto LookupCameraColorMatrices(const std::string& camera_make,
                               const std::string& camera_model,
                               double cm1_out[9], double cm2_out[9]) -> bool {
  const auto full_key  = NormalizeCameraName(camera_make + " " + camera_model);
  const auto model_key = NormalizeCameraName(camera_model);

  const auto& db = CameraColorMatrixDatabaseSorted();

  const CameraColorMatrixEntry* found = FindInSortedColorMatrixDB(db, full_key);
  if (!found) {
    found = FindInSortedColorMatrixDB(db, model_key);
  }
  if (!found && !model_key.empty()) {
    const auto make_key = NormalizeCameraName(camera_make);
    if (!make_key.empty() && model_key.starts_with(make_key + " ")) {
      found = FindInSortedColorMatrixDB(db, model_key);
    }
  }
  if (!found) {
    const auto alias_key = ResolveCameraColorMatrixAlias(camera_make, camera_model);
    if (!alias_key.empty()) {
      found = FindInSortedColorMatrixDB(db, alias_key);
    }
  }
  if (!found) {
    found = FindApproximateColorMatrixMatch(db, camera_make, camera_model);
  }

  if (found) {
    std::memcpy(cm1_out, found->cm1_, 9 * sizeof(double));
    std::memcpy(cm2_out, found->cm2_, 9 * sizeof(double));
    return true;
  }
  return false;
}

/// Populate a RawRuntimeColorContext directly from libraw's open-but-not-processed state.
/// Only requires open_file / unpack to have been called so that imgdata.rawdata.color,
/// imgdata.idata, imgdata.other, imgdata.lens are populated.
void PopulateMetadataRuntimeContext(LibRaw& raw_processor, RawRuntimeColorContext& ctx) {
  const auto& color = raw_processor.imgdata.rawdata.color;
  for (int i = 0; i < 3; ++i) {
    ctx.cam_mul_[i] = color.cam_mul[i];
    ctx.pre_mul_[i] = color.pre_mul[i];
  }
  for (int r = 0; r < 3; ++r) {
    for (int c = 0; c < 3; ++c) {
      ctx.cam_xyz_[r * 3 + c] = color.cam_xyz[r][c];
      ctx.rgb_cam_[r * 3 + c] = color.rgb_cam[r][c];
    }
  }

  ctx.camera_make_  = TrimAscii(raw_processor.imgdata.idata.make);
  ctx.camera_model_ = TrimAscii(raw_processor.imgdata.idata.model);

  ctx.lens_make_  = TrimTrailingZeroPadded(raw_processor.imgdata.lens.LensMake);
  ctx.lens_model_ = TrimTrailingZeroPadded(raw_processor.imgdata.lens.Lens);
  if (ctx.lens_model_.empty()) {
    ctx.lens_model_ = TrimTrailingZeroPadded(raw_processor.imgdata.lens.makernotes.Lens);
  }

  if (IsNikonCamera(ctx.camera_make_, ctx.camera_model_)) {
    if (ctx.lens_make_.empty()) {
      ctx.lens_make_ = "Nikon";
    }
    if (ctx.lens_model_.empty()) {
      ctx.lens_model_ = ResolveNikonLensModel(raw_processor.imgdata.lens);
    }
  }

  ctx.focal_length_mm_ = raw_processor.imgdata.other.focal_len;
  if (!IsFinitePositive(ctx.focal_length_mm_)) {
    ctx.focal_length_mm_ = raw_processor.imgdata.lens.makernotes.CurFocal;
  }
  ctx.aperture_f_number_ = raw_processor.imgdata.other.aperture;
  if (!IsFinitePositive(ctx.aperture_f_number_)) {
    ctx.aperture_f_number_ = raw_processor.imgdata.lens.makernotes.CurAp;
  }
  ctx.focus_distance_m_ = 0.0f;
  if (std::isfinite(raw_processor.imgdata.lens.makernotes.FocusRangeIndex) &&
      raw_processor.imgdata.lens.makernotes.FocusRangeIndex > 0.0f) {
    ctx.focus_distance_m_ = raw_processor.imgdata.lens.makernotes.FocusRangeIndex;
  }

  ctx.focal_35mm_mm_ = 0.0f;
  if (raw_processor.imgdata.lens.FocalLengthIn35mmFormat > 0) {
    ctx.focal_35mm_mm_ = static_cast<float>(raw_processor.imgdata.lens.FocalLengthIn35mmFormat);
  } else if (raw_processor.imgdata.lens.makernotes.FocalLengthIn35mmFormat > 0) {
    ctx.focal_35mm_mm_ =
        static_cast<float>(raw_processor.imgdata.lens.makernotes.FocalLengthIn35mmFormat);
  }
  ctx.crop_factor_hint_ = ResolveCropFactorHint(ctx.focal_length_mm_, ctx.focal_35mm_mm_);

  ctx.lens_metadata_valid_ = !ctx.lens_model_.empty() && std::isfinite(ctx.focal_length_mm_) &&
                             ctx.focal_length_mm_ > 0.0f;

  // Resolve Adobe DNG colour matrices from the camera-model database once and
  // store them in the context so that downstream operators (ColorTempOp) never
  // need to repeat the lookup.
  ctx.color_matrices_valid_ = LookupCameraColorMatrices(
      ctx.camera_make_, ctx.camera_model_,
      ctx.color_matrix_1_, ctx.color_matrix_2_);

  // Mark as valid for downstream consumers (not an actual decode, just metadata)
  ctx.valid_                  = true;
  ctx.output_in_camera_space_ = true;  // no processing done, color data is camera-space
}

/// Populate ExifDisplayMetaData from a RawRuntimeColorContext + libraw other/sizes data.
void PopulateDisplayMetadataFromLibRaw(LibRaw& raw_processor, const RawRuntimeColorContext& ctx,
                                       ExifDisplayMetaData& display) {
  display.make_          = ctx.camera_make_;
  display.model_         = ctx.camera_model_;
  display.lens_          = ctx.lens_model_;
  display.lens_make_     = ctx.lens_make_;
  display.focal_         = ctx.focal_length_mm_;
  display.focal_35mm_    = ctx.focal_35mm_mm_;
  display.aperture_      = ctx.aperture_f_number_;
  display.focus_distance_m_ = ctx.focus_distance_m_;

  display.iso_ = static_cast<uint64_t>(raw_processor.imgdata.other.iso_speed);

  const float shutter_sec = raw_processor.imgdata.other.shutter;
  if (std::isfinite(shutter_sec) && shutter_sec > 0.0f) {
    if (shutter_sec >= 1.0f) {
      display.shutter_speed_ = {static_cast<int>(shutter_sec), 1};
    } else {
      display.shutter_speed_ = {1, static_cast<int>(1.0f / shutter_sec + 0.5f)};
    }
  }

  display.width_  = static_cast<uint32_t>(raw_processor.imgdata.sizes.width);
  display.height_ = static_cast<uint32_t>(raw_processor.imgdata.sizes.height);

  // Timestamp
  const time_t ts = raw_processor.imgdata.other.timestamp;
  if (ts > 0) {
    struct tm t {};
#if defined(_WIN32)
    gmtime_s(&t, &ts);
#else
    gmtime_r(&ts, &t);
#endif
    char buf[64] = {};
    std::strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", &t);
    display.date_time_str_ = buf;
  }
}

/// Set of raw file extensions (lowercase).
static const std::unordered_set<std::string> kRawExtensions = {
    ".arw", ".cr2", ".cr3", ".nef", ".dng", ".raw", ".raf", ".3fr", ".rw2"};

auto IsRawExtension(const std::filesystem::path& path) -> bool {
  std::string ext = path.extension().string();
  std::transform(ext.begin(), ext.end(), ext.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return kRawExtensions.count(ext) > 0;
}
}  // namespace

void MetadataExtractor::MergeMetadataHint(const ExifDisplayMetaData* metadata_hint,
                                          RawRuntimeColorContext&    ctx) {
  if (!metadata_hint) {
    return;
  }

  const std::string hint_make      = TrimAscii(metadata_hint->make_);
  const std::string hint_model     = TrimAscii(metadata_hint->model_);
  const std::string hint_lens_make = TrimAscii(metadata_hint->lens_make_);
  const std::string hint_lens      = TrimAscii(metadata_hint->lens_);

  if (ctx.camera_make_.empty() && !hint_make.empty()) {
    ctx.camera_make_ = hint_make;
  }
  if (ctx.camera_model_.empty() && !hint_model.empty()) {
    ctx.camera_model_ = hint_model;
  }
  if (ctx.lens_make_.empty() && !hint_lens_make.empty()) {
    ctx.lens_make_ = hint_lens_make;
  }
  if (ctx.lens_model_.empty() && !hint_lens.empty()) {
    ctx.lens_model_ = hint_lens;
  }

  if (!IsFinitePositive(ctx.focal_length_mm_) && IsFinitePositive(metadata_hint->focal_)) {
    ctx.focal_length_mm_ = metadata_hint->focal_;
  }
  if (!IsFinitePositive(ctx.aperture_f_number_) && IsFinitePositive(metadata_hint->aperture_)) {
    ctx.aperture_f_number_ = metadata_hint->aperture_;
  }
  if (!IsFinitePositive(ctx.focus_distance_m_) &&
      IsFinitePositive(metadata_hint->focus_distance_m_)) {
    ctx.focus_distance_m_ = metadata_hint->focus_distance_m_;
  }
  if (!IsFinitePositive(ctx.focal_35mm_mm_) && IsFinitePositive(metadata_hint->focal_35mm_)) {
    ctx.focal_35mm_mm_ = metadata_hint->focal_35mm_;
  }

  if (!IsFinitePositive(ctx.crop_factor_hint_)) {
    ctx.crop_factor_hint_ = ResolveCropFactorHint(ctx.focal_length_mm_, ctx.focal_35mm_mm_);
  }
}

void MetadataExtractor::PopulateRuntimeContextFromOpenLibRaw(LibRaw&                 raw_processor,
                                                             RawRuntimeColorContext& ctx) {
  PopulateMetadataRuntimeContext(raw_processor, ctx);
}

static void GetDisplayMetadataFromExif(Exiv2::ExifData&     exif_data,
                                       ExifDisplayMetaData& display_metadata) {
  if (exif_data.empty()) {
    return;
  }
  if (exif_data.findKey(Exiv2::ExifKey("Exif.Image.Make")) != exif_data.end()) {
    display_metadata.make_ = exif_data["Exif.Image.Make"].toString();
  }
  if (exif_data.findKey(Exiv2::ExifKey("Exif.Image.Model")) != exif_data.end()) {
    display_metadata.model_ = exif_data["Exif.Image.Model"].toString();
  }
  if (exif_data.findKey(Exiv2::ExifKey("Exif.Photo.LensModel")) != exif_data.end()) {
    display_metadata.lens_ = exif_data["Exif.Photo.LensModel"].toString();
  }
  if (exif_data.findKey(Exiv2::ExifKey("Exif.Photo.LensMake")) != exif_data.end()) {
    display_metadata.lens_make_ = exif_data["Exif.Photo.LensMake"].toString();
  }
  if (exif_data.findKey(Exiv2::ExifKey("Exif.Photo.FNumber")) != exif_data.end()) {
    display_metadata.aperture_ = RationalToFloat(exif_data["Exif.Photo.FNumber"].toRational());
  }
  if (exif_data.findKey(Exiv2::ExifKey("Exif.Photo.FocalLength")) != exif_data.end()) {
    display_metadata.focal_ = RationalToFloat(exif_data["Exif.Photo.FocalLength"].toRational());
  }
  if (exif_data.findKey(Exiv2::ExifKey("Exif.Photo.FocalLengthIn35mmFilm")) != exif_data.end()) {
    display_metadata.focal_35mm_ =
        static_cast<float>(exif_data["Exif.Photo.FocalLengthIn35mmFilm"].toInt64());
  }
  if (exif_data.findKey(Exiv2::ExifKey("Exif.Photo.SubjectDistance")) != exif_data.end()) {
    display_metadata.focus_distance_m_ =
        RationalToFloat(exif_data["Exif.Photo.SubjectDistance"].toRational());
  }
  if (exif_data.findKey(Exiv2::ExifKey("Exif.Photo.ISOSpeedRatings")) != exif_data.end()) {
    display_metadata.iso_ = exif_data["Exif.Photo.ISOSpeedRatings"].toInt64();
  }
  if (exif_data.findKey(Exiv2::ExifKey("Exif.Image.ShutterSpeedValue")) != exif_data.end()) {
    display_metadata.shutter_speed_ = exif_data["Exif.Image.ShutterSpeedValue"].toRational();
  }
  if (exif_data.findKey(Exiv2::ExifKey("Exif.Image.ImageLength")) != exif_data.end()) {
    display_metadata.height_ = exif_data["Exif.Image.ImageLength"].toUint32();
  }
  if (exif_data.findKey(Exiv2::ExifKey("Exif.Image.ImageWidth")) != exif_data.end()) {
    display_metadata.width_ = exif_data["Exif.Image.ImageWidth"].toUint32();
  }
  if (exif_data.findKey(Exiv2::ExifKey("Exif.Image.DateTime")) != exif_data.end()) {
    display_metadata.date_time_str_ = exif_data["Exif.Image.DateTime"].toString();
    if (display_metadata.date_time_str_.size() >= 10) {
      display_metadata.date_time_str_[4] =
          '-';  // Change from "YYYY:MM:DD HH:MM:SS" to "YYYY-MM-DD HH:MM:SS"
      display_metadata.date_time_str_[7] = '-';
    }
  }
}

auto MetadataExtractor::ExtractEXIF(const image_path_t& image_path) -> Exiv2::Image::UniquePtr {
  Exiv2::Image::UniquePtr image = Exiv2::ImageFactory::open(image_path.string());
  image->readMetadata();
  return image;
}

auto MetadataExtractor::ExtractEXIFFromBuffer(const uint8_t* buffer, size_t size)
    -> Exiv2::Image::UniquePtr {
  if (!buffer || size == 0) {
    throw std::runtime_error("MetadataExtractor: empty buffer");
  }
  Exiv2::Image::UniquePtr image =
      Exiv2::ImageFactory::open(reinterpret_cast<const Exiv2::byte*>(buffer), size);
  image->readMetadata();
  return image;
}

auto MetadataExtractor::EXIFToDisplayMetaData(const Exiv2::Image::UniquePtr& exif_data)
    -> ExifDisplayMetaData {
  ExifDisplayMetaData display_metadata;
  if (exif_data->exifData().empty()) {
    return display_metadata;
  }
  GetDisplayMetadataFromExif(exif_data->exifData(), display_metadata);
  return display_metadata;
}

auto MetadataExtractor::BufferToDisplayMetaData(const uint8_t* buffer, size_t size)
    -> ExifDisplayMetaData {
  ExifDisplayMetaData display_metadata;
  try {
    auto exif_data = ExtractEXIFFromBuffer(buffer, size);
    if (!exif_data || exif_data->exifData().empty()) {
      return display_metadata;
    }
    GetDisplayMetadataFromExif(exif_data->exifData(), display_metadata);
  } catch (...) {
    return ExifDisplayMetaData{};
  }
  return display_metadata;
}


auto MetadataExtractor::EXIFToJSON(const Exiv2::Image::UniquePtr& exif_data) -> nlohmann::json {
  // The full EXIF is too large, we only convert the display-friendly metadata to JSON
  nlohmann::json exif_json;
  auto           display_metadata = EXIFToDisplayMetaData(exif_data);
  exif_json                       = display_metadata.ToJson();
  return exif_json;
}

void MetadataExtractor::ExtractEXIF_ToImage(const image_path_t& image_path, Image& image) {
  // For raw files, prefer the libraw-based extraction path which also provides
  // RawRuntimeColorContext for pipeline operators.
  if (IsRawExtension(image_path)) {
    if (ExtractRawMetadata_ToImage(image_path, image)) {
      return;
    }
    // Fall through to Exiv2 if libraw extraction fails
  }

  auto exif_data = ExtractEXIF(image_path);
  if (exif_data->exifData().empty()) {
    return;
  }
  ExifDisplayMetaData display_metadata;
  GetDisplayMetadataFromExif(exif_data->exifData(), display_metadata);
  image.SetExifDisplayMetaData(std::move(display_metadata));
}

auto MetadataExtractor::ExtractRawMetadata_ToImage(const image_path_t& image_path, Image& image)
    -> bool {
  // LibRaw is large enough to blow worker-thread stacks under ASan if allocated locally.
  auto raw_processor = std::make_unique<LibRaw>();

#if defined(_WIN32)
  int ret = raw_processor->open_file(image_path.wstring().c_str());
#else
  int ret = raw_processor->open_file(image_path.string().c_str());
#endif
  if (ret != LIBRAW_SUCCESS) {
    std::cerr << "MetadataExtractor: libraw open_file failed for '"
              << image_path.string() << "' (error " << ret << ")" << std::endl;
    return false;
  }

  ret = libraw_guard::Unpack(*raw_processor);
  if (ret != LIBRAW_SUCCESS) {
    std::cerr << "MetadataExtractor: libraw unpack failed for '"
              << image_path.string() << "' (error " << ret << ")" << std::endl;
    raw_processor->recycle();
    return false;
  }

  RawRuntimeColorContext ctx{};
  PopulateMetadataRuntimeContext(*raw_processor, ctx);

  ExifDisplayMetaData display{};
  PopulateDisplayMetadataFromLibRaw(*raw_processor, ctx, display);

  raw_processor->recycle();

  image.SetExifDisplayMetaData(std::move(display));
  image.SetRawColorContext(std::move(ctx));
  return true;
}
}  // namespace puerhlab
