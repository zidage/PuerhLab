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

// TODO: Migrate to static pipeline architecture

#include "decoders/processor/raw_processor.hpp"

#include <libraw/libraw.h>  // Add this header for libraw_rawdata_t
#include <libraw/libraw_const.h>
#include <opencv2/core/hal/interface.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <opencv2/core.hpp>
#include <opencv2/core/base.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/utility.hpp>
#include <sstream>
#include <string>
#include <unordered_map>

#include "decoders/processor/operators/cpu/debayer_rcd.hpp"
#include "decoders/processor/operators/cpu/raw_proc_utils.hpp"
#include "type/type.hpp"

#ifdef HAVE_CUDA
#include "decoders/processor/operators/gpu/cuda_image_ops.hpp"
#endif

#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <stdexcept>

#include "decoders/processor/operators/cpu/color_space_conv.hpp"
#include "decoders/processor/operators/cpu/debayer_ahd.hpp"
#include "decoders/processor/operators/cpu/debayer_amaze.hpp"
#include "decoders/processor/operators/cpu/highlight_reconstruct.hpp"
#include "decoders/processor/operators/cpu/white_balance.hpp"

#ifdef HAVE_CUDA
#include "decoders/processor/operators/gpu/cuda_color_space_conv.hpp"
#include "decoders/processor/operators/gpu/cuda_debayer_ahd.hpp"
#include "decoders/processor/operators/gpu/cuda_debayer_rcd.hpp"
#include "decoders/processor/operators/gpu/cuda_highlight_reconstruct.hpp"
#include "decoders/processor/operators/gpu/cuda_rotate.hpp"
#include "decoders/processor/operators/gpu/cuda_white_balance.hpp"
#endif
#include "image/image_buffer.hpp"
#include "image/metadata.hpp"
#include "json.hpp"

namespace puerhlab {
namespace {
template <typename T>
auto DownsampleBayerRGGB2xTyped(const cv::Mat& src) -> cv::Mat {
  const int out_rows = src.rows / 2;
  const int out_cols = src.cols / 2;
  cv::Mat   dst(out_rows, out_cols, src.type());

  for (int y = 0; y < out_rows; ++y) {
    const T* row0 = src.ptr<T>(2 * y);
    const T* row1 = src.ptr<T>(2 * y + 1);
    T*       drow = dst.ptr<T>(y);
    for (int x = 0; x < out_cols; ++x) {
      const int sx = 2 * x;
      if ((y & 1) == 0) {
        // R G
        // G B
        drow[x] = (x & 1) == 0 ? row0[sx] : row0[sx + 1];
      } else {
        drow[x] = (x & 1) == 0 ? row1[sx] : row1[sx + 1];
      }
    }
  }

  return dst;
}

auto DownsampleBayerRGGB2x(const cv::Mat& src) -> cv::Mat {
  switch (src.type()) {
    case CV_32FC1:
      return DownsampleBayerRGGB2xTyped<float>(src);
    case CV_16UC1:
      return DownsampleBayerRGGB2xTyped<uint16_t>(src);
    default:
      throw std::runtime_error("RawProcessor: Unsupported Bayer type for downsample");
  }
}

auto TrimTrailingZeroPadded(const char* value) -> std::string {
  if (!value) {
    return {};
  }
  std::string out(value);
  while (!out.empty() && (out.back() == '\0' || out.back() == ' ')) {
    out.pop_back();
  }
  return out;
}

auto ResolveCropFactorHint(float focal_mm, float focal_35mm_mm) -> float {
  if (!std::isfinite(focal_mm) || !std::isfinite(focal_35mm_mm) || focal_mm <= 0.0f ||
      focal_35mm_mm <= 0.0f) {
    return 0.0f;
  }
  return focal_35mm_mm / focal_mm;
}

auto IsFinitePositive(float value) -> bool { return std::isfinite(value) && value > 0.0f; }

auto ContainsCaseInsensitive(const std::string& text, const std::string& pattern) -> bool {
  if (text.empty() || pattern.empty()) {
    return false;
  }

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

struct NikonLensIdLookup {
  std::unordered_map<std::string, std::string> hex_id_map;
  std::unordered_map<uint64_t, std::string>    numeric_id_map;
  bool                                         valid = false;
};

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
    const auto byte = static_cast<unsigned>((value >> (index * 8)) & 0xFFULL);
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

void MergeMetadataHint(const ExifDisplayMetaData* metadata_hint, RawRuntimeColorContext& ctx) {
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
  if (!IsFinitePositive(ctx.focus_distance_m_) && IsFinitePositive(metadata_hint->focus_distance_m_)) {
    ctx.focus_distance_m_ = metadata_hint->focus_distance_m_;
  }
  if (!IsFinitePositive(ctx.focal_35mm_mm_) && IsFinitePositive(metadata_hint->focal_35mm_)) {
    ctx.focal_35mm_mm_ = metadata_hint->focal_35mm_;
  }

  if (!IsFinitePositive(ctx.crop_factor_hint_)) {
    ctx.crop_factor_hint_ = ResolveCropFactorHint(ctx.focal_length_mm_, ctx.focal_35mm_mm_);
  }
}

static void PrintRuntimeContext(const RawRuntimeColorContext& ctx) {
  std::cout << "Camera: " << ctx.camera_make_ << " " << ctx.camera_model_ << std::endl;

  std::cout << "Lens: " << ctx.lens_make_ << " " << ctx.lens_model_ << std::endl;
  std::cout << "Focal Length: " << ctx.focal_length_mm_ << " mm" << std::endl;
  std::cout << "Aperture: f/" << ctx.aperture_f_number_ << std::endl;
  if (std::isfinite(ctx.focus_distance_m_) && ctx.focus_distance_m_ > 0.0f) {
    std::cout << "Focus Distance: " << ctx.focus_distance_m_ << " m" << std::endl;
  }
  if (std::isfinite(ctx.focal_35mm_mm_) && ctx.focal_35mm_mm_ > 0.0f) {
    std::cout << "35mm Equivalent Focal Length: " << ctx.focal_35mm_mm_ << " mm" << std::endl;
  }
  if (std::isfinite(ctx.crop_factor_hint_) && ctx.crop_factor_hint_ > 0.0f) {
    std::cout << "Crop Factor Hint: " << ctx.crop_factor_hint_ << std::endl;
  }
}

void PopulateRuntimeColorContext(const libraw_rawdata_t& raw_data, const LibRaw& raw_processor,
                                 RawRuntimeColorContext& ctx,
                                 const ExifDisplayMetaData* metadata_hint) {
  for (int i = 0; i < 3; ++i) {
    ctx.cam_mul_[i] = raw_data.color.cam_mul[i];
    ctx.pre_mul_[i] = raw_data.color.pre_mul[i];
  }

  for (int r = 0; r < 3; ++r) {
    for (int c = 0; c < 3; ++c) {
      ctx.cam_xyz_[r * 3 + c] = raw_data.color.cam_xyz[r][c];
      ctx.rgb_cam_[r * 3 + c] = raw_data.color.rgb_cam[r][c];
    }
  }

  ctx.camera_make_       = raw_processor.imgdata.idata.make;
  ctx.camera_model_      = raw_processor.imgdata.idata.model;
  ctx.lens_make_         = TrimTrailingZeroPadded(raw_processor.imgdata.lens.LensMake);
  ctx.lens_model_        = TrimTrailingZeroPadded(raw_processor.imgdata.lens.Lens);
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

  ctx.focal_length_mm_   = raw_processor.imgdata.other.focal_len;
  if (!IsFinitePositive(ctx.focal_length_mm_)) {
    ctx.focal_length_mm_ = raw_processor.imgdata.lens.makernotes.CurFocal;
  }
  ctx.aperture_f_number_ = raw_processor.imgdata.other.aperture;
  if (!IsFinitePositive(ctx.aperture_f_number_)) {
    ctx.aperture_f_number_ = raw_processor.imgdata.lens.makernotes.CurAp;
  }
  ctx.focus_distance_m_  = 0.0f;
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
  ctx.crop_factor_hint_    = ResolveCropFactorHint(ctx.focal_length_mm_, ctx.focal_35mm_mm_);
  MergeMetadataHint(metadata_hint, ctx);

  ctx.lens_metadata_valid_ = !ctx.lens_model_.empty() && std::isfinite(ctx.focal_length_mm_) &&
                             ctx.focal_length_mm_ > 0.0f;
  ctx.valid_ = true;

  PrintRuntimeContext(ctx);
}
}  // namespace

RawProcessor::RawProcessor(const RawParams& params, const libraw_rawdata_t& rawdata,
                           LibRaw& raw_processor, const ExifDisplayMetaData* metadata_hint)
    : params_(params),
      raw_data_(rawdata),
      raw_processor_(raw_processor),
      metadata_hint_(metadata_hint) {}

void RawProcessor::SetDecodeRes() {
  // Adjust internal parameters based on decode resolution
  auto& cpu_data = process_buffer_.GetCPUData();
  switch (params_.decode_res_) {
    case DecodeRes::FULL:
      // No changes needed for full resolution
      break;
    case DecodeRes::HALF:
      // Downscale by factor of 2
      cpu_data = DownsampleBayerRGGB2x(cpu_data);
      break;
    case DecodeRes::QUARTER:
      // Downscale by factor of 4
      cpu_data = DownsampleBayerRGGB2x(DownsampleBayerRGGB2x(cpu_data));
      break;
    default:
      throw std::runtime_error("RawProcessor: Unknown decode resolution");
  }
}

void RawProcessor::ApplyLinearization() {
  auto& pre_debayer_buffer = process_buffer_;
#ifdef HAVE_CUDA
  if (params_.cuda_) {
    auto& gpu_img = pre_debayer_buffer.GetGPUData();
    CUDA::ToLinearRef(gpu_img, raw_processor_);
    return;
  }
#endif
  auto& cpu_img = pre_debayer_buffer.GetCPUData();
  // Apply "as shot" white balance multipliers for highlight reconstruction
  // This step will stretch the image to original 16-bit range
  // Because 0-65535 is mapped to 0-1 float range, we can also think this as
  // stretching to [0, 1] range
  CPU::ToLinearRef(cpu_img, raw_processor_);
}

void RawProcessor::ApplyDebayer() {
  auto& pre_debayer_buffer = process_buffer_;
#ifdef HAVE_CUDA
  if (params_.cuda_) {
    auto& gpu_img = pre_debayer_buffer.GetGPUData();
    CUDA::BayerRGGB2RGB_RCD(gpu_img);
    if (params_.decode_res_ == DecodeRes::FULL) {
      cv::Rect crop_rect(raw_data_.sizes.left_margin, raw_data_.sizes.top_margin,
                         raw_data_.sizes.width, raw_data_.sizes.height);
      gpu_img = gpu_img(crop_rect);
    }

    return;
  }
#endif
  auto& img = pre_debayer_buffer.GetCPUData();
  CPU::BayerRGGB2RGB_RCD(img);
  // Crop to valid area
  // cv::Rect crop_rect(_raw_data.sizes.raw_inset_crops[0].cleft,
  // _raw_data.sizes.raw_inset_crops[0].ctop,
  //                    _raw_data.sizes.raw_inset_crops[0].cwidth,
  //                    _raw_data.sizes.raw_inset_crops[0].cheight);
  if (params_.decode_res_ == DecodeRes::FULL) {
    cv::Rect crop_rect(raw_data_.sizes.left_margin, raw_data_.sizes.top_margin,
                       raw_data_.sizes.width, raw_data_.sizes.height);
    img = img(crop_rect);
  }
}

void RawProcessor::ApplyHighlightReconstruct() {
#ifdef HAVE_CUDA
  if (params_.cuda_) {
    auto& gpu_img = process_buffer_.GetGPUData();
    if (!params_.highlights_reconstruct_) {
      CUDA::Clamp01(gpu_img);
      return;
    }
    CUDA::HighlightReconstruct(gpu_img, raw_processor_);
    return;
  }
#endif

  auto& img = process_buffer_.GetCPUData();
  if (!params_.highlights_reconstruct_) {
    // clamp to [0, 1]
    cv::threshold(img, img, 1.0f, 1.0f, cv::THRESH_TRUNC);
    cv::threshold(img, img, 0.0f, 0.0f, cv::THRESH_TOZERO);
    return;
  }
  CPU::HighlightReconstruct(img, raw_processor_);
}

void RawProcessor::ApplyGeometricCorrections() {
  // TODO: Add lens distortion correction if needed

#ifdef HAVE_CUDA
  if (params_.cuda_) {
    auto& gpu_img = process_buffer_.GetGPUData();

    switch (raw_data_.sizes.flip) {
      case 3:
        // 180 degree
        CUDA::Rotate180(gpu_img);
        break;
      case 5:
        // Rotate 90 CCW
        CUDA::Rotate90CCW(gpu_img);
        break;
      case 6:
        // Rotate 90 CW
        CUDA::Rotate90CW(gpu_img);
        break;
      default:
        // Do nothing
        break;
    }
    return;
  }
#endif

  switch (raw_data_.sizes.flip) {
    case 3:
      // 180 degree
      cv::rotate(process_buffer_.GetCPUData(), process_buffer_.GetCPUData(), cv::ROTATE_180);
      break;
    case 5:
      // Rotate 90 CCW
      cv::rotate(process_buffer_.GetCPUData(), process_buffer_.GetCPUData(),
                 cv::ROTATE_90_COUNTERCLOCKWISE);
      break;
    case 6:
      // Rotate 90 CW
      cv::rotate(process_buffer_.GetCPUData(), process_buffer_.GetCPUData(),
                 cv::ROTATE_90_CLOCKWISE);
      break;
    default:
      // Do nothing
      break;
  }
}

void RawProcessor::ConvertToWorkingSpace() {
  auto& debayer_buffer = process_buffer_;
  auto  color_coeffs   = raw_data_.color.rgb_cam;
#ifdef HAVE_CUDA
  if (params_.cuda_) {
    auto& gpu_img = debayer_buffer.GetGPUData();
    auto  cam_mul = raw_data_.color.cam_mul;
    CUDA::ApplyInverseCamMul(gpu_img, cam_mul);
    runtime_color_context_.output_in_camera_space_ = true;
    return;
  }
#endif
  auto& img = debayer_buffer.GetCPUData();
  img.convertTo(img, CV_32FC3);
  auto pre_mul   = raw_data_.color.pre_mul;
  auto cam_mul   = raw_data_.color.cam_mul;
  auto wb_coeffs = raw_data_.color.WB_Coeffs;  // EXIF Lightsource Values
  auto cam_xyz   = raw_data_.color.cam_xyz;
  if (!params_.use_camera_wb_) {
    // User specified white balance temperature
    auto user_temp_indices = CPU::GetWBIndicesForTemp(static_cast<float>(params_.user_wb_));
    CPU::ApplyColorMatrix(img, color_coeffs, pre_mul, cam_mul, wb_coeffs, user_temp_indices,
                          params_.user_wb_, cam_xyz);
    runtime_color_context_.output_in_camera_space_ = false;
    return;
  }
  CPU::ApplyColorMatrix(img, color_coeffs, pre_mul, cam_mul, cam_xyz);
  runtime_color_context_.output_in_camera_space_ = false;
}

auto RawProcessor::Process() -> ImageBuffer {
  auto    img_unpacked = raw_data_.raw_image;
  auto&   img_sizes    = raw_data_.sizes;

  cv::Mat unpacked_mat{img_sizes.raw_height, img_sizes.raw_width, CV_16UC1, img_unpacked};
  process_buffer_ = {std::move(unpacked_mat)};

  PopulateRuntimeColorContext(raw_data_, raw_processor_, runtime_color_context_, metadata_hint_);
  runtime_color_context_.output_in_camera_space_ = false;

  // std::cout << _raw_processor.COLOR(0, 0) << " " << _raw_processor.COLOR(0, 1) << " "
  //           << _raw_processor.COLOR(1, 0) << " " << _raw_processor.COLOR(1, 1) << "\n";
  CV_Assert(raw_processor_.COLOR(0, 0) == 0 && raw_processor_.COLOR(0, 1) == 1 &&
            raw_processor_.COLOR(1, 0) == 3 && raw_processor_.COLOR(1, 1) == 2);

#ifdef HAVE_CUDA
  if (params_.cuda_) {
    // Keep raw data in 16-bit until it reaches the CUDA linearization stage.
    SetDecodeRes();
    process_buffer_.SyncToGPU();

    ApplyLinearization();
    ApplyHighlightReconstruct();
    ApplyDebayer();
    ConvertToWorkingSpace();

    // Ensure GPU buffer is CV_32FC4 (RGBA float32).
    CUDA::RGBToRGBA(process_buffer_.GetGPUData());
    ApplyGeometricCorrections();

    return {std::move(process_buffer_)};
    // process_buffer_.SyncToCPU();
    // process_buffer_.ReleaseGPUData();
  } else
#endif
  {
    // CPU pipeline expects float32 input in [0, 1].
    process_buffer_.GetCPUData().convertTo(process_buffer_.GetCPUData(), CV_32FC1, 1.0f / 65535.0f);

    SetDecodeRes();
    ApplyLinearization();
    ApplyHighlightReconstruct();
    ApplyDebayer();
    ApplyGeometricCorrections();
    ConvertToWorkingSpace();
  }

  cv::Mat final_img = cv::Mat();
  final_img.create(process_buffer_.GetCPUData().rows, process_buffer_.GetCPUData().cols, CV_32FC4);
  cv::cvtColor(process_buffer_.GetCPUData(), final_img, cv::COLOR_RGB2RGBA);
  process_buffer_ = {std::move(final_img)};

  return {std::move(process_buffer_)};
}
}  // namespace puerhlab
