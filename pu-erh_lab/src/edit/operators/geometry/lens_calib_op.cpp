//  Copyright 2026 Yurun Zi
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

#include "edit/operators/geometry/lens_calib_op.hpp"

#include <lensfun/lensfun.h>

#include <filesystem>

#include "utils/string/convert.hpp"

namespace puerhlab {
static bool LoadPortableDBXMLs(lfDatabase* db, const std::filesystem::path& root) {
  if (!db) return false;
  if (root.empty() || !std::filesystem::exists(root) || !std::filesystem::is_directory(root)) {
    return false;
  }

  bool any_loaded = false;

  // Accept either:
  //  - a folder directly containing *.xml
  //  - a folder containing version_x subfolders
  //  - any nested structure; we just scan for *.xml
  for (auto const& entry : std::filesystem::recursive_directory_iterator(root)) {
    if (!entry.is_regular_file()) continue;
    auto p = entry.path();
    if (p.extension() == ".xml") {
      lfError e = lf_db_load_file(db, p.string().c_str());
      if (e == LF_NO_ERROR) {
        any_loaded = true;
      } else {
        std::cerr << "[portable-db] failed to load: " << p << " (lfError=" << int(e) << ")\n";
      }
    }
  }
  return any_loaded;
}

static const lfCamera* FindBestCamera(const lfDatabase* db, const std::string& maker,
                                      const std::string& model) {
  if (!db || maker.empty() || model.empty()) return nullptr;

  // Exact match
  const lfCamera** cameras = lf_db_find_cameras(db, maker.c_str(), model.c_str());
  if (!cameras) return nullptr;

  const lfCamera* best = cameras[0];
  lf_free((void*)cameras);
  return best;
}

static int ScoreLensCandidate(const lfLens* l, const InputMeta& in) {
  // Start from Lensfun's own fuzzy-match score.
  int s = l ? l->Score : -999999;
  if (!l) return s;

  // Add heuristics so the “first pick” aligns better with your focal/aperture metadata.
  if (in.focal_length_mm_ > 0.0f && l->MinFocal > 0.0f && l->MaxFocal > 0.0f) {
    if (in.focal_length_mm_ >= l->MinFocal - 0.2f && in.focal_length_mm_ <= l->MaxFocal + 0.2f)
      s += 2000;
    else
      s -= 2000;
  }
  if (in.aperture_f_number_ > 0.0f && l->MinAperture > 0.0f && l->MaxAperture > 0.0f) {
    if (in.aperture_f_number_ >= l->MinAperture - 0.05f &&
        in.aperture_f_number_ <= l->MaxAperture + 0.05f)
      s += 200;
    else
      s -= 200;
  }
  return s;
}

static const lfLens* FindBestLens(const lfDatabase* db, const lfCamera* cam, const InputMeta& in) {
  if (!db || in.lens_model_.empty()) return nullptr;

  int            flags       = LF_SEARCH_SORT_AND_UNIQUIFY | LF_SEARCH_LOOSE;

  const char*    makerOrNull = in.lens_maker_.empty() ? nullptr : in.lens_maker_.c_str();

  // Human-readable lens search
  const lfLens** lenses = lf_db_find_lenses_hd(db, cam, makerOrNull, in.lens_model_.c_str(), flags);
  if (!lenses) return nullptr;

  const lfLens* best       = nullptr;
  int           best_score = -999999;
  for (int i = 0; lenses[i]; ++i) {
    int score = ScoreLensCandidate(lenses[i], in);
    if (score > best_score) {
      best_score = score;
      best       = lenses[i];
    }
  }

  lf_free((void*)lenses);
  return best;
}

// FIXME: Dump interpolated coeff. to a struct that can be passed to GPU.
static void DumpInterpolatedCoeffs(const lfLens* lens, const InputMeta& in) {
  if (!lens) return;

  std::cout << "\n=== Interpolated correction metadata (for your CUDA pipeline) ===\n";
  std::cout << "Input:\n";
  std::cout << "  focal(mm)=" << in.focal_length_mm_ << ", aperture(f)=" << in.aperture_f_number_
            << ", distance(m)=" << (in.distance_m_ > 0.0f ? in.distance_m_ : 0.0f) << "\n";

  // 1) Distortion (depends on focal; coefficients in Terms[5])
  {
    lfLensCalibDistortion d{};
    bool                  ok = lens->InterpolateDistortion(in.focal_length_mm_, d);
    std::cout << "\n[Distortion]\n";
    if (!ok || d.Model == LF_DIST_MODEL_NONE) {
      std::cout << "  <no data>\n";
    } else {
      std::cout << "  model: " << lfLens::GetDistortionModelDesc(d.Model, nullptr, nullptr) << "\n";
      std::cout << "  focal(mm): " << d.Focal << "\n";
      std::cout << "  terms[5]: ";
      for (int i = 0; i < 5; ++i) std::cout << d.Terms[i] << (i + 1 < 5 ? ", " : "");
      std::cout << "\n";
    }
  }

  // 2) TCA (depends mainly on focal; coefficients in Terms[12])
  {
    lfLensCalibTCA t{};
    bool           ok = lens->InterpolateTCA(in.focal_length_mm_, t);
    std::cout << "\n[TCA]\n";
    if (!ok || t.Model == LF_TCA_MODEL_NONE) {
      std::cout << "  <no data>\n";
    } else {
      std::cout << "  model: " << lfLens::GetTCAModelDesc(t.Model, nullptr, nullptr) << "\n";
      std::cout << "  focal(mm): " << t.Focal << "\n";
      std::cout << "  terms[12]: ";
      for (int i = 0; i < 12; ++i) std::cout << t.Terms[i] << (i + 1 < 12 ? ", " : "");
      std::cout << "\n";
    }
  }

  // 3) Vignetting (depends on focal/aperture/distance; coefficients in Terms[3])
  {
    float                 dist = in.distance_m_ > 0.0f ? in.distance_m_
                                                     : 1000.0f;  // if unknown, use "far" as a pragmatic default
    lfLensCalibVignetting v{};
    bool                  ok = lens->InterpolateVignetting(in.focal_length_mm_, in.aperture_f_number_, dist, v);
    std::cout << "\n[Vignetting]\n";
    if (!ok || v.Model == LF_VIGNETTING_MODEL_NONE) {
      std::cout << "  <no data>\n";
    } else {
      std::cout << "  model: " << lfLens::GetVignettingModelDesc(v.Model, nullptr, nullptr) << "\n";
      std::cout << "  focal(mm): " << v.Focal << "\n";
      std::cout << "  aperture(f): " << v.Aperture << "\n";
      std::cout << "  distance(m): " << v.Distance << "\n";
      std::cout << "  terms[3]: " << v.Terms[0] << ", " << v.Terms[1] << ", " << v.Terms[2] << "\n";
    }
  }

  // 4) Crop (depends on focal; crop rectangle/circle data in Crop[4])
  {
    lfLensCalibCrop c{};
    bool            ok = lens->InterpolateCrop(in.focal_length_mm_, c);
    std::cout << "\n[Crop]\n";
    if (!ok || c.CropMode == LF_NO_CROP) {
      std::cout << "  <no crop>\n";
    } else {
      std::cout << "  focal(mm): " << c.Focal << "\n";
      std::cout << "  cropMode: " << int(c.CropMode) << "  (1=rect, 2=circle)\n";
      std::cout << "  crop[4]: " << c.Crop[0] << ", " << c.Crop[1] << ", " << c.Crop[2] << ", "
                << c.Crop[3] << "\n";
    }
  }

  std::cout << "\n(Next step) Feed the printed coefficients into your CUDA kernels.\n";
}

LensCalibOp::LensCalibOp(const nlohmann::json& params) { SetParams(params); }

void LensCalibOp::Apply(std::shared_ptr<ImageBuffer>) {
  throw std::runtime_error("LensCalibOp does not support CPU processing. Use ApplyGPU instead.");
}

void LensCalibOp::ApplyGPU(std::shared_ptr<ImageBuffer> input) {
  // GPU implementation of lens calibration goes here.
  // This is a placeholder and should be replaced with actual GPU code.
  // The implementation would typically involve using the input_meta_ to compute
  // the necessary transformations and applying them to the input image on the GPU.
}

auto LensCalibOp::GetParams() const -> nlohmann::json {
  nlohmann::json params;
  nlohmann::json outer;
  params["enabled"]              = enabled_;
  params["cam_maker"]            = input_meta_.cam_maker_;
  params["cam_model"]            = input_meta_.cam_model_;
  params["lens_maker"]           = input_meta_.lens_maker_;
  params["lens_model"]           = input_meta_.lens_model_;
  params["focal_length_mm"]      = input_meta_.focal_length_mm_;
  params["aperture_f_number"]    = input_meta_.aperture_f_number_;
  params["distance_m"]           = input_meta_.distance_m_;
  params["sensor_width_mm"]      = input_meta_.sensor_width_mm_;
  params["lens_profile_db_path"] = conv::ToBytes(lens_profile_db_path_.wstring());
  outer[script_name_]            = params;
  return outer;
}

void LensCalibOp::SetParams(const nlohmann::json& params) {
  nlohmann::json inner =
      params.contains(script_name_) ? params[script_name_] : nlohmann::json::object();
  enabled_                       = inner.value("enabled", true);
  input_meta_.cam_maker_         = inner.value("cam_maker", "");
  input_meta_.cam_model_         = inner.value("cam_model", "");
  input_meta_.lens_maker_        = inner.value("lens_maker", "");
  input_meta_.lens_model_        = inner.value("lens_model", "");
  input_meta_.focal_length_mm_   = inner.value("focal_length_mm", 0.0f);
  input_meta_.aperture_f_number_ = inner.value("aperture_f_number", 0.0f);
  input_meta_.distance_m_        = inner.value("distance_m", 0.0f);
  input_meta_.sensor_width_mm_   = inner.value("sensor_width_mm", 0.0f);
  lens_profile_db_path_ =
      std::filesystem::path(conv::FromBytes(inner.value("lens_profile_db_path", "")));
}

void LensCalibOp::SetGlobalParams(OperatorParams&) const {
  // This operator does not have global parameters to set.
}

void LensCalibOp::EnableGlobalParams(OperatorParams&, bool) {
  // This operator does not have global parameters to enable or disable.
}
};  // namespace puerhlab