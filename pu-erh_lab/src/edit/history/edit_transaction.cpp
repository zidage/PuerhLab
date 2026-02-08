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

#include "edit/history/edit_transaction.hpp"

#include <algorithm>
#include <sstream>
#include <string>
#include <vector>

#include "edit/pipeline/pipeline.hpp"

namespace puerhlab {
namespace {
static std::string TruncateForUi(std::string s, std::size_t max_chars) {
  if (max_chars == 0 || s.size() <= max_chars) {
    return s;
  }
  if (max_chars <= 3) {
    return s.substr(0, max_chars);
  }
  s.resize(max_chars - 3);
  s += "...";
  return s;
}

static std::string JsonScalarToString(const nlohmann::json& v) {
  if (v.is_string()) {
    return v.get<std::string>();
  }
  if (v.is_number_float()) {
    std::ostringstream oss;
    oss.setf(std::ios::fixed);
    oss.precision(3);
    oss << v.get<double>();
    return oss.str();
  }
  if (v.is_number_integer()) {
    return std::to_string(v.get<long long>());
  }
  if (v.is_number_unsigned()) {
    return std::to_string(v.get<unsigned long long>());
  }
  if (v.is_boolean()) {
    return v.get<bool>() ? "true" : "false";
  }
  if (v.is_null()) {
    return "null";
  }
  return v.dump();
}

static bool IsLutPathEmpty(const nlohmann::json& params) {
  if (!params.is_object() || !params.contains("ocio_lmt")) {
    return true;
  }
  try {
    const auto path = params["ocio_lmt"].get<std::string>();
    return path.empty();
  } catch (...) {
    return true;
  }
}
}  // namespace

auto EditTransaction::TransactionTypeToString(TransactionType type) -> const char* {
  switch (type) {
    case TransactionType::_ADD:
      return "ADD";
    case TransactionType::_DELETE:
      return "DELETE";
    case TransactionType::_EDIT:
      return "EDIT";
  }
  return "UNKNOWN";
}

auto EditTransaction::OperatorTypeToString(OperatorType type) -> const char* {
  switch (type) {
    case OperatorType::RAW_DECODE:
      return "RAW_DECODE";
    case OperatorType::RESIZE:
      return "RESIZE";
    case OperatorType::EXPOSURE:
      return "EXPOSURE";
    case OperatorType::CONTRAST:
      return "CONTRAST";
    case OperatorType::WHITE:
      return "WHITE";
    case OperatorType::BLACK:
      return "BLACK";
    case OperatorType::SHADOWS:
      return "SHADOWS";
    case OperatorType::HIGHLIGHTS:
      return "HIGHLIGHTS";
    case OperatorType::CURVE:
      return "CURVE";
    case OperatorType::HLS:
      return "HLS";
    case OperatorType::SATURATION:
      return "SATURATION";
    case OperatorType::TINT:
      return "TINT";
    case OperatorType::VIBRANCE:
      return "VIBRANCE";
    case OperatorType::CST:
      return "CST";
    case OperatorType::TO_WS:
      return "TO_WS";
    case OperatorType::TO_OUTPUT:
      return "TO_OUTPUT";
    case OperatorType::LMT:
      return "LMT";
    case OperatorType::ODT:
      return "ODT";
    case OperatorType::CLARITY:
      return "CLARITY";
    case OperatorType::SHARPEN:
      return "SHARPEN";
    case OperatorType::COLOR_WHEEL:
      return "COLOR_WHEEL";
    case OperatorType::ACES_TONE_MAPPING:
      return "ACES_TONE_MAPPING";
    case OperatorType::AUTO_EXPOSURE:
      return "AUTO_EXPOSURE";
    case OperatorType::UNKNOWN:
      return "UNKNOWN";
  }
  return "UNKNOWN";
}

auto EditTransaction::StageNameToString(PipelineStageName stage) -> const char* {
  switch (stage) {
    case PipelineStageName::Image_Loading:
      return "Image_Loading";
    case PipelineStageName::Geometry_Adjustment:
      return "Geometry_Adjustment";
    case PipelineStageName::To_WorkingSpace:
      return "To_WorkingSpace";
    case PipelineStageName::Basic_Adjustment:
      return "Basic_Adjustment";
    case PipelineStageName::Color_Adjustment:
      return "Color_Adjustment";
    case PipelineStageName::Detail_Adjustment:
      return "Detail_Adjustment";
    case PipelineStageName::Output_Transform:
      return "Output_Transform";
    case PipelineStageName::Stage_Count:
      return "Stage_Count";
    case PipelineStageName::Merged_Stage:
      return "Merged_Stage";
  }
  return "UnknownStage";
}

auto EditTransaction::Describe(bool include_params, std::size_t max_params_chars) const
    -> std::string {
  std::ostringstream oss;
  oss << "#" << tx_id_ << " " << TransactionTypeToString(type_) << " "
      << StageNameToString(stage_name_)
      << "/" << OperatorTypeToString(operator_type_);
  if (!include_params) {
    return oss.str();
  }

  if (!operator_params_.is_object()) {
    oss << " " << TruncateForUi(operator_params_.dump(), max_params_chars);
    return oss.str();
  }

  std::vector<std::string> parts;
  parts.reserve(operator_params_.size());

  for (auto it = operator_params_.begin(); it != operator_params_.end(); ++it) {
    const std::string& key = it.key();
    const auto&        val = it.value();

    if (last_operator_params_.has_value() && last_operator_params_->is_object() &&
        last_operator_params_->contains(key)) {
      const auto& old_val = (*last_operator_params_)[key];
      if (old_val == val) {
        continue;
      }
      parts.push_back(key + ": " + JsonScalarToString(old_val) + " -> " + JsonScalarToString(val));
    } else {
      parts.push_back(key + ": " + JsonScalarToString(val));
    }
  }

  if (parts.empty()) {
    oss << " " << TruncateForUi(operator_params_.dump(), max_params_chars);
    return oss.str();
  }

  // Keep the summary compact for UI lists.
  constexpr std::size_t kMaxKeys = 3;
  if (parts.size() > kMaxKeys) {
    parts.resize(kMaxKeys);
    parts.push_back("...");
  }

  std::string joined;
  for (std::size_t i = 0; i < parts.size(); ++i) {
    if (i) {
      joined += ", ";
    }
    joined += parts[i];
  }

  oss << " " << TruncateForUi(joined, max_params_chars);
  return oss.str();
}

auto EditTransaction::ApplyTransaction(PipelineExecutor& pipeline) const -> bool {
  auto& stage = pipeline.GetStage(stage_name_);
  auto& global_params = pipeline.GetGlobalParams();
  switch (type_) {
    case TransactionType::_ADD:
      stage.SetOperator(operator_type_, operator_params_, global_params);
      stage.EnableOperator(operator_type_, true, global_params);
      return true;
    case TransactionType::_DELETE:
      stage.EnableOperator(operator_type_, false, global_params);
      return true;
    case TransactionType::_EDIT:
      stage.SetOperator(operator_type_, operator_params_, global_params);
      if (operator_type_ == OperatorType::LMT && IsLutPathEmpty(operator_params_)) {
        stage.EnableOperator(operator_type_, false, global_params);
      } else {
        stage.EnableOperator(operator_type_, true, global_params);
      }
      return true;
  }
  return false;
}

auto EditTransaction::ToJSON() const -> nlohmann::json {
  nlohmann::json j;
  j["id"]              = tx_id_;
  j["type"]            = static_cast<int>(type_);
  j["operator_type"]   = static_cast<int>(operator_type_);
  j["stage_name"]      = static_cast<int>(stage_name_);
  j["operator_params"] = operator_params_;
  if (last_operator_params_) {
    j["last_operator_params"] = *last_operator_params_;
  }

  return j;
}

void EditTransaction::FromJSON(const nlohmann::json& j) {
  tx_id_           = j["id"].get<tx_id_t>();
  type_            = static_cast<TransactionType>(j["type"].get<int>());
  operator_type_   = static_cast<OperatorType>(j["operator_type"].get<int>());
  stage_name_      = static_cast<PipelineStageName>(j["stage_name"].get<int>());
  operator_params_ = j["operator_params"];
  if (j.contains("last_operator_params")) {
    last_operator_params_ = j["last_operator_params"];
  } else {
    last_operator_params_ = std::nullopt;
  }
}

};  // namespace puerhlab
