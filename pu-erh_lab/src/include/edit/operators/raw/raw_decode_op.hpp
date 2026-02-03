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

#include <memory>

#include "decoders/processor/raw_processor.hpp"
#include "edit/operators/op_base.hpp"

namespace puerhlab {

enum class RawProcessBackend { PUERH, LIBRAW };

class RawDecodeOp : public OperatorBase<RawDecodeOp> {
 public:
  static constexpr PriorityLevel     priority_level_    = 0;
  static constexpr std::string_view  canonical_name_    = "RawDecode";
  static constexpr std::string_view  script_name_       = "raw_decode";
  static constexpr PipelineStageName affiliation_stage_ = PipelineStageName::Image_Loading;
  static constexpr OperatorType      operator_type_     = OperatorType::RAW_DECODE;

  RawParams                          params_;
  RawProcessBackend                  backend_;

  RawDecodeOp() = delete;

  RawDecodeOp(const nlohmann::json& params);

  void Apply(std::shared_ptr<ImageBuffer> input) override;

  auto GetParams() const -> nlohmann::json override;
  void SetParams(const nlohmann::json& params) override;

  void SetGlobalParams(OperatorParams& params) const override;
  void EnableGlobalParams(OperatorParams& params, bool enable) override;
};

}  // namespace puerhlab