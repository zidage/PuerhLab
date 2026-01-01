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
#include <mutex>

#include "edit/operators/CPU_kernels/cpu_kernels.hpp"
#include "edit/operators/op_base.hpp"
#include "edit/operators/op_kernel.hpp"
#include "edit/operators/operator_factory.hpp"
#include "edit/pipeline/pipeline_stage.hpp"
#include "image/image_buffer.hpp"
#include "pipeline.hpp"
#include "pipeline_stage.hpp"
#include "utils/cache/lru_cache.hpp"

namespace puerhlab {
class CPUPipelineExecutor : public PipelineExecutor {
 private:
  bool                                                                        enable_cache_ = true;
  std::array<PipelineStage, static_cast<int>(PipelineStageName::Stage_Count)> stages_;

  OperatorParams                                                              global_params_;

  bool                                                                        is_thumbnail_ = false;

  nlohmann::json                                                              thumbnail_params_;

  static constexpr PipelineBackend backend_ = PipelineBackend::CPU;

  std::vector<PipelineStage*>      exec_stages_;
  std::unique_ptr<PipelineStage>   merged_stages_;

  void                             ResetStages();

 public:
  CPUPipelineExecutor();
  CPUPipelineExecutor(bool enable_cache);

  void SetEnableCache(bool enable_cache);
  auto GetBackend() -> PipelineBackend override;

  auto GetStage(PipelineStageName stage) -> PipelineStage& override;
  auto Apply(std::shared_ptr<ImageBuffer> input) -> std::shared_ptr<ImageBuffer> override;

  void SetPreviewMode(bool is_preview);
  void SetExecutionStages();
  void ResetExecutionStages();

  auto GetGlobalParams() -> OperatorParams& override { return global_params_; }

  /**
   * @brief Serialize the pipeline parameters to JSON
   *
   * @return nlohmann::json
   */
  auto ExportPipelineParams() const -> nlohmann::json override;
  /**
   * @brief Set the pipeline parameters from JSON. It will reset all stages and operators, as well
   * as cache. After importing, you need to call SetExecutionStages() to rebuild the execution
   * stages.
   *
   * @param j
   */
  void ImportPipelineParams(const nlohmann::json& j) override;
};
};  // namespace puerhlab