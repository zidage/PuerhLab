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

#include "edit/operators/op_base.hpp"
#include "image/image_buffer.hpp"
#include "pipeline_stage.hpp"

namespace puerhlab {
enum class PipelineBackend { CPU, CUDA, OpenCL };
class PipelineExecutor {
 public:
  virtual void SetBoundFile(sl_element_id_t file_id)                                     = 0;
  virtual auto GetBoundFile() const -> sl_element_id_t                                   = 0;
  virtual auto GetStage(PipelineStageName stage) -> PipelineStage&                       = 0;
  virtual auto Apply(std::shared_ptr<ImageBuffer> input) -> std::shared_ptr<ImageBuffer> = 0;
  virtual auto GetBackend() -> PipelineBackend                                           = 0;
  virtual auto ExportPipelineParams() const -> nlohmann::json                            = 0;
  virtual void ImportPipelineParams(const nlohmann::json& j)                             = 0;

  virtual auto GetGlobalParams() -> OperatorParams&                                      = 0;
};

}  // namespace puerhlab
