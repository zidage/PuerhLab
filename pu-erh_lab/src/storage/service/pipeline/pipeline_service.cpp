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

#include <memory>

#include "storage/service/pipeline/pipeline_service.hpp"
#include "storage/mapper/pipeline/pipeline_mapper.hpp"

namespace puerhlab {
auto PipelineService::ToParams(const std::shared_ptr<CPUPipelineExecutor> source)
    -> PipelineMapperParams {
  PipelineMapperParams param;
  param.file_id    = source->GetBoundFile();
  param.param_json = std::make_unique<std::string>(source->ExportPipelineParams().dump());
  return param;
}

auto PipelineService::FromParams(PipelineMapperParams&& param)
    -> std::shared_ptr<CPUPipelineExecutor> {
  auto pipeline = std::make_shared<CPUPipelineExecutor>();
  pipeline->SetBoundFile(param.file_id);
  if (param.param_json) {
    pipeline->ImportPipelineParams(nlohmann::json::parse(std::move(*param.param_json)));
  }
  return pipeline;
}

auto PipelineService::GetPipelineParamByFileId(const sl_element_id_t file_id)
    -> std::shared_ptr<CPUPipelineExecutor> {
  auto result = GetByPredicate(std::format(PipelineMapper::PrimeKeyClause(), file_id));
  if (result.size() > 1) {
    throw std::runtime_error(
        "[ERROR] PipelineService: Broken image database. Multiple pipeline params found for "
        "file_id " +
        std::to_string(file_id));
  }

  if (result.empty()) {
    return nullptr;
  }

  return result.front();
}
};  // namespace puerhlab