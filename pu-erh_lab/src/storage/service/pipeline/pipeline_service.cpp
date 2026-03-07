//  Copyright 2025 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#include "storage/service/pipeline/pipeline_service.hpp"

#include <memory>

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
    pipeline->SetExecutionStages();
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

void PipelineService::UpdatePipelineParamByFileId(
    const sl_element_id_t file_id, const std::shared_ptr<CPUPipelineExecutor> pipeline) {
  // Now the duckorm::update use the upsert semantics (I should change the interface, but
  // anyways...), so we can directly call Update even if the record does not exist.
  Update(pipeline, file_id);
}
};  // namespace puerhlab