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

#include "edit/pipeline/pipeline_cpu.hpp"
#include "storage/mapper/duckorm/duckdb_orm.hpp"
#include "storage/mapper/pipeline/pipeline_mapper.hpp"
#include "storage/service/service_interface.hpp"
#include "type/type.hpp"


namespace puerhlab {
class PipelineService
    : public ServiceInterface<PipelineService, std::shared_ptr<CPUPipelineExecutor>,
                              PipelineMapperParams, PipelineMapper, sl_element_id_t> {
 public:
  using ServiceInterface::ServiceInterface;

  static auto ToParams(const std::shared_ptr<CPUPipelineExecutor> source) -> PipelineMapperParams;
  static auto FromParams(PipelineMapperParams&& param) -> std::shared_ptr<CPUPipelineExecutor>;

  auto GetPipelineParamByFileId(const sl_element_id_t file_id) -> std::shared_ptr<CPUPipelineExecutor>;
};
};  // namespace puerhlab
