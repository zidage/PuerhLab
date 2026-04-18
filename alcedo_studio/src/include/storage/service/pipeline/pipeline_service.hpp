//  Copyright 2025 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#pragma once

#include <memory>

#include "edit/pipeline/pipeline_cpu.hpp"
#include "storage/mapper/duckorm/duckdb_orm.hpp"
#include "storage/mapper/pipeline/pipeline_mapper.hpp"
#include "storage/service/service_interface.hpp"
#include "type/type.hpp"

namespace alcedo {
class PipelineService
    : public ServiceInterface<PipelineService, std::shared_ptr<CPUPipelineExecutor>,
                              PipelineMapperParams, PipelineMapper, sl_element_id_t> {
 public:
  using ServiceInterface::ServiceInterface;

  static auto ToParams(const std::shared_ptr<CPUPipelineExecutor> source) -> PipelineMapperParams;
  static auto FromParams(PipelineMapperParams&& param) -> std::shared_ptr<CPUPipelineExecutor>;

  auto        GetPipelineParamByFileId(const sl_element_id_t file_id)
      -> std::shared_ptr<CPUPipelineExecutor>;
  void UpdatePipelineParamByFileId(const sl_element_id_t                      file_id,
                                   const std::shared_ptr<CPUPipelineExecutor> pipeline);
};
};  // namespace alcedo
