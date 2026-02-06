#pragma once

#include <memory>

#include <QWidget>

#include "app/history_mgmt_service.hpp"
#include "app/image_pool_service.hpp"
#include "app/pipeline_service.hpp"

namespace puerhlab::demo {

// Opens the OpenGL editor dialog and runs it modally.
auto OpenEditorDialog(std::shared_ptr<ImagePoolService> image_pool,
                      std::shared_ptr<PipelineGuard> pipeline_guard,
                      std::shared_ptr<EditHistoryMgmtService> history_service,
                      std::shared_ptr<EditHistoryGuard> history_guard,
                      sl_element_id_t element_id, image_id_t image_id,
                      QWidget* parent = nullptr) -> bool;

}  // namespace puerhlab::demo
