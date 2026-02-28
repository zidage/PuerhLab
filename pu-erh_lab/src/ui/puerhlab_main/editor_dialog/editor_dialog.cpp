#include "ui/puerhlab_main/editor_dialog/editor_dialog.hpp"

#include "ui/puerhlab_main/editor_dialog/dialog.hpp"

namespace puerhlab::ui {

auto OpenEditorDialog(std::shared_ptr<ImagePoolService> image_pool,
                      std::shared_ptr<PipelineGuard> pipeline_guard,
                      std::shared_ptr<EditHistoryMgmtService> history_service,
                      std::shared_ptr<EditHistoryGuard> history_guard,
                      sl_element_id_t element_id, image_id_t image_id,
                      QWidget* parent) -> bool {
  return RunEditorDialog(std::move(image_pool), std::move(pipeline_guard),
                         std::move(history_service), std::move(history_guard),
                         element_id, image_id, parent);
}

}  // namespace puerhlab::ui
