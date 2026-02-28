#include "ui/puerhlab_main/editor_dialog/controllers/history_controller.hpp"

namespace puerhlab::ui::controllers {

auto SeedWorkingVersionFromLatest(sl_element_id_t element_id,
                                  const std::shared_ptr<EditHistoryGuard>& history_guard)
    -> Version {
  try {
    if (history_guard && history_guard->history_) {
      const auto parent_id = history_guard->history_->GetLatestVersion().ver_ref_.GetVersionID();
      return Version(element_id, parent_id);
    }
  } catch (...) {
  }
  return Version(element_id);
}

auto SeedWorkingVersionFromParent(sl_element_id_t element_id,
                                  const Hash128& parent_id,
                                  bool incremental_mode) -> Version {
  if (incremental_mode) {
    return Version(element_id, parent_id);
  }
  return Version(element_id);
}

auto CommitWorkingVersion(const std::shared_ptr<EditHistoryMgmtService>& history_service,
                          const std::shared_ptr<EditHistoryGuard>& history_guard,
                          Version&& working_version) -> history_id_t {
  return history_service->CommitVersion(history_guard, std::move(working_version));
}

}  // namespace puerhlab::ui::controllers
