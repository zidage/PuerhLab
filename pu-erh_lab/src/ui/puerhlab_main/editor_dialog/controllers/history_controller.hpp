#pragma once

#include <memory>

#include "app/history_mgmt_service.hpp"
#include "edit/history/version.hpp"

namespace puerhlab::ui::controllers {

auto SeedWorkingVersionFromLatest(sl_element_id_t element_id,
                                  const std::shared_ptr<EditHistoryGuard>& history_guard)
    -> Version;

auto SeedWorkingVersionFromParent(sl_element_id_t element_id,
                                  const Hash128& parent_id,
                                  bool incremental_mode) -> Version;

auto CommitWorkingVersion(const std::shared_ptr<EditHistoryMgmtService>& history_service,
                          const std::shared_ptr<EditHistoryGuard>& history_guard,
                          Version&& working_version) -> history_id_t;

}  // namespace puerhlab::ui::controllers
