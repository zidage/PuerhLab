//  Copyright 2026 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#pragma once

#include <memory>

#include "app/history_mgmt_service.hpp"
#include "edit/history/version.hpp"

namespace alcedo::ui::controllers {

auto SeedWorkingVersionFromLatest(sl_element_id_t element_id,
                                  const std::shared_ptr<EditHistoryGuard>& history_guard)
    -> Version;

auto SeedWorkingVersionFromParent(sl_element_id_t element_id,
                                  const Hash128& parent_id,
                                  bool incremental_mode) -> Version;

auto CommitWorkingVersion(const std::shared_ptr<EditHistoryMgmtService>& history_service,
                          const std::shared_ptr<EditHistoryGuard>& history_guard,
                          Version&& working_version) -> history_id_t;

}  // namespace alcedo::ui::controllers
