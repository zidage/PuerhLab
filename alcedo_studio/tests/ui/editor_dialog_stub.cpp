//  Copyright 2026 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

/// @file editor_dialog_stub.cpp
/// @brief No-op stub for OpenEditorDialog — used in headless unit tests.
///
/// The real implementation lives in editor_dialog.cpp and opens a modal
/// QWidget-based OpenGL editor.  Tests don't need (and can't run) that GUI
/// code, so this stub returns false immediately.

#include "ui/alcedo_main/editor_dialog/editor_dialog.hpp"

namespace alcedo::ui {

auto OpenEditorDialog(std::shared_ptr<ImagePoolService> /*image_pool*/,
                      std::shared_ptr<PipelineGuard> /*pipeline_guard*/,
                      std::shared_ptr<EditHistoryMgmtService> /*history_service*/,
                      std::shared_ptr<EditHistoryGuard> /*history_guard*/,
                      sl_element_id_t /*element_id*/, image_id_t /*image_id*/,
                      QWidget* /*parent*/) -> bool {
  return false;  // no-op for unit tests
}

}  // namespace alcedo::ui
