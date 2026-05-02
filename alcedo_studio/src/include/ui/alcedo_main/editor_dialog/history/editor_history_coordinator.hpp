//  Copyright 2026 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#pragma once

#include <functional>
#include <memory>
#include <optional>

#include <QString>
#include <json.hpp>

#include "app/history_mgmt_service.hpp"
#include "app/pipeline_service.hpp"
#include "edit/history/version.hpp"
#include "ui/alcedo_main/editor_dialog/modules/versioning.hpp"

class QListWidgetItem;
class QWidget;

namespace alcedo::ui {

class EditorHistoryCoordinator {
 public:
  struct Dependencies {
    std::shared_ptr<EditHistoryMgmtService> history_service;
    std::shared_ptr<EditHistoryGuard>       history_guard;
    std::shared_ptr<PipelineGuard>          pipeline_guard;
    sl_element_id_t                         element_id = 0;
    QWidget*                                message_parent = nullptr;
  };

  struct Callbacks {
    std::function<bool(bool)>                    reload_ui_state_from_pipeline;
    std::function<void()>                        after_pipeline_params_imported;
    std::function<bool()>                        is_plain_working_mode;
    std::function<void()>                        refresh_version_log_selection_styles;
  };

  EditorHistoryCoordinator(Dependencies dependencies, Callbacks callbacks);

  auto WorkingVersion() -> Version&;
  auto WorkingVersion() const -> const Version&;

  void SetUiContext(const versioning::VersionUiContext& ui);
  void SeedWorkingVersionFromLatest();

  auto ReconstructPipelineParamsForVersion(Version& version) const
      -> std::optional<nlohmann::json>;
  auto ReloadUiStateFromPipeline(bool reset_to_defaults_if_missing) -> bool;
  auto ApplyPipelineParamsToEditor(const nlohmann::json& params) -> bool;
  auto ReloadEditorFromHistoryVersion(Version& version, QString* error) -> bool;

  void CheckoutSelectedVersion(QListWidgetItem* item);
  void UndoLastTransaction();
  void UpdateVersionUi();
  void CommitWorkingVersion();
  void StartNewWorkingVersionFromUi();
  void StartNewWorkingVersionFromCommit(const Hash128& committed_id);

 private:
  auto IsPlainWorkingMode() const -> bool;
  auto IsIncrementalWorkingMode() const -> bool;

  Dependencies                 dependencies_;
  Callbacks                    callbacks_;
  versioning::VersionUiContext ui_{};
  Version                      working_version_{};
};

}  // namespace alcedo::ui
