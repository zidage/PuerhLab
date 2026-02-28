#pragma once

#include <cstddef>
#include <functional>
#include <memory>
#include <optional>

#include <QString>
#include <json.hpp>

#include "app/history_mgmt_service.hpp"
#include "app/pipeline_service.hpp"
#include "edit/history/version.hpp"

class QComboBox;
class QLabel;
class QListWidget;
class QListWidgetItem;
class QPushButton;

namespace puerhlab::ui::versioning {

constexpr int kWorkingModeIncrementalValue = 0;
constexpr int kWorkingModePlainValue       = 1;

struct VersionUiContext {
  QLabel*      version_status     = nullptr;
  QPushButton* commit_version_btn = nullptr;
  QPushButton* undo_tx_btn        = nullptr;
  QComboBox*   working_mode_combo = nullptr;
  QListWidget* version_log        = nullptr;
  QListWidget* tx_stack           = nullptr;
};

struct ResolvedVersionSelection {
  Hash128 version_id{};
  Version* version = nullptr;
};

struct UndoResult {
  bool    undone         = false;
  bool    no_transaction = false;
  QString error;
};

struct CommitResult {
  std::optional<history_id_t> committed_id;
  bool                        no_transactions = false;
  QString                     error;
  std::optional<Version>      recovery_working_version;
};

auto MakeTxCountLabel(size_t tx_count) -> QString;
auto IsPlainModeSelected(const QComboBox* working_mode_combo) -> bool;

auto ReconstructPipelineParamsForVersion(Version& version,
                                         const std::shared_ptr<EditHistoryGuard>& history_guard)
    -> std::optional<nlohmann::json>;

auto ResolveSelectedVersion(QListWidgetItem* item,
                            const std::shared_ptr<EditHistoryGuard>& history_guard,
                            ResolvedVersionSelection* out_selection,
                            QString* error) -> bool;

auto UndoLastTransaction(Version& working_version,
                         const std::shared_ptr<PipelineGuard>& pipeline_guard) -> UndoResult;

auto CommitWorkingVersion(const std::shared_ptr<EditHistoryMgmtService>& history_service,
                          const std::shared_ptr<EditHistoryGuard>& history_guard,
                          const std::shared_ptr<PipelineGuard>& pipeline_guard,
                          sl_element_id_t element_id, Version&& working_version)
    -> CommitResult;

auto SeedWorkingVersionFromUi(sl_element_id_t element_id,
                              const std::shared_ptr<EditHistoryGuard>& history_guard,
                              const std::shared_ptr<PipelineGuard>& pipeline_guard,
                              bool plain_mode) -> Version;

auto SeedWorkingVersionFromCommit(sl_element_id_t element_id, const Hash128& committed_id,
                                  const std::shared_ptr<PipelineGuard>& pipeline_guard,
                                  bool incremental_mode) -> Version;

void UpdateVersionUi(const VersionUiContext& ui, const Version& working_version,
                     const std::shared_ptr<EditHistoryGuard>& history_guard,
                     const std::function<void()>& refresh_selection_styles);

}  // namespace puerhlab::ui::versioning
