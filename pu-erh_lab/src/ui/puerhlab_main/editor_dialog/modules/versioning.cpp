#include "ui/puerhlab_main/editor_dialog/modules/versioning.hpp"

#include <QColor>
#include <QComboBox>
#include <QDateTime>
#include <QFontDatabase>
#include <QHBoxLayout>
#include <QLabel>
#include <QListWidget>
#include <QPushButton>
#include <QVBoxLayout>
#include <QWidget>
#include <algorithm>
#include <json.hpp>
#include <utility>
#include <vector>

#include "edit/history/edit_transaction.hpp"
#include "edit/pipeline/pipeline_cpu.hpp"
#include "ui/puerhlab_main/editor_dialog/controllers/history_controller.hpp"
#include "ui/puerhlab_main/editor_dialog/widgets/history_cards.hpp"

namespace puerhlab::ui::versioning {

auto MakeTxCountLabel(size_t tx_count) -> QString {
  return QString("Uncommitted: %1 tx").arg(static_cast<qulonglong>(tx_count));
}

auto IsPlainModeSelected(const QComboBox* working_mode_combo) -> bool {
  if (!working_mode_combo) {
    return false;
  }
  return working_mode_combo->currentData().toInt() == kWorkingModePlainValue;
}

auto ReconstructPipelineParamsForVersion(Version& version,
                                         const std::shared_ptr<EditHistoryGuard>& history_guard)
    -> std::optional<nlohmann::json> {
  if (const auto snapshot = version.GetFinalPipelineParams(); snapshot.has_value()) {
    return snapshot;
  }

  if (!history_guard || !history_guard->history_) {
    return std::nullopt;
  }

  std::vector<Version*> lineage;
  lineage.push_back(&version);
  while (lineage.back()->HasParentVersion()) {
    try {
      lineage.push_back(
          &history_guard->history_->GetVersion(lineage.back()->GetParentVersionID()));
    } catch (...) {
      return std::nullopt;
    }
  }
  std::reverse(lineage.begin(), lineage.end());

  auto replay_exec = std::make_shared<CPUPipelineExecutor>();

  size_t replay_from = 0;
  for (size_t i = lineage.size(); i > 0; --i) {
    if (const auto snapshot = lineage[i - 1]->GetFinalPipelineParams();
        snapshot.has_value()) {
      replay_exec->ImportPipelineParams(*snapshot);
      replay_exec->SetExecutionStages();
      replay_from = i;
      break;
    }
  }

  for (size_t i = replay_from; i < lineage.size(); ++i) {
    const auto& txs = lineage[i]->GetAllEditTransactions();
    for (auto it = txs.rbegin(); it != txs.rend(); ++it) {
      (void)it->ApplyTransaction(*replay_exec);
    }
  }
  return replay_exec->ExportPipelineParams();
}

auto ResolveSelectedVersion(QListWidgetItem* item,
                            const std::shared_ptr<EditHistoryGuard>& history_guard,
                            ResolvedVersionSelection* out_selection,
                            QString* error) -> bool {
  if (!item || !history_guard || !history_guard->history_) {
    return false;
  }

  const auto version_id_str = item->data(Qt::UserRole).toString().toStdString();
  if (version_id_str.empty()) {
    return false;
  }

  Hash128 version_id{};
  try {
    version_id = Hash128::FromString(version_id_str);
  } catch (const std::exception& e) {
    if (error) {
      *error = QString("Invalid version ID: %1").arg(e.what());
    }
    return false;
  }

  Version* selected_version = nullptr;
  try {
    selected_version = &history_guard->history_->GetVersion(version_id);
  } catch (const std::exception& e) {
    if (error) {
      *error = QString("Failed to load selected version: %1").arg(e.what());
    }
    return false;
  }

  if (out_selection) {
    out_selection->version_id = version_id;
    out_selection->version    = selected_version;
  }
  return true;
}

auto UndoLastTransaction(Version& working_version,
                         const std::shared_ptr<PipelineGuard>& pipeline_guard) -> UndoResult {
  UndoResult result{};

  if (!pipeline_guard || !pipeline_guard->pipeline_) {
    return result;
  }

  if (working_version.GetAllEditTransactions().empty()) {
    result.no_transaction = true;
    return result;
  }

  EditTransaction last_tx{TransactionType::_EDIT, OperatorType::UNKNOWN,
                          PipelineStageName::Basic_Adjustment, nlohmann::json::object()};
  try {
    last_tx = working_version.RemoveLastEditTransaction();
  } catch (const std::exception& e) {
    result.error = QString("Undo failed: %1").arg(e.what());
    return result;
  }

  auto exec = pipeline_guard->pipeline_;

  EditTransaction undo_delete_tx(TransactionType::_DELETE, last_tx.GetTxOperatorType(),
                                 last_tx.GetTxOpStageName(), nlohmann::json::object());
  if (const auto prev = last_tx.GetLastOperatorParams(); prev.has_value()) {
    undo_delete_tx.SetLastOperatorParams(*prev);
  }
  (void)undo_delete_tx.ApplyTransaction(*exec);

  bool restored_from_stack = false;
  for (const auto& tx : working_version.GetAllEditTransactions()) {
    if (tx.GetTxOpStageName() == last_tx.GetTxOpStageName() &&
        tx.GetTxOperatorType() == last_tx.GetTxOperatorType()) {
      (void)tx.ApplyTransaction(*exec);
      restored_from_stack = true;
      break;
    }
  }

  if (!restored_from_stack) {
    if (const auto prev = last_tx.GetLastOperatorParams();
        prev.has_value() && prev->is_object() && !prev->empty()) {
      EditTransaction restore_tx(TransactionType::_EDIT, last_tx.GetTxOperatorType(),
                                 last_tx.GetTxOpStageName(), *prev);
      (void)restore_tx.ApplyTransaction(*exec);
    }
  }

  pipeline_guard->dirty_ = true;
  result.undone          = true;
  return result;
}

auto CommitWorkingVersion(const std::shared_ptr<EditHistoryMgmtService>& history_service,
                          const std::shared_ptr<EditHistoryGuard>& history_guard,
                          const std::shared_ptr<PipelineGuard>& pipeline_guard,
                          sl_element_id_t element_id, Version&& working_version)
    -> CommitResult {
  CommitResult result{};

  if (!history_service || !history_guard || !history_guard->history_) {
    result.error = "Edit history service not available.";
    return result;
  }

  if (working_version.GetAllEditTransactions().empty()) {
    result.no_transactions = true;
    return result;
  }

  try {
    if (pipeline_guard && pipeline_guard->pipeline_) {
      working_version.SetFinalPipelineParams(pipeline_guard->pipeline_->ExportPipelineParams());
    }
    result.committed_id = controllers::CommitWorkingVersion(
        history_service, history_guard, std::move(working_version));
    return result;
  } catch (const std::exception& e) {
    result.error = QString("Commit failed: %1").arg(e.what());
  }

  Version recovery(element_id);
  if (pipeline_guard && pipeline_guard->pipeline_) {
    recovery.SetBasePipelineExecutor(pipeline_guard->pipeline_);
  }
  result.recovery_working_version = std::move(recovery);
  return result;
}

auto SeedWorkingVersionFromUi(sl_element_id_t element_id,
                              const std::shared_ptr<EditHistoryGuard>& history_guard,
                              const std::shared_ptr<PipelineGuard>& pipeline_guard,
                              bool plain_mode) -> Version {
  Version working =
      plain_mode ? Version(element_id)
                 : controllers::SeedWorkingVersionFromLatest(element_id, history_guard);
  if (pipeline_guard && pipeline_guard->pipeline_) {
    working.SetBasePipelineExecutor(pipeline_guard->pipeline_);
  }
  return working;
}

auto SeedWorkingVersionFromCommit(sl_element_id_t element_id,
                                  const Hash128& committed_id,
                                  const std::shared_ptr<PipelineGuard>& pipeline_guard,
                                  bool incremental_mode) -> Version {
  Version working = controllers::SeedWorkingVersionFromParent(
      element_id, committed_id, incremental_mode);
  if (pipeline_guard && pipeline_guard->pipeline_) {
    working.SetBasePipelineExecutor(pipeline_guard->pipeline_);
  }
  return working;
}

void UpdateVersionUi(const VersionUiContext& ui, const Version& working_version,
                     const std::shared_ptr<EditHistoryGuard>& history_guard,
                     const std::function<void()>& refresh_selection_styles) {
  if (!ui.version_status || !ui.commit_version_btn) {
    return;
  }

  const size_t tx_count = working_version.GetAllEditTransactions().size();
  QString      label    = MakeTxCountLabel(tx_count);

  if (working_version.HasParentVersion()) {
    label +=
        QString(" | parent: %1")
            .arg(QString::fromStdString(
                working_version.GetParentVersionID().ToString().substr(0, 8)));
  } else {
    label += " | plain";
  }

  if (ui.working_mode_combo) {
    label += IsPlainModeSelected(ui.working_mode_combo) ? " | mode: plain"
                                                         : " | mode: incremental";
  }

  if (history_guard && history_guard->history_) {
    try {
      const auto latest_id =
          history_guard->history_->GetLatestVersion().ver_ref_.GetVersionID();
      label +=
          QString(" | Latest: %1").arg(QString::fromStdString(latest_id.ToString().substr(0, 8)));
    } catch (...) {
    }
  }

  ui.version_status->setText(label);
  ui.version_status->setToolTip(label);
  ui.commit_version_btn->setEnabled(tx_count > 0);
  if (ui.undo_tx_btn) {
    ui.undo_tx_btn->setEnabled(tx_count > 0);
  }

  if (ui.tx_stack) {
    ui.tx_stack->clear();
    const auto&  txs   = working_version.GetAllEditTransactions();
    const size_t total = txs.size();
    size_t       i     = 0;
    for (const auto& tx : txs) {
      const QString title = QString::fromStdString(tx.Describe(true, 110));

      auto* item = new QListWidgetItem(ui.tx_stack);
      item->setToolTip(QString::fromStdString(tx.ToJSON().dump(2)));
      item->setSizeHint(QSize(0, 58));

      auto* card = new HistoryCardWidget(ui.tx_stack);
      auto* row  = new QHBoxLayout(card);
      row->setContentsMargins(10, 8, 10, 8);
      row->setSpacing(10);

      const QColor dot  = QColor(0xFC, 0xC7, 0x04);
      const QColor line = QColor(0x2A, 0x2A, 0x2A);
      auto* lane        = new HistoryLaneWidget(dot, line, /*draw_top*/ i > 0,
                                                /*draw_bottom*/ (i + 1) < total, card);
      row->addWidget(lane, 0);

      auto* body = new QVBoxLayout();
      body->setContentsMargins(0, 0, 0, 0);
      body->setSpacing(2);

      auto* title_l = new QLabel(title, card);
      title_l->setWordWrap(true);
      title_l->setStyleSheet("QLabel {"
                             "  color: #E6E6E6;"
                             "  font-size: 12px;"
                             "  font-weight: 500;"
                             "}");

      auto* meta_l =
          new QLabel(QString("uncommitted | #%1").arg(static_cast<qulonglong>(i + 1)), card);
      meta_l->setStyleSheet("QLabel {"
                            "  color: #A3A3A3;"
                            "  font-size: 11px;"
                            "}");

      body->addWidget(title_l);
      body->addWidget(meta_l);
      row->addLayout(body, 1);

      ui.tx_stack->setItemWidget(item, card);
      ++i;
    }
  }

  if (ui.version_log) {
    QString prev_selected_id;
    if (auto* cur = ui.version_log->currentItem()) {
      prev_selected_id = cur->data(Qt::UserRole).toString();
    }

    ui.version_log->clear();
    if (history_guard && history_guard->history_) {
      const auto& tree = history_guard->history_->GetCommitTree();
      Hash128     latest_id{};
      try {
        latest_id = history_guard->history_->GetLatestVersion().ver_ref_.GetVersionID();
      } catch (...) {
      }

      const Hash128 base_parent = working_version.GetParentVersionID();

      int       row_index  = 0;
      const int total_rows = static_cast<int>(tree.size());

      for (auto it = tree.rbegin(); it != tree.rend(); ++it, ++row_index) {
        const auto& ver      = it->ver_ref_;
        const auto  ver_id   = ver.GetVersionID();
        const auto  short_id = QString::fromStdString(ver_id.ToString().substr(0, 8));
        const auto  when =
            QDateTime::fromSecsSinceEpoch(static_cast<qint64>(ver.GetLastModifiedTime()))
                .toString("yyyy-MM-dd HH:mm:ss");
        const auto committed_tx_count =
            static_cast<qulonglong>(ver.GetAllEditTransactions().size());

        QString     msg;
        const auto& txs = ver.GetAllEditTransactions();
        if (!txs.empty()) {
          msg = QString::fromStdString(txs.front().Describe(true, 70));
        } else {
          msg = "(empty)";
        }

        const bool is_head  = (ver_id == latest_id);
        const bool is_base  = (base_parent == ver_id && working_version.HasParentVersion());
        const bool is_plain = !ver.HasParentVersion();

        auto* item = new QListWidgetItem(ui.version_log);
        item->setData(Qt::UserRole, QString::fromStdString(ver_id.ToString()));
        item->setToolTip(QString("version=%1\nparent=%2\ntx=%3")
                             .arg(QString::fromStdString(ver_id.ToString()))
                             .arg(QString::fromStdString(ver.GetParentVersionID().ToString()))
                             .arg(committed_tx_count));
        item->setSizeHint(QSize(0, 74));

        auto* card = new HistoryCardWidget(ui.version_log);
        auto* row  = new QHBoxLayout(card);
        row->setContentsMargins(10, 9, 10, 9);
        row->setSpacing(10);

        const QColor dot =
            is_head ? QColor(0xFC, 0xC7, 0x04)
                    : (is_base ? QColor(0xFC, 0xC7, 0x04) : QColor(0x8C, 0x8C, 0x8C));
        const QColor line = QColor(0x2A, 0x2A, 0x2A);
        auto* lane        = new HistoryLaneWidget(dot, line, /*draw_top*/ row_index > 0,
                                                  /*draw_bottom*/ (row_index + 1) < total_rows, card);
        row->addWidget(lane, 0);

        auto* body = new QVBoxLayout();
        body->setContentsMargins(0, 0, 0, 0);
        body->setSpacing(4);

        auto* top = new QHBoxLayout();
        top->setContentsMargins(0, 0, 0, 0);
        top->setSpacing(8);

        const QFont mono   = QFontDatabase::systemFont(QFontDatabase::FixedFont);
        auto*       hash_l = new QLabel(short_id, card);
        hash_l->setFont(mono);
        hash_l->setStyleSheet("QLabel {"
                              "  color: #E6E6E6;"
                              "  font-size: 12px;"
                              "  font-weight: 600;"
                              "}");

        top->addWidget(hash_l, 0);

        if (is_head) {
          top->addWidget(MakePillLabel("HEAD", "#121212", "rgba(252, 199, 4, 0.95)",
                                       "rgba(252, 199, 4, 0.95)", card),
                         0);
        }
        if (is_base) {
          top->addWidget(MakePillLabel("BASE", "#121212", "rgba(252, 199, 4, 0.88)",
                                       "rgba(252, 199, 4, 0.88)", card),
                         0);
        }
        if (is_plain) {
          top->addWidget(MakePillLabel("PLAIN", "#1A1A1A", "rgba(252, 199, 4, 0.22)",
                                       "rgba(252, 199, 4, 0.40)", card),
                         0);
        } else {
          const auto parent_short =
              QString::fromStdString(ver.GetParentVersionID().ToString().substr(0, 8));
          top->addWidget(
              MakePillLabel(QString("PARENT %1").arg(parent_short), "#A3A3A3",
                            "rgba(252, 199, 4, 0.16)", "rgba(252, 199, 4, 0.32)", card),
              0);
        }

        top->addStretch(1);

        auto* tx_pill = MakePillLabel(QString("tx %1").arg(committed_tx_count), "#A3A3A3",
                                      "rgba(252, 199, 4, 0.16)", "rgba(252, 199, 4, 0.32)", card);
        top->addWidget(tx_pill, 0);

        auto* msg_l = new QLabel(msg, card);
        msg_l->setWordWrap(true);
        msg_l->setStyleSheet("QLabel {"
                             "  color: #E6E6E6;"
                             "  font-size: 12px;"
                             "}");

        auto* meta_l = new QLabel(when, card);
        meta_l->setStyleSheet("QLabel {"
                              "  color: #A3A3A3;"
                              "  font-size: 11px;"
                              "}");

        body->addLayout(top);
        body->addWidget(msg_l);
        body->addWidget(meta_l);
        row->addLayout(body, 1);

        ui.version_log->setItemWidget(item, card);

        const QString ver_id_str = QString::fromStdString(ver_id.ToString());
        if (!prev_selected_id.isEmpty() && ver_id_str == prev_selected_id) {
          ui.version_log->setCurrentItem(item);
          item->setSelected(true);
        } else if (prev_selected_id.isEmpty() && is_head) {
          ui.version_log->setCurrentItem(item);
          item->setSelected(true);
        }
      }
    }

    if (refresh_selection_styles) {
      refresh_selection_styles();
    }
  }
}

}  // namespace puerhlab::ui::versioning
