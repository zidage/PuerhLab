//  Copyright 2026 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#include "ui/alcedo_main/editor_dialog/modules/versioning.hpp"

#include <QColor>
#include <QComboBox>
#include <QCoreApplication>
#include <QDateTime>
#include <QHBoxLayout>
#include <QLabel>
#include <QListWidget>
#include <QPushButton>
#include <QVBoxLayout>
#include <QWidget>
#include <algorithm>
#include <json.hpp>
#include <string>
#include <utility>
#include <vector>

#include "edit/history/edit_transaction.hpp"
#include "edit/pipeline/pipeline_cpu.hpp"
#include "ui/alcedo_main/app_theme.hpp"
#include "ui/alcedo_main/i18n.hpp"
#include "ui/alcedo_main/editor_dialog/controllers/history_controller.hpp"
#include "ui/alcedo_main/editor_dialog/widgets/history_cards.hpp"

namespace alcedo::ui::versioning {

namespace {

auto Tr(const char* text) -> QString {
  return QCoreApplication::translate(ALCEDO_I18N_CONTEXT, text);
}

void AddEmptyStateItem(QListWidget* list_widget, const QString& text) {
  if (!list_widget) {
    return;
  }
  auto* item = new QListWidgetItem(list_widget);
  item->setFlags(Qt::NoItemFlags);
  item->setSizeHint(QSize(0, 56));

  auto* label = new QLabel(text, list_widget);
  label->setAlignment(Qt::AlignCenter);
  label->setStyleSheet(
      AppTheme::EditorLabelStyle(AppTheme::Instance().textMutedColor()));
  AppTheme::MarkFontRole(label, AppTheme::FontRole::UiCaption);
  label->setAttribute(Qt::WA_TransparentForMouseEvents, true);
  list_widget->setItemWidget(item, label);
}

}  // namespace

auto MakeTxCountLabel(size_t tx_count) -> QString {
  return Tr("Uncommitted: %1").arg(static_cast<qulonglong>(tx_count));
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
      *error = Tr("Invalid version ID: %1").arg(e.what());
    }
    return false;
  }

  Version* selected_version = nullptr;
  try {
    selected_version = &history_guard->history_->GetVersion(version_id);
  } catch (const std::exception& e) {
    if (error) {
      *error = Tr("Failed to load selected version: %1").arg(e.what());
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
    result.error = Tr("Undo failed: %1").arg(e.what());
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
    result.error = Tr("Edit history service not available.");
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
    result.error = Tr("Commit failed: %1").arg(e.what());
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
  const QString label = MakeTxCountLabel(tx_count);

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
    if (total == 0) {
      AddEmptyStateItem(ui.tx_stack, Tr("No data"));
    }
    size_t i = 0;
    for (const auto& tx : txs) {
      auto* item = new QListWidgetItem(ui.tx_stack);
      item->setSizeHint(QSize(0, 56));

      auto* card = BuildTxHistoryCard(tx, /*draw_top*/ i > 0,
                                      /*draw_bottom*/ (i + 1) < total, ui.tx_stack);
      card->SetSelected(i == 0);
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
                .toString("MM-dd HH:mm");
        const auto committed_tx_count =
            static_cast<qulonglong>(ver.GetAllEditTransactions().size());

        const auto& txs = ver.GetAllEditTransactions();

        const bool is_head  = (ver_id == latest_id);
        const bool is_base  = (base_parent == ver_id && working_version.HasParentVersion());
        const bool is_plain = !ver.HasParentVersion();
        const int  version_number = std::max(1, total_rows - row_index);
        const QString version_title =
            is_plain ? Tr("Plain") : Tr("Version %1").arg(static_cast<qulonglong>(version_number));

        QString last_tx_summary;
        QString last_tx_full;
        if (!txs.empty()) {
          const QString detail = CompactTxDelta(txs.front());
          last_tx_summary = detail.isEmpty()
                                ? OperatorDisplayName(txs.front().GetTxOperatorType())
                                : QStringLiteral("%1 | %2")
                                      .arg(OperatorDisplayName(txs.front().GetTxOperatorType()),
                                           detail);
          last_tx_full    = QString::fromStdString(txs.front().Describe(true, 4096));
        } else {
          last_tx_summary = Tr("(empty)");
          last_tx_full    = last_tx_summary;
        }

        const QString parent_id =
            is_plain ? Tr("(none)") : QString::fromStdString(ver.GetParentVersionID().ToString());
        const QString meta = Tr("%1 · %2 · tx %3").arg(short_id, when).arg(committed_tx_count);
        const QString card_tooltip =
            Tr("Version: %1\nID: %2\nParent: %3\nUpdated: %4\nTransactions: %5\nLast operation: %6")
                .arg(version_title)
                .arg(QString::fromStdString(ver_id.ToString()))
                .arg(parent_id)
                .arg(when)
                .arg(committed_tx_count)
                .arg(last_tx_full);

        auto* item = new QListWidgetItem(ui.version_log);
        item->setData(Qt::UserRole, QString::fromStdString(ver_id.ToString()));
        item->setToolTip(card_tooltip);
        item->setSizeHint(QSize(0, 56));

        auto* card = new HistoryCardWidget(ui.version_log);
        card->setToolTip(card_tooltip);
        auto* row  = new QHBoxLayout(card);
        row->setContentsMargins(8, 4, 8, 4);
        row->setSpacing(4);

        const QColor dot =
            is_head ? AppTheme::Instance().accentColor() : AppTheme::Instance().textMutedColor();
        const QColor line = AppTheme::Instance().dividerColor();
        auto* lane        = new HistoryLaneWidget(dot, line, /*draw_top*/ row_index > 0,
                                                  /*draw_bottom*/ (row_index + 1) < total_rows,
                                                  /*solid_dot*/ is_head, card);
        row->addWidget(lane, 0);

        auto* body = new QVBoxLayout();
        body->setContentsMargins(0, 0, 0, 0);
        body->setSpacing(0);

        auto* title_l = new ElidedLabel(version_title, card);
        QFont title_font = AppTheme::Font(AppTheme::FontRole::UiBodyStrong);
        title_font.setPointSizeF(9.5);
        title_font.setWeight(QFont::DemiBold);
        title_font.setStyleStrategy(QFont::PreferAntialias);
        title_l->setFont(title_font);
        title_l->setStyleSheet(
            AppTheme::EditorLabelStyle(AppTheme::Instance().textColor()));
        title_l->setAttribute(Qt::WA_TransparentForMouseEvents, true);

        auto* subtitle_l = new ElidedLabel(last_tx_summary, card);
        QFont subtitle_font = AppTheme::Font(AppTheme::FontRole::UiCaption);
        subtitle_font.setPointSizeF(8.0);
        subtitle_font.setWeight(QFont::Normal);
        subtitle_font.setStyleStrategy(QFont::PreferAntialias);
        subtitle_l->setFont(subtitle_font);
        subtitle_l->setStyleSheet(
            AppTheme::EditorLabelStyle(AppTheme::Instance().textMutedColor()));
        subtitle_l->setAttribute(Qt::WA_TransparentForMouseEvents, true);

        auto* meta_l = new ElidedLabel(meta, card);
        QFont meta_font = AppTheme::Font(AppTheme::FontRole::DataCaption);
        meta_font.setPointSizeF(7.5);
        meta_font.setWeight(QFont::Normal);
        meta_font.setStyleStrategy(QFont::PreferAntialias);
        meta_l->setFont(meta_font);
        meta_l->setStyleSheet(
            AppTheme::EditorLabelStyle(AppTheme::Instance().textMutedColor()));
        meta_l->setAttribute(Qt::WA_TransparentForMouseEvents, true);

        body->addWidget(title_l);
        body->addWidget(subtitle_l);
        body->addWidget(meta_l);
        row->addLayout(body, 1);

        auto* badges = new QVBoxLayout();
        badges->setContentsMargins(0, 0, 0, 0);
        badges->setSpacing(1);
        badges->setAlignment(Qt::AlignTop | Qt::AlignRight);

        if (is_head) {
          badges->addWidget(MakePillLabel(Tr("HEAD"), card), 0, Qt::AlignRight);
        }
        if (is_base) {
          badges->addWidget(MakePillLabel(Tr("BASE"), card), 0, Qt::AlignRight);
        }
        if (is_plain) {
          badges->addWidget(MakePillLabel(Tr("PLAIN"), card), 0, Qt::AlignRight);
        }

        if (badges->count() > 0) {
          row->addLayout(badges, 0);
        } else {
          delete badges;
        }

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
    if (ui.version_log->count() == 0) {
      AddEmptyStateItem(ui.version_log, Tr("No data"));
    }

    if (refresh_selection_styles) {
      refresh_selection_styles();
    }
  }
}

}  // namespace alcedo::ui::versioning
