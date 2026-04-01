//  Copyright 2026 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#pragma once

#include <QProcess>
#include <QString>
#include <QVariantList>

#include <filesystem>
#include <memory>
#include <vector>

#include "app/import_service.hpp"
#include "ui/puerhlab_main/i18n.hpp"
#include "ui/puerhlab_main/album_backend/image_controller.hpp"
#include "ui/puerhlab_main/album_backend/nikon_he_recovery_types.hpp"
#include "utils/import/import_log.hpp"

namespace puerhlab::ui {

class AlbumBackend;

class NikonHeRecoveryController {
 public:
  explicit NikonHeRecoveryController(AlbumBackend& backend);

  void BeginRecovery(const std::vector<ImportLogEntry>& unsupported_entries,
                     sl_element_id_t                  import_target_folder_id,
                     const std::filesystem::path& import_target_folder_path);
  void BrowseConverter();
  void StartConversion();
  void ExitRecovery();
  void UpdateReimportProgress(uint32_t completed, uint32_t total, uint32_t failed);
  void HandleReimportFinished(const ImportResult& result);

  [[nodiscard]] bool         is_active() const { return active_; }
  [[nodiscard]] bool         is_busy() const { return busy_; }
  [[nodiscard]] bool         is_reimporting() const {
    return active_ && phase_ == NikonHeRecoveryPhase::REIMPORTING_DNG;
  }
  [[nodiscard]] auto         phase() const -> NikonHeRecoveryPhase { return phase_; }
  [[nodiscard]] auto         phase_text() const -> QString;
  [[nodiscard]] auto         status_text() const -> QString { return status_text_.Render(); }
  [[nodiscard]] auto         unsupported_files() const -> QVariantList;
  [[nodiscard]] const QString& converter_path() const { return converter_path_; }

 private:
  void NotifyStateChanged();
  void SetPhase(NikonHeRecoveryPhase phase, const i18n::LocalizedText& status, int progress = 0);
  void FinishAndClose(const i18n::LocalizedText& status, int progress = 100);
  void ClearState();
  void HandleConverterFinished(int exit_code, int exit_status);
  auto ValidateConvertedDngs(QString* warning_summary) -> std::vector<image_path_t>;
  auto BuildDeleteTargets() const -> std::vector<ImageController::DeleteTarget>;
  void RemoveUnsupportedEntriesAndContinue(const std::vector<image_path_t>& converted_paths,
                                           const QString& completion_note);

  AlbumBackend&            backend_;
  NikonHeRecoveryItemList  items_{};
  sl_element_id_t          import_target_folder_id_ = 0;
  std::filesystem::path    import_target_folder_path_{};
  QString                  converter_path_{};
  std::unique_ptr<QProcess> process_{};
  NikonHeRecoveryPhase     phase_ = NikonHeRecoveryPhase::IDLE;
  i18n::LocalizedText      status_text_{};
  QString                  completion_note_{};
  bool                     active_ = false;
  bool                     busy_   = false;
};

}  // namespace puerhlab::ui
