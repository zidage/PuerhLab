//  Copyright 2025 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#pragma once

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "edit/pipeline/pipeline_cpu.hpp"
#include "edit/history/edit_history.hpp"
#include "sleeve/sleeve_element/sleeve_element.hpp"
#include "sleeve/sleeve_filter/filter_combo.hpp"
#include "storage/controller/controller_types.hpp"
#include "storage/service/pipeline/pipeline_service.hpp"
#include "storage/service/sleeve/edit_history/history_service.hpp"
#include "storage/service/sleeve/element/element_id_service.hpp"
#include "storage/service/sleeve/element/element_service.hpp"
#include "storage/service/sleeve/element/file_service.hpp"
#include "storage/service/sleeve/element/folder_service.hpp"
#include "type/type.hpp"

namespace puerhlab {
struct StorageStatsBucket {
  std::string label_{};
  int         count_ = 0;
};

struct FolderStatsView {
  int                             total_photo_count_ = 0;
  std::vector<StorageStatsBucket> date_stats_{};
  std::vector<StorageStatsBucket> camera_stats_{};
  std::vector<StorageStatsBucket> lens_stats_{};
};

class ElementController {
 private:
  ConnectionGuard    guard_;

  ElementService     element_service_;
  ElementIdService  element_id_service_;
  
  FileService        file_service_;
  FolderService      folder_service_;
  EditHistoryService history_service_;
  PipelineService    pipeline_service_;
  EditHistoryService edit_history_service_;

 public:
  ElementController(ConnectionGuard&& guard);

  void AddElement(const std::shared_ptr<SleeveElement> element);

  void AddFolderContent(sl_element_id_t folder_id, sl_element_id_t content_id);
  auto GetFolderContent(const sl_element_id_t folder_id) -> std::vector<sl_element_id_t>;

  void RemoveElement(const sl_element_id_t id);
  void RemoveElement(const std::shared_ptr<SleeveElement> element);
  void UpdateElement(const std::shared_ptr<SleeveElement> element);
  auto GetElementById(const sl_element_id_t id) -> std::shared_ptr<SleeveElement>;

  auto GetElementsInFolderByFilter(const std::shared_ptr<FilterCombo> filter,
                                   const sl_element_id_t              folder_id)
      -> std::vector<std::shared_ptr<SleeveElement>>;

  auto GetElementIdsInFolderByFilter(const std::shared_ptr<FilterCombo> filter,
                                      const sl_element_id_t folder_id)
      -> std::vector<sl_element_id_t>;
  auto BuildFolderStats(sl_element_id_t                           folder_id,
                        const std::optional<std::wstring>& extra_filter_where = std::nullopt)
      -> FolderStatsView;

  void EnsureChildrenLoaded(sl_element_id_t folder_id);

  auto GetPipelineByElementId(const sl_element_id_t element_id)
      -> std::shared_ptr<CPUPipelineExecutor>;
  auto UpdatePipelineByElementId(const sl_element_id_t                      element_id,
                                 const std::shared_ptr<CPUPipelineExecutor> pipeline) -> void;
  auto RemovePipelineByElementId(const sl_element_id_t element_id) -> void;

  auto GetEditHistoryByFileId(const sl_element_id_t file_id) -> std::shared_ptr<EditHistory>;
  auto UpdateEditHistoryByFileId(const sl_element_id_t file_id,
                                 const std::shared_ptr<EditHistory> history) -> void;
  auto RemoveEditHistoryByFileId(const sl_element_id_t file_id) -> void;

  auto GetEditHistoryService() -> std::shared_ptr<EditHistoryService>;

  void UpdateEditHistoryService(const std::shared_ptr<EditHistoryService> new_service);
};
};  // namespace puerhlab
