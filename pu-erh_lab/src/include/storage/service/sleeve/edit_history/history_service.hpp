#pragma once

#include <duckdb.h>

#include <memory>

#include "edit/history/edit_history.hpp"
#include "storage/mapper/sleeve/edit_history/history_mapper.hpp"
#include "storage/service/service_interface.hpp"

namespace puerhlab {
class EditHistoryService
    : public ServiceInterface<EditHistoryService, std::shared_ptr<EditHistory>,
                              EditHistoryMapperParams, EditHistoryMapper, sl_element_id_t> {
 public:
  using ServiceInterface::ServiceInterface;

  static auto ToParams(const std::shared_ptr<EditHistory> source) -> EditHistoryMapperParams;
  static auto FromParams(EditHistoryMapperParams&& param) -> std::shared_ptr<EditHistory>;

  auto        GetEditHistoryByFileId(const sl_element_id_t file_id) -> std::shared_ptr<EditHistory>;
};
};  // namespace puerhlab