#include "storage/service/sleeve/edit_history/history_service.hpp"

#include <format>

#include "edit/history/edit_history.hpp"

namespace puerhlab {
auto EditHistoryService::ToParams(const std::shared_ptr<EditHistory> source)
    -> EditHistoryMapperParams {
  EditHistoryMapperParams param;
  param.file_id = source->GetBoundImage();
  param.history = std::make_unique<std::string>(source->ToJSON().dump());
  return param;
}

auto EditHistoryService::FromParams(EditHistoryMapperParams&& param)
    -> std::shared_ptr<EditHistory> {
  auto history = std::make_shared<EditHistory>(param.file_id);
  if (param.history) {
    history->FromJSON(nlohmann::json::parse(std::move(*param.history)));
  }
  return history;
}

auto EditHistoryService::GetEditHistoryByFileId(const sl_element_id_t file_id)
    -> std::shared_ptr<EditHistory> {
  auto result = GetByPredicate(std::format("file_id={}", file_id));
  if (result.size() > 1) {
    throw std::runtime_error("EditHistoryService: Multiple edit history found for file_id " +
                             std::to_string(file_id));
  }
  if (result.empty()) {
    throw std::runtime_error(std::format(
        "EditHistoryService: No history bound with sleeve file id {} is stored in DB", file_id));
  }
  return result.front();
}
};  // namespace puerhlab