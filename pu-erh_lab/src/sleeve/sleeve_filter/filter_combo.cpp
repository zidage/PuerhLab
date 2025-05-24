#include "sleeve/sleeve_filter/filter_combo.hpp"

#include <cstddef>
#include <set>
#include <unordered_map>

namespace puerhlab {
auto FilterCombo::GetFilters() -> std::vector<SleeveFilter> & { return _filters; }
auto FilterCombo::CreateIndexOn(std::shared_ptr<std::set<sl_element_id_t>> _lists)
    -> std::shared_ptr<std::set<sl_element_id_t>> {
  return nullptr;
}
};  // namespace puerhlab