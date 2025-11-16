#include "sleeve/sleeve_filter/filter_combo.hpp"

#include <cstddef>
#include <set>
#include <unordered_map>
#include <vector>

#include "sleeve/sleeve_element/sleeve_element.hpp"
#include "type/type.hpp"


namespace puerhlab {
auto FilterCombo::GenerateSQLOn(sl_element_id_t parent_id) const -> std::wstring {
  // For now, we just compile the filter node to SQL
  std::wstring sql = std::format(
      L"SELECT\n"
      L"  e.id AS file_id\n"
      L"FROM\n"
      L"  FolderContent fc\n"
      L"JOIN Element e ON e.id = fc.element_id\n"
      L"JOIN FileImage fi ON fi.file_id = e.id\n"
      L"JOIN Image img ON img.id = fi.image_id\n"
      L"WHERE\n"
      L"  fc.folder_id = :{}\n"
      L"  AND e.type = {}\n"
      L"  AND ({});",
      parent_id, static_cast<uint32_t>(ElementType::FILE), FilterSQLCompiler::Compile(_root));
  return sql;
}
};  // namespace puerhlab