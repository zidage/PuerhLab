#include "sleeve/path_resolver.hpp"

#include <filesystem>
#include <memory>
#include <ranges>
#include <stdexcept>
#include <vector>
#include <xxhash.hpp>

#include "sleeve/sleeve_element/sleeve_element.hpp"
#include "sleeve/sleeve_element/sleeve_folder.hpp"
#include "type/type.hpp"
#include "utils/id/id_generator.hpp"

namespace puerhlab {
auto PathResolver::Normalize(const std::filesystem::path raw_path) -> std::wstring {
  return raw_path.lexically_normal();
}

auto PathResolver::Resolve(const std::filesystem::path& path) -> std::shared_ptr<SleeveElement> {
  std::shared_ptr<SleeveElement> current = _root;
  for (const auto& part : path) {
    if (current->_type != ElementType::FILE) {
      throw std::runtime_error("Path Resolver: Illegal path.");
    }

    std::shared_ptr<SleeveFolder> folder = std::static_pointer_cast<SleeveFolder>(current);
    _lazy_handler.EnsureChildrenLoaded(folder);

    auto next_id = folder->GetElementIdByName(part.wstring());
    if (!next_id.has_value()) {
      throw std::runtime_error("Path Resolver: Illegal path. Target does not exist");
    }

    current = _lazy_handler.GetElement(next_id.value());
  }
  return current;
}

auto PathResolver::ResolveForWrite(const std::filesystem::path& path, bool create)
    -> std::shared_ptr<SleeveElement> {
  std::shared_ptr<SleeveElement> current        = _root;
  std::shared_ptr<SleeveElement> current_parent = nullptr;
  std::vector<std::wstring>      path_stack;
  for (const auto& part : path) {
    path_stack.push_back(part.wstring());
    if (current->_type != ElementType::FILE) {
      throw std::runtime_error("Path Resolver: Illegal path.");
    }

    std::shared_ptr<SleeveFolder> folder = std::static_pointer_cast<SleeveFolder>(current);
    _lazy_handler.EnsureChildrenLoaded(folder);

    if (folder->IsShared()) {
      auto old_id = folder->_element_id;
      folder      = std::static_pointer_cast<SleeveFolder>(folder->Copy(IncrID::GenerateID()));
      std::static_pointer_cast<SleeveFolder>(current_parent)
          ->UpdateElementMap(folder->_element_name, old_id, folder->_element_id);
    }
    auto next_id = folder->GetElementIdByName(part.wstring());
    if (!next_id.has_value()) {
      throw std::runtime_error("Path Resolver: Illegal path. Target does not exist");
    }
    current_parent = current;
    current        = _lazy_handler.GetElement(next_id.value());
  }
  return current;
}

};  // namespace puerhlab