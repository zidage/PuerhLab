#include "sleeve/path_resolver.hpp"

#include <algorithm>
#include <exception>
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
PathResolver::PathResolver(LazyNodeHandler& lazy_handler) : _lazy_handler(lazy_handler) {}

void PathResolver::SetRoot(std::shared_ptr<SleeveFolder> root) { _root = root; }

auto PathResolver::Normalize(const std::filesystem::path raw_path) -> std::wstring {
  return raw_path.lexically_normal();
}

auto PathResolver::IsSubpath(const std::filesystem::path& base, const std::filesystem::path& target)
    -> bool {
  auto mismatch = std::mismatch(base.begin(), base.end(), target.begin());
  return mismatch.first == base.end();
}

auto PathResolver::Contains(const std::filesystem::path& path) -> bool {
  try {
    Resolve(path);
  } catch (std::exception& e) {
    return false;
  }
  return true;
}

auto PathResolver::Contains(const std::filesystem::path& path, ElementType type) -> bool {
  try {
    auto target = Resolve(path);
    if (target->_type != type) {
      return false;
    }
  } catch (std::exception& e) {
    return false;
  }
  return true;
}

auto PathResolver::Resolve(const std::filesystem::path& path) -> std::shared_ptr<SleeveElement> {
  std::shared_ptr<SleeveElement> current    = _root;
  auto                           visit_path = path.relative_path();
  for (const auto& part : visit_path) {
    if (current->_type == ElementType::FILE) {
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

auto PathResolver::ResolveForWrite(const std::filesystem::path& path)
    -> std::shared_ptr<SleeveElement> {
  std::shared_ptr<SleeveElement> current        = _root;
  std::shared_ptr<SleeveElement> current_parent = nullptr;
  auto                           visit_path     = path.relative_path();
  for (const auto& part : visit_path) {
    if (current->_type == ElementType::FILE) {
      throw std::runtime_error("Path Resolver: Illegal path.");
    }

    std::shared_ptr<SleeveFolder> folder = std::static_pointer_cast<SleeveFolder>(current);
    _lazy_handler.EnsureChildrenLoaded(folder);

    if (folder->IsShared()) {
      auto old_id = folder->_element_id;
      folder->DecrementRefCount();
      folder = std::static_pointer_cast<SleeveFolder>(folder->Copy(IncrID::GenerateID()));
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