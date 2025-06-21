#include "sleeve/path_resolver.hpp"

#include <algorithm>
#include <exception>
#include <filesystem>
#include <memory>
#include <ranges>
#include <sstream>
#include <stack>
#include <stdexcept>
#include <string>
#include <vector>
#include <xxhash.hpp>

#include "sleeve/sleeve_element/sleeve_element.hpp"
#include "sleeve/sleeve_element/sleeve_folder.hpp"
#include "type/type.hpp"
#include "utils/id/id_generator.hpp"

namespace puerhlab {
PathResolver::PathResolver(NodeStorageHandler& lazy_handler, IncrID::IDGenerator<uint32_t>& id_gen)
    : _storage_handler(lazy_handler), _id_gen(id_gen) {}

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
      // TODO: add customized exception class
      throw std::runtime_error("Path Resolver: Illegal path.");
    }

    std::shared_ptr<SleeveFolder> folder = std::static_pointer_cast<SleeveFolder>(current);
    _storage_handler.EnsureChildrenLoaded(folder);

    auto next_id = folder->GetElementIdByName(part.wstring());
    if (!next_id.has_value()) {
      throw std::runtime_error("Path Resolver: Illegal path. Target does not exist");
    }

    current = _storage_handler.GetElement(next_id.value());
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
    _storage_handler.EnsureChildrenLoaded(folder);

    if (folder->IsShared()) {
      folder = std::static_pointer_cast<SleeveFolder>(
          CoWHandler(current, std::static_pointer_cast<SleeveFolder>(current_parent)));
    }
    auto next_id = folder->GetElementIdByName(part.wstring());
    if (!next_id.has_value()) {
      throw std::runtime_error("Path Resolver: Illegal path. Target does not exist");
    }
    current_parent = current;
    current        = _storage_handler.GetElement(next_id.value());
  }
  if (current->IsShared()) {
    current = CoWHandler(current, std::static_pointer_cast<SleeveFolder>(current_parent));
  }
  return current;
}

auto PathResolver::CoWHandler(const std::shared_ptr<SleeveElement> to_copy,
                              const std::shared_ptr<SleeveFolder>  parent_folder)
    -> std::shared_ptr<SleeveElement> {
  auto old_id = to_copy->_element_id;
  to_copy->DecrementRefCount();
  auto copied = to_copy->Copy(_id_gen.GenerateID());
  parent_folder->UpdateElementMap(copied->_element_name, old_id, copied->_element_id);
  copied->IncrementRefCount();
  _storage_handler.AddToStorage(copied);
  return copied;
}

auto PathResolver::Tree(const std::filesystem::path& path) -> std::wstring {
  struct TreeNode {
    sl_element_id_t id;
    int             depth;
    bool            is_last;
  };

  std::wostringstream tree_str;
  auto                start_node = Resolve(path);
  if (start_node->_type != ElementType::FOLDER) {
    throw std::runtime_error("Filesystem: Cannot call Tree() on a file node");
  }

  auto dfs_stack    = std::stack<TreeNode>();

  auto visit_folder = std::static_pointer_cast<SleeveFolder>(start_node);
  _storage_handler.EnsureChildrenLoaded(visit_folder);

  auto contains = visit_folder->ListElements();
  for (auto& e : *contains) {
    dfs_stack.push({e, 0, _storage_handler.GetElement(e)->_type == ElementType::FILE});
  }

  tree_str << visit_folder->_element_name << L"id:" << std::to_wstring(visit_folder->_element_id)
           << L"\n";
  std::shared_ptr<SleeveElement> parent = nullptr;

  while (!dfs_stack.empty()) {
    auto next_visit = dfs_stack.top();
    dfs_stack.pop();
    auto next_visit_element = _storage_handler.GetElement(next_visit.id);

    if (next_visit_element->_type == ElementType::FOLDER) {
      for (int i = 0; i < next_visit.depth; ++i) {
        tree_str << L"    ";
      }
      tree_str << L"├── " << next_visit_element->_element_name << L" id:"
               << std::to_wstring(next_visit_element->_element_id) << L"\n";
      auto sub_folder = std::static_pointer_cast<SleeveFolder>(next_visit_element);
      contains        = sub_folder->ListElements();
      for (auto& e : *contains) {
        dfs_stack.push(
            {e, next_visit.depth + 1, _storage_handler.GetElement(e)->_type == ElementType::FILE});
      }
    } else {
      for (int i = 0; i < next_visit.depth; ++i) {
        tree_str << L"    ";
      }
      tree_str << L"└── " + next_visit_element->_element_name << L" id:"
               << std::to_wstring(next_visit_element->_element_id) << L"\n";
    }
  }
  return tree_str.str();
}
};  // namespace puerhlab