#include "sleeve/path_resolver.hpp"

#include <memory>
#include <stdexcept>
#include <xxhash.hpp>

#include "sleeve/sleeve_element/sleeve_element.hpp"
#include "sleeve/sleeve_element/sleeve_folder.hpp"
#include "type/type.hpp"

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
  std::shared_ptr<SleeveElement> current = _root;
  for (const auto& part : path) {
    if (current->_type != ElementType::FILE) {
      throw std::runtime_error("Path Resolver: Illegal path.");
    }

    std::shared_ptr<SleeveFolder> folder = std::static_pointer_cast<SleeveFolder>(current);
    _lazy_handler.EnsureChildrenLoaded(folder);

    folder       = std::static_pointer_cast<SleeveFolder>(CoWIfNeeded(folder));
    auto next_id = folder->GetElementIdByName(part.wstring());
    if (!next_id.has_value()) {
      throw std::runtime_error("Path Resolver: Illegal path. Target does not exist");
    }

    current = _lazy_handler.GetElement(next_id.value());
  }
  return current;
}

auto PathResolver::CoWIfNeeded(std::shared_ptr<SleeveFolder> folder)
    -> std::shared_ptr<SleeveElement> {
  if (folder->_ref_count <= 1) {
    return folder;
  }
  return folder->Copy(xxh::xxhash<32>(folder.get(), sizeof(*folder)));
}

};  // namespace puerhlab