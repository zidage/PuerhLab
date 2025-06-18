#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "dentry_cache_manager.hpp"
#include "sleeve/sleeve_element/sleeve_element.hpp"
#include "sleeve/sleeve_element/sleeve_folder.hpp"
#include "storage/controller/sleeve/element_controller.hpp"
#include "storage_service.hpp"

namespace puerhlab {
class PathResolver {
 private:
  std::shared_ptr<SleeveFolder> _root;
  DCacheManager                 _directory_cache;
  LazyNodeHandler&              _lazy_handler;

 public:
  PathResolver(std::shared_ptr<SleeveFolder> root);
  static auto Normalize(const std::filesystem::path raw_path) -> std::wstring;

  auto        Resolve(const std::filesystem::path& path) -> std::shared_ptr<SleeveElement>;
  auto        ResolveForWrite(const std::filesystem::path& path, bool create)
      -> std::shared_ptr<SleeveElement>;
};
};  // namespace puerhlab