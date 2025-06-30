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
#include "type/type.hpp"
#include "utils/cache/lru_cache.hpp"
#include "utils/id/id_generator.hpp"

namespace puerhlab {
class PathResolver {
 private:
  std::shared_ptr<SleeveFolder>        _root;
  LRUCache<sl_path_t, sl_element_id_t> _directory_cache;
  NodeStorageHandler&                  _storage_handler;
  IncrID::IDGenerator<uint32_t>&       _id_gen;

  auto                                 CoWHandler(const std::shared_ptr<SleeveElement> to_copy,
                                                  const std::shared_ptr<SleeveFolder>  parent_folder)
      -> std::shared_ptr<SleeveElement>;

 public:
  PathResolver();
  PathResolver(NodeStorageHandler& lazy_handler, IncrID::IDGenerator<uint32_t>& id_gen);
  void        SetRoot(std::shared_ptr<SleeveFolder> root);
  static auto Normalize(const std::filesystem::path raw_path) -> std::wstring;

  auto IsSubpath(const std::filesystem::path& base, const std::filesystem::path& target) -> bool;
  auto Contains(const std::filesystem::path& path, ElementType type) -> bool;
  auto Contains(const std::filesystem::path& path) -> bool;
  auto Resolve(const std::filesystem::path& path) -> std::shared_ptr<SleeveElement>;
  auto ResolveForWrite(const std::filesystem::path& path) -> std::shared_ptr<SleeveElement>;

  auto Tree(const std::filesystem::path& path) -> std::wstring;
};
};  // namespace puerhlab