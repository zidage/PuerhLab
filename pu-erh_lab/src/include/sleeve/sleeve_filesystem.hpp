#pragma once

#include <filesystem>
#include <memory>
#include <string>
#include <unordered_map>

#include "path_resolver.hpp"
#include "sleeve/sleeve_element/sleeve_element.hpp"
#include "sleeve/sleeve_element/sleeve_folder.hpp"
#include "storage/image_pool/image_pool_manager.hpp"
#include "storage_service.hpp"
#include "type/type.hpp"
#include "utils/id/id_generator.hpp"

namespace puerhlab {
class FileSystem {
  using NodeMapping = std::unordered_map<sl_element_id_t, std::shared_ptr<SleeveElement>>;

 private:
  // A mapping between node id and pointers to actual nodes.
  NodeMapping                          _storage;
  std::shared_ptr<SleeveFolder>        _root;

  // ID Generation
  IncrID::IDGenerator<sl_element_id_t> _id_gen;
  /** @name Database interaction */
  ///@{
  std::filesystem::path                _db_path;
  StorageService                       _storage_service;
  LazyNodeHandler                      _lazy_handler;
  PathResolver                         _resolver;
  ///@}

 public:
  FileSystem(std::filesystem::path db_path, sl_element_id_t start_id);

  auto InitRoot() -> bool;

  void Create(std::filesystem::path dest, std::wstring filename, ElementType type);
  void Delete(std::filesystem::path target);
  auto Get(std::filesystem::path target, bool write) -> std::shared_ptr<SleeveElement>;
  void Copy(std::filesystem::path from, std::filesystem::path dest);
};
};  // namespace puerhlab