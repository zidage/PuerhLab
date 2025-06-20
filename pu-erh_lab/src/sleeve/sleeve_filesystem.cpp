#include "sleeve/sleeve_filesystem.hpp"

#include <exception>
#include <filesystem>
#include <memory>
#include <stdexcept>

#include "sleeve/sleeve_element/sleeve_element.hpp"
#include "sleeve/sleeve_element/sleeve_element_factory.hpp"
#include "sleeve/sleeve_element/sleeve_folder.hpp"

namespace puerhlab {
FileSystem::FileSystem(std::filesystem::path db_path, sl_element_id_t start_id)
    : _id_gen(start_id),
      _db_path(db_path),
      _storage_service(db_path),
      _storage_handler(_storage_service.GetElementController(), _storage),
      _resolver(_storage_handler, _id_gen) {
  _root = nullptr;
}

auto FileSystem::InitRoot() -> bool {
  if (_root != nullptr) {
    throw std::runtime_error("Filesystem: root has already been initialized");
  }
  // root's id is always 0
  std::shared_ptr<SleeveElement> root;
  try {
    root = _storage_service.GetElementController().GetElementById(0);
  } catch (std::exception& e) {
    root = SleeveElementFactory::CreateElement(ElementType::FOLDER, 0, L"");
  }
  _storage[0] = root;
  _root       = std::static_pointer_cast<SleeveFolder>(root);
  _resolver.SetRoot(_root);
  return true;
}

auto FileSystem::Create(std::filesystem::path dest, std::wstring filename, ElementType type)
    -> std::shared_ptr<SleeveElement> {
  auto dest_element = _resolver.ResolveForWrite(dest);
  if (dest_element->_type != ElementType::FOLDER) {
    throw std::runtime_error("Filesystem: Cannot create element under a file");
  }
  auto dest_folder = std::static_pointer_cast<SleeveFolder>(dest_element);
  while (dest_folder->Contains(filename)) {
    filename = filename + L"@";
  }
  auto new_id      = _id_gen.GenerateID();
  auto new_element = SleeveElementFactory::CreateElement(type, new_id, filename);
  _storage[new_id] = new_element;
  dest_folder->AddElementToMap(new_element);

  return new_element;
}

auto FileSystem::Get(sl_element_id_t id) -> std::shared_ptr<SleeveElement> {
  return _storage_handler.GetElement(id);
}

auto FileSystem::Get(std::filesystem::path target, bool write) -> std::shared_ptr<SleeveElement> {
  if (write) {
    return _resolver.ResolveForWrite(target);
  }
  return _resolver.Resolve(target);
}

void FileSystem::Delete(std::filesystem::path target) {
  if (!target.has_parent_path()) {
    throw std::runtime_error("Filesystem: root cannot be deleted");
  }
  auto parent           = target.parent_path();
  auto delete_node_name = target.filename();
  if (!_resolver.Contains(parent) || !_resolver.Contains(target)) {
    throw std::runtime_error("Filesystem: Deleting node does not exist");
  }
  // Now re-acquire parent_node using write method
  auto parent_write_node =
      std::static_pointer_cast<SleeveFolder>(_resolver.ResolveForWrite(parent));
  auto delete_node_id = parent_write_node->GetElementIdByName(delete_node_name);
  auto delete_node    = _storage.at(delete_node_id.value());
  delete_node->DecrementRefCount();

  parent_write_node->RemoveNameFromMap(delete_node_name);
}

void FileSystem::Copy(std::filesystem::path from, std::filesystem::path dest) {
  // Path resolver will do the sanity check
  if (_resolver.IsSubpath(from, dest)) {
    throw std::runtime_error(
        "Filesystem: Target folder cannot be a subfolder of the original folder");
  }
  if (!_resolver.Contains(from) || !_resolver.Contains(dest, ElementType::FOLDER)) {
    throw std::runtime_error("Filesystem: Origin path or destination path does not exist");
  }

  auto from_node = _resolver.ResolveForWrite(from);
  auto dest_node = std::static_pointer_cast<SleeveFolder>(_resolver.ResolveForWrite(dest));
  dest_node->AddElementToMap(from_node);

  from_node->IncrementRefCount();
}

auto FileSystem::Tree(const std::filesystem::path& path) -> std::wstring {
  return _resolver.Tree(path);
}
};  // namespace puerhlab