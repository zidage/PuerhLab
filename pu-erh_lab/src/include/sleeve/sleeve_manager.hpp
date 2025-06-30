#pragma once

#include <opencv2/core/hal/interface.h>

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <unordered_map>

#include "io/image/image_loader.hpp"
#include "sleeve/sleeve_filesystem.hpp"
#include "sleeve_base.hpp"
#include "sleeve_view.hpp"
#include "storage/image_pool/image_pool_manager.hpp"
#include "type/type.hpp"

namespace puerhlab {
class SleeveManager {
 private:
  std::shared_ptr<FileSystem>       _fs;
  std::shared_ptr<SleeveView>       _view;
  std::shared_ptr<ImagePoolManager> _image_pool;

 public:
  explicit SleeveManager(std::filesystem::path db_path);

  auto GetFilesystem() -> std::shared_ptr<FileSystem>;
  auto GetView() -> std::shared_ptr<SleeveView>;
  auto GetPool() -> std::shared_ptr<ImagePoolManager>;
  auto GetImgCount() -> uint32_t;
  auto LoadToPath(std::vector<image_path_t> img_os_path, sl_path_t dest) -> uint32_t;

  auto RestoreSleeveFromDB(sleeve_id_t sleeve_id) -> std::shared_ptr<FileSystem>;
};
};  // namespace puerhlab