//  Copyright 2026 Yurun Zi
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.

#include "app/import_service.hpp"

#include <filesystem>

#include "image/metadata_extractor.hpp"
#include "sleeve/sleeve_element/sleeve_element.hpp"
#include "type/supported_file_type.hpp"

namespace puerhlab {
auto ImportServiceImpl::ImportToFolder(const std::vector<image_path_t>& paths,
                                       const image_path_t& dest, const ImportOptions& options,
                                       std::shared_ptr<ImportJob> job)
    -> std::shared_ptr<ImportJob> {
  ImportProgress progress;
  progress.total_ = static_cast<uint32_t>(paths.size());

  for (const auto& image_path : paths) {
    if (job && job->IsCancelled()) {
      break;
    }
    // Check image file type
    if (!std::filesystem::is_regular_file(image_path) ||
        !std::filesystem::is_regular_file(image_path)) {
      progress.failed_.fetch_add(1);
      if (job) {
        job->on_progress_(progress);
      }
      continue;
    }
    auto element = fs_->Create(dest, image_path.filename(), ElementType::FILE);
    if (!element) {
      progress.failed_.fetch_add(1);
      if (job) {
        job->on_progress_(progress);
      }
      continue;
    }
    // Create the corresponding image file
    auto sleeve_file = std::static_pointer_cast<SleeveFile>(element);
    auto image_ptr   = image_pool_manager_->InsertEmpty();

    if (!image_ptr) {
      progress.failed_.fetch_add(1);
      if (job) {
        job->on_progress_(progress);
      }
      continue;
    }

    image_ptr->image_path_ = image_path;
    image_ptr->image_name_ = image_path.filename().wstring();
    // TODO: Parse image type for future use

    // Link the image to the SleeveFile
    sleeve_file->SetImage(image_ptr);

    // Submit the metadata extraction task to thread pool
    thread_pool_.Submit([image_ptr, image_path, &progress, job]() {
      if (job && job->IsCancelled()) {
        return;
      }
      // Extract metadata
      try {
        MetadataExtractor::ExtractEXIF_ToImage(image_ptr->image_path_, *image_ptr);
        // Update progress
        progress.metadata_done_.fetch_add(1);
      } catch (...) {
        progress.failed_.fetch_add(1);
      }

      if (job) {
        job->on_progress_(progress);
      }

      if (progress.metadata_done_.load() + progress.failed_.load() >= progress.total_ ||
          (job && job->IsCancelled())) {
        // All done
        ImportResult result;
        result.requested_ = progress.total_;
        result.imported_  = progress.metadata_done_.load();
        result.failed_    = progress.failed_.load();
        if (job) {
          job->on_finished_(result);
        }
      }
    });
  }
  return job;
}
};  // namespace puerhlab