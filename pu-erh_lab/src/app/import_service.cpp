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

#include <cstdint>
#include <filesystem>
#include <memory>

#include "image/metadata_extractor.hpp"
#include "sleeve/sleeve_element/sleeve_element.hpp"
#include "sleeve/sleeve_filesystem.hpp"
#include "type/supported_file_type.hpp"

namespace puerhlab {

static void SetImportResult(std::shared_ptr<ImportJob> job, uint32_t requested, uint32_t imported,
                            uint32_t failed) {
  ImportResult result;
  result.requested_ = requested;
  result.imported_  = imported;
  result.failed_    = failed;
  if (job && job->on_finished_ && !job->cancelation_acked_.exchange(true)) {
    job->on_finished_(result);
  }
}

auto ImportServiceImpl::ImportToFolder(const std::vector<image_path_t>& paths,
                                       const image_path_t& dest, const ImportOptions& options,
                                       std::shared_ptr<ImportJob> job)
    -> std::shared_ptr<ImportJob> {
  // TODO: Use sleeve service to interact with FS
  // The current implementation is a temporary solution
  auto import_log = std::make_shared<ImportLog>();
  if (job) {
    job->import_log_ = import_log;
  }
  std::shared_ptr<ImportProgress> progress_ptr = std::make_shared<ImportProgress>();
  progress_ptr->total_                         = static_cast<uint32_t>(paths.size());

  if (paths.empty()) {
    // Immediately finish
    SetImportResult(job, 0, 0, 0);
    return job;
  }

  bool any_threadpool_submission = false;

  for (const auto& image_path : paths) {
    if (job && job->IsCancelled()) {
      break;
    }
    // Check image file type
    if (!std::filesystem::is_regular_file(image_path) || !is_supported_file(image_path)) {
      progress_ptr->failed_.fetch_add(1);
      if (job && job->on_progress_) {
        job->on_progress_(*progress_ptr);
      }
      continue;
    }
    std::shared_ptr<SleeveElement> element = nullptr;
    try {
      element =
          fs_service_.Write_NoSync<std::shared_ptr<SleeveElement>>([&dest, &image_path](FileSystem& fs) {
            return fs.Create(dest, image_path.filename(), ElementType::FILE);
          });
    } catch (...) {
      progress_ptr->failed_.fetch_add(1);
      if (job && job->on_progress_) {
        job->on_progress_(*progress_ptr);
      }
      continue;
    }
    if (!element) {
      progress_ptr->failed_.fetch_add(1);
      if (job && job->on_progress_) {
        job->on_progress_(*progress_ptr);
      }
      continue;
    }
    // Create the corresponding image file
    auto sleeve_file = std::static_pointer_cast<SleeveFile>(element);

    auto image_ptr   = image_pool_manager_->InsertEmpty();

    if (!image_ptr) {
      progress_ptr->failed_.fetch_add(1);
      if (job && job->on_progress_) {
        job->on_progress_(*progress_ptr);
      }
      continue;
    }

    image_ptr->image_path_ = image_path;
    image_ptr->image_name_ = image_path.filename().wstring();
    // TODO: Parse image type for future use

    // Link the image to the SleeveFile
    sleeve_file->SetImage(image_ptr);
    progress_ptr->placeholders_created_.fetch_add(1);
    if (import_log) {
      import_log->AddPlaceholder(image_ptr->image_id_, sleeve_file->element_id_,
                                 image_path.filename().wstring());
    }

    any_threadpool_submission = true;

    // Submit the metadata extraction task to thread pool
    thread_pool_.Submit([image_ptr, image_path, progress_ptr, job, import_log]() {
      if (job && job->IsCancelationAcked()) {
        return;
      }
      // Extract metadata
      try {
        MetadataExtractor::ExtractEXIF_ToImage(image_ptr->image_path_, *image_ptr);
        if (import_log) {
          import_log->MarkMetadataSuccess(image_ptr->image_id_);
        }
        // Update progress
        progress_ptr->metadata_done_.fetch_add(1);
      } catch (...) {
        progress_ptr->failed_.fetch_add(1);
      }

      if (job && job->on_progress_) {
        job->on_progress_(*progress_ptr);
      }

      if (progress_ptr->metadata_done_.load() + progress_ptr->failed_.load() >=
              progress_ptr->total_ ||
          (job && job->IsCancelled())) {
        // Job finished
        SetImportResult(job, progress_ptr->total_, progress_ptr->metadata_done_.load(),
                        progress_ptr->failed_.load());
      }
    });
  }

  if (!any_threadpool_submission) {
    // No valid submissions, immediately finish
    SetImportResult(job, progress_ptr->total_, progress_ptr->metadata_done_.load(),
                    progress_ptr->failed_.load());
  }
  return job;
}

void ImportServiceImpl::SyncImports(const ImportLogSnapshot& log_snapshot,
                                             const image_path_t&      dest) {
  if (!image_pool_manager_) {
    return;
  }

  for (const auto& entry : log_snapshot.metadata_failed_) {
    if (!entry.file_name_.empty()) {
      try {
        // fs_->Delete(dest / entry.file_name_);
        fs_service_.Write_NoSync<void>([&dest, &entry](FileSystem& fs) {
          fs.Delete(dest / image_path_t(entry.file_name_));
        });
      } catch (...) {
      }
    }

    auto& pool = image_pool_manager_->GetPool();
    pool.erase(entry.image_id_);

    fs_service_.Sync();
  }
}
};  // namespace puerhlab