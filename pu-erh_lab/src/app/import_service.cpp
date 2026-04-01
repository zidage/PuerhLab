//  Copyright 2026 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

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
  (void)options;
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
    const std::wstring file_name = image_path.filename().wstring();

    std::shared_ptr<SleeveElement> element = nullptr;
    try {
      element = fs_service_->Write_NoSync<std::shared_ptr<SleeveElement>>(
          [&dest, &file_name](FileSystem& fs) {
            return fs.Create(dest, file_name, ElementType::FILE);
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
    auto sleeve_file   = std::static_pointer_cast<SleeveFile>(element);

    // auto image_ptr   = image_pool_manager_->InsertEmpty();
    auto image_handler = image_pool_service_->CreateAndReturnPinnedEmpty();

    if (!image_handler) {
      progress_ptr->failed_.fetch_add(1);
      if (job && job->on_progress_) {
        job->on_progress_(*progress_ptr);
      }
      continue;
    }

    auto image_handler_ptr =
        std::make_shared<ImagePoolManager::PinnedImageHandle>(std::move(image_handler));
    auto  image_ptr = image_handler_ptr->Get();
    image_ptr->image_path_ = image_path;
    image_ptr->image_name_ = file_name;
    // TODO: Parse image type for future use

    // Link the image to the SleeveFile
    sleeve_file->SetImage(image_ptr);
      progress_ptr->placeholders_created_.fetch_add(1);
    if (import_log) {
      import_log->AddPlaceholder(image_ptr->image_id_, sleeve_file->element_id_,
                                 file_name, image_path);
    }

    any_threadpool_submission = true;

    // Submit the metadata extraction task to thread pool
    thread_pool_.Submit([image_handler_ptr, image_ptr, image_path, progress_ptr, job,
                         import_log]() {
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

      } catch (const MetadataExtractionError& e) {
        if (import_log) {
          import_log->MarkMetadataFailure(image_ptr->image_id_, e.code(), e.message());
        }
        progress_ptr->failed_.fetch_add(1);
      } catch (const std::exception& e) {
        if (import_log) {
          import_log->MarkMetadataFailure(image_ptr->image_id_,
                                          ImportErrorCode::METADATA_EXTRACTION_FAILED, e.what());
        }
        progress_ptr->failed_.fetch_add(1);
      } catch (...) {
        if (import_log) {
          import_log->MarkMetadataFailure(image_ptr->image_id_,
                                          ImportErrorCode::METADATA_EXTRACTION_FAILED);
        }
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
  if (!image_pool_service_) {
    return;
  }

  for (const auto& entry : log_snapshot.metadata_failed_) {
    if (!entry.file_name_.empty()) {
      try {
        // fs_->Delete(dest / entry.file_name_);
        fs_service_->Write_NoSync<void>(
            [&dest, &entry](FileSystem& fs) { fs.Delete(dest / image_path_t(entry.file_name_)); });
      } catch (...) {
      }
    }
  }
  image_pool_service_->SyncWithStorage();
  fs_service_->Sync();
}
};  // namespace puerhlab
