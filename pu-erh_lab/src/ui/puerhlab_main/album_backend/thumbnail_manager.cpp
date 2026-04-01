//  Copyright 2026 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#include "ui/puerhlab_main/album_backend/thumbnail_manager.hpp"

#include "ui/puerhlab_main/album_backend/album_backend.hpp"
#include "ui/puerhlab_main/album_backend/path_utils.hpp"

#include <QCoreApplication>
#include <QImage>
#include <QMetaObject>
#include <QPointer>

#include <filesystem>
#include <system_error>
#include <thread>

#include <opencv2/opencv.hpp>

#include "app/thumbnail_service.hpp"
#include "image/image.hpp"
#include "image/image_buffer.hpp"

namespace puerhlab::ui {

auto ThumbnailManager::PathExists(const std::filesystem::path& path) -> bool {
  if (path.empty()) {
    return false;
  }

  std::error_code exists_error;
  return std::filesystem::exists(path, exists_error) && !exists_error;
}

auto ThumbnailManager::ResolveThumbnailSourcePath(sl_element_id_t elementId,
                                                  image_id_t imageId) const
    -> std::filesystem::path {
  if (const auto* item = backend_.FindAlbumItem(elementId);
      item != nullptr && !item->file_path_.empty()) {
    return item->file_path_;
  }

  auto proj = backend_.project_handler_.project();
  if (!proj) {
    return {};
  }

  try {
    return proj->GetImagePoolService()->Read<std::filesystem::path>(
        imageId, [](const std::shared_ptr<Image>& image) -> std::filesystem::path {
          if (!image) {
            return {};
          }
          return image->image_path_;
        });
  } catch (...) {
    return {};
  }
}

ThumbnailManager::ThumbnailManager(AlbumBackend& backend) : backend_(backend) {}

void ThumbnailManager::SetThumbnailVisible(sl_element_id_t elementId, image_id_t imageId,
                                          bool visible) {
  if (elementId == 0 || imageId == 0) {
    return;
  }

  auto thumb_svc = backend_.project_handler_.thumbnail_service();

  if (visible) {
    if (!thumb_svc) {
      return;
    }
    auto& ref = thumbnail_pin_ref_counts_[elementId];
    ref++;
    if (ref == 1) {
      const auto* item = backend_.FindAlbumItem(elementId);
      const bool known_missing = item != nullptr && item->thumb_missing_source;
      const auto source_path = ResolveThumbnailSourcePath(elementId, imageId);
      if (known_missing && !source_path.empty() && !PathExists(source_path)) {
        UpdateThumbnailState(elementId, QString(), false, true);
        return;
      }
      RequestThumbnail(elementId, imageId);
    }
    return;
  }

  const auto it = thumbnail_pin_ref_counts_.find(elementId);
  if (it == thumbnail_pin_ref_counts_.end()) {
    return;
  }

  if (it->second > 1) {
    it->second--;
    return;
  }

  thumbnail_pin_ref_counts_.erase(it);
  const auto* item = backend_.FindAlbumItem(elementId);
  const bool  missing_source = item != nullptr && item->thumb_missing_source;
  UpdateThumbnailState(elementId, QString(), false, missing_source);
  if (thumb_svc) {
    try {
      thumb_svc->ReleaseThumbnail(elementId);
    } catch (...) {
    }
  }
}

void ThumbnailManager::RequestThumbnail(sl_element_id_t elementId, image_id_t imageId) {
  auto thumb_svc = backend_.project_handler_.thumbnail_service();
  if (!thumb_svc) {
    return;
  }

  UpdateThumbnailState(elementId, QString(), true, false);

  auto                   service = thumb_svc;
  QPointer<AlbumBackend> self(&backend_);

  CallbackDispatcher dispatcher = [](std::function<void()> fn) {
    auto* app = QCoreApplication::instance();
    if (!app) {
      fn();
      return;
    }
    QMetaObject::invokeMethod(app, std::move(fn), Qt::QueuedConnection);
  };

  service->GetThumbnail(
      elementId, imageId,
      [self, service, elementId, imageId](std::shared_ptr<ThumbnailGuard> guard) {
        if (!guard || !guard->thumbnail_buffer_) {
          if (self) {
            const auto source_path = self->thumb_.ResolveThumbnailSourcePath(elementId, imageId);
            const bool missing_source = !source_path.empty() && !PathExists(source_path);
            self->thumb_.UpdateThumbnailState(elementId, QString(), false, missing_source);
          }
          if (self && !self->thumb_.IsThumbnailPinned(elementId) && service) {
            try {
              service->ReleaseThumbnail(elementId);
            } catch (...) {
            }
          }
          return;
        }
        if (!self) {
          try {
            if (service) {
              service->ReleaseThumbnail(elementId);
            }
          } catch (...) {
          }
          return;
        }

        std::thread([self, service, elementId, guard = std::move(guard)]() mutable {
          QString dataUrl;
          try {
            auto* buffer = guard->thumbnail_buffer_.get();
            if (buffer) {
              if (!buffer->cpu_data_valid_ && buffer->gpu_data_valid_) {
                buffer->SyncToCPU();
              }
              if (buffer->cpu_data_valid_) {
                QImage image = album_util::MatRgba32fToQImageCopy(buffer->GetCPUData());
                if (!image.isNull()) {
                  QImage scaled = image.scaled(220, 160, Qt::KeepAspectRatio,
                                               Qt::SmoothTransformation);
                  dataUrl = album_util::DataUrlFromImage(scaled);
                }
              }
            }
          } catch (...) {
          }

          if (self) {
            QMetaObject::invokeMethod(
                self,
                [self, service, elementId, dataUrl]() {
                  if (!self) {
                    return;
                  }
                  const bool pinned = self->thumb_.IsThumbnailPinned(elementId);
                  if (pinned) {
                    self->thumb_.UpdateThumbnailState(elementId, dataUrl, false, false);
                  } else {
                    self->thumb_.UpdateThumbnailState(elementId, QString(), false, false);
                  }
                  if (!pinned && service) {
                    try {
                      service->ReleaseThumbnail(elementId);
                    } catch (...) {
                    }
                  }
                },
                Qt::QueuedConnection);
          }
        }).detach();
      },
      true, dispatcher);
}

void ThumbnailManager::UpdateThumbnailState(sl_element_id_t elementId, const QString& dataUrl,
                                            bool loading, bool missingSource) {
  auto* item = backend_.FindAlbumItem(elementId);
  if (!item) {
    return;
  }

  if (item->thumb_data_url == dataUrl && item->thumb_loading == loading &&
      item->thumb_missing_source == missingSource) {
    return;
  }

  item->thumb_data_url        = dataUrl;
  item->thumb_loading         = loading;
  item->thumb_missing_source  = missingSource;

  for (qsizetype i = 0; i < backend_.view_state_.visible_thumbnails_.size(); ++i) {
    QVariantMap row = backend_.view_state_.visible_thumbnails_.at(i).toMap();
    if (static_cast<sl_element_id_t>(row.value("elementId").toUInt()) != elementId) {
      continue;
    }
    row.insert("thumbUrl", dataUrl);
    row.insert("thumbLoading", loading);
    row.insert("thumbMissingSource", missingSource);
    backend_.view_state_.visible_thumbnails_[i] = row;
    break;
  }

  emit backend_.ThumbnailUpdated(static_cast<uint>(elementId), dataUrl, loading, missingSource);
  emit backend_.thumbnailUpdated(static_cast<uint>(elementId), dataUrl, loading, missingSource);
}

bool ThumbnailManager::IsThumbnailPinned(sl_element_id_t elementId) const {
  const auto it = thumbnail_pin_ref_counts_.find(elementId);
  return it != thumbnail_pin_ref_counts_.end() && it->second > 0;
}

void ThumbnailManager::RemoveThumbnailState(sl_element_id_t elementId, image_id_t imageId) {
  (void)imageId;
  if (elementId == 0) {
    return;
  }

  thumbnail_pin_ref_counts_.erase(elementId);
  UpdateThumbnailState(elementId, QString(), false, false);

  auto thumb_svc = backend_.project_handler_.thumbnail_service();
  if (!thumb_svc) {
    return;
  }

  try {
    thumb_svc->InvalidateThumbnail(elementId);
  } catch (...) {
  }
  try {
    thumb_svc->ReleaseThumbnail(elementId);
  } catch (...) {
  }
}

void ThumbnailManager::ReleaseVisibleThumbnailPins() {
  if (thumbnail_pin_ref_counts_.empty()) {
    return;
  }

  auto thumb_svc = backend_.project_handler_.thumbnail_service();

  for (const auto& [id, _] : thumbnail_pin_ref_counts_) {
    auto* item = backend_.FindAlbumItem(id);
    if (item) {
      item->thumb_data_url.clear();
      item->thumb_loading = false;
    }
    if (thumb_svc) {
      try {
        thumb_svc->ReleaseThumbnail(id);
      } catch (...) {
      }
    }
  }
  thumbnail_pin_ref_counts_.clear();
}

}  // namespace puerhlab::ui
