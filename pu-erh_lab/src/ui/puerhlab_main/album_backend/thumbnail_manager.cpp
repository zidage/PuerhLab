#include "ui/puerhlab_main/album_backend/thumbnail_manager.hpp"

#include "ui/puerhlab_main/album_backend/album_backend.hpp"
#include "ui/puerhlab_main/album_backend/path_utils.hpp"

#include <QCoreApplication>
#include <QImage>
#include <QMetaObject>
#include <QPointer>

#include <thread>

#include <opencv2/opencv.hpp>

#include "app/thumbnail_service.hpp"
#include "image/image_buffer.hpp"

namespace puerhlab::ui {

ThumbnailManager::ThumbnailManager(AlbumBackend& backend) : backend_(backend) {}

void ThumbnailManager::SetThumbnailVisible(sl_element_id_t elementId, image_id_t imageId, bool visible) {
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
  UpdateThumbnailDataUrl(elementId, QString());
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
      [self, service, elementId](std::shared_ptr<ThumbnailGuard> guard) {
        if (!guard || !guard->thumbnail_buffer_) {
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
                    self->thumb_.UpdateThumbnailDataUrl(elementId, dataUrl);
                  } else {
                    self->thumb_.UpdateThumbnailDataUrl(elementId, QString());
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

void ThumbnailManager::UpdateThumbnailDataUrl(sl_element_id_t elementId, const QString& dataUrl) {
  const auto it = backend_.index_by_element_id_.find(elementId);
  if (it == backend_.index_by_element_id_.end()) {
    return;
  }

  auto& item = backend_.all_images_[it->second];
  if (item.thumb_data_url == dataUrl) {
    return;
  }

  item.thumb_data_url = dataUrl;

  for (qsizetype i = 0; i < backend_.visible_thumbnails_.size(); ++i) {
    QVariantMap row = backend_.visible_thumbnails_.at(i).toMap();
    if (static_cast<sl_element_id_t>(row.value("elementId").toUInt()) != elementId) {
      continue;
    }
    row.insert("thumbUrl", dataUrl);
    backend_.visible_thumbnails_[i] = row;
    break;
  }

  emit backend_.ThumbnailUpdated(static_cast<uint>(elementId), dataUrl);
  emit backend_.thumbnailUpdated(static_cast<uint>(elementId), dataUrl);
}

bool ThumbnailManager::IsThumbnailPinned(sl_element_id_t elementId) const {
  const auto it = thumbnail_pin_ref_counts_.find(elementId);
  return it != thumbnail_pin_ref_counts_.end() && it->second > 0;
}

void ThumbnailManager::ReleaseVisibleThumbnailPins() {
  if (thumbnail_pin_ref_counts_.empty()) {
    return;
  }

  auto thumb_svc = backend_.project_handler_.thumbnail_service();

  for (const auto& [id, _] : thumbnail_pin_ref_counts_) {
    const auto index_it = backend_.index_by_element_id_.find(id);
    if (index_it != backend_.index_by_element_id_.end()) {
      backend_.all_images_[index_it->second].thumb_data_url.clear();
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
