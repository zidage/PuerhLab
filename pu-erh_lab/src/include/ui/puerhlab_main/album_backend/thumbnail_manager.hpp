#pragma once

#include <cstdint>
#include <unordered_map>

#include <QString>

#include "type/type.hpp"

namespace puerhlab::ui {

class AlbumBackend;

/// Manages thumbnail pin reference counts and async data-URL generation.
class ThumbnailManager {
 public:
  explicit ThumbnailManager(AlbumBackend& backend);

  void SetThumbnailVisible(sl_element_id_t elementId, image_id_t imageId, bool visible);
  void RequestThumbnail(sl_element_id_t elementId, image_id_t imageId);
  void UpdateThumbnailDataUrl(sl_element_id_t elementId, const QString& dataUrl);
  [[nodiscard]] bool IsThumbnailPinned(sl_element_id_t elementId) const;
  void ReleaseVisibleThumbnailPins();

 private:
  AlbumBackend& backend_;
  std::unordered_map<sl_element_id_t, uint32_t> thumbnail_pin_ref_counts_{};
};

}  // namespace puerhlab::ui
