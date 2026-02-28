#pragma once

#include <QVariantList>
#include <QVariantMap>

#include <ctime>
#include <optional>
#include <unordered_set>

#include "ui/puerhlab_main/album_backend/album_types.hpp"
#include "sleeve/sleeve_filter/filter_combo.hpp"

namespace puerhlab::ui {

class AlbumBackend;

/// Builds filter trees, evaluates them against the sleeve, and manages active
/// filter state.
class FilterEngine {
 public:
  explicit FilterEngine(AlbumBackend& backend);

  void ApplyFilters(int joinOpValue);
  void ClearFilters();
  void ReapplyCurrentFilters();

  [[nodiscard]] auto BuildFilterNode(FilterOp joinOp) const -> FilterBuildResult;
  [[nodiscard]] auto ParseFilterValue(FilterField field, const QString& text,
                                      QString& error) const
      -> std::optional<FilterValue>;
  [[nodiscard]] static auto ParseDate(const QString& text) -> std::optional<std::tm>;
  [[nodiscard]] bool IsImageInCurrentFolder(const AlbumItem& image) const;
  [[nodiscard]] auto FormatFilterInfo(int shown, int total) const -> QString;
  [[nodiscard]] auto MakeThumbMap(const AlbumItem& image, int index) const -> QVariantMap;

  void RebuildThumbnailView(
      const std::optional<std::unordered_set<sl_element_id_t>>& allowedElementIds);

  [[nodiscard]] auto active_filter_ids() const
      -> const std::optional<std::unordered_set<sl_element_id_t>>& {
    return active_filter_ids_;
  }
  void ResetActiveFilterIds() { active_filter_ids_.reset(); }

 private:
  AlbumBackend& backend_;
  std::optional<std::unordered_set<sl_element_id_t>> active_filter_ids_{};
  FilterOp last_join_op_ = FilterOp::AND;
};

}  // namespace puerhlab::ui
