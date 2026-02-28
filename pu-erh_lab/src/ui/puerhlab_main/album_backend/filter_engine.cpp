#include "ui/puerhlab_main/album_backend/filter_engine.hpp"

#include "ui/puerhlab_main/album_backend/album_backend.hpp"
#include "ui/puerhlab_main/album_backend/path_utils.hpp"

#include <QDate>
#include <QStringList>

namespace puerhlab::ui {

FilterEngine::FilterEngine(AlbumBackend& backend) : backend_(backend) {}

void FilterEngine::ApplyFilters(int joinOpValue) {
  auto parsedJoin = static_cast<FilterOp>(joinOpValue);
  if (parsedJoin != FilterOp::AND && parsedJoin != FilterOp::OR) {
    parsedJoin = FilterOp::AND;
  }

  last_join_op_ = parsedJoin;

  const FilterBuildResult result = BuildFilterNode(parsedJoin);
  if (!result.error.isEmpty()) {
    if (backend_.validation_error_ != result.error) {
      backend_.validation_error_ = result.error;
      emit backend_.ValidationErrorChanged();
    }
    return;
  }

  if (!backend_.validation_error_.isEmpty()) {
    backend_.validation_error_.clear();
    emit backend_.ValidationErrorChanged();
  }

  QString nextSql;
  if (result.node.has_value()) {
    nextSql = QString::fromStdWString(FilterSQLCompiler::Compile(result.node.value()));
  }
  if (backend_.sql_preview_ != nextSql) {
    backend_.sql_preview_ = nextSql;
    emit backend_.SqlPreviewChanged();
  }

  if (!result.node.has_value()) {
    active_filter_ids_.reset();
    RebuildThumbnailView(std::nullopt);
    return;
  }

  auto* fsvc = backend_.project_handler_.filter_service();
  if (!fsvc) {
    active_filter_ids_ = std::unordered_set<sl_element_id_t>{};
    RebuildThumbnailView(active_filter_ids_);
    return;
  }

  try {
    const auto filterId = fsvc->CreateFilterCombo(result.node.value());
    const auto idsOpt =
        fsvc->ApplyFilterOn(filterId, backend_.folder_ctrl_.current_folder_id());
    fsvc->RemoveFilterCombo(filterId);

    std::unordered_set<sl_element_id_t> nextIds;
    if (idsOpt.has_value()) {
      nextIds.reserve(idsOpt->size() * 2 + 1);
      for (const auto id : idsOpt.value()) {
        nextIds.insert(id);
      }
    }

    active_filter_ids_ = std::move(nextIds);
    RebuildThumbnailView(active_filter_ids_);
  } catch (const std::exception& e) {
    const QString error =
        QString("Filter execution failed: %1").arg(QString::fromUtf8(e.what()));
    if (backend_.validation_error_ != error) {
      backend_.validation_error_ = error;
      emit backend_.ValidationErrorChanged();
    }
    active_filter_ids_ = std::unordered_set<sl_element_id_t>{};
    RebuildThumbnailView(active_filter_ids_);
  }
}

void FilterEngine::ClearFilters() {
  backend_.rule_model_.ClearAndReset();
  last_join_op_ = FilterOp::AND;

  if (!backend_.validation_error_.isEmpty()) {
    backend_.validation_error_.clear();
    emit backend_.ValidationErrorChanged();
  }
  if (!backend_.sql_preview_.isEmpty()) {
    backend_.sql_preview_.clear();
    emit backend_.SqlPreviewChanged();
  }

  active_filter_ids_.reset();
  RebuildThumbnailView(std::nullopt);
}

void FilterEngine::ReapplyCurrentFilters() {
  ApplyFilters(static_cast<int>(last_join_op_));
  if (!backend_.validation_error_.isEmpty()) {
    RebuildThumbnailView(active_filter_ids_);
  }
}

auto FilterEngine::BuildFilterNode(FilterOp joinOp) const -> FilterBuildResult {
  std::optional<FilterNode> rules_node;
  std::vector<FilterNode>   conditions;

  for (const auto& rule : backend_.rule_model_.Rules()) {
    if (rule.value.trimmed().isEmpty()) {
      continue;
    }

    QString error;
    const auto value_opt = ParseFilterValue(rule.field, rule.value, error);
    if (!value_opt.has_value()) {
      return FilterBuildResult{.node = std::nullopt, .error = error};
    }

    FieldCondition condition{
        .field_        = rule.field,
        .op_           = rule.op,
        .value_        = value_opt.value(),
        .second_value_ = std::nullopt,
    };

    if (rule.op == CompareOp::BETWEEN) {
      if (rule.value2.trimmed().isEmpty()) {
        return FilterBuildResult{.node = std::nullopt, .error = "BETWEEN requires two values."};
      }
      const auto second_opt = ParseFilterValue(rule.field, rule.value2, error);
      if (!second_opt.has_value()) {
        return FilterBuildResult{.node = std::nullopt, .error = error};
      }
      condition.second_value_ = second_opt.value();
    }

    conditions.push_back(FilterNode{
        FilterNode::Type::Condition, {}, {}, std::move(condition), std::nullopt});
  }

  if (!conditions.empty()) {
    if (conditions.size() == 1) {
      rules_node = conditions.front();
    } else {
      rules_node = FilterNode{
          FilterNode::Type::Logical, joinOp, std::move(conditions), {}, std::nullopt};
    }
  }

  if (rules_node.has_value()) {
    return FilterBuildResult{.node = rules_node, .error = QString()};
  }
  return FilterBuildResult{.node = std::nullopt, .error = QString()};
}

auto FilterEngine::ParseFilterValue(FilterField field, const QString& text,
                                    QString& error) const -> std::optional<FilterValue> {
  const QString trimmed = text.trimmed();
  const auto    kind    = FilterRuleModel::KindForField(field);

  if (kind == FilterValueKind::String) {
    return FilterValue{trimmed.toStdWString()};
  }

  if (kind == FilterValueKind::Int64) {
    bool       ok = false;
    const auto v  = trimmed.toLongLong(&ok);
    if (!ok) {
      error = "Expected an integer value.";
      return std::nullopt;
    }
    return FilterValue{static_cast<int64_t>(v)};
  }

  if (kind == FilterValueKind::Double) {
    bool       ok = false;
    const auto v  = trimmed.toDouble(&ok);
    if (!ok) {
      error = "Expected a numeric value.";
      return std::nullopt;
    }
    return FilterValue{v};
  }

  const auto date_opt = ParseDate(trimmed);
  if (!date_opt.has_value()) {
    error = "Expected a date in YYYY-MM-DD format.";
    return std::nullopt;
  }
  return FilterValue{date_opt.value()};
}

auto FilterEngine::ParseDate(const QString& text) -> std::optional<std::tm> {
  const QStringList parts = text.trimmed().split('-', Qt::SkipEmptyParts);
  if (parts.size() != 3) {
    return std::nullopt;
  }

  bool      ok_year = false;
  bool      ok_mon  = false;
  bool      ok_day  = false;
  const int year    = parts[0].toInt(&ok_year);
  const int month   = parts[1].toInt(&ok_mon);
  const int day     = parts[2].toInt(&ok_day);
  if (!ok_year || !ok_mon || !ok_day) {
    return std::nullopt;
  }

  const QDate date(year, month, day);
  if (!date.isValid()) {
    return std::nullopt;
  }

  std::tm tm{};
  tm.tm_year = year - 1900;
  tm.tm_mon  = month - 1;
  tm.tm_mday = day;
  return tm;
}

bool FilterEngine::IsImageInCurrentFolder(const AlbumItem& image) const {
  return image.parent_folder_id == backend_.folder_ctrl_.current_folder_id();
}

auto FilterEngine::FormatFilterInfo(int shown, int total) const -> QString {
  if (total <= 0) {
    return "No images loaded.";
  }
  if (shown == total) {
    return QString("Showing %1 images").arg(total);
  }
  return QString("Showing %1 of %2").arg(shown).arg(total);
}

auto FilterEngine::MakeThumbMap(const AlbumItem& image, int index) const -> QVariantMap {
  const QString aperture =
      image.aperture > 0.0 ? QString::number(image.aperture, 'f', 1) : "--";
  const QString focal =
      image.focal_length > 0.0 ? QString::number(image.focal_length, 'f', 0) : "--";

  return QVariantMap{
      {"elementId", static_cast<uint>(image.element_id)},
      {"imageId", static_cast<uint>(image.image_id)},
      {"fileName", image.file_name.isEmpty() ? "(unnamed)" : image.file_name},
      {"cameraModel", image.camera_model.isEmpty() ? "Unknown" : image.camera_model},
      {"extension", image.extension.isEmpty() ? "--" : image.extension},
      {"iso", image.iso},
      {"aperture", aperture},
      {"focalLength", focal},
      {"captureDate",
       image.capture_date.isValid() ? image.capture_date.toString("yyyy-MM-dd") : "--"},
      {"rating", image.rating},
      {"tags", image.tags},
      {"accent", image.accent.isEmpty() ? album_util::AccentForIndex(static_cast<size_t>(index))
                                        : image.accent},
      {"thumbUrl", image.thumb_data_url},
  };
}

void FilterEngine::RebuildThumbnailView(
    const std::optional<std::unordered_set<sl_element_id_t>>& allowedElementIds) {
  backend_.thumb_.ReleaseVisibleThumbnailPins();

  QVariantList next;
  next.reserve(static_cast<qsizetype>(backend_.all_images_.size()));

  int index = 0;
  for (const AlbumItem& image : backend_.all_images_) {
    if (!IsImageInCurrentFolder(image)) {
      continue;
    }
    if (allowedElementIds.has_value() && !allowedElementIds->contains(image.element_id)) {
      continue;
    }
    next.push_back(MakeThumbMap(image, index++));
  }

  backend_.visible_thumbnails_ = std::move(next);
  emit backend_.ThumbnailsChanged();
  emit backend_.thumbnailsChanged();
  emit backend_.CountsChanged();
}

}  // namespace puerhlab::ui
