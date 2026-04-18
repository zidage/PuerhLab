//  Copyright 2026 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#include "ui/alcedo_main/editor_dialog/widgets/history_cards.hpp"

#include <QFile>
#include <QFileInfo>
#include <QFontMetrics>
#include <QHBoxLayout>
#include <QPainter>
#include <QPen>
#include <QPixmap>
#include <QStyle>
#include <QSvgRenderer>
#include <QVBoxLayout>

#include <algorithm>
#include <array>
#include <cmath>
#include <json.hpp>
#include <optional>
#include <string>
#include <utility>

#include "ui/alcedo_main/app_theme.hpp"
#include "ui/alcedo_main/editor_dialog/modules/curve.hpp"

namespace alcedo::ui {

namespace {

constexpr int   kLaneWidth      = 12;
constexpr qreal kLineWidth      = 1.5;
constexpr qreal kDotRadius      = 3.0;
constexpr qreal kDotGap         = 2.0;
constexpr qreal kRingRadius     = 5.2;
constexpr int   kIconTileSize   = 24;
constexpr int   kIconGlyphSize  = 12;

struct TxCardSummary {
  QString value;
  QString detail;
};

auto FormatNumber(double v) -> QString {
  if (std::fabs(v - std::round(v)) < 1e-4 && std::fabs(v) < 1e9) {
    return QString::number(static_cast<long long>(std::llround(v)));
  }
  QString s = QString::number(v, 'f', 2);
  while (s.endsWith(QLatin1Char('0'))) {
    s.chop(1);
  }
  if (s.endsWith(QLatin1Char('.'))) {
    s.chop(1);
  }
  return s;
}

auto FormatSigned(double v) -> QString {
  const QString body = FormatNumber(std::fabs(v));
  if (v > 0.0) {
    return QStringLiteral("+") + body;
  }
  if (v < 0.0) {
    return QStringLiteral("-") + body;
  }
  return body;
}

auto WithUnit(const QString& value, const QString& unit, bool space_before_unit = false)
    -> QString {
  if (value.isEmpty() || unit.isEmpty()) {
    return value;
  }
  return space_before_unit ? value + QStringLiteral(" ") + unit : value + unit;
}

auto FormatBool(bool value) -> QString {
  return value ? QStringLiteral("On") : QStringLiteral("Off");
}

auto FormatJsonScalar(const nlohmann::json& v, bool signed_if_numeric) -> QString {
  if (v.is_number_float()) {
    const double d = v.get<double>();
    return signed_if_numeric ? FormatSigned(d) : FormatNumber(d);
  }
  if (v.is_number_integer()) {
    const double d = static_cast<double>(v.get<long long>());
    return signed_if_numeric ? FormatSigned(d) : FormatNumber(d);
  }
  if (v.is_number_unsigned()) {
    return FormatNumber(static_cast<double>(v.get<unsigned long long>()));
  }
  if (v.is_boolean()) {
    return v.get<bool>() ? QStringLiteral("on") : QStringLiteral("off");
  }
  if (v.is_string()) {
    return QString::fromStdString(v.get<std::string>());
  }
  if (v.is_null()) {
    return QStringLiteral("--");
  }
  return QString::fromStdString(v.dump());
}

auto JsonAtPath(const nlohmann::json& root, std::initializer_list<const char*> path)
    -> const nlohmann::json* {
  const nlohmann::json* node = &root;
  for (const char* key : path) {
    if (!node->is_object()) {
      return nullptr;
    }
    const auto it = node->find(key);
    if (it == node->end()) {
      return nullptr;
    }
    node = &(*it);
  }
  return node;
}

auto JsonNumberAtPath(const nlohmann::json& root, std::initializer_list<const char*> path)
    -> std::optional<double> {
  const auto* node = JsonAtPath(root, path);
  if (!node) {
    return std::nullopt;
  }
  if (node->is_number_float()) {
    return node->get<double>();
  }
  if (node->is_number_integer()) {
    return static_cast<double>(node->get<long long>());
  }
  if (node->is_number_unsigned()) {
    return static_cast<double>(node->get<unsigned long long>());
  }
  return std::nullopt;
}

auto JsonBoolAtPath(const nlohmann::json& root, std::initializer_list<const char*> path)
    -> std::optional<bool> {
  const auto* node = JsonAtPath(root, path);
  if (!node || !node->is_boolean()) {
    return std::nullopt;
  }
  return node->get<bool>();
}

auto JsonStringAtPath(const nlohmann::json& root, std::initializer_list<const char*> path)
    -> std::optional<QString> {
  const auto* node = JsonAtPath(root, path);
  if (!node || !node->is_string()) {
    return std::nullopt;
  }
  return QString::fromStdString(node->get<std::string>());
}

auto JsonArrayNumber(const nlohmann::json* array, size_t index) -> std::optional<double> {
  if (!array || !array->is_array() || index >= array->size()) {
    return std::nullopt;
  }
  const auto& node = (*array)[index];
  if (node.is_number_float()) {
    return node.get<double>();
  }
  if (node.is_number_integer()) {
    return static_cast<double>(node.get<long long>());
  }
  if (node.is_number_unsigned()) {
    return static_cast<double>(node.get<unsigned long long>());
  }
  return std::nullopt;
}

auto PrettyToken(QString raw) -> QString {
  if (raw.isEmpty()) {
    return raw;
  }

  static const std::pair<const char*, const char*> kMap[] = {
      {"as_shot", "As Shot"},
      {"custom", "Custom"},
      {"REC709", "Rec.709"},
      {"DISPLAY_P3", "Display P3"},
      {"SRGB", "sRGB"},
      {"ACESCG", "ACEScg"},
      {"GAMMA_2_2", "Gamma 2.2"},
      {"GAMMA_2_4", "Gamma 2.4"},
      {"ST2084", "PQ"},
      {"OPEN_DRT", "OpenDRT"},
      {"HLG", "HLG"},
  };
  for (const auto& [from, to] : kMap) {
    if (raw.compare(QLatin1String(from), Qt::CaseInsensitive) == 0) {
      return QString::fromLatin1(to);
    }
  }

  raw.replace(QLatin1Char('_'), QLatin1Char(' '));
  raw.replace(QLatin1Char('-'), QLatin1Char(' '));
  const QStringList parts = raw.split(QLatin1Char(' '), Qt::SkipEmptyParts);
  QStringList normalized;
  normalized.reserve(parts.size());
  for (const QString& part : parts) {
    const QString lower = part.toLower();
    if (part == part.toUpper() && part.size() <= 4) {
      normalized.push_back(part);
    } else {
      normalized.push_back(lower.left(1).toUpper() + lower.mid(1));
    }
  }
  return normalized.join(QLatin1Char(' '));
}

auto ShortKey(const std::string& key) -> QString {
  static const std::pair<const char*, const char*> kMap[] = {
      {"exposure_val", "Exp"},      {"exposure", "Exp"},        {"contrast", "Con"},
      {"white", "Wht"},             {"black", "Blk"},           {"highlights", "Hi"},
      {"shadows", "Sh"},            {"saturation", "Sat"},      {"vibrance", "Vib"},
      {"temperature", "Temp"},      {"tint", "Tint"},           {"clarity", "Clar"},
      {"sharpen", "Sharp"},         {"strength", "Strength"},   {"amount", "Amt"},
      {"angle", "Ang"},             {"angle_degrees", "Angle"}, {"hue", "Hue"},
      {"luminance", "Lum"},         {"opacity", "Opa"},         {"enabled", "Enabled"},
      {"radius", "Rad"},            {"threshold", "Thr"},       {"peak_luminance", "Peak"},
      {"encoding_space", "Space"},  {"encoding_eotf", "EOTF"},  {"look_preset", "Look"},
      {"method", "Method"},         {"highlights_reconstruct", "Recover"},
      {"mode", "Mode"},             {"cct", "CCT"},
  };
  for (const auto& [from, to] : kMap) {
    if (key == from) {
      return QString::fromLatin1(to);
    }
  }

  QString out = QString::fromStdString(key);
  if (out.size() > 12) {
    out = out.left(11) + QChar(0x2026);
  }
  return out;
}

auto FirstMeaningfulDeltaNode(const nlohmann::json& node, const nlohmann::json* old_node,
                              bool skip_enabled) -> QString {
  if (!node.is_object()) {
    return {};
  }

  for (auto it = node.begin(); it != node.end(); ++it) {
    const std::string& key = it.key();
    if (skip_enabled && key == "enabled") {
      continue;
    }

    const auto& val = it.value();
    const nlohmann::json* old_val = nullptr;
    if (old_node && old_node->is_object()) {
      const auto old_it = old_node->find(key);
      if (old_it != old_node->end()) {
        old_val = &(*old_it);
      }
    }

    if (val.is_object()) {
      if (const QString nested = FirstMeaningfulDeltaNode(val, old_val, skip_enabled);
          !nested.isEmpty()) {
        return nested;
      }
      continue;
    }
    if (val.is_array()) {
      continue;
    }

    const bool is_numeric =
        val.is_number_float() || val.is_number_integer() || val.is_number_unsigned();
    if (old_val) {
      if (*old_val == val) {
        continue;
      }
      return QStringLiteral("%1 %2 \u2192 %3")
          .arg(ShortKey(key), FormatJsonScalar(*old_val, is_numeric),
               FormatJsonScalar(val, is_numeric));
    }
    return QStringLiteral("%1 %2").arg(ShortKey(key), FormatJsonScalar(val, is_numeric));
  }

  return {};
}

auto FirstMeaningfulDelta(const EditTransaction& tx) -> QString {
  const auto& params = tx.GetOperatorParams();
  const auto prev = tx.GetLastOperatorParams();
  if (!params.is_object() || params.empty()) {
    return {};
  }

  QString picked = FirstMeaningfulDeltaNode(params, prev ? &(*prev) : nullptr,
                                            /*skip_enabled*/ true);
  if (picked.isEmpty()) {
    picked = FirstMeaningfulDeltaNode(params, prev ? &(*prev) : nullptr,
                                      /*skip_enabled*/ false);
  }
  return picked;
}

auto MakeActionSummary(TransactionType type, const QString& detail) -> TxCardSummary {
  switch (type) {
    case TransactionType::_ADD:
      return {QStringLiteral("Applied"),
              detail.isEmpty() ? QStringLiteral("Operator added") : detail};
    case TransactionType::_DELETE:
      return {QStringLiteral("Removed"),
              detail.isEmpty() ? QStringLiteral("Operator removed") : detail};
    case TransactionType::_EDIT:
      return {QStringLiteral("Updated"),
              detail.isEmpty() ? QStringLiteral("Operator updated") : detail};
  }
  return {QStringLiteral("Updated"), detail};
}

auto MakeSignedNumberSummary(std::optional<double> old_v, std::optional<double> new_v,
                             const QString& unit = QString(),
                             bool space_before_unit = false) -> TxCardSummary {
  if (!new_v.has_value()) {
    return {};
  }
  const QString value = WithUnit(FormatSigned(*new_v), unit, space_before_unit);
  const QString detail =
      old_v.has_value()
          ? QStringLiteral("%1 \u2192 %2")
                .arg(WithUnit(FormatSigned(*old_v), unit, space_before_unit), value)
          : QStringLiteral("Set to %1").arg(value);
  return {value, detail};
}

auto MakeUnsignedNumberSummary(std::optional<double> old_v, std::optional<double> new_v,
                               const QString& unit = QString(),
                               bool space_before_unit = false) -> TxCardSummary {
  if (!new_v.has_value()) {
    return {};
  }
  const QString value = WithUnit(FormatNumber(*new_v), unit, space_before_unit);
  const QString detail =
      old_v.has_value()
          ? QStringLiteral("%1 \u2192 %2")
                .arg(WithUnit(FormatNumber(*old_v), unit, space_before_unit), value)
          : QStringLiteral("Set to %1").arg(value);
  return {value, detail};
}

auto LensName(const nlohmann::json& params) -> QString {
  const QString maker =
      JsonStringAtPath(params, {"lens_calib", "lens_maker"}).value_or(QString());
  const QString model =
      JsonStringAtPath(params, {"lens_calib", "lens_model"}).value_or(QString());
  if (!maker.isEmpty() && !model.isEmpty()) {
    return maker + QStringLiteral(" ") + model;
  }
  if (!model.isEmpty()) {
    return model;
  }
  if (!maker.isEmpty()) {
    return maker;
  }
  return {};
}

auto ColorTempLabel(const nlohmann::json& params) -> QString {
  const QString mode =
      PrettyToken(JsonStringAtPath(params, {"color_temp", "mode"}).value_or(QString()));
  if (mode.compare(QStringLiteral("Custom"), Qt::CaseInsensitive) == 0) {
    const auto cct = JsonNumberAtPath(params, {"color_temp", "cct"});
    const auto tint = JsonNumberAtPath(params, {"color_temp", "tint"});
    if (cct.has_value()) {
      QString label = WithUnit(FormatNumber(*cct), QStringLiteral("K"));
      if (tint.has_value() && std::fabs(*tint) > 1e-4) {
        label += QStringLiteral(" / ") + FormatSigned(*tint);
      }
      return label;
    }
  }
  return mode.isEmpty() ? QStringLiteral("Color Temp") : mode;
}

auto OdtTargetLabel(const nlohmann::json& params) -> QString {
  const QString space =
      PrettyToken(JsonStringAtPath(params, {"odt", "encoding_space"}).value_or(QString()));
  const QString eotf =
      PrettyToken(JsonStringAtPath(params, {"odt", "encoding_eotf"}).value_or(QString()));
  if (!space.isEmpty() && !eotf.isEmpty()) {
    return space + QStringLiteral(" / ") + eotf;
  }
  if (!space.isEmpty()) {
    return space;
  }
  if (!eotf.isEmpty()) {
    return eotf;
  }
  return QStringLiteral("Output");
}

auto CropAreaPercent(const nlohmann::json& params) -> std::optional<double> {
  const auto w = JsonNumberAtPath(params, {"crop_rotate", "crop_rect", "w"});
  const auto h = JsonNumberAtPath(params, {"crop_rotate", "crop_rect", "h"});
  if (!w.has_value() || !h.has_value()) {
    return std::nullopt;
  }
  return std::clamp(*w * *h * 100.0, 0.0, 100.0);
}

auto FormatCropAspect(const nlohmann::json& params) -> QString {
  const QString preset =
      JsonStringAtPath(params, {"crop_rotate", "aspect_ratio_preset"}).value_or(QString());
  if (!preset.isEmpty() && preset.compare(QStringLiteral("free"), Qt::CaseInsensitive) != 0) {
    return PrettyToken(preset);
  }

  const auto w = JsonNumberAtPath(params, {"crop_rotate", "aspect_ratio", "width"});
  const auto h = JsonNumberAtPath(params, {"crop_rotate", "aspect_ratio", "height"});
  if (!w.has_value() || !h.has_value() || *w <= 0.0 || *h <= 0.0) {
    return QStringLiteral("Free");
  }
  return QStringLiteral("%1:%2").arg(FormatNumber(*w), FormatNumber(*h));
}

auto RenderSvgPixmap(const QString& resource_path, const QColor& color, const QSize& size,
                     qreal device_pixel_ratio) -> QPixmap {
  QFile svg_file(resource_path);
  if (!svg_file.open(QIODevice::ReadOnly)) {
    return {};
  }

  QByteArray svg_data = svg_file.readAll();
  svg_data.replace("currentColor", color.name(QColor::HexRgb).toUtf8());

  QSvgRenderer renderer(svg_data);
  if (!renderer.isValid()) {
    return {};
  }

  const qreal scale = std::max<qreal>(1.0, device_pixel_ratio);
  const QSize physical_size(std::max(1, qRound(size.width() * scale)),
                            std::max(1, qRound(size.height() * scale)));

  QPixmap pixmap(physical_size);
  pixmap.fill(Qt::transparent);
  pixmap.setDevicePixelRatio(scale);

  QPainter painter(&pixmap);
  painter.setRenderHint(QPainter::Antialiasing, true);
  renderer.render(&painter, QRectF(QPointF(0.0, 0.0), QSizeF(size)));
  return pixmap;
}

auto MakeHistoryIconTile(OperatorType op, QWidget* parent) -> QFrame* {
  auto* tile = new QFrame(parent);
  tile->setObjectName(QStringLiteral("HistoryTxIconTile"));
  tile->setFixedSize(kIconTileSize, kIconTileSize);
  tile->setAttribute(Qt::WA_TransparentForMouseEvents, true);

  auto* layout = new QHBoxLayout(tile);
  layout->setContentsMargins(0, 0, 0, 0);
  layout->setSpacing(0);

  auto* icon = new QLabel(tile);
  icon->setAlignment(Qt::AlignCenter);
  icon->setAttribute(Qt::WA_TransparentForMouseEvents, true);
  const QColor icon_color = QColor(232, 236, 243);
  icon->setPixmap(RenderSvgPixmap(
      [&]() -> QString {
        switch (op) {
          case OperatorType::RAW_DECODE:
            return QStringLiteral(":/history_icons/scan-search.svg");
          case OperatorType::RESIZE:
            return QStringLiteral(":/history_icons/scaling.svg");
          case OperatorType::CROP_ROTATE:
            return QStringLiteral(":/history_icons/crop.svg");
          case OperatorType::EXPOSURE:
            return QStringLiteral(":/history_icons/sun-medium.svg");
          case OperatorType::CONTRAST:
            return QStringLiteral(":/history_icons/contrast.svg");
          case OperatorType::WHITE:
            return QStringLiteral(":/history_icons/sun.svg");
          case OperatorType::BLACK:
            return QStringLiteral(":/history_icons/moon.svg");
          case OperatorType::SHADOWS:
            return QStringLiteral(":/history_icons/square-split-horizontal.svg");
          case OperatorType::HIGHLIGHTS:
            return QStringLiteral(":/history_icons/sparkles.svg");
          case OperatorType::CURVE:
            return QStringLiteral(":/history_icons/chart-spline.svg");
          case OperatorType::HLS:
            return QStringLiteral(":/history_icons/swatch-book.svg");
          case OperatorType::SATURATION:
            return QStringLiteral(":/history_icons/droplets.svg");
          case OperatorType::TINT:
            return QStringLiteral(":/history_icons/pipette.svg");
          case OperatorType::VIBRANCE:
            return QStringLiteral(":/history_icons/sparkles.svg");
          case OperatorType::CST:
            return QStringLiteral(":/history_icons/arrow-right-left.svg");
          case OperatorType::TO_WS:
            return QStringLiteral(":/history_icons/workflow.svg");
          case OperatorType::TO_OUTPUT:
            return QStringLiteral(":/history_icons/monitor-up.svg");
          case OperatorType::LMT:
            return QStringLiteral(":/history_icons/file-sliders.svg");
          case OperatorType::ODT:
            return QStringLiteral(":/history_icons/monitor.svg");
          case OperatorType::CLARITY:
            return QStringLiteral(":/history_icons/focus.svg");
          case OperatorType::SHARPEN:
            return QStringLiteral(":/history_icons/scan-line.svg");
          case OperatorType::COLOR_WHEEL:
            return QStringLiteral(":/history_icons/palette.svg");
          case OperatorType::ACES_TONE_MAPPING:
            return QStringLiteral(":/history_icons/git-commit-horizontal.svg");
          case OperatorType::AUTO_EXPOSURE:
            return QStringLiteral(":/history_icons/wand-sparkles.svg");
          case OperatorType::LENS_CALIBRATION:
            return QStringLiteral(":/history_icons/aperture.svg");
          case OperatorType::COLOR_TEMP:
            return QStringLiteral(":/history_icons/thermometer.svg");
          case OperatorType::UNKNOWN:
            return QStringLiteral(":/history_icons/sliders-horizontal.svg");
        }
        return QStringLiteral(":/history_icons/sliders-horizontal.svg");
      }(),
      icon_color, QSize(kIconGlyphSize, kIconGlyphSize), tile->devicePixelRatioF()));
  layout->addWidget(icon, 1);
  return tile;
}

auto SummarizeRawDecode(const EditTransaction& tx) -> TxCardSummary {
  const auto& params = tx.GetOperatorParams();
  const auto prev = tx.GetLastOperatorParams();
  const auto next = JsonBoolAtPath(params, {"raw", "highlights_reconstruct"});
  const auto old = prev ? JsonBoolAtPath(*prev, {"raw", "highlights_reconstruct"})
                        : std::nullopt;
  if (!next.has_value()) {
    return {};
  }
  return {next.value() ? QStringLiteral("Recover") : QStringLiteral("Basic"),
          old.has_value()
              ? QStringLiteral("Highlights %1 \u2192 %2")
                    .arg(old.value() ? QStringLiteral("on") : QStringLiteral("off"),
                         next.value() ? QStringLiteral("on") : QStringLiteral("off"))
              : QStringLiteral("Highlights %1")
                    .arg(next.value() ? QStringLiteral("on") : QStringLiteral("off"))};
}

auto SummarizeLens(const EditTransaction& tx) -> TxCardSummary {
  const auto& params = tx.GetOperatorParams();
  const auto prev = tx.GetLastOperatorParams();

  const bool enabled = JsonBoolAtPath(params, {"lens_calib", "enabled"}).value_or(true);
  const auto old_enabled =
      prev ? JsonBoolAtPath(*prev, {"lens_calib", "enabled"}) : std::nullopt;
  const QString lens = LensName(params);
  const QString old_lens = prev ? LensName(*prev) : QString();

  QString detail;
  if (!old_lens.isEmpty() && lens != old_lens) {
    detail = QStringLiteral("%1 \u2192 %2").arg(old_lens, lens.isEmpty() ? QStringLiteral("Auto")
                                                                         : lens);
  } else if (!lens.isEmpty()) {
    detail = lens;
  } else if (old_enabled.has_value() && *old_enabled != enabled) {
    detail = QStringLiteral("Profile %1 \u2192 %2")
                 .arg(old_enabled.value() ? QStringLiteral("on") : QStringLiteral("off"),
                      enabled ? QStringLiteral("on") : QStringLiteral("off"));
  } else {
    detail = enabled ? QStringLiteral("Auto lens profile") : QStringLiteral("Profile off");
  }

  return {enabled ? QStringLiteral("Applied") : QStringLiteral("Off"), detail};
}

auto SummarizeColorTemp(const EditTransaction& tx) -> TxCardSummary {
  const auto& params = tx.GetOperatorParams();
  const auto prev = tx.GetLastOperatorParams();
  const QString value = ColorTempLabel(params);
  const QString old_value = prev ? ColorTempLabel(*prev) : QString();
  if (!old_value.isEmpty()) {
    return {value, QStringLiteral("%1 \u2192 %2").arg(old_value, value)};
  }
  return {value, QStringLiteral("White balance updated")};
}

auto SummarizeHls(const EditTransaction& tx) -> TxCardSummary {
  constexpr std::array<const char*, 8> kHueLabels = {"Red",    "Orange", "Yellow", "Green",
                                                     "Cyan",   "Blue",   "Purple", "Magenta"};
  constexpr std::array<const char*, 3> kComponentLabels = {"Hue", "Light", "Sat"};

  const auto& params = tx.GetOperatorParams();
  const auto prev = tx.GetLastOperatorParams();
  const auto* table = JsonAtPath(params, {"HLS", "hls_adj_table"});
  const auto* old_table = prev ? JsonAtPath(*prev, {"HLS", "hls_adj_table"}) : nullptr;
  const auto* range_table = JsonAtPath(params, {"HLS", "h_range_table"});
  const auto* old_range_table = prev ? JsonAtPath(*prev, {"HLS", "h_range_table"}) : nullptr;

  if (table && table->is_array()) {
    for (size_t i = 0; i < table->size() && i < kHueLabels.size(); ++i) {
      const auto& row = (*table)[i];
      const nlohmann::json* old_row =
          (old_table && old_table->is_array() && i < old_table->size()) ? &(*old_table)[i]
                                                                         : nullptr;
      if (!row.is_array()) {
        continue;
      }
      for (size_t j = 0; j < 3 && j < row.size(); ++j) {
        const auto next = JsonArrayNumber(&row, j);
        const auto old = old_row ? JsonArrayNumber(old_row, j) : std::nullopt;
        if (!next.has_value()) {
          continue;
        }
        const bool changed = !old.has_value() || std::fabs(*old - *next) > 1e-6;
        if (!changed) {
          continue;
        }

        const bool is_hue = j == 0;
        const double scale = is_hue ? 1.0 : 1000.0;
        const QString unit = is_hue ? QStringLiteral("°") : QString();
        const QString value = WithUnit(FormatSigned(*next * scale), unit);
        const QString old_value =
            old.has_value() ? WithUnit(FormatSigned(*old * scale), unit) : QString();
        const QString detail =
            old_value.isEmpty()
                ? QStringLiteral("%1 %2 %3")
                      .arg(QString::fromLatin1(kHueLabels[i]),
                           QString::fromLatin1(kComponentLabels[j]), value)
                : QStringLiteral("%1 %2 %3 \u2192 %4")
                      .arg(QString::fromLatin1(kHueLabels[i]),
                           QString::fromLatin1(kComponentLabels[j]), old_value, value);
        return {value, detail};
      }
    }
  }

  if (range_table && range_table->is_array()) {
    for (size_t i = 0; i < range_table->size() && i < kHueLabels.size(); ++i) {
      const auto next = JsonArrayNumber(range_table, i);
      const auto old = old_range_table ? JsonArrayNumber(old_range_table, i) : std::nullopt;
      if (!next.has_value()) {
        continue;
      }
      const bool changed = !old.has_value() || std::fabs(*old - *next) > 1e-6;
      if (!changed) {
        continue;
      }
      const QString value = WithUnit(FormatNumber(*next), QStringLiteral("°"));
      const QString detail =
          old.has_value()
              ? QStringLiteral("%1 Range %2 \u2192 %3")
                    .arg(QString::fromLatin1(kHueLabels[i]),
                         WithUnit(FormatNumber(*old), QStringLiteral("°")), value)
              : QStringLiteral("%1 Range %2").arg(QString::fromLatin1(kHueLabels[i]), value);
      return {value, detail};
    }
  }

  return {};
}

auto SummarizeColorWheel(const EditTransaction& tx) -> TxCardSummary {
  static const std::array<std::pair<const char*, const char*>, 3> kWheels = {
      {{"lift", "Lift"}, {"gamma", "Gamma"}, {"gain", "Gain"}}};

  const auto& params = tx.GetOperatorParams();
  const auto prev = tx.GetLastOperatorParams();
  for (const auto& [key, label] : kWheels) {
    const auto strength = JsonNumberAtPath(params, {"color_wheel", key, "strength"});
    const auto old_strength =
        prev ? JsonNumberAtPath(*prev, {"color_wheel", key, "strength"}) : std::nullopt;
    if (strength.has_value() &&
        (!old_strength.has_value() || std::fabs(*strength - *old_strength) > 1e-6)) {
      const QString value = FormatNumber(*strength);
      const QString detail =
          old_strength.has_value()
              ? QStringLiteral("%1 Strength %2 \u2192 %3")
                    .arg(QString::fromLatin1(label), FormatNumber(*old_strength), value)
              : QStringLiteral("%1 Strength %2").arg(QString::fromLatin1(label), value);
      return {value, detail};
    }

    const auto luminance =
        JsonNumberAtPath(params, {"color_wheel", key, "luminance_offset"});
    const auto old_luminance =
        prev ? JsonNumberAtPath(*prev, {"color_wheel", key, "luminance_offset"})
             : std::nullopt;
    if (luminance.has_value() &&
        (!old_luminance.has_value() || std::fabs(*luminance - *old_luminance) > 1e-6)) {
      const QString value = FormatSigned(*luminance);
      const QString detail =
          old_luminance.has_value()
              ? QStringLiteral("%1 Lum %2 \u2192 %3")
                    .arg(QString::fromLatin1(label), FormatSigned(*old_luminance), value)
              : QStringLiteral("%1 Lum %2").arg(QString::fromLatin1(label), value);
      return {value, detail};
    }

    const auto disc_x = JsonNumberAtPath(params, {"color_wheel", key, "disc", "x"});
    const auto disc_y = JsonNumberAtPath(params, {"color_wheel", key, "disc", "y"});
    const auto old_disc_x =
        prev ? JsonNumberAtPath(*prev, {"color_wheel", key, "disc", "x"}) : std::nullopt;
    const auto old_disc_y =
        prev ? JsonNumberAtPath(*prev, {"color_wheel", key, "disc", "y"}) : std::nullopt;
    if (disc_x.has_value() && disc_y.has_value() &&
        (!old_disc_x.has_value() || !old_disc_y.has_value() ||
         std::fabs(*disc_x - *old_disc_x) > 1e-6 ||
         std::fabs(*disc_y - *old_disc_y) > 1e-6)) {
      return {QString::fromLatin1(label),
              QStringLiteral("%1 balance updated").arg(QString::fromLatin1(label))};
    }
  }
  return {};
}

auto SummarizeCurve(const EditTransaction& tx) -> TxCardSummary {
  const auto next_points = curve::ParseCurveControlPointsFromParams(tx.GetOperatorParams());
  const auto old_points =
      tx.GetLastOperatorParams().has_value()
          ? curve::ParseCurveControlPointsFromParams(*tx.GetLastOperatorParams())
          : std::nullopt;
  if (!next_points.has_value()) {
    return {};
  }
  const QString value = QStringLiteral("%1 pts").arg(static_cast<int>(next_points->size()));
  const QString detail =
      old_points.has_value()
          ? QStringLiteral("%1 \u2192 %2 control points")
                .arg(static_cast<int>(old_points->size()))
                .arg(static_cast<int>(next_points->size()))
          : QStringLiteral("Tone curve updated");
  return {value, detail};
}

auto SummarizeLut(const EditTransaction& tx) -> TxCardSummary {
  const auto& params = tx.GetOperatorParams();
  const QString path = JsonStringAtPath(params, {"ocio_lmt"}).value_or(QString());
  const QString file_name = path.isEmpty() ? QString() : QFileInfo(path).fileName();
  if (path.isEmpty()) {
    return {QStringLiteral("Cleared"), QStringLiteral("No LUT selected")};
  }
  return {QStringLiteral("Loaded"),
          file_name.isEmpty() ? QStringLiteral("LUT selected") : file_name};
}

auto SummarizeOdt(const EditTransaction& tx) -> TxCardSummary {
  const auto& params = tx.GetOperatorParams();
  const auto prev = tx.GetLastOperatorParams();
  const auto peak = JsonNumberAtPath(params, {"odt", "peak_luminance"});
  const auto old_peak = prev ? JsonNumberAtPath(*prev, {"odt", "peak_luminance"}) : std::nullopt;
  const QString target = OdtTargetLabel(params);
  const QString old_target = prev ? OdtTargetLabel(*prev) : QString();

  if (peak.has_value() &&
      (!old_peak.has_value() || std::fabs(*peak - *old_peak) > 1e-6)) {
    const QString value = WithUnit(FormatNumber(*peak), QStringLiteral("nit"),
                                   /*space_before_unit*/ true);
    const QString detail =
        old_peak.has_value()
            ? QStringLiteral("%1 \u2192 %2")
                  .arg(WithUnit(FormatNumber(*old_peak), QStringLiteral("nit"), true), value)
            : target;
    return {value, detail};
  }

  if (!target.isEmpty() && target != old_target && !old_target.isEmpty()) {
    return {target, QStringLiteral("%1 \u2192 %2").arg(old_target, target)};
  }

  return {target.isEmpty() ? QStringLiteral("Output") : target,
          QStringLiteral("Output transform updated")};
}

auto SummarizeCropRotate(const EditTransaction& tx) -> TxCardSummary {
  const auto& params = tx.GetOperatorParams();
  const auto prev = tx.GetLastOperatorParams();

  const auto angle = JsonNumberAtPath(params, {"crop_rotate", "angle_degrees"});
  const auto old_angle =
      prev ? JsonNumberAtPath(*prev, {"crop_rotate", "angle_degrees"}) : std::nullopt;
  if (angle.has_value() &&
      (!old_angle.has_value() || std::fabs(*angle - *old_angle) > 1e-6)) {
    return MakeSignedNumberSummary(old_angle, angle, QStringLiteral("°"));
  }

  const bool crop_enabled = JsonBoolAtPath(params, {"crop_rotate", "enable_crop"}).value_or(false);
  const bool old_crop_enabled =
      prev ? JsonBoolAtPath(*prev, {"crop_rotate", "enable_crop"}).value_or(false) : false;
  const auto area = CropAreaPercent(params);
  const auto old_area = prev ? CropAreaPercent(*prev) : std::nullopt;
  if ((crop_enabled || old_crop_enabled) && area.has_value() &&
      (!old_area.has_value() || std::fabs(*area - *old_area) > 1e-4 ||
       crop_enabled != old_crop_enabled)) {
    const QString value = QStringLiteral("%1%").arg(FormatNumber(*area));
    const QString detail =
        old_area.has_value()
            ? QStringLiteral("%1 \u2192 %2")
                  .arg(QStringLiteral("%1%").arg(FormatNumber(*old_area)), value)
            : QStringLiteral("Crop area %1").arg(value);
    return {value, detail};
  }

  const QString aspect = FormatCropAspect(params);
  const QString old_aspect = prev ? FormatCropAspect(*prev) : QString();
  if (!aspect.isEmpty() && aspect != old_aspect && !old_aspect.isEmpty()) {
    return {aspect, QStringLiteral("%1 \u2192 %2").arg(old_aspect, aspect)};
  }

  return {QStringLiteral("Crop"), QStringLiteral("Geometry updated")};
}

auto BuildTxSummary(const EditTransaction& tx) -> TxCardSummary {
  if (tx.GetTransactionType() == TransactionType::_DELETE) {
    return MakeActionSummary(tx.GetTransactionType(), FirstMeaningfulDelta(tx));
  }

  switch (tx.GetTxOperatorType()) {
    case OperatorType::EXPOSURE:
      return MakeSignedNumberSummary(
          tx.GetLastOperatorParams() ? JsonNumberAtPath(*tx.GetLastOperatorParams(), {"exposure"})
                                     : std::nullopt,
          JsonNumberAtPath(tx.GetOperatorParams(), {"exposure"}));
    case OperatorType::CONTRAST:
      return MakeSignedNumberSummary(
          tx.GetLastOperatorParams() ? JsonNumberAtPath(*tx.GetLastOperatorParams(), {"contrast"})
                                     : std::nullopt,
          JsonNumberAtPath(tx.GetOperatorParams(), {"contrast"}));
    case OperatorType::WHITE:
      return MakeSignedNumberSummary(
          tx.GetLastOperatorParams() ? JsonNumberAtPath(*tx.GetLastOperatorParams(), {"white"})
                                     : std::nullopt,
          JsonNumberAtPath(tx.GetOperatorParams(), {"white"}));
    case OperatorType::BLACK:
      return MakeSignedNumberSummary(
          tx.GetLastOperatorParams() ? JsonNumberAtPath(*tx.GetLastOperatorParams(), {"black"})
                                     : std::nullopt,
          JsonNumberAtPath(tx.GetOperatorParams(), {"black"}));
    case OperatorType::SHADOWS:
      return MakeSignedNumberSummary(
          tx.GetLastOperatorParams() ? JsonNumberAtPath(*tx.GetLastOperatorParams(), {"shadows"})
                                     : std::nullopt,
          JsonNumberAtPath(tx.GetOperatorParams(), {"shadows"}));
    case OperatorType::HIGHLIGHTS:
      return MakeSignedNumberSummary(
          tx.GetLastOperatorParams()
              ? JsonNumberAtPath(*tx.GetLastOperatorParams(), {"highlights"})
              : std::nullopt,
          JsonNumberAtPath(tx.GetOperatorParams(), {"highlights"}));
    case OperatorType::SATURATION:
      return MakeSignedNumberSummary(
          tx.GetLastOperatorParams()
              ? JsonNumberAtPath(*tx.GetLastOperatorParams(), {"saturation"})
              : std::nullopt,
          JsonNumberAtPath(tx.GetOperatorParams(), {"saturation"}));
    case OperatorType::CLARITY:
      return MakeSignedNumberSummary(
          tx.GetLastOperatorParams() ? JsonNumberAtPath(*tx.GetLastOperatorParams(), {"clarity"})
                                     : std::nullopt,
          JsonNumberAtPath(tx.GetOperatorParams(), {"clarity"}));
    case OperatorType::SHARPEN:
      return MakeSignedNumberSummary(
          tx.GetLastOperatorParams()
              ? JsonNumberAtPath(*tx.GetLastOperatorParams(), {"sharpen", "offset"})
              : std::nullopt,
          JsonNumberAtPath(tx.GetOperatorParams(), {"sharpen", "offset"}));
    case OperatorType::RAW_DECODE:
      return SummarizeRawDecode(tx);
    case OperatorType::LENS_CALIBRATION:
      return SummarizeLens(tx);
    case OperatorType::COLOR_TEMP:
      return SummarizeColorTemp(tx);
    case OperatorType::HLS:
      return SummarizeHls(tx);
    case OperatorType::COLOR_WHEEL:
      return SummarizeColorWheel(tx);
    case OperatorType::CURVE:
      return SummarizeCurve(tx);
    case OperatorType::LMT:
      return SummarizeLut(tx);
    case OperatorType::ODT:
      return SummarizeOdt(tx);
    case OperatorType::CROP_ROTATE:
      return SummarizeCropRotate(tx);
    default:
      break;
  }

  return MakeActionSummary(tx.GetTransactionType(), FirstMeaningfulDelta(tx));
}

}  // namespace

HistoryLaneWidget::HistoryLaneWidget(QColor dot, QColor line, bool draw_top, bool draw_bottom,
                                     bool solid_dot, QWidget* parent)
    : QWidget(parent),
      dot_(std::move(dot)),
      line_(std::move(line)),
      draw_top_(draw_top),
      draw_bottom_(draw_bottom),
      solid_dot_(solid_dot) {
  setFixedWidth(kLaneWidth);
  setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Expanding);
  setAttribute(Qt::WA_TransparentForMouseEvents);
}

void HistoryLaneWidget::SetConnectors(bool draw_top, bool draw_bottom) {
  draw_top_ = draw_top;
  draw_bottom_ = draw_bottom;
  update();
}

void HistoryLaneWidget::paintEvent(QPaintEvent*) {
  QPainter p(this);
  p.setRenderHint(QPainter::Antialiasing, true);

  const qreal cx = width() / 2.0;
  const qreal cy = height() / 2.0;

  {
    QPen pen(line_);
    pen.setWidthF(kLineWidth);
    pen.setCapStyle(Qt::FlatCap);
    p.setPen(pen);

    if (draw_top_) {
      p.drawLine(QPointF(cx, 0.0), QPointF(cx, cy - kDotRadius - kDotGap));
    }
    if (draw_bottom_) {
      p.drawLine(QPointF(cx, cy + kDotRadius + kDotGap), QPointF(cx, height()));
    }
  }

  if (solid_dot_) {
    QColor ring = dot_;
    ring.setAlpha(92);
    p.setPen(Qt::NoPen);
    p.setBrush(ring);
    p.drawEllipse(QPointF(cx, cy), kRingRadius, kRingRadius);

    p.setBrush(dot_);
    p.drawEllipse(QPointF(cx, cy), kDotRadius, kDotRadius);
  } else {
    QColor fill = dot_;
    fill.setAlpha(34);
    p.setPen(Qt::NoPen);
    p.setBrush(fill);
    p.drawEllipse(QPointF(cx, cy), kDotRadius, kDotRadius);

    QPen pen(dot_);
    pen.setWidthF(1.5);
    p.setPen(pen);
    p.setBrush(Qt::NoBrush);
    p.drawEllipse(QPointF(cx, cy), kDotRadius, kDotRadius);
  }
}

HistoryCardWidget::HistoryCardWidget(QWidget* parent) : QFrame(parent) {
  setObjectName("HistoryCard");
  setAttribute(Qt::WA_StyledBackground, true);
  setAttribute(Qt::WA_Hover, true);
  setProperty("selected", false);

  setStyleSheet(AppTheme::EditorHistoryCardStyle());
}

void HistoryCardWidget::SetSelected(bool selected) {
  if (property("selected").toBool() == selected) {
    return;
  }
  setProperty("selected", selected);
  style()->unpolish(this);
  style()->polish(this);
  update();
}

ElidedLabel::ElidedLabel(const QString& text, QWidget* parent) : QLabel(parent) {
  setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Minimum);
  setTextFormat(Qt::PlainText);
  setContentsMargins(0, 0, 0, 2);
  SetRawText(text);
}

void ElidedLabel::SetRawText(const QString& text) {
  raw_text_ = text;
  UpdateElidedText();
}

void ElidedLabel::resizeEvent(QResizeEvent* event) {
  QLabel::resizeEvent(event);
  UpdateElidedText();
}

void ElidedLabel::UpdateElidedText() {
  if (raw_text_.isEmpty()) {
    QLabel::setText(QString());
    return;
  }

  const int available_width = std::max(0, contentsRect().width());
  if (available_width <= 0) {
    QLabel::setText(raw_text_);
  } else {
    const QFontMetrics metrics(font());
    QLabel::setText(metrics.elidedText(raw_text_, Qt::ElideRight, available_width));
  }
  setToolTip(raw_text_);
}

auto MakePillLabel(const QString& text, QWidget* parent) -> QLabel* {
  auto* l = new QLabel(text, parent);
  const auto& theme = AppTheme::Instance();
  QFont badge_font = AppTheme::Font(AppTheme::FontRole::UiCaptionStrong);
  badge_font.setPointSizeF(7.5);
  badge_font.setWeight(QFont::DemiBold);
  badge_font.setStyleStrategy(QFont::PreferAntialias);
  l->setFont(badge_font);
  l->setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Fixed);
  l->setStyleSheet(QStringLiteral("QLabel {"
                                  "  color: %1;"
                                  "  background: %2;"
                                  "  border: 1px solid %3;"
                                  "  border-radius: 4px;"
                                  "  font-size: 8px;"
                                  "  font-weight: 600;"
                                  "  padding: 1px 4px;"
                                  "}")
                       .arg(theme.accentColor().name(QColor::HexRgb),
                            QColor(theme.accentColor().red(), theme.accentColor().green(),
                                   theme.accentColor().blue(), 32)
                                .name(QColor::HexArgb),
                            QColor(theme.accentColor().red(), theme.accentColor().green(),
                                   theme.accentColor().blue(), 72)
                                .name(QColor::HexArgb)));
  return l;
}

auto OperatorDisplayName(OperatorType op) -> QString {
  switch (op) {
    case OperatorType::RAW_DECODE:
      return QStringLiteral("RAW Decode");
    case OperatorType::RESIZE:
      return QStringLiteral("Resize");
    case OperatorType::CROP_ROTATE:
      return QStringLiteral("Crop / Rotate");
    case OperatorType::EXPOSURE:
      return QStringLiteral("Exposure");
    case OperatorType::CONTRAST:
      return QStringLiteral("Contrast");
    case OperatorType::WHITE:
      return QStringLiteral("Whites");
    case OperatorType::BLACK:
      return QStringLiteral("Blacks");
    case OperatorType::SHADOWS:
      return QStringLiteral("Shadows");
    case OperatorType::HIGHLIGHTS:
      return QStringLiteral("Highlights");
    case OperatorType::CURVE:
      return QStringLiteral("Curve");
    case OperatorType::HLS:
      return QStringLiteral("HSL");
    case OperatorType::SATURATION:
      return QStringLiteral("Saturation");
    case OperatorType::TINT:
      return QStringLiteral("Tint");
    case OperatorType::VIBRANCE:
      return QStringLiteral("Vibrance");
    case OperatorType::CST:
      return QStringLiteral("Color Space");
    case OperatorType::TO_WS:
      return QStringLiteral("To Working Space");
    case OperatorType::TO_OUTPUT:
      return QStringLiteral("To Output");
    case OperatorType::LMT:
      return QStringLiteral("LUT");
    case OperatorType::ODT:
      return QStringLiteral("ODT");
    case OperatorType::CLARITY:
      return QStringLiteral("Clarity");
    case OperatorType::SHARPEN:
      return QStringLiteral("Sharpen");
    case OperatorType::COLOR_WHEEL:
      return QStringLiteral("Color Wheel");
    case OperatorType::ACES_TONE_MAPPING:
      return QStringLiteral("ACES Tone");
    case OperatorType::AUTO_EXPOSURE:
      return QStringLiteral("Auto Exposure");
    case OperatorType::LENS_CALIBRATION:
      return QStringLiteral("Lens Profile");
    case OperatorType::COLOR_TEMP:
      return QStringLiteral("Color Temp");
    case OperatorType::UNKNOWN:
      return QStringLiteral("Edit");
  }
  return QStringLiteral("Edit");
}

auto TxActionGlyph(TransactionType type) -> QString {
  switch (type) {
    case TransactionType::_ADD:
      return QStringLiteral("+");
    case TransactionType::_DELETE:
      return QStringLiteral("\u2212");
    case TransactionType::_EDIT:
      return QStringLiteral("~");
  }
  return QStringLiteral("~");
}

auto CompactTxDelta(const EditTransaction& tx) -> QString {
  const TxCardSummary summary = BuildTxSummary(tx);
  const QString detail_text =
      !summary.detail.isEmpty() ? summary.detail
                                : (!summary.value.isEmpty() ? summary.value : CompactTxDelta(tx));
  if (!summary.detail.isEmpty()) {
    return summary.detail;
  }
  return FirstMeaningfulDelta(tx);
}

auto BuildTxHistoryCard(const EditTransaction& tx, bool draw_top, bool draw_bottom,
                        QWidget* parent) -> HistoryCardWidget* {
  auto* card = new HistoryCardWidget(parent);

  auto* row = new QHBoxLayout(card);
  row->setContentsMargins(8, 5, 8, 5);
  row->setSpacing(6);

  const auto& theme = AppTheme::Instance();

  QColor dot_color;
  switch (tx.GetTransactionType()) {
    case TransactionType::_ADD:
      dot_color = theme.accentColor();
      break;
    case TransactionType::_DELETE:
      dot_color = theme.textMutedColor();
      break;
    case TransactionType::_EDIT:
      dot_color = theme.accentSecondaryColor();
      break;
  }
  if (!dot_color.isValid()) {
    dot_color = theme.accentColor();
  }

  auto* lane = new HistoryLaneWidget(dot_color, theme.dividerColor(), draw_top, draw_bottom,
                                     /*solid_dot*/ tx.GetTransactionType() != TransactionType::_DELETE,
                                     card);
  row->addWidget(lane, 0);

  row->addWidget(MakeHistoryIconTile(tx.GetTxOperatorType(), card), 0, Qt::AlignTop);

  const TxCardSummary summary = BuildTxSummary(tx);
  const QString detail_text =
      !summary.detail.isEmpty() ? summary.detail
                                : (!summary.value.isEmpty() ? summary.value : CompactTxDelta(tx));

  auto* body = new QVBoxLayout();
  body->setContentsMargins(0, 0, 0, 0);
  body->setSpacing(1);

  auto* title_l = new ElidedLabel(OperatorDisplayName(tx.GetTxOperatorType()), card);
  title_l->setObjectName(QStringLiteral("HistoryTxTitle"));
  QFont title_font = AppTheme::Font(AppTheme::FontRole::UiBodyStrong);
  title_font.setPointSizeF(9.5);
  title_font.setWeight(QFont::DemiBold);
  title_font.setStyleStrategy(QFont::PreferAntialias);
  title_l->setFont(title_font);
  title_l->setAttribute(Qt::WA_TransparentForMouseEvents, true);
  title_l->setMinimumHeight(QFontMetrics(title_font).lineSpacing() + 2);
  body->addWidget(title_l);

  auto* detail_l = new ElidedLabel(detail_text, card);
  detail_l->setObjectName(QStringLiteral("HistoryTxSubtitle"));
  QFont detail_font = AppTheme::Font(AppTheme::FontRole::DataCaption);
  detail_font.setPointSizeF(8.25);
  detail_font.setWeight(QFont::Medium);
  detail_font.setStyleStrategy(QFont::PreferAntialias);
  detail_l->setFont(detail_font);
  detail_l->setAttribute(Qt::WA_TransparentForMouseEvents, true);
  detail_l->setMinimumHeight(QFontMetrics(detail_font).lineSpacing() + 2);
  body->addWidget(detail_l);

  row->addLayout(body, 1);

  const QString tooltip =
      QStringLiteral("%1  %2\n%3")
          .arg(OperatorDisplayName(tx.GetTxOperatorType()), TxActionGlyph(tx.GetTransactionType()),
               QString::fromStdString(tx.Describe(true, 256)));
  card->setToolTip(tooltip);

  return card;
}

}  // namespace alcedo::ui
