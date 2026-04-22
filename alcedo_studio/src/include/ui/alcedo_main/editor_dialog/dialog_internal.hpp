//  Copyright 2026 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#pragma once

#include "ui/alcedo_main/editor_dialog/dialog.hpp"

#include <QAbstractItemView>
#include <QAbstractSpinBox>
#include <QApplication>
#include <QByteArray>
#include <QCheckBox>
#include <QColor>
#include <QComboBox>
#include <QCoreApplication>
#include <QDesktopServices>
#include <QDialog>
#include <QDoubleSpinBox>
#include <QEasingCurve>
#include <QEvent>
#include <QFile>
#include <QFrame>
#include <QGraphicsDropShadowEffect>
#include <QGraphicsOpacityEffect>
#include <QGridLayout>
#include <QHBoxLayout>
#include <QIcon>
#include <QImage>
#include <QLabel>
#include <QLineEdit>
#include <QListWidget>
#include <QMessageBox>
#include <QMouseEvent>
#include <QPainter>
#include <QResizeEvent>
#include <QPainterPath>
#include <QPen>
#include <QPlainTextEdit>
#include <QPushButton>
#include <QPixmap>
#include <QScrollArea>
#include <QShowEvent>
#include <QSlider>
#include <QSpinBox>
#include <QStackedWidget>
#include <QSplitter>
#include <QStyle>
#include <QSurfaceFormat>
#include <QTimer>
#include <QTextEdit>
#include <QUrl>
#include <QVariantAnimation>
#include <QVBoxLayout>
#include <QSvgRenderer>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <deque>
#include <filesystem>
#include <format>
#include <functional>
#include <future>
#include <fstream>
#include <json.hpp>
#include <map>
#include <numbers>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "app/render_service.hpp"
#include "edit/operators/basic/color_temp_op.hpp"
#include "edit/history/edit_transaction.hpp"
#include "edit/history/version.hpp"
#include "edit/pipeline/default_pipeline_params.hpp"
#include "edit/pipeline/pipeline_cpu.hpp"
#include "edit/scope/final_display_frame_tap.hpp"
#include "edit/scope/scope_analyzer.hpp"
#include "image/image.hpp"
#include "io/image/image_loader.hpp"
#include "renderer/pipeline_task.hpp"
#include "ui/alcedo_main/editor_dialog/controllers/history_controller.hpp"
#include "ui/alcedo_main/editor_dialog/controllers/image_controller.hpp"
#include "ui/alcedo_main/editor_dialog/controllers/lut_controller.hpp"
#include "ui/alcedo_main/editor_dialog/controllers/pipeline_controller.hpp"
#include "ui/alcedo_main/editor_dialog/controllers/render_controller.hpp"
#include "ui/alcedo_main/editor_dialog/frame/editor_frame_manager.hpp"
#include "ui/alcedo_main/editor_dialog/modules/color_temp.hpp"
#include "ui/alcedo_main/editor_dialog/modules/color_wheel.hpp"
#include "ui/alcedo_main/editor_dialog/modules/curve.hpp"
#include "ui/alcedo_main/editor_dialog/modules/geometry.hpp"
#include "ui/alcedo_main/editor_dialog/modules/histogram.hpp"
#include "ui/alcedo_main/editor_dialog/modules/hls.hpp"
#include "ui/alcedo_main/editor_dialog/modules/lens_calib.hpp"
#include "ui/alcedo_main/editor_dialog/modules/lut_catalog.hpp"
#include "ui/alcedo_main/editor_dialog/modules/pipeline_io.hpp"
#include "ui/alcedo_main/editor_dialog/modules/versioning.hpp"
#include "ui/alcedo_main/editor_dialog/scope/scope_panel.hpp"
#include "ui/alcedo_main/editor_dialog/state.hpp"
#include "ui/alcedo_main/editor_dialog/widgets/display_transform_panel_widget.hpp"
#include "ui/alcedo_main/editor_dialog/widgets/editor_control_panel_widget.hpp"
#include "ui/alcedo_main/editor_dialog/widgets/editor_viewer_pane.hpp"
#include "ui/alcedo_main/editor_dialog/widgets/geometry_panel_widget.hpp"
#include "ui/alcedo_main/editor_dialog/widgets/history_cards.hpp"
#include "ui/alcedo_main/editor_dialog/widgets/look_control_panel_widget.hpp"
#include "ui/alcedo_main/editor_dialog/widgets/lut_browser_widget.hpp"
#include "ui/alcedo_main/editor_dialog/widgets/raw_decode_panel_widget.hpp"
#include "ui/alcedo_main/editor_dialog/widgets/spinner.hpp"
#include "ui/alcedo_main/editor_dialog/widgets/tone_curve_widget.hpp"
#include "ui/alcedo_main/editor_dialog/widgets/tone_control_panel_widget.hpp"
#include "ui/alcedo_main/editor_dialog/widgets/trackball.hpp"
#include "ui/alcedo_main/editor_dialog/widgets/versioning_panel_widget.hpp"
#include "ui/alcedo_main/app_theme.hpp"
#include "ui/alcedo_main/i18n.hpp"
#include "ui/alcedo_main/shortcut_registry.hpp"
#include "ui/edit_viewer/edit_viewer.hpp"

namespace alcedo::ui {
namespace {

const auto kShortcutUndoHistoryId   = QStringLiteral("editor_dialog.undo_history_transaction");
const auto kShortcutResetGeometryId = QStringLiteral("editor_dialog.reset_geometry");
const auto kShortcutSelectPrevLutId = QStringLiteral("editor_dialog.select_prev_lut");
const auto kShortcutSelectNextLutId = QStringLiteral("editor_dialog.select_next_lut");
constexpr char kPanelIconPathProperty[] = "puerhlabPanelIconPath";
const QSize     kPanelToggleIconSize(18, 18);
constexpr int   kPanelToggleButtonHeight = 44;
const QSize     kVersioningRailIconSize(24, 24);
constexpr int   kVersioningRailButtonSize = 46;
constexpr int   kEditorOuterMargin = 14;
constexpr int   kVersioningCollapsedWidth = 64;
constexpr int   kVersioningExpandedMinWidth = 320;
constexpr int   kVersioningExpandedMaxWidth = 420;
constexpr int   kVersioningExpandedMinHeight = 300;
constexpr int   kVersioningExpandedMaxHeight = 460;
constexpr int   kVersioningAnimationMs = 250;
constexpr int   kControlsPanelMinWidth = 260;

using namespace std::chrono_literals;

using LensCatalog = lens_calib::LensCatalog;
using lens_calib::LoadLensCatalog;
using lens_calib::SortAndUniqueStrings;
using color_wheel::ClampDiscPoint;
using color_wheel::DiscToCdlDelta;
using color_wheel::CdlSliderUiToMaster;
using color_wheel::CdlMasterToSliderUi;
using curve::Clamp01;
using curve::DefaultCurveControlPoints;
using curve::NormalizeCurveControlPoints;
using curve::CurveControlPointsEqual;
using curve::CurveHermiteCache;
using curve::BuildCurveHermiteCache;
using curve::EvaluateCurveHermite;
using curve::CurveControlPointsToParams;
using curve::ParseCurveControlPointsFromParams;
using geometry::ClampCropRect;
using geometry::CropAspectPreset;
using hls::WrapHueDegrees;
using hls::HueDistanceDegrees;
using hls::HlsProfileArray;
inline auto ClosestHlsCandidateHueIndex(float hue) -> int { return hls::ClosestCandidateHueIndex(hue); }
inline auto HlsCandidateColor(float hue_degrees) -> QColor { return hls::CandidateColor(hue_degrees); }
inline auto MakeHlsFilledArray(float value) -> HlsProfileArray { return hls::MakeFilledArray(value); }

// Tonal-slider scale aliases (defined in pipeline_io).
constexpr float kBlackSliderFromGlobalScale      = pipeline_io::kBlackSliderFromGlobalScale;
constexpr float kWhiteSliderFromGlobalScale      = pipeline_io::kWhiteSliderFromGlobalScale;
constexpr float kShadowsSliderFromGlobalScale    = pipeline_io::kShadowsSliderFromGlobalScale;
constexpr float kHighlightsSliderFromGlobalScale = pipeline_io::kHighlightsSliderFromGlobalScale;

// HLS constant aliases.
constexpr auto& kHlsCandidateHues        = hls::kCandidateHues;
constexpr float kHlsFixedTargetLightness  = hls::kFixedTargetLightness;
constexpr float kHlsFixedTargetSaturation = hls::kFixedTargetSaturation;
constexpr float kHlsDefaultHueRange       = hls::kDefaultHueRange;
constexpr float kHlsFixedLightnessRange   = hls::kFixedLightnessRange;
constexpr float kHlsFixedSaturationRange  = hls::kFixedSaturationRange;
constexpr float kHlsMaxHueShiftDegrees    = hls::kMaxHueShiftDegrees;
constexpr float kHlsAdjUiMin              = hls::kAdjUiMin;
constexpr float kHlsAdjUiMax              = hls::kAdjUiMax;
constexpr float kHlsAdjUiToParamScale     = hls::kAdjUiToParamScale;

// Aliases for module constants (preserve local names used throughout EditorDialog).
constexpr int   kColorTempCctMin         = color_temp::kCctMin;
constexpr int   kColorTempCctMax         = color_temp::kCctMax;
constexpr int   kColorTempTintMin        = color_temp::kTintMin;
constexpr int   kColorTempTintMax        = color_temp::kTintMax;
constexpr int   kColorTempSliderUiMin    = color_temp::kSliderUiMin;
constexpr int   kColorTempSliderUiMax    = color_temp::kSliderUiMax;
constexpr float kRotationSliderScale     = geometry::kRotationSliderScale;
constexpr float kCropRectSliderScale     = geometry::kCropRectSliderScale;
constexpr double kCropAspectSpinMin      = 0.01;
constexpr double kCropAspectSpinMax      = 100.0;
constexpr int   kCdlWheelSliderUiMin     = color_wheel::kSliderUiMin;
constexpr int   kCdlWheelSliderUiMax     = color_wheel::kSliderUiMax;
constexpr float kCdlWheelStrengthDefault = color_wheel::kStrengthDefault;

auto RenderPanelToggleIcon(const QString& resource_path, const QColor& color, const QSize& size,
                           qreal device_pixel_ratio) -> QIcon {
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

  const qreal scale = std::max<qreal>(2.0, std::ceil(std::max<qreal>(1.0, device_pixel_ratio) * 2.0));
  const QSize physical_size(std::max(1, qRound(size.width() * scale)),
                            std::max(1, qRound(size.height() * scale)));
  QPixmap     pixmap(physical_size);
  pixmap.fill(Qt::transparent);
  pixmap.setDevicePixelRatio(scale);

  QPainter painter(&pixmap);
  painter.setRenderHint(QPainter::Antialiasing, true);
  painter.setRenderHint(QPainter::TextAntialiasing, true);
  painter.setRenderHint(QPainter::SmoothPixmapTransform, true);
  renderer.render(&painter, QRectF(QPointF(0.0, 0.0), QSizeF(size)));
  return QIcon(pixmap);
}

auto RenderDockRailIcon(const QString& resource_path, const QColor& icon_color,
                        const QColor& chevron_color, const QSize& size,
                        qreal device_pixel_ratio, qreal chevron_angle_degrees) -> QIcon {
  QFile svg_file(resource_path);
  if (!svg_file.open(QIODevice::ReadOnly)) {
    return {};
  }

  QByteArray svg_data = svg_file.readAll();
  svg_data.replace("currentColor", icon_color.name(QColor::HexRgb).toUtf8());

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

  const qreal glyph_span =
      std::max(4.0, std::min<qreal>(size.height() - 2.0, size.width() - 9.0));
  const QRectF glyph_rect(1.0, (size.height() - glyph_span) * 0.5, glyph_span, glyph_span);
  renderer.render(&painter, glyph_rect);

  const qreal cx = size.width() - 3.8;
  const qreal cy = size.height() * 0.5;

  painter.save();
  painter.translate(cx, cy);
  painter.rotate(chevron_angle_degrees);
  painter.translate(-cx, -cy);
  QPen chevron_pen(chevron_color, 1.8, Qt::SolidLine, Qt::RoundCap, Qt::RoundJoin);
  painter.setPen(chevron_pen);
  painter.setBrush(Qt::NoBrush);
  QPainterPath chevron_path;
  chevron_path.moveTo(cx - 3.2, cy - 2.6);
  chevron_path.lineTo(cx - 0.3, cy);
  chevron_path.lineTo(cx - 3.2, cy + 2.6);
  painter.drawPath(chevron_path);
  painter.restore();

  return QIcon(pixmap);
}

void ConfigurePanelToggleButton(QPushButton*    button,
                                const QString& tooltip,
                                const QString& icon_resource_path) {
  if (!button) {
    return;
  }

  button->setText(QString());
  button->setCheckable(true);
  button->setAutoDefault(false);
  button->setDefault(false);
  button->setCursor(Qt::PointingHandCursor);
  button->setFixedHeight(kPanelToggleButtonHeight);
  button->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
  button->setToolTip(tooltip);
  button->setAccessibleName(tooltip);
  button->setIconSize(kPanelToggleIconSize);
  button->setProperty(kPanelIconPathProperty, icon_resource_path);
}

template <typename T>
struct EnumOption {
  T           value_;
  const char* label_;
};

constexpr std::array<EnumOption<ColorUtils::ColorSpace>, 6> kDisplayEncodingSpaceOptions = {{
    {ColorUtils::ColorSpace::REC709, "Rec.709"},
    {ColorUtils::ColorSpace::P3_D65, "P3-D65"},
    {ColorUtils::ColorSpace::P3_D60, "P3-D60"},
    {ColorUtils::ColorSpace::P3_DCI, "P3-DCI"},
    {ColorUtils::ColorSpace::XYZ, "XYZ"},
    {ColorUtils::ColorSpace::REC2020, "Rec.2020"},
}};

constexpr std::array<EnumOption<ColorUtils::ColorSpace>, 7> kAcesLimitingSpaceOptions = {{
    {ColorUtils::ColorSpace::REC709, "Rec.709"},
    {ColorUtils::ColorSpace::REC2020, "Rec.2020"},
    {ColorUtils::ColorSpace::P3_D65, "P3-D65"},
    {ColorUtils::ColorSpace::P3_D60, "P3-D60"},
    {ColorUtils::ColorSpace::P3_DCI, "P3-DCI"},
    {ColorUtils::ColorSpace::PROPHOTO, "ProPhoto RGB"},
    {ColorUtils::ColorSpace::ADOBE_RGB, "Adobe RGB"},
}};

constexpr std::array<EnumOption<odt_cpu::OpenDRTLookPreset>, 7> kOpenDrtLookPresetOptions = {{
    {odt_cpu::OpenDRTLookPreset::STANDARD, "Standard"},
    {odt_cpu::OpenDRTLookPreset::ARRIBA, "Arriba"},
    {odt_cpu::OpenDRTLookPreset::SYLVAN, "Sylvan"},
    {odt_cpu::OpenDRTLookPreset::COLORFUL, "Colorful"},
    {odt_cpu::OpenDRTLookPreset::AERY, "Aery"},
    {odt_cpu::OpenDRTLookPreset::DYSTOPIC, "Dystopic"},
    {odt_cpu::OpenDRTLookPreset::UMBRA, "Umbra"},
}};

constexpr std::array<EnumOption<odt_cpu::OpenDRTTonescalePreset>, 14> kOpenDrtTonescaleOptions = {{
    {odt_cpu::OpenDRTTonescalePreset::USE_LOOK_PRESET, "Use Look Preset"},
    {odt_cpu::OpenDRTTonescalePreset::LOW_CONTRAST, "Low Contrast"},
    {odt_cpu::OpenDRTTonescalePreset::MEDIUM_CONTRAST, "Medium Contrast"},
    {odt_cpu::OpenDRTTonescalePreset::HIGH_CONTRAST, "High Contrast"},
    {odt_cpu::OpenDRTTonescalePreset::ARRIBA_TONESCALE, "Arriba Tonescale"},
    {odt_cpu::OpenDRTTonescalePreset::SYLVAN_TONESCALE, "Sylvan Tonescale"},
    {odt_cpu::OpenDRTTonescalePreset::COLORFUL_TONESCALE, "Colorful Tonescale"},
    {odt_cpu::OpenDRTTonescalePreset::AERY_TONESCALE, "Aery Tonescale"},
    {odt_cpu::OpenDRTTonescalePreset::DYSTOPIC_TONESCALE, "Dystopic Tonescale"},
    {odt_cpu::OpenDRTTonescalePreset::UMBRA_TONESCALE, "Umbra Tonescale"},
    {odt_cpu::OpenDRTTonescalePreset::ACES_1_X, "ACES 1.x"},
    {odt_cpu::OpenDRTTonescalePreset::ACES_2_0, "ACES 2.0"},
    {odt_cpu::OpenDRTTonescalePreset::MARVELOUS_TONESCAPE, "Marvelous Tonscape"},
    {odt_cpu::OpenDRTTonescalePreset::DAGRINCHI_TONEGROAN, "Dagrinchi Tonegroan"},
}};

constexpr std::array<EnumOption<odt_cpu::OpenDRTCreativeWhitePreset>, 7>
    kOpenDrtCreativeWhiteOptions = {{
        {odt_cpu::OpenDRTCreativeWhitePreset::USE_LOOK_PRESET, "Use Look Preset"},
        {odt_cpu::OpenDRTCreativeWhitePreset::D93, "D93"},
        {odt_cpu::OpenDRTCreativeWhitePreset::D75, "D75"},
        {odt_cpu::OpenDRTCreativeWhitePreset::D65, "D65"},
        {odt_cpu::OpenDRTCreativeWhitePreset::D60, "D60"},
        {odt_cpu::OpenDRTCreativeWhitePreset::D55, "D55"},
        {odt_cpu::OpenDRTCreativeWhitePreset::D50, "D50"},
    }};

auto ColorTempSliderPosToCct(int pos) -> float { return color_temp::SliderPosToCct(pos); }
auto ColorTempCctToSliderPos(float cct) -> int { return color_temp::CctToSliderPos(cct); }

auto SupportedDisplayEotfOptions(ColorUtils::ColorSpace encoding_space)
    -> std::vector<EnumOption<ColorUtils::EOTF>> {
  switch (encoding_space) {
    case ColorUtils::ColorSpace::REC709:
      return {{ColorUtils::EOTF::BT1886, "BT.1886"},
              {ColorUtils::EOTF::GAMMA_2_2, "Gamma 2.2"}};
    case ColorUtils::ColorSpace::P3_D65:
      return {{ColorUtils::EOTF::GAMMA_2_2, "Gamma 2.2"},
              {ColorUtils::EOTF::ST2084, "ST 2084 (PQ)"}};
    case ColorUtils::ColorSpace::P3_D60:
    case ColorUtils::ColorSpace::P3_DCI:
    case ColorUtils::ColorSpace::XYZ:
      return {{ColorUtils::EOTF::GAMMA_2_6, "Gamma 2.6"}};
    case ColorUtils::ColorSpace::REC2020:
      return {{ColorUtils::EOTF::ST2084, "ST 2084 (PQ)"},
              {ColorUtils::EOTF::HLG, "HLG"}};
    default:
      return {{ColorUtils::EOTF::GAMMA_2_2, "Gamma 2.2"}};
  }
}

auto IsSupportedDisplayEncoding(ColorUtils::ColorSpace encoding_space,
                                ColorUtils::EOTF       encoding_eotf) -> bool {
  const auto options = SupportedDisplayEotfOptions(encoding_space);
  return std::any_of(options.begin(), options.end(),
                     [encoding_eotf](const auto& option) { return option.value_ == encoding_eotf; });
}

auto DefaultDisplayEotfForSpace(ColorUtils::ColorSpace encoding_space) -> ColorUtils::EOTF {
  const auto options = SupportedDisplayEotfOptions(encoding_space);
  return options.empty() ? ColorUtils::EOTF::GAMMA_2_2 : options.front().value_;
}

void SanitizeOdtStateForUi(OdtState& odt_state) {
  const bool encoding_space_supported =
      std::any_of(kDisplayEncodingSpaceOptions.begin(), kDisplayEncodingSpaceOptions.end(),
                  [&odt_state](const auto& option) {
                    return option.value_ == odt_state.encoding_space_;
                  });
  if (!encoding_space_supported) {
    odt_state.encoding_space_ = ColorUtils::ColorSpace::REC709;
  }
  if (!IsSupportedDisplayEncoding(odt_state.encoding_space_, odt_state.encoding_eotf_)) {
    odt_state.encoding_eotf_ = DefaultDisplayEotfForSpace(odt_state.encoding_space_);
  }
  const bool limiting_space_supported =
      std::any_of(kAcesLimitingSpaceOptions.begin(), kAcesLimitingSpaceOptions.end(),
                  [&odt_state](const auto& option) {
                    return option.value_ == odt_state.aces_.limiting_space_;
                  });
  if (!limiting_space_supported) {
    odt_state.aces_.limiting_space_ = ColorUtils::ColorSpace::REC709;
  }
  odt_state.peak_luminance_ = std::clamp(odt_state.peak_luminance_, 100.0f, 1000.0f);
}

auto ResolveEditorWindowTitle(const std::shared_ptr<ImagePoolService>& image_pool,
                              image_id_t image_id,
                              sl_element_id_t element_id) -> QString {
  if (image_pool && image_id != 0) {
    try {
      const std::wstring image_name = image_pool->Read<std::wstring>(
          image_id,
          [](const std::shared_ptr<Image>& image) -> std::wstring {
            if (!image) {
              return {};
            }
            if (!image->image_name_.empty()) {
              return image->image_name_;
            }
            if (!image->image_path_.empty()) {
              return image->image_path_.filename().wstring();
            }
            return {};
          });
      if (!image_name.empty()) {
        return Tr("Editor - %1").arg(QString::fromStdWString(image_name));
      }
    } catch (...) {
    }
  }

  return Tr("Editor - image #%1").arg(static_cast<qulonglong>(element_id));
}

}  // namespace

class EditorDialog final : public QDialog {
 public:
  enum class WorkingMode : int { Incremental = 0, Plain = 1 };
  enum class VersioningFlyoutPage : int { History = 0, Versions = 1 };

  EditorDialog(std::shared_ptr<ImagePoolService>       image_pool,
               std::shared_ptr<PipelineGuard>          pipeline_guard,
               std::shared_ptr<EditHistoryMgmtService> history_service,
               std::shared_ptr<EditHistoryGuard> history_guard, sl_element_id_t element_id,
               image_id_t image_id, QWidget* parent = nullptr);

 private:

  void BuildViewerAndPanelShell();
  auto BuildControlPanelShell(const QString& panel_style) -> EditorControlPanelWidget*;
  void BuildLookControlPanel(EditorControlPanelWidget* controls_panel, const QString& scroll_style);
  void WireLookControlPanel();
  void BuildToneControlPanel();
  void BuildDisplayTransformPanel();
  void BuildGeometryRawPanels();
  void BuildRawDecodePanel();
  void BuildVersioningPanel();
  // Types, enums, and state helpers are defined in state.hpp / state.cpp.

  void RegisterShortcuts();

  auto ShouldConsumeUndoShortcutLocally() const -> bool;

  auto ShouldConsumeLutNavigationShortcut() const -> bool;

  void RefreshHlsTargetUi();

  void RefreshCdlOffsetLabels();

  void UpdateGeometryCropRectLabel();

  auto CurrentGeometrySourceAspect() const -> float;

  auto CurrentGeometryAspectRatio() const -> std::optional<float>;

  void SyncGeometryCropSlidersFromState();

  void SyncCropAspectControlsFromState();

  void PushGeometryStateToViewer();

  void SetCropRectState(float x, float y, float w, float h, bool sync_controls = true,
                        bool sync_viewer = true);

  void ApplyAspectPresetToCurrentCrop();

  void ResizeCropRectWithAspect(float proposed_value, bool use_width_driver);

  void SetCropAspectPresetState(CropAspectPreset preset);

  static auto DefaultAdjustmentState() -> const AdjustmentState&;

  void CacheAsShotColorTemp(float cct, float tint);

  void PrimeColorTempDisplayForAsShot();

  void WarmAsShotColorTempCacheFromRawMetadata();

  void RegisterSliderReset(QSlider* slider, std::function<void()> on_reset);

  void RegisterCurveReset(ToneCurveWidget* widget, std::function<void()> on_reset);

  void ResetFieldToDefault(AdjustmentField field,
                           const std::function<void(const AdjustmentState&)>& apply_default);

  void ResetColorTempToAsShot();

  void ResetCurveToDefault();

  void ResetCropAndRotation();

  bool eventFilter(QObject* obj, QEvent* event) override;

  void changeEvent(QEvent* event) override;

  void showEvent(QShowEvent* event) override;

  void resizeEvent(QResizeEvent* event) override;

  void ApplyInitialSplitterSizes();

  void UpdateViewerZoomLabel(float zoom);

  void RetranslateUi();

  void RefreshGeometryModeUi();

  void EnsureLensCatalogLoaded();

  void RefreshLensBrandComboFromState();

  void RefreshLensModelComboFromState();

  void RefreshLensComboFromState();

  void RefreshLutBrowserUi();

  void ForceRefreshLutBrowserUi();

  void OpenLutFolder();

  void RefreshPanelSwitchUi();

  void RefreshVersioningCollapseUi();

  void SetVersioningCollapsed(bool collapsed, bool animate = true);

  void RepositionVersioningFlyout();

  void SetActiveControlPanel(ControlPanelKind panel);

  void RefreshOdtMethodUi();

  void RefreshOdtEncodingEotfComboFromState();

  void PromoteColorTempToCustomForEditing();

  // Returns true if any resolved color temp value actually changed.
  auto RefreshColorTempRuntimeStateFromGlobalParams() -> bool;

  void SyncColorTempControlsFromState();

  void RefreshVersionLogSelectionStyles();

  void TriggerQualityPreviewRenderFromPipeline();

  void SyncControlsFromState();

  auto ReconstructPipelineParamsForVersion(Version& version) -> std::optional<nlohmann::json>;

  auto ReloadUiStateFromPipeline(bool reset_to_defaults_if_missing) -> bool;

  auto ApplyPipelineParamsToEditor(const nlohmann::json& params) -> bool;

  auto ReloadEditorFromHistoryVersion(Version& version, QString* error) -> bool;

  void CheckoutSelectedVersion(QListWidgetItem* item);

  void UndoLastTransaction();

  void UpdateVersionUi();

  void CommitWorkingVersion();

  auto CurrentWorkingMode() const -> WorkingMode;

  void StartNewWorkingVersionFromUi();

  void StartNewWorkingVersionFromCommit(const Hash128& committed_id);

  auto ReadCurrentOperatorParams(PipelineStageName stage_name, OperatorType op_type) const
      -> std::optional<nlohmann::json>;

  std::pair<PipelineStageName, OperatorType> FieldSpec(AdjustmentField field) const;

  nlohmann::json ParamsForField(AdjustmentField field, const AdjustmentState& s) const;

  bool FieldChanged(AdjustmentField field) const;

  void CommitAdjustment(AdjustmentField field);

  bool LoadStateFromPipelineIfPresent();

  void SetupPipeline();

  void ApplyStateToPipeline(const AdjustmentState& render_state);

  static constexpr std::chrono::milliseconds kFastPreviewMinSubmitInterval =
      controllers::render::kFastPreviewMinSubmitInterval;
  static constexpr std::chrono::milliseconds kQualityPreviewDebounceInterval =
      controllers::render::kQualityPreviewDebounceInterval;
  static constexpr std::chrono::milliseconds kViewportDetailDebounceInterval{120};

  void AdvancePreviewGeneration();
  void InvalidateDetailPreviewState();
  auto BuildPreviewMetadata(RenderType render_type) const -> FramePreviewMetadata;
  auto IsDetailPreviewGeometryFallbackActive() const -> bool;
  auto CanScheduleDetailPreview() const -> bool;
  void MaybeScheduleDetailPreviewRenderFromViewport();

  void EnsureQualityPreviewTimer();
  void EnsureDetailPreviewTimer();

  void ScheduleQualityPreviewRenderFromPipeline();
  void ScheduleDetailPreviewRenderFromViewport();
  void TriggerDetailPreviewRenderFromViewport();

  auto CanSubmitFastPreviewNow() const -> bool;

  void EnsureFastPreviewSubmitTimer();

  void ArmFastPreviewSubmitTimer();

  void EnqueueRenderRequest(const AdjustmentState& snapshot,
                            const FramePreviewMetadata& frame_metadata, bool apply_state,
                            bool use_viewport_region = true);

  void RequestRender(bool use_viewport_region = true, bool bump_preview_generation = true);

  void RequestRenderWithoutApplyingState(bool use_viewport_region = true,
                                         bool bump_preview_generation = false);

  void EnsurePollTimer();

  void PollInflight();

  void StartNext();

  void OnRenderFinished();

  std::shared_ptr<ImagePoolService>                        image_pool_;
  std::shared_ptr<PipelineGuard>                           pipeline_guard_;
  std::shared_ptr<EditHistoryMgmtService>                  history_service_;
  std::shared_ptr<EditHistoryGuard>                        history_guard_;
  sl_element_id_t                                          element_id_ = 0;
  image_id_t                                               image_id_   = 0;

  std::shared_ptr<PipelineScheduler>                       scheduler_;
  PipelineTask                                             base_task_{};

  QtEditViewer*                                            viewer_                 = nullptr;
  QWidget*                                                 viewer_container_       = nullptr;
  QLabel*                                                  viewer_zoom_label_      = nullptr;
  ScopePanel*                                              scope_panel_            = nullptr;
  QScrollArea*                                             controls_scroll_        = nullptr;
  QScrollArea*                                             tone_controls_scroll_   = nullptr;
  QScrollArea*                                             look_controls_scroll_   = nullptr;
  QScrollArea*                                             drt_controls_scroll_    = nullptr;
  QScrollArea*                                             geometry_controls_scroll_ = nullptr;
  QScrollArea*                                             raw_controls_scroll_    = nullptr;
  QSplitter*                                               main_splitter_          = nullptr;
  QWidget*                                                 versioning_panel_host_  = nullptr;
  QWidget*                                                 versioning_flyout_      = nullptr;
  QWidget*                                                 versioning_collapsed_nav_ = nullptr;
  QGraphicsOpacityEffect*                                  versioning_panel_opacity_effect_ = nullptr;
  QVariantAnimation*                                       versioning_panel_anim_ = nullptr;
  QStackedWidget*                                          control_panels_stack_   = nullptr;
  SpinnerWidget*                                           spinner_                = nullptr;
  QWidget*                                                 controls_               = nullptr;
  QWidget*                                                 tone_controls_          = nullptr;
  QWidget*                                                 look_controls_          = nullptr;
  QWidget*                                                 drt_controls_           = nullptr;
  QWidget*                                                 geometry_controls_      = nullptr;
  QWidget*                                                 raw_controls_           = nullptr;
  QVBoxLayout*                                             controls_layout_        = nullptr;
  QVBoxLayout*                                             look_controls_layout_   = nullptr;
  QVBoxLayout*                                             drt_controls_layout_    = nullptr;
  QVBoxLayout*                                             geometry_controls_layout_ = nullptr;
  QVBoxLayout*                                             raw_controls_layout_    = nullptr;
  QWidget*                                                 shared_versioning_root_ = nullptr;
  QVBoxLayout*                                             shared_versioning_layout_ = nullptr;
  QStackedWidget*                                          versioning_pages_stack_ = nullptr;
  QPushButton*                                             tone_panel_btn_         = nullptr;
  QPushButton*                                             look_panel_btn_         = nullptr;
  QPushButton*                                             drt_panel_btn_          = nullptr;
  QPushButton*                                             geometry_panel_btn_     = nullptr;
  QPushButton*                                             raw_panel_btn_          = nullptr;
  LutBrowserWidget*                                        lut_browser_widget_     = nullptr;
  QSlider*                                                 exposure_slider_        = nullptr;
  QSlider*                                                 contrast_slider_        = nullptr;
  QSlider*                                                 saturation_slider_      = nullptr;
  QCheckBox*                                               raw_highlights_reconstruct_checkbox_ = nullptr;
  QCheckBox*                                               lens_calib_enabled_checkbox_ = nullptr;
  QComboBox*                                               lens_brand_combo_       = nullptr;
  QComboBox*                                               lens_model_combo_       = nullptr;
  QLabel*                                                  lens_catalog_status_label_ = nullptr;
  CdlTrackballDiscWidget*                                  lift_disc_widget_       = nullptr;
  CdlTrackballDiscWidget*                                  gamma_disc_widget_      = nullptr;
  CdlTrackballDiscWidget*                                  gain_disc_widget_       = nullptr;
  QLabel*                                                  lift_offset_label_      = nullptr;
  QLabel*                                                  gamma_offset_label_     = nullptr;
  QLabel*                                                  gain_offset_label_      = nullptr;
  QSlider*                                                 lift_master_slider_     = nullptr;
  QSlider*                                                 gamma_master_slider_    = nullptr;
  QSlider*                                                 gain_master_slider_     = nullptr;
  QComboBox*                                               color_temp_mode_combo_  = nullptr;
  QSlider*                                                 color_temp_cct_slider_  = nullptr;
  QSlider*                                                 color_temp_tint_slider_ = nullptr;
  QLabel*                                                  color_temp_unsupported_label_ = nullptr;
  QLabel*                                                  hls_target_label_       = nullptr;
  std::vector<QPushButton*>                                hls_candidate_buttons_{};
  QSlider*                                                 hls_hue_adjust_slider_        = nullptr;
  QSlider*                                                 hls_lightness_adjust_slider_  = nullptr;
  QSlider*                                                 hls_saturation_adjust_slider_ = nullptr;
  QSlider*                                                 hls_hue_range_slider_         = nullptr;
  QSlider*                                                 blacks_slider_                = nullptr;
  QSlider*                                                 whites_slider_                = nullptr;
  QSlider*                                                 shadows_slider_               = nullptr;
  QSlider*                                                 highlights_slider_            = nullptr;
  ToneCurveWidget*                                         curve_widget_                 = nullptr;
  QSlider*                                                 sharpen_slider_               = nullptr;
  QSlider*                                                 clarity_slider_               = nullptr;
  QComboBox*                                               odt_encoding_space_combo_     = nullptr;
  QComboBox*                                               odt_encoding_eotf_combo_      = nullptr;
  QSlider*                                                 odt_peak_luminance_slider_    = nullptr;
  QPushButton*                                             odt_aces_method_card_         = nullptr;
  QPushButton*                                             odt_open_drt_method_card_     = nullptr;
  QStackedWidget*                                          odt_method_stack_             = nullptr;
  QComboBox*                                               odt_aces_limiting_space_combo_ = nullptr;
  QComboBox*                                               odt_open_drt_look_preset_combo_ = nullptr;
  QComboBox*                                               odt_open_drt_tonescale_preset_combo_ = nullptr;
  QComboBox*                                               odt_open_drt_creative_white_combo_ = nullptr;
  QSlider*                                                 rotate_slider_                = nullptr;
  QSlider*                                                 geometry_crop_x_slider_       = nullptr;
  QSlider*                                                 geometry_crop_y_slider_       = nullptr;
  QSlider*                                                 geometry_crop_w_slider_       = nullptr;
  QSlider*                                                 geometry_crop_h_slider_       = nullptr;
  QComboBox*                                               geometry_crop_aspect_preset_combo_ = nullptr;
  QDoubleSpinBox*                                          geometry_crop_aspect_width_spin_ = nullptr;
  QDoubleSpinBox*                                          geometry_crop_aspect_height_spin_ = nullptr;
  QLabel*                                                  geometry_crop_rect_label_     = nullptr;
  QPushButton*                                             geometry_apply_btn_           = nullptr;
  QPushButton*                                             geometry_reset_btn_           = nullptr;
  QLabel*                                                  version_status_               = nullptr;
  QPushButton*                                             undo_tx_btn_                  = nullptr;
  QPushButton*                                             commit_version_btn_           = nullptr;
  QPushButton*                                             versioning_history_btn_       = nullptr;
  QPushButton*                                             versioning_versions_btn_      = nullptr;
  std::unique_ptr<ShortcutRegistry>                        shortcut_registry_{};
  QComboBox*                                               working_mode_combo_           = nullptr;
  QPushButton*                                             new_working_btn_              = nullptr;
  QListWidget*                                             version_log_                  = nullptr;
  QListWidget*                                             tx_stack_                     = nullptr;
  QTimer*                                                  poll_timer_                   = nullptr;
  QTimer*                                                  detail_preview_timer_         = nullptr;
  std::optional<std::future<std::shared_ptr<ImageBuffer>>> inflight_future_{};
  std::optional<PendingRenderRequest>                      inflight_request_{};

  controllers::LutController                               lut_controller_{};
  LensCatalog                                              lens_catalog_{};

  std::string                                              last_applied_lut_path_{};
  std::optional<ColorTempRequestSnapshot>                  last_submitted_color_temp_request_{};
  AdjustmentState                                          state_{};
  AdjustmentState                                          committed_state_{};
  Version                                                  working_version_{};
  std::optional<PendingRenderRequest>                      pending_fast_preview_request_{};
  std::optional<PendingRenderRequest>                      pending_quality_base_render_request_{};
  std::optional<PendingRenderRequest>                      pending_detail_render_request_{};
  ControlPanelKind                                         active_panel_               = ControlPanelKind::Tone;
  bool                                                     pipeline_initialized_       = false;
  bool                                                     inflight_                   = false;
  QTimer*                                                  quality_preview_timer_      = nullptr;
  QTimer*                                                  fast_preview_submit_timer_  = nullptr;
  std::chrono::steady_clock::time_point                    last_fast_preview_submit_time_{};
  std::uint64_t                                            preview_generation_         = 0;
  std::uint64_t                                            detail_serial_              = 0;
  std::uint64_t                                            latest_quality_base_generation_ready_ = 0;
  bool                                                     syncing_controls_           = false;
  bool                                                     versioning_collapsed_       = true;
  bool                                                     initial_splitter_sizes_applied_ = false;
  qreal                                                    versioning_panel_progress_  = 0.0;
  VersioningFlyoutPage                                     versioning_active_page_ =
      VersioningFlyoutPage::History;
  float                                                    last_known_as_shot_cct_     = 6500.0f;
  float                                                    last_known_as_shot_tint_    = 0.0f;
  bool                                                     has_last_known_as_shot_color_temp_ = false;
  EditorFrameManager                                      frame_manager_{};
  std::map<QSlider*, std::function<void()>>                slider_reset_callbacks_{};
  std::function<void()>                                    curve_reset_callback_{};
};

}  // namespace alcedo::ui
