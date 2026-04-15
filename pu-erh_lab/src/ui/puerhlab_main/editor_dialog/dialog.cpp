//  Copyright 2026 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#include "ui/puerhlab_main/editor_dialog/dialog.hpp"

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
#include <QPainterPath>
#include <QPen>
#include <QPlainTextEdit>
#include <QPushButton>
#include <QPixmap>
#include <QScrollArea>
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
#include "ui/puerhlab_main/editor_dialog/controllers/history_controller.hpp"
#include "ui/puerhlab_main/editor_dialog/controllers/image_controller.hpp"
#include "ui/puerhlab_main/editor_dialog/controllers/lut_controller.hpp"
#include "ui/puerhlab_main/editor_dialog/controllers/pipeline_controller.hpp"
#include "ui/puerhlab_main/editor_dialog/controllers/render_controller.hpp"
#include "ui/puerhlab_main/editor_dialog/modules/color_temp.hpp"
#include "ui/puerhlab_main/editor_dialog/modules/color_wheel.hpp"
#include "ui/puerhlab_main/editor_dialog/modules/curve.hpp"
#include "ui/puerhlab_main/editor_dialog/modules/geometry.hpp"
#include "ui/puerhlab_main/editor_dialog/modules/histogram.hpp"
#include "ui/puerhlab_main/editor_dialog/modules/hls.hpp"
#include "ui/puerhlab_main/editor_dialog/modules/lens_calib.hpp"
#include "ui/puerhlab_main/editor_dialog/modules/lut_catalog.hpp"
#include "ui/puerhlab_main/editor_dialog/modules/pipeline_io.hpp"
#include "ui/puerhlab_main/editor_dialog/modules/versioning.hpp"
#include "ui/puerhlab_main/editor_dialog/scope/scope_panel.hpp"
#include "ui/puerhlab_main/editor_dialog/state.hpp"
#include "ui/puerhlab_main/editor_dialog/widgets/history_cards.hpp"
#include "ui/puerhlab_main/editor_dialog/widgets/lut_browser_widget.hpp"
#include "ui/puerhlab_main/editor_dialog/widgets/spinner.hpp"
#include "ui/puerhlab_main/editor_dialog/widgets/tone_curve_widget.hpp"
#include "ui/puerhlab_main/editor_dialog/widgets/trackball.hpp"
#include "ui/puerhlab_main/app_theme.hpp"
#include "ui/puerhlab_main/i18n.hpp"
#include "ui/puerhlab_main/shortcut_registry.hpp"
#include "ui/edit_viewer/edit_viewer.hpp"

namespace puerhlab::ui {
namespace {

auto Tr(const char* text) -> QString {
  return QCoreApplication::translate(PUERHLAB_I18N_CONTEXT, text);
}

const auto kShortcutUndoHistoryId   = QStringLiteral("editor_dialog.undo_history_transaction");
const auto kShortcutResetGeometryId = QStringLiteral("editor_dialog.reset_geometry");
const auto kShortcutSelectPrevLutId = QStringLiteral("editor_dialog.select_prev_lut");
const auto kShortcutSelectNextLutId = QStringLiteral("editor_dialog.select_next_lut");
constexpr char kPanelIconPathProperty[] = "puerhlabPanelIconPath";
const QSize     kPanelToggleIconSize(18, 18);
constexpr int   kPanelToggleButtonHeight = 44;
const QSize     kVersioningRailIconSize(22, 22);
constexpr int   kVersioningRailButtonSize = 46;
constexpr int   kEditorOuterMargin = 14;
constexpr int   kVersioningCollapsedWidth = 64;
constexpr int   kVersioningExpandedMinWidth = 220;
constexpr int   kVersioningExpandedMaxWidth = 600;
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

  const qreal scale = std::max<qreal>(1.0, device_pixel_ratio);
  const QSize physical_size(std::max(1, qRound(size.width() * scale)),
                            std::max(1, qRound(size.height() * scale)));
  QPixmap     pixmap(physical_size);
  pixmap.fill(Qt::transparent);
  pixmap.setDevicePixelRatio(scale);

  QPainter painter(&pixmap);
  painter.setRenderHint(QPainter::Antialiasing, true);
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

class EditorDialog final : public QDialog {
 public:
  enum class WorkingMode : int { Incremental = 0, Plain = 1 };

  EditorDialog(std::shared_ptr<ImagePoolService>       image_pool,
               std::shared_ptr<PipelineGuard>          pipeline_guard,
               std::shared_ptr<EditHistoryMgmtService> history_service,
               std::shared_ptr<EditHistoryGuard> history_guard, sl_element_id_t element_id,
               image_id_t image_id, QWidget* parent = nullptr)
      : QDialog(parent),
        image_pool_(std::move(image_pool)),
        pipeline_guard_(std::move(pipeline_guard)),
        history_service_(std::move(history_service)),
        history_guard_(std::move(history_guard)),
        element_id_(element_id),
        image_id_(image_id),
        scheduler_(RenderService::GetPreviewScheduler()) {
    if (!image_pool_ || !pipeline_guard_ || !pipeline_guard_->pipeline_ || !history_service_ ||
        !history_guard_ || !history_guard_->history_ || !scheduler_) {
      throw std::runtime_error("EditorDialog: missing services");
    }

    setModal(true);
    setSizeGripEnabled(true);
    setWindowFlag(Qt::WindowMinMaxButtonsHint, true);
    setWindowFlag(Qt::MSWindowsFixedSizeDialogHint, false);
    setWindowTitle(ResolveEditorWindowTitle(image_pool_, image_id_, element_id_));
    setMinimumSize(1080, 680);
    resize(1500, 1000);
    AppTheme::ApplyFont(this, AppTheme::FontRole::UiBody);


    // --- Constructor body split into .inc files for maintainability ---
#include "ctor_viewer_and_panels.inc"
#include "ctor_tone_color.inc"
#include "ctor_display_transform.inc"
#include "ctor_geometry_raw.inc"
#include "ctor_versioning.inc"

    shortcut_registry_ = std::make_unique<ShortcutRegistry>(this);
    RegisterShortcuts();

    AppTheme::ApplyFontsRecursively(this);

    UpdateVersionUi();

    SetupPipeline();
    pipeline_initialized_ = true;
    if (viewer_) {
      viewer_->SetCropOverlayAspectLock(false, 1.0f);
      viewer_->SetCropOverlayRectNormalized(state_.crop_x_, state_.crop_y_, state_.crop_w_,
                                            state_.crop_h_);
      viewer_->SetCropOverlayRotationDegrees(state_.rotate_degrees_);
      viewer_->SetCropOverlayVisible(false);
      viewer_->SetCropToolEnabled(false);
    }

    // Load with a full-res preview first; scheduler transitions back to fast-preview baseline.
    QTimer::singleShot(0, this, [this]() {
      state_.type_ = RenderType::FULL_RES_PREVIEW;
      RequestRender();
      state_.type_ = RenderType::FAST_PREVIEW;
    });
  }

 private:
  // Types, enums, and state helpers are defined in state.hpp / state.cpp.

  void RegisterShortcuts() {
    if (!shortcut_registry_) {
      return;
    }

    shortcut_registry_->Register({
        .id               = kShortcutUndoHistoryId,
        .description      = Tr("Undo last uncommitted transaction"),
        .default_sequence = QKeySequence(QKeySequence::Undo),
        .context          = Qt::WidgetWithChildrenShortcut,
        .on_trigger       = [this]() {
          if (!ShouldConsumeUndoShortcutLocally()) {
            UndoLastTransaction();
          }
        },
    });
    shortcut_registry_->Register({
        .id               = kShortcutResetGeometryId,
        .description      = Tr("Reset crop & rotation"),
        .default_sequence = QKeySequence(Qt::CTRL | Qt::Key_R),
        .context          = Qt::WidgetWithChildrenShortcut,
        .on_trigger       = [this]() { ResetCropAndRotation(); },
    });
    shortcut_registry_->Register({
        .id               = kShortcutSelectPrevLutId,
        .description      = Tr("Select previous LUT"),
        .default_sequence = QKeySequence(Qt::Key_Up),
        .context          = Qt::WidgetWithChildrenShortcut,
        .on_trigger       = [this]() {
          if (ShouldConsumeLutNavigationShortcut()) {
            return;
          }
          if (lut_browser_widget_) {
            lut_browser_widget_->SelectRelativeEntry(-1);
          }
        },
    });
    shortcut_registry_->Register({
        .id               = kShortcutSelectNextLutId,
        .description      = Tr("Select next LUT"),
        .default_sequence = QKeySequence(Qt::Key_Down),
        .context          = Qt::WidgetWithChildrenShortcut,
        .on_trigger       = [this]() {
          if (ShouldConsumeLutNavigationShortcut()) {
            return;
          }
          if (lut_browser_widget_) {
            lut_browser_widget_->SelectRelativeEntry(1);
          }
        },
    });

    if (undo_tx_btn_) {
      undo_tx_btn_->setToolTip(shortcut_registry_->DecorateTooltip(
          Tr("Undo last uncommitted transaction"), kShortcutUndoHistoryId));
    }
    if (geometry_reset_btn_) {
      geometry_reset_btn_->setToolTip(
          shortcut_registry_->DecorateTooltip(Tr("Reset crop & rotation"),
                                             kShortcutResetGeometryId));
    }
  }

  auto ShouldConsumeUndoShortcutLocally() const -> bool {
    QWidget* const focus_widget = QApplication::focusWidget();
    if (!focus_widget || !isAncestorOf(focus_widget)) {
      return false;
    }

    return qobject_cast<QLineEdit*>(focus_widget) != nullptr ||
           qobject_cast<QTextEdit*>(focus_widget) != nullptr ||
           qobject_cast<QPlainTextEdit*>(focus_widget) != nullptr ||
           qobject_cast<QAbstractSpinBox*>(focus_widget) != nullptr;
  }

  auto ShouldConsumeLutNavigationShortcut() const -> bool {
    if (active_panel_ != ControlPanelKind::Look || !lut_browser_widget_) {
      return true;
    }
    if (QApplication::activePopupWidget()) {
      return true;
    }

    QWidget* const focus_widget = QApplication::focusWidget();
    if (!focus_widget || !isAncestorOf(focus_widget)) {
      return false;
    }

    if (qobject_cast<QComboBox*>(focus_widget) != nullptr ||
        qobject_cast<QAbstractSpinBox*>(focus_widget) != nullptr ||
        qobject_cast<QTextEdit*>(focus_widget) != nullptr ||
        qobject_cast<QPlainTextEdit*>(focus_widget) != nullptr) {
      return true;
    }
    if (auto* line_edit = qobject_cast<QLineEdit*>(focus_widget)) {
      return !lut_browser_widget_->isAncestorOf(line_edit);
    }
    return false;
  }

  void RefreshHlsTargetUi() {
    if (!hls_target_label_ && hls_candidate_buttons_.empty()) {
      return;
    }

    const float hue = WrapHueDegrees(state_.hls_target_hue_);
    if (hls_target_label_) {
      hls_target_label_->setText(Tr("Target Hue: %1 deg").arg(hue, 0, 'f', 0));
    }

    const int selected_idx = ClosestHlsCandidateHueIndex(hue);
    for (int i = 0; i < static_cast<int>(hls_candidate_buttons_.size()); ++i) {
      auto* btn = hls_candidate_buttons_[i];
      if (!btn) {
        continue;
      }
      const bool   selected    = (i == selected_idx);
      const QColor swatch      = HlsCandidateColor(kHlsCandidateHues[static_cast<size_t>(i)]);
      const auto   border_w_px = selected ? "3px" : "1px";
      const QString border_col = selected
                                     ? AppTheme::Instance().accentColor().name(QColor::HexRgb)
                                     : AppTheme::Instance().glassStrokeColor().name(QColor::HexArgb);
      btn->setToolTip(Tr("Hue %1 deg").arg(kHlsCandidateHues[static_cast<size_t>(i)], 0, 'f', 0));
      btn->setStyleSheet(QString("QPushButton {"
                                 "  background: %1;"
                                 "  border: %2 solid %3;"
                                 "  border-radius: 11px;"
                                 "}"
                                 "QPushButton:hover {"
                                 "  border-color: %4;"
                                 "}")
                             .arg(swatch.name(QColor::HexRgb), border_w_px, border_col,
                                  AppTheme::Instance().accentSecondaryColor().name(QColor::HexRgb)));
    }
  }

  void RefreshCdlOffsetLabels() {
    if (lift_offset_label_) {
      lift_offset_label_->setText(FormatWheelDeltaText(state_.lift_wheel_, false));
    }
    if (gamma_offset_label_) {
      gamma_offset_label_->setText(FormatWheelDeltaText(state_.gamma_wheel_, true));
    }
    if (gain_offset_label_) {
      gain_offset_label_->setText(FormatWheelDeltaText(state_.gain_wheel_, true));
    }
  }

  void UpdateGeometryCropRectLabel() {
    if (!geometry_crop_rect_label_) {
      return;
    }
    const int source_width  = viewer_ ? viewer_->GetWidth() : 0;
    const int source_height = viewer_ ? viewer_->GetHeight() : 0;
    const float image_aspect =
        source_height > 0 ? static_cast<float>(source_width) / static_cast<float>(source_height)
                          : 1.0f;
    const auto resolved = geometry::ResolveCropRect(
        state_.crop_x_, state_.crop_y_, state_.crop_w_, state_.crop_h_, image_aspect,
        state_.crop_aspect_preset_, state_.crop_aspect_width_, state_.crop_aspect_height_,
        source_width, source_height);

    QString text = Tr("Crop Rect: x=%1 y=%2 w=%3 h=%4")
                       .arg(state_.crop_x_, 0, 'f', 3)
                       .arg(state_.crop_y_, 0, 'f', 3)
                       .arg(state_.crop_w_, 0, 'f', 3)
                       .arg(state_.crop_h_, 0, 'f', 3);
    if (resolved.pixel_width_ > 0 && resolved.pixel_height_ > 0) {
      text += Tr(" | Output: %1x%2").arg(resolved.pixel_width_).arg(resolved.pixel_height_);
    }
    text += Tr(" | Aspect: %1:1").arg(resolved.aspect_ratio_, 0, 'f', 3);
    geometry_crop_rect_label_->setText(text);
  }

  auto CurrentGeometrySourceAspect() const -> float {
    if (viewer_ && viewer_->GetWidth() > 0 && viewer_->GetHeight() > 0) {
      return static_cast<float>(viewer_->GetWidth()) / static_cast<float>(viewer_->GetHeight());
    }
    return 1.0f;
  }

  auto CurrentGeometryAspectRatio() const -> std::optional<float> {
    return geometry::AspectRatioFromSize(state_.crop_aspect_width_, state_.crop_aspect_height_);
  }

  void SyncGeometryCropSlidersFromState() {
    const bool prev_sync = syncing_controls_;
    syncing_controls_    = true;
    if (geometry_crop_x_slider_) {
      geometry_crop_x_slider_->setValue(
          static_cast<int>(std::lround(state_.crop_x_ * kCropRectSliderScale)));
    }
    if (geometry_crop_y_slider_) {
      geometry_crop_y_slider_->setValue(
          static_cast<int>(std::lround(state_.crop_y_ * kCropRectSliderScale)));
    }
    if (geometry_crop_w_slider_) {
      geometry_crop_w_slider_->setValue(
          static_cast<int>(std::lround(state_.crop_w_ * kCropRectSliderScale)));
    }
    if (geometry_crop_h_slider_) {
      geometry_crop_h_slider_->setValue(
          static_cast<int>(std::lround(state_.crop_h_ * kCropRectSliderScale)));
    }
    syncing_controls_ = prev_sync;
  }

  void SyncCropAspectControlsFromState() {
    const bool prev_sync = syncing_controls_;
    syncing_controls_    = true;
    if (geometry_crop_aspect_preset_combo_) {
      const int preset_index =
          geometry_crop_aspect_preset_combo_->findData(static_cast<int>(state_.crop_aspect_preset_));
      geometry_crop_aspect_preset_combo_->setCurrentIndex(std::max(0, preset_index));
    }
    if (geometry_crop_aspect_width_spin_) {
      geometry_crop_aspect_width_spin_->setValue(state_.crop_aspect_width_);
    }
    if (geometry_crop_aspect_height_spin_) {
      geometry_crop_aspect_height_spin_->setValue(state_.crop_aspect_height_);
    }
    syncing_controls_ = prev_sync;
  }

  void PushGeometryStateToViewer() {
    if (!viewer_) {
      return;
    }
    const bool locked_aspect =
        geometry::HasLockedAspect(state_.crop_aspect_preset_, state_.crop_aspect_width_,
                                  state_.crop_aspect_height_);
    viewer_->SetCropOverlayAspectLock(locked_aspect,
                                      CurrentGeometryAspectRatio().value_or(1.0f));
    viewer_->SetCropOverlayRectNormalized(state_.crop_x_, state_.crop_y_, state_.crop_w_,
                                          state_.crop_h_);
    viewer_->SetCropOverlayRotationDegrees(state_.rotate_degrees_);
  }

  void SyncViewerDisplayEncoding() {
    if (!viewer_) {
      return;
    }
    viewer_->SetDisplayEncoding(state_.odt_.encoding_space_, state_.odt_.encoding_eotf_);
  }

  auto CurrentFrameSink() -> IFrameSink* {
    if (final_display_frame_tap_) {
      return final_display_frame_tap_.get();
    }
    return viewer_;
  }

  void SetCropRectState(float x, float y, float w, float h, bool sync_controls = true,
                        bool sync_viewer = true) {
    const auto clamped = ClampCropRect(x, y, w, h);
    state_.crop_x_     = clamped[0];
    state_.crop_y_     = clamped[1];
    state_.crop_w_     = clamped[2];
    state_.crop_h_     = clamped[3];
    state_.crop_enabled_ = true;
    if (sync_controls) {
      SyncGeometryCropSlidersFromState();
    }
    UpdateGeometryCropRectLabel();
    if (sync_viewer) {
      PushGeometryStateToViewer();
    }
  }

  void ApplyAspectPresetToCurrentCrop() {
    if (!geometry::HasLockedAspect(state_.crop_aspect_preset_, state_.crop_aspect_width_,
                                   state_.crop_aspect_height_)) {
      UpdateGeometryCropRectLabel();
      PushGeometryStateToViewer();
      return;
    }

    const auto aspect_ratio = CurrentGeometryAspectRatio();
    if (!aspect_ratio.has_value()) {
      UpdateGeometryCropRectLabel();
      PushGeometryStateToViewer();
      return;
    }

    const auto max_rect =
        geometry::MakeMaxAspectCropRect(CurrentGeometrySourceAspect(), *aspect_ratio);
    SetCropRectState(max_rect[0], max_rect[1], max_rect[2], max_rect[3], true, true);
  }

  void ResizeCropRectWithAspect(float proposed_value, bool use_width_driver) {
    if (!geometry::HasLockedAspect(state_.crop_aspect_preset_, state_.crop_aspect_width_,
                                   state_.crop_aspect_height_)) {
      if (use_width_driver) {
        SetCropRectState(state_.crop_x_, state_.crop_y_, proposed_value, state_.crop_h_);
      } else {
        SetCropRectState(state_.crop_x_, state_.crop_y_, state_.crop_w_, proposed_value);
      }
      return;
    }

    const auto aspect_ratio = CurrentGeometryAspectRatio();
    if (!aspect_ratio.has_value()) {
      return;
    }

    const auto resized = geometry::ResizeAspectRectAroundCenter(
        state_.crop_x_, state_.crop_y_, use_width_driver ? proposed_value : state_.crop_w_,
        use_width_driver ? state_.crop_h_ : proposed_value, CurrentGeometrySourceAspect(),
        *aspect_ratio, use_width_driver);
    SetCropRectState(resized[0], resized[1], resized[2], resized[3], true, true);
  }

  void SetCropAspectPresetState(CropAspectPreset preset) {
    state_.crop_aspect_preset_ = preset;
    if (const auto preset_ratio = geometry::CropAspectPresetRatio(preset); preset_ratio.has_value()) {
      state_.crop_aspect_width_  = preset_ratio->at(0);
      state_.crop_aspect_height_ = preset_ratio->at(1);
    } else if (!geometry::NormalizeCropAspect(state_.crop_aspect_width_, state_.crop_aspect_height_)
                    .has_value()) {
      state_.crop_aspect_width_  = 1.0f;
      state_.crop_aspect_height_ = 1.0f;
    }
    SyncCropAspectControlsFromState();
    ApplyAspectPresetToCurrentCrop();
  }

  static auto DefaultAdjustmentState() -> const AdjustmentState& {
    static const AdjustmentState defaults{};
    return defaults;
  }

  void CacheAsShotColorTemp(float cct, float tint) {
    last_known_as_shot_cct_ =
        std::clamp(cct, static_cast<float>(kColorTempCctMin), static_cast<float>(kColorTempCctMax));
    last_known_as_shot_tint_ = std::clamp(tint, static_cast<float>(kColorTempTintMin),
                                          static_cast<float>(kColorTempTintMax));
    has_last_known_as_shot_color_temp_ = true;
  }

  void PrimeColorTempDisplayForAsShot() {
    if (!has_last_known_as_shot_color_temp_) {
      return;
    }
    state_.color_temp_resolved_cct_   = last_known_as_shot_cct_;
    state_.color_temp_resolved_tint_  = last_known_as_shot_tint_;
    committed_state_.color_temp_resolved_cct_  = last_known_as_shot_cct_;
    committed_state_.color_temp_resolved_tint_ = last_known_as_shot_tint_;
  }

  void WarmAsShotColorTempCacheFromRawMetadata() {
    if (has_last_known_as_shot_color_temp_ || !pipeline_guard_ || !pipeline_guard_->pipeline_) {
      return;
    }

    auto params = pipeline_guard_->pipeline_->GetGlobalParams();
    try {
      static const nlohmann::json kAsShotColorTempParams = {
          {"color_temp", {{"mode", "as_shot"}, {"cct", 6500.0f}, {"tint", 0.0f}}}};
      ColorTempOp                  as_shot_probe(kAsShotColorTempParams);
      as_shot_probe.SetGlobalParams(params);
    } catch (...) {
      return;
    }

    if (!params.color_temp_matrices_valid_) {
      return;
    }

    CacheAsShotColorTemp(params.color_temp_resolved_cct_, params.color_temp_resolved_tint_);
    if (state_.color_temp_mode_ == ColorTempMode::AS_SHOT) {
      PrimeColorTempDisplayForAsShot();
    }
  }

  void RegisterSliderReset(QSlider* slider, std::function<void()> on_reset) {
    if (!slider || !on_reset) {
      return;
    }
    slider->installEventFilter(this);
    slider_reset_callbacks_[slider] = std::move(on_reset);
  }

  void RegisterCurveReset(ToneCurveWidget* widget, std::function<void()> on_reset) {
    if (!widget || !on_reset) {
      return;
    }
    widget->installEventFilter(this);
    curve_reset_callback_ = std::move(on_reset);
  }

  void ResetFieldToDefault(AdjustmentField field,
                           const std::function<void(const AdjustmentState&)>& apply_default) {
    if (!apply_default) {
      return;
    }
    const auto& defaults = DefaultAdjustmentState();
    apply_default(defaults);
    SyncControlsFromState();
    RequestRender();
    CommitAdjustment(field);
  }

  void ResetColorTempToAsShot() {
    if (state_.color_temp_mode_ == ColorTempMode::AS_SHOT) {
      return;
    }
    state_.color_temp_mode_ = ColorTempMode::AS_SHOT;
    PrimeColorTempDisplayForAsShot();
    SyncColorTempControlsFromState();
    RequestRender();
    CommitAdjustment(AdjustmentField::ColorTemp);
  }

  void ResetCurveToDefault() {
    state_.curve_points_ = DefaultCurveControlPoints();
    SyncControlsFromState();
    RequestRender();
    CommitAdjustment(AdjustmentField::Curve);
  }

  void ResetCropAndRotation() {
    state_.crop_x_         = 0.0f;
    state_.crop_y_         = 0.0f;
    state_.crop_w_         = 1.0f;
    state_.crop_h_         = 1.0f;
    state_.crop_enabled_   = true;
    state_.rotate_degrees_ = 0.0f;
    state_.crop_aspect_preset_ = CropAspectPreset::Free;
    state_.crop_aspect_width_  = 1.0f;
    state_.crop_aspect_height_ = 1.0f;

    const bool prev_sync = syncing_controls_;
    syncing_controls_     = true;
    if (geometry_crop_x_slider_) {
      geometry_crop_x_slider_->setValue(0);
    }
    if (geometry_crop_y_slider_) {
      geometry_crop_y_slider_->setValue(0);
    }
    if (geometry_crop_w_slider_) {
      geometry_crop_w_slider_->setValue(static_cast<int>(kCropRectSliderScale));
    }
    if (geometry_crop_h_slider_) {
      geometry_crop_h_slider_->setValue(static_cast<int>(kCropRectSliderScale));
    }
    if (rotate_slider_) {
      rotate_slider_->setValue(0);
    }
    if (geometry_crop_aspect_preset_combo_) {
      const int free_index =
          geometry_crop_aspect_preset_combo_->findData(static_cast<int>(CropAspectPreset::Free));
      geometry_crop_aspect_preset_combo_->setCurrentIndex(std::max(0, free_index));
    }
    if (geometry_crop_aspect_width_spin_) {
      geometry_crop_aspect_width_spin_->setValue(1.0);
    }
    if (geometry_crop_aspect_height_spin_) {
      geometry_crop_aspect_height_spin_->setValue(1.0);
    }
    syncing_controls_ = prev_sync;

    UpdateGeometryCropRectLabel();
    if (viewer_) {
      viewer_->SetCropOverlayAspectLock(false, 1.0f);
      viewer_->SetCropOverlayRectNormalized(0.0f, 0.0f, 1.0f, 1.0f);
      viewer_->SetCropOverlayRotationDegrees(0.0f);
    }
  }

  bool eventFilter(QObject* obj, QEvent* event) override {
    if (obj == versioning_nav_btn_ && event) {
      if (event->type() == QEvent::Enter || event->type() == QEvent::HoverEnter) {
        versioning_nav_hovered_ = true;
        RefreshVersioningCollapseUi();
      } else if (event->type() == QEvent::Leave || event->type() == QEvent::HoverLeave) {
        versioning_nav_hovered_ = false;
        RefreshVersioningCollapseUi();
      }
    }

    if (event && event->type() == QEvent::MouseButtonDblClick) {
      if (auto* slider = qobject_cast<QSlider*>(obj)) {
        const auto it = slider_reset_callbacks_.find(slider);
        if (it != slider_reset_callbacks_.end()) {
          if (!syncing_controls_ && it->second) {
            it->second();
          }
          return true;
        }
      }
      if (obj == curve_widget_ && curve_reset_callback_) {
        if (!syncing_controls_) {
          curve_reset_callback_();
        }
        return true;
      }
    }
    return QDialog::eventFilter(obj, event);
  }

  void changeEvent(QEvent* event) override {
    if (event && event->type() == QEvent::LanguageChange) {
      RetranslateUi();
    }
    QDialog::changeEvent(event);
  }

  void UpdateViewerZoomLabel(float zoom) {
    if (!viewer_zoom_label_) {
      return;
    }
    const float clamped = std::max(1.0f, zoom);
    viewer_zoom_label_->setText(
        Tr("Zoom %1% (%2x)")
            .arg(clamped * 100.0f, 0, 'f', 0)
            .arg(clamped, 0, 'f', 2));
  }

  void RetranslateUi() {
    setWindowTitle(ResolveEditorWindowTitle(image_pool_, image_id_, element_id_));

    if (tone_panel_btn_) {
      tone_panel_btn_->setText(Tr("Tone"));
    }
    if (look_panel_btn_) {
      look_panel_btn_->setText(Tr("Look"));
    }
    if (drt_panel_btn_) {
      drt_panel_btn_->setText(Tr("Display RT"));
      drt_panel_btn_->setToolTip(Tr("Display Rendering Transform"));
    }
    if (geometry_panel_btn_) {
      geometry_panel_btn_->setText(Tr("Geometry"));
    }
    if (raw_panel_btn_) {
      raw_panel_btn_->setText(Tr("RAW Decode"));
    }
    if (geometry_apply_btn_) {
      geometry_apply_btn_->setText(Tr("Apply Crop"));
    }
    if (geometry_reset_btn_) {
      geometry_reset_btn_->setText(Tr("Reset"));
      geometry_reset_btn_->setToolTip(Tr("Reset crop & rotation (Ctrl+R)"));
    }
    if (undo_tx_btn_) {
      undo_tx_btn_->setText(Tr("Undo"));
    }
    if (commit_version_btn_) {
      commit_version_btn_->setText(Tr("Commit Version"));
    }
    if (new_working_btn_) {
      new_working_btn_->setText(Tr("New"));
    }
    if (color_temp_unsupported_label_) {
      color_temp_unsupported_label_->setText(
          Tr("Color temperature/tint is unavailable for this image."));
    }
    if (working_mode_combo_) {
      const int current_value = working_mode_combo_->currentData().toInt();
      const bool prev_sync = syncing_controls_;
      syncing_controls_ = true;
      working_mode_combo_->clear();
      working_mode_combo_->addItem(Tr("Plain"), static_cast<int>(WorkingMode::Plain));
      working_mode_combo_->addItem(Tr("Incr"), static_cast<int>(WorkingMode::Incremental));
      const int index = working_mode_combo_->findData(current_value);
      working_mode_combo_->setCurrentIndex(std::max(0, index));
      syncing_controls_ = prev_sync;
    }
    if (geometry_crop_aspect_preset_combo_) {
      const int current_value = geometry_crop_aspect_preset_combo_->currentData().toInt();
      const bool prev_sync = syncing_controls_;
      syncing_controls_ = true;
      geometry_crop_aspect_preset_combo_->clear();
      for (const auto& option : geometry::CropAspectPresetOptions()) {
        geometry_crop_aspect_preset_combo_->addItem(Tr(option.label_),
                                                    static_cast<int>(option.value_));
      }
      const int index = geometry_crop_aspect_preset_combo_->findData(current_value);
      geometry_crop_aspect_preset_combo_->setCurrentIndex(std::max(0, index));
      syncing_controls_ = prev_sync;
    }
    if (lut_browser_widget_) {
      lut_browser_widget_->RetranslateUi();
      RefreshLutBrowserUi();
    }

    RefreshLensComboFromState();
    RefreshOdtEncodingEotfComboFromState();
    RefreshHlsTargetUi();
    UpdateGeometryCropRectLabel();
    UpdateViewerZoomLabel(viewer_ ? viewer_->GetViewZoom() : 1.0f);
    RefreshVersioningCollapseUi();
    UpdateVersionUi();
  }

  void RefreshGeometryModeUi() {
    if (geometry_crop_aspect_width_spin_) {
      geometry_crop_aspect_width_spin_->setEnabled(true);
    }
    if (geometry_crop_aspect_height_spin_) {
      geometry_crop_aspect_height_spin_->setEnabled(true);
    }
  }

  void EnsureLensCatalogLoaded() {
    if (!lens_catalog_.brands_.empty() || !lens_catalog_.models_by_brand_.empty()) {
      return;
    }
    lens_catalog_ = LoadLensCatalog();
  }

  void RefreshLensBrandComboFromState() {
    if (!lens_brand_combo_) {
      return;
    }
    EnsureLensCatalogLoaded();

    const bool prev_sync = syncing_controls_;
    syncing_controls_    = true;

    lens_brand_combo_->clear();
    lens_brand_combo_->addItem(Tr("Auto (metadata)"), QString());
    for (const auto& brand : lens_catalog_.brands_) {
      lens_brand_combo_->addItem(QString::fromStdString(brand), QString::fromStdString(brand));
    }

    int selected_index = 0;
    if (!state_.lens_override_make_.empty()) {
      selected_index =
          lens_brand_combo_->findData(QString::fromStdString(state_.lens_override_make_));
      if (selected_index < 0) {
        lens_brand_combo_->addItem(QString::fromStdString(state_.lens_override_make_),
                                   QString::fromStdString(state_.lens_override_make_));
        selected_index = lens_brand_combo_->count() - 1;
      }
    }
    lens_brand_combo_->setCurrentIndex(std::max(0, selected_index));

    syncing_controls_ = prev_sync;
  }

  void RefreshLensModelComboFromState() {
    if (!lens_model_combo_) {
      return;
    }
    EnsureLensCatalogLoaded();

    const bool prev_sync = syncing_controls_;
    syncing_controls_    = true;

    lens_model_combo_->clear();

    if (state_.lens_override_make_.empty()) {
      lens_model_combo_->addItem(Tr("Auto (metadata)"), QString());
      lens_model_combo_->setCurrentIndex(0);
      lens_model_combo_->setEnabled(false);
      state_.lens_override_model_.clear();
    } else {
      std::vector<std::string> models;
      if (const auto it = lens_catalog_.models_by_brand_.find(state_.lens_override_make_);
          it != lens_catalog_.models_by_brand_.end()) {
        models = it->second;
      }
      if (!state_.lens_override_model_.empty() &&
          std::find(models.begin(), models.end(), state_.lens_override_model_) == models.end()) {
        models.push_back(state_.lens_override_model_);
      }
      SortAndUniqueStrings(&models);
      for (const auto& model : models) {
        lens_model_combo_->addItem(QString::fromStdString(model), QString::fromStdString(model));
      }

      int selected_index = 0;
      if (!state_.lens_override_model_.empty()) {
        selected_index =
            lens_model_combo_->findData(QString::fromStdString(state_.lens_override_model_));
      }
      if (selected_index < 0 && lens_model_combo_->count() > 0) {
        selected_index = 0;
      }
      lens_model_combo_->setCurrentIndex(selected_index);
      lens_model_combo_->setEnabled(lens_model_combo_->count() > 0);

      if (lens_model_combo_->count() > 0) {
        state_.lens_override_model_ = lens_model_combo_->currentData().toString().toStdString();
      } else {
        state_.lens_override_model_.clear();
      }
    }

    if (lens_catalog_status_label_) {
      if (lens_catalog_.brands_.empty()) {
        lens_catalog_status_label_->setText(
            Tr("Lens catalog not found. You can still use Auto (metadata) mode."));
      } else {
        lens_catalog_status_label_->setText(
            Tr("Lens catalog: %1 brands").arg(static_cast<int>(lens_catalog_.brands_.size())));
      }
    }

    syncing_controls_ = prev_sync;
  }

  void RefreshLensComboFromState() {
    RefreshLensBrandComboFromState();
    RefreshLensModelComboFromState();
  }

  void RefreshLutBrowserUi() {
    if (!lut_browser_widget_) {
      return;
    }
    const auto view_model = lut_controller_.Refresh(state_.lut_path_, false);
    lut_browser_widget_->SetDirectoryInfo(view_model.directory_text_,
                                          view_model.status_text_,
                                          view_model.can_open_directory_);
    lut_browser_widget_->SetEntries(view_model.entries_, view_model.selected_path_);
  }

  void ForceRefreshLutBrowserUi() {
    if (!lut_browser_widget_) {
      return;
    }
    const auto view_model = lut_controller_.Refresh(state_.lut_path_, true);
    lut_browser_widget_->SetDirectoryInfo(view_model.directory_text_,
                                          view_model.status_text_,
                                          view_model.can_open_directory_);
    lut_browser_widget_->SetEntries(view_model.entries_, view_model.selected_path_);
  }

  void OpenLutFolder() {
    const auto& directory = lut_controller_.directory();
    std::error_code ec;
    if (directory.empty() || !std::filesystem::is_directory(directory, ec) || ec) {
      QMessageBox::warning(this, Tr("LUT"), Tr("LUT folder is unavailable."));
      return;
    }
    if (!QDesktopServices::openUrl(QUrl::fromLocalFile(QString::fromStdWString(directory.wstring())))) {
      QMessageBox::warning(this, Tr("LUT"), Tr("Failed to open the LUT folder."));
    }
  }

  void RefreshPanelSwitchUi() {
    if (!tone_panel_btn_ || !look_panel_btn_ || !drt_panel_btn_ ||
        !geometry_panel_btn_ || !raw_panel_btn_) {
      return;
    }
    const bool tone_active     = (active_panel_ == ControlPanelKind::Tone);
    const bool look_active     = (active_panel_ == ControlPanelKind::Look);
    const bool drt_active      = (active_panel_ == ControlPanelKind::DisplayRenderingTransform);
    const bool geometry_active = (active_panel_ == ControlPanelKind::Geometry);
    const bool raw_active      = (active_panel_ == ControlPanelKind::RawDecode);

    const auto apply_panel_button_state = [](QPushButton* button, bool active, bool is_first,
                                             bool is_last) {
      if (!button) {
        return;
      }

      button->setChecked(active);
      button->setIcon(RenderPanelToggleIcon(
          button->property(kPanelIconPathProperty).toString(),
          active ? AppTheme::Instance().bgCanvasColor() : AppTheme::Instance().textColor(),
          kPanelToggleIconSize, button->devicePixelRatioF()));
      button->setStyleSheet(AppTheme::EditorPanelToggleStyle(active, is_first, is_last));
    };

    apply_panel_button_state(tone_panel_btn_, tone_active, true, false);
    apply_panel_button_state(look_panel_btn_, look_active, false, false);
    apply_panel_button_state(drt_panel_btn_, drt_active, false, false);
    apply_panel_button_state(geometry_panel_btn_, geometry_active, false, false);
    apply_panel_button_state(raw_panel_btn_, raw_active, false, true);
  }

  void RefreshVersioningCollapseUi() {
    if (!versioning_panel_host_ || !versioning_panel_content_ || !versioning_collapsed_nav_) {
      return;
    }

    const auto& theme = AppTheme::Instance();
    const qreal progress =
        std::clamp(versioning_panel_progress_, static_cast<qreal>(0.0), static_cast<qreal>(1.0));

    const int panel_width =
        qRound(static_cast<qreal>(versioning_expanded_width_) * progress);
    const int gap_width = qRound(static_cast<qreal>(versioning_collapsed_gap_base_width_) *
                                 (static_cast<qreal>(1.0) - progress));

    if (versioning_panel_opacity_effect_) {
      versioning_panel_opacity_effect_->setOpacity(progress);
    }
    versioning_panel_content_->setVisible(panel_width > 0 || progress > 0.0);
    versioning_panel_content_->setMinimumWidth(panel_width);
    versioning_panel_content_->setMaximumWidth(panel_width);

    if (versioning_collapsed_gap_) {
      versioning_collapsed_gap_->setVisible(gap_width > 0);
      versioning_collapsed_gap_->setFixedWidth(gap_width);
    }

    const int host_width = kVersioningCollapsedWidth + panel_width + gap_width;
    versioning_panel_host_->setMinimumWidth(host_width);
    versioning_panel_host_->setMaximumWidth(host_width);
    versioning_collapsed_nav_->setVisible(true);

    const bool panel_expanded = progress >= 0.5;
    versioning_collapsed_     = !panel_expanded;

    if (versioning_nav_btn_) {
      versioning_nav_btn_->setChecked(panel_expanded);
      versioning_nav_btn_->setIconSize(versioning_nav_hovered_ ? QSize(24, 24)
                                                               : kVersioningRailIconSize);
      versioning_nav_btn_->setIcon(RenderDockRailIcon(
          QStringLiteral(":/panel_icons/git-branch.svg"),
          panel_expanded ? theme.bgCanvasColor() : theme.textColor(),
          panel_expanded ? theme.bgCanvasColor() : theme.textMutedColor(), QSize(24, 24),
          versioning_nav_btn_->devicePixelRatioF(), 180.0 * progress));
      const QString nav_tip = panel_expanded ? Tr("Collapse Versioning") : Tr("Expand Versioning");
      versioning_nav_btn_->setToolTip(nav_tip);
      versioning_nav_btn_->setAccessibleName(nav_tip);
    }

    if (main_splitter_) {
      auto sizes = main_splitter_->sizes();
      if (sizes.size() >= 3) {
        const int delta = host_width - sizes[0];
        if (delta != 0) {
          int center_width = sizes[1] - delta;
          int right_width  = sizes[2];
          if (center_width < 0) {
            right_width += center_width;
            center_width = 0;
          }
          if (right_width < 0) {
            center_width += right_width;
            right_width = 0;
          }

          sizes[0] = host_width;
          sizes[1] = std::max(0, center_width);
          sizes[2] = std::max(0, right_width);
          main_splitter_->setSizes(sizes);
        }
      }
    }
  }

  void SetVersioningCollapsed(bool collapsed, bool animate = true) {
    const qreal target_progress = collapsed ? 0.0 : 1.0;
    if (!animate || std::abs(versioning_panel_progress_ - target_progress) < 0.001) {
      if (versioning_panel_anim_) {
        versioning_panel_anim_->stop();
      }
      versioning_panel_progress_ = target_progress;
      versioning_collapsed_      = collapsed;
      RefreshVersioningCollapseUi();
      return;
    }

    if (!versioning_panel_anim_) {
      versioning_panel_anim_ = new QVariantAnimation(this);
      versioning_panel_anim_->setDuration(kVersioningAnimationMs);
      versioning_panel_anim_->setEasingCurve(QEasingCurve::OutCubic);
      QObject::connect(versioning_panel_anim_, &QVariantAnimation::valueChanged, this,
                       [this](const QVariant& value) {
                         versioning_panel_progress_ = value.toReal();
                         RefreshVersioningCollapseUi();
                       });
      QObject::connect(versioning_panel_anim_, &QVariantAnimation::finished, this, [this]() {
        if (!versioning_panel_anim_) {
          return;
        }
        versioning_panel_progress_ = versioning_panel_anim_->endValue().toReal();
        versioning_collapsed_      = versioning_panel_progress_ < 0.5;
        RefreshVersioningCollapseUi();
      });
    }

    if (collapsed && main_splitter_) {
      const auto sizes = main_splitter_->sizes();
      if (sizes.size() >= 3) {
        const int current_panel_width =
            sizes[0] - kVersioningCollapsedWidth - versioning_collapsed_gap_base_width_;
        versioning_expanded_width_ =
            std::clamp(current_panel_width > 0 ? current_panel_width : versioning_expanded_width_,
                       kVersioningExpandedMinWidth, kVersioningExpandedMaxWidth);
      }
    }

    versioning_panel_anim_->stop();
    versioning_panel_anim_->setStartValue(versioning_panel_progress_);
    versioning_panel_anim_->setEndValue(target_progress);
    versioning_panel_anim_->start();
  }

  void SetActiveControlPanel(ControlPanelKind panel) {
    const ControlPanelKind previous_panel = active_panel_;
    active_panel_ = panel;
    if (control_panels_stack_) {
      int panel_index = 0;
      if (panel == ControlPanelKind::Look) {
        panel_index = 1;
      } else if (panel == ControlPanelKind::DisplayRenderingTransform) {
        panel_index = 2;
      } else if (panel == ControlPanelKind::Geometry) {
        panel_index = 3;
      } else if (panel == ControlPanelKind::RawDecode) {
        panel_index = 4;
      }
      control_panels_stack_->setCurrentIndex(panel_index);
    }

    const bool geometry_active = (panel == ControlPanelKind::Geometry);

    if (viewer_) {
      PushGeometryStateToViewer();
      viewer_->SetCropOverlayVisible(geometry_active);
      viewer_->SetCropToolEnabled(geometry_active);
    }
    RefreshGeometryModeUi();
    RefreshPanelSwitchUi();
    if (pipeline_initialized_) {
      const bool force_full_frame_preview =
          previous_panel == ControlPanelKind::Geometry &&
          panel != ControlPanelKind::Geometry && force_next_full_frame_preview_;
      RequestRender(!force_full_frame_preview);
      if (force_full_frame_preview) {
        force_next_full_frame_preview_ = false;
      }
      ScheduleQualityPreviewRenderFromPipeline();
    }
  }

  void RefreshOdtMethodUi() {
    const bool aces_active = state_.odt_.method_ == ColorUtils::ODTMethod::ACES_2_0;
    const bool open_drt_active = state_.odt_.method_ == ColorUtils::ODTMethod::OPEN_DRT;

    const QString active_style = AppTheme::EditorMethodCardStyle(true);
    const QString inactive_style = AppTheme::EditorMethodCardStyle(false);

    if (odt_aces_method_card_) {
      odt_aces_method_card_->setChecked(aces_active);
      odt_aces_method_card_->setStyleSheet(aces_active ? active_style : inactive_style);
    }
    if (odt_open_drt_method_card_) {
      odt_open_drt_method_card_->setChecked(open_drt_active);
      odt_open_drt_method_card_->setStyleSheet(open_drt_active ? active_style : inactive_style);
    }
    if (odt_method_stack_) {
      odt_method_stack_->setCurrentIndex(aces_active ? 0 : 1);
    }
  }

  void RefreshOdtEncodingEotfComboFromState() {
    if (!odt_encoding_eotf_combo_) {
      return;
    }

    SanitizeOdtStateForUi(state_.odt_);
    const bool prev_sync = syncing_controls_;
    syncing_controls_    = true;

    odt_encoding_eotf_combo_->clear();
    const auto options = SupportedDisplayEotfOptions(state_.odt_.encoding_space_);
    for (const auto& option : options) {
      odt_encoding_eotf_combo_->addItem(Tr(option.label_), static_cast<int>(option.value_));
    }

    int selected_index = odt_encoding_eotf_combo_->findData(
        static_cast<int>(state_.odt_.encoding_eotf_));
    if (selected_index < 0) {
      state_.odt_.encoding_eotf_ = DefaultDisplayEotfForSpace(state_.odt_.encoding_space_);
      selected_index = odt_encoding_eotf_combo_->findData(
          static_cast<int>(state_.odt_.encoding_eotf_));
    }
    odt_encoding_eotf_combo_->setCurrentIndex(std::max(0, selected_index));

    syncing_controls_ = prev_sync;
  }

  void PromoteColorTempToCustomForEditing() {
    if (state_.color_temp_mode_ == ColorTempMode::CUSTOM) {
      return;
    }
    state_.color_temp_custom_cct_  = DisplayedColorTempCct(state_);
    state_.color_temp_custom_tint_ = DisplayedColorTempTint(state_);
    state_.color_temp_mode_        = ColorTempMode::CUSTOM;

    const bool prev_sync           = syncing_controls_;
    syncing_controls_              = true;
    if (color_temp_mode_combo_) {
      color_temp_mode_combo_->setCurrentIndex(ColorTempModeToComboIndex(state_.color_temp_mode_));
    }
    syncing_controls_ = prev_sync;
  }

  // Returns true if any resolved color temp value actually changed.
  auto RefreshColorTempRuntimeStateFromGlobalParams() -> bool {
    if (!pipeline_guard_ || !pipeline_guard_->pipeline_) {
      return false;
    }

    const auto& global = pipeline_guard_->pipeline_->GetGlobalParams();
    const float new_cct  = std::clamp(global.color_temp_resolved_cct_,
                                      static_cast<float>(kColorTempCctMin),
                                      static_cast<float>(kColorTempCctMax));
    const float new_tint = std::clamp(global.color_temp_resolved_tint_,
                                      static_cast<float>(kColorTempTintMin),
                                      static_cast<float>(kColorTempTintMax));
    const bool  new_sup  = global.color_temp_matrices_valid_;

    const bool changed = !NearlyEqual(state_.color_temp_resolved_cct_, new_cct) ||
                         !NearlyEqual(state_.color_temp_resolved_tint_, new_tint) ||
                         state_.color_temp_supported_ != new_sup;

    state_.color_temp_resolved_cct_  = new_cct;
    state_.color_temp_resolved_tint_ = new_tint;
    state_.color_temp_supported_     = new_sup;

    committed_state_.color_temp_resolved_cct_  = new_cct;
    committed_state_.color_temp_resolved_tint_ = new_tint;
    committed_state_.color_temp_supported_     = new_sup;
    if (state_.color_temp_mode_ == ColorTempMode::AS_SHOT && new_sup) {
      CacheAsShotColorTemp(new_cct, new_tint);
    }

    return changed;
  }

  void SyncColorTempControlsFromState() {
    const bool prev_sync = syncing_controls_;
    syncing_controls_    = true;

    if (color_temp_mode_combo_) {
      color_temp_mode_combo_->setCurrentIndex(ColorTempModeToComboIndex(state_.color_temp_mode_));
    }
    if (color_temp_cct_slider_) {
      color_temp_cct_slider_->setValue(ColorTempCctToSliderPos(DisplayedColorTempCct(state_)));
      color_temp_cct_slider_->setEnabled(state_.color_temp_supported_);
    }
    if (color_temp_tint_slider_) {
      color_temp_tint_slider_->setValue(
          static_cast<int>(std::lround(DisplayedColorTempTint(state_))));
      color_temp_tint_slider_->setEnabled(state_.color_temp_supported_);
    }
    if (color_temp_unsupported_label_) {
      color_temp_unsupported_label_->setVisible(!state_.color_temp_supported_);
    }

    syncing_controls_ = prev_sync;
  }

  void RefreshVersionLogSelectionStyles() {
    if (!version_log_) {
      return;
    }
    for (int i = 0; i < version_log_->count(); ++i) {
      auto* item = version_log_->item(i);
      if (!item) {
        continue;
      }
      auto* w = version_log_->itemWidget(item);
      if (!w) {
        continue;
      }
      if (auto* card = dynamic_cast<HistoryCardWidget*>(w)) {
        card->SetSelected(item->isSelected());
      }
    }
  }

  void TriggerQualityPreviewRenderFromPipeline() {
    if (quality_preview_timer_ && quality_preview_timer_->isActive()) {
      quality_preview_timer_->stop();
    }
    state_.type_ = RenderType::FULL_RES_PREVIEW;
    RequestRenderWithoutApplyingState();
    state_.type_ = RenderType::FAST_PREVIEW;
  }

  void SyncControlsFromState() {
    if (!controls_) {
      return;
    }

    syncing_controls_ = true;
    LoadActiveHlsProfile(state_);
    SanitizeOdtStateForUi(state_.odt_);
    RefreshLutBrowserUi();

    if (exposure_slider_) {
      exposure_slider_->setValue(static_cast<int>(std::lround(state_.exposure_ * 100.0f)));
    }
    if (contrast_slider_) {
      contrast_slider_->setValue(static_cast<int>(std::lround(state_.contrast_)));
    }
    if (saturation_slider_) {
      saturation_slider_->setValue(static_cast<int>(std::lround(state_.saturation_)));
    }
    if (raw_highlights_reconstruct_checkbox_) {
      raw_highlights_reconstruct_checkbox_->setChecked(state_.raw_highlights_reconstruct_);
    }
    if (lens_calib_enabled_checkbox_) {
      lens_calib_enabled_checkbox_->setChecked(state_.lens_calib_enabled_);
    }
    RefreshLensComboFromState();
    if (lift_disc_widget_) {
      lift_disc_widget_->SetPosition(state_.lift_wheel_.disc_position_);
    }
    if (gamma_disc_widget_) {
      gamma_disc_widget_->SetPosition(state_.gamma_wheel_.disc_position_);
    }
    if (gain_disc_widget_) {
      gain_disc_widget_->SetPosition(state_.gain_wheel_.disc_position_);
    }
    if (lift_master_slider_) {
      lift_master_slider_->setValue(CdlMasterToSliderUi(state_.lift_wheel_.master_offset_));
    }
    if (gamma_master_slider_) {
      gamma_master_slider_->setValue(CdlMasterToSliderUi(-state_.gamma_wheel_.master_offset_));
    }
    if (gain_master_slider_) {
      gain_master_slider_->setValue(CdlMasterToSliderUi(state_.gain_wheel_.master_offset_));
    }
    if (color_temp_mode_combo_) {
      color_temp_mode_combo_->setCurrentIndex(ColorTempModeToComboIndex(state_.color_temp_mode_));
    }
    if (color_temp_cct_slider_) {
      color_temp_cct_slider_->setValue(ColorTempCctToSliderPos(DisplayedColorTempCct(state_)));
    }
    if (color_temp_tint_slider_) {
      color_temp_tint_slider_->setValue(
          static_cast<int>(std::lround(DisplayedColorTempTint(state_))));
    }
    if (color_temp_cct_slider_) {
      color_temp_cct_slider_->setEnabled(state_.color_temp_supported_);
    }
    if (color_temp_tint_slider_) {
      color_temp_tint_slider_->setEnabled(state_.color_temp_supported_);
    }
    if (color_temp_unsupported_label_) {
      color_temp_unsupported_label_->setVisible(!state_.color_temp_supported_);
    }
    if (hls_hue_adjust_slider_) {
      hls_hue_adjust_slider_->setValue(static_cast<int>(std::lround(state_.hls_hue_adjust_)));
    }
    if (hls_lightness_adjust_slider_) {
      hls_lightness_adjust_slider_->setValue(
          static_cast<int>(std::lround(state_.hls_lightness_adjust_)));
    }
    if (hls_saturation_adjust_slider_) {
      hls_saturation_adjust_slider_->setValue(
          static_cast<int>(std::lround(state_.hls_saturation_adjust_)));
    }
    if (hls_hue_range_slider_) {
      hls_hue_range_slider_->setValue(static_cast<int>(std::lround(state_.hls_hue_range_)));
    }
    if (blacks_slider_) {
      blacks_slider_->setValue(static_cast<int>(std::lround(state_.blacks_)));
    }
    if (whites_slider_) {
      whites_slider_->setValue(static_cast<int>(std::lround(state_.whites_)));
    }
    if (shadows_slider_) {
      shadows_slider_->setValue(static_cast<int>(std::lround(state_.shadows_)));
    }
    if (highlights_slider_) {
      highlights_slider_->setValue(static_cast<int>(std::lround(state_.highlights_)));
    }
    if (sharpen_slider_) {
      sharpen_slider_->setValue(static_cast<int>(std::lround(state_.sharpen_)));
    }
    if (clarity_slider_) {
      clarity_slider_->setValue(static_cast<int>(std::lround(state_.clarity_)));
    }
    if (odt_encoding_space_combo_) {
      const int encoding_space_index = odt_encoding_space_combo_->findData(
          static_cast<int>(state_.odt_.encoding_space_));
      odt_encoding_space_combo_->setCurrentIndex(std::max(0, encoding_space_index));
    }
    RefreshOdtEncodingEotfComboFromState();
    if (odt_peak_luminance_slider_) {
      odt_peak_luminance_slider_->setValue(
          static_cast<int>(std::lround(state_.odt_.peak_luminance_)));
    }
    if (odt_aces_limiting_space_combo_) {
      const int limiting_space_index = odt_aces_limiting_space_combo_->findData(
          static_cast<int>(state_.odt_.aces_.limiting_space_));
      odt_aces_limiting_space_combo_->setCurrentIndex(std::max(0, limiting_space_index));
    }
    if (odt_open_drt_look_preset_combo_) {
      const int look_preset_index = odt_open_drt_look_preset_combo_->findData(
          static_cast<int>(state_.odt_.open_drt_.look_preset_));
      odt_open_drt_look_preset_combo_->setCurrentIndex(std::max(0, look_preset_index));
    }
    if (odt_open_drt_tonescale_preset_combo_) {
      const int tonescale_index = odt_open_drt_tonescale_preset_combo_->findData(
          static_cast<int>(state_.odt_.open_drt_.tonescale_preset_));
      odt_open_drt_tonescale_preset_combo_->setCurrentIndex(std::max(0, tonescale_index));
    }
    if (odt_open_drt_creative_white_combo_) {
      const int creative_white_index = odt_open_drt_creative_white_combo_->findData(
          static_cast<int>(state_.odt_.open_drt_.creative_white_));
      odt_open_drt_creative_white_combo_->setCurrentIndex(std::max(0, creative_white_index));
    }
    RefreshOdtMethodUi();
    if (rotate_slider_) {
      rotate_slider_->setValue(
          static_cast<int>(std::lround(state_.rotate_degrees_ * kRotationSliderScale)));
    }
    SyncGeometryCropSlidersFromState();
    SyncCropAspectControlsFromState();
    if (curve_widget_) {
      curve_widget_->SetControlPoints(state_.curve_points_);
    }
    UpdateGeometryCropRectLabel();
    RefreshGeometryModeUi();
    if (viewer_) {
      SyncViewerDisplayEncoding();
      PushGeometryStateToViewer();
      const bool geometry_active = (active_panel_ == ControlPanelKind::Geometry);
      viewer_->SetCropOverlayVisible(geometry_active);
      viewer_->SetCropToolEnabled(geometry_active);
    }
    RefreshHlsTargetUi();
    RefreshCdlOffsetLabels();

    syncing_controls_ = false;
  }

  auto ReconstructPipelineParamsForVersion(Version& version) -> std::optional<nlohmann::json> {
    return versioning::ReconstructPipelineParamsForVersion(version, history_guard_);
  }

  auto ReloadUiStateFromPipeline(bool reset_to_defaults_if_missing) -> bool {
    const bool loaded = LoadStateFromPipelineIfPresent();
    if (!loaded && !reset_to_defaults_if_missing) {
      return false;
    }
    if (!loaded) {
      state_ = AdjustmentState{};
      SanitizeOdtStateForUi(state_.odt_);
      UpdateAllCdlWheelDerivedColors(state_);
      last_submitted_color_temp_request_.reset();
    } else {
      last_submitted_color_temp_request_ = BuildColorTempRequest(state_);
    }
    committed_state_ = state_;
    SyncControlsFromState();
    TriggerQualityPreviewRenderFromPipeline();
    return true;
  }

  auto ApplyPipelineParamsToEditor(const nlohmann::json& params) -> bool {
    if (!pipeline_guard_ || !pipeline_guard_->pipeline_) {
      return false;
    }

    auto exec = pipeline_guard_->pipeline_;
    exec->ImportPipelineParams(params);
    exec->SetExecutionStages(CurrentFrameSink());
    pipeline_guard_->dirty_ = true;
    last_applied_lut_path_.clear();

    return ReloadUiStateFromPipeline(/*reset_to_defaults_if_missing=*/true);
  }

  auto ReloadEditorFromHistoryVersion(Version& version, QString* error) -> bool {
    const auto selected_params = ReconstructPipelineParamsForVersion(version);
    if (!selected_params.has_value()) {
      if (error) {
        *error = Tr("Could not reconstruct pipeline params for the selected version.");
      }
      return false;
    }

    if (!ApplyPipelineParamsToEditor(*selected_params)) {
      if (error) {
        *error = Tr("Failed to apply selected version to the editor.");
      }
      return false;
    }
    return true;
  }

  void CheckoutSelectedVersion(QListWidgetItem* item) {
    versioning::ResolvedVersionSelection selection{};
    QString                              selection_error;
    if (!versioning::ResolveSelectedVersion(item, history_guard_, &selection,
                                            &selection_error)) {
      if (!selection_error.isEmpty()) {
        QMessageBox::warning(this, Tr("History"), selection_error);
      }
      return;
    }

    QString reload_error;
    if (!selection.version || !ReloadEditorFromHistoryVersion(*selection.version, &reload_error)) {
      QMessageBox::warning(this, Tr("History"), reload_error);
      return;
    }

    working_version_ = versioning::SeedWorkingVersionFromCommit(
        element_id_, selection.version_id, pipeline_guard_,
        CurrentWorkingMode() != WorkingMode::Plain);
    UpdateVersionUi();
  }

  void UndoLastTransaction() {
    if (!pipeline_guard_ || !pipeline_guard_->pipeline_) {
      return;
    }

    const auto undo_result = versioning::UndoLastTransaction(working_version_, pipeline_guard_);
    if (undo_result.no_transaction) {
      QMessageBox::information(this, Tr("History"), Tr("No transaction to undo."));
      return;
    }
    if (!undo_result.error.isEmpty()) {
      QMessageBox::warning(this, Tr("History"), undo_result.error);
      return;
    }
    if (!undo_result.undone) {
      return;
    }
    if (!ReloadUiStateFromPipeline(/*reset_to_defaults_if_missing=*/false)) {
      QMessageBox::warning(this, Tr("History"),
                           Tr("Undo failed while reloading pipeline state."));
      return;
    }
    UpdateVersionUi();
  }

  void UpdateVersionUi() {
    const versioning::VersionUiContext ui{
        .version_status     = version_status_,
        .commit_version_btn = commit_version_btn_,
        .undo_tx_btn        = undo_tx_btn_,
        .working_mode_combo = working_mode_combo_,
        .version_log        = version_log_,
        .tx_stack           = tx_stack_,
    };
    versioning::UpdateVersionUi(ui, working_version_, history_guard_,
                                [this]() { RefreshVersionLogSelectionStyles(); });
  }

  void CommitWorkingVersion() {
    const auto commit_result = versioning::CommitWorkingVersion(
        history_service_, history_guard_, pipeline_guard_, element_id_,
        std::move(working_version_));
    if (commit_result.no_transactions) {
      QMessageBox::information(this, Tr("History"), Tr("No uncommitted transactions."));
      return;
    }
    if (!commit_result.committed_id.has_value()) {
      QMessageBox::warning(this, Tr("History"),
                           commit_result.error.isEmpty() ? Tr("Commit failed.")
                                                         : commit_result.error);
      if (commit_result.recovery_working_version.has_value()) {
        working_version_ = std::move(*commit_result.recovery_working_version);
        UpdateVersionUi();
      }
      return;
    }

    StartNewWorkingVersionFromCommit(*commit_result.committed_id);
    UpdateVersionUi();
  }

  auto CurrentWorkingMode() const -> WorkingMode {
    return versioning::IsPlainModeSelected(working_mode_combo_) ? WorkingMode::Plain
                                                                : WorkingMode::Incremental;
  }

  void StartNewWorkingVersionFromUi() {
    working_version_ = versioning::SeedWorkingVersionFromUi(
        element_id_, history_guard_, pipeline_guard_,
        CurrentWorkingMode() == WorkingMode::Plain);
    UpdateVersionUi();
  }

  void StartNewWorkingVersionFromCommit(const Hash128& committed_id) {
    working_version_ = versioning::SeedWorkingVersionFromCommit(
        element_id_, committed_id, pipeline_guard_,
        CurrentWorkingMode() != WorkingMode::Plain);
  }

  auto ReadCurrentOperatorParams(PipelineStageName stage_name, OperatorType op_type) const
      -> std::optional<nlohmann::json> {
    if (!pipeline_guard_ || !pipeline_guard_->pipeline_) {
      return std::nullopt;
    }
    return pipeline_io::ReadCurrentOperatorParams(*pipeline_guard_->pipeline_, stage_name, op_type);
  }

  std::pair<PipelineStageName, OperatorType> FieldSpec(AdjustmentField field) const {
    return pipeline_io::FieldSpec(field);
  }

  nlohmann::json ParamsForField(AdjustmentField field, const AdjustmentState& s) const {
    return pipeline_io::ParamsForField(
        field, s, (pipeline_guard_ && pipeline_guard_->pipeline_)
                      ? pipeline_guard_->pipeline_.get()
                      : nullptr);
  }

  bool FieldChanged(AdjustmentField field) const {
    return pipeline_io::FieldChanged(field, state_, committed_state_);
  }

  void CommitAdjustment(AdjustmentField field) {
    if (!FieldChanged(field) || !pipeline_guard_ || !pipeline_guard_->pipeline_) {
      // Still fulfill the "full res on release/change" behavior.
      ScheduleQualityPreviewRenderFromPipeline();
      return;
    }

    const auto [stage_name, op_type] = FieldSpec(field);
    const auto            old_params = ParamsForField(field, committed_state_);
    const auto            new_params = ParamsForField(field, state_);

    auto                  exec       = pipeline_guard_->pipeline_;
    auto&                 stage      = exec->GetStage(stage_name);
    const auto            op         = stage.GetOperator(op_type);
    const TransactionType tx_type =
        (op.has_value() && op.value() != nullptr) ? TransactionType::_EDIT : TransactionType::_ADD;

    EditTransaction tx{tx_type, op_type, stage_name, new_params};
    tx.SetLastOperatorParams(old_params);
    (void)tx.ApplyTransaction(*exec);

    working_version_.AppendEditTransaction(std::move(tx));
    pipeline_guard_->dirty_ = true;

    CopyFieldState(field, state_, committed_state_);
    if (field == AdjustmentField::CropRotate) {
      force_next_full_frame_preview_ = true;
    }
    UpdateVersionUi();

    ScheduleQualityPreviewRenderFromPipeline();
  }

  bool LoadStateFromPipelineIfPresent() {
    auto exec = pipeline_guard_ ? pipeline_guard_->pipeline_ : nullptr;
    if (!exec) {
      return false;
    }
    auto [loaded_state, has_loaded_any] = pipeline_io::LoadStateFromPipeline(*exec, state_);
    if (!has_loaded_any) {
      return false;
    }
    SanitizeOdtStateForUi(loaded_state.odt_);
    if (loaded_state.color_temp_mode_ == ColorTempMode::AS_SHOT &&
        loaded_state.color_temp_supported_) {
      CacheAsShotColorTemp(loaded_state.color_temp_resolved_cct_,
                           loaded_state.color_temp_resolved_tint_);
    }
    state_ = loaded_state;
    last_submitted_color_temp_request_ = BuildColorTempRequest(state_);
    return true;
  }

  void SetupPipeline() {
    base_task_.input_             = controllers::LoadImageInputBuffer(image_pool_, image_id_);
    base_task_.pipeline_executor_ = pipeline_guard_->pipeline_;

    auto           exec           = pipeline_guard_->pipeline_;
    controllers::EnsureLoadingOperatorDefaults(exec);
    if (!scope_analyzer_) {
      scope_analyzer_ = CreateDefaultScopeAnalyzer();
    }
    if (!final_display_frame_tap_) {
      final_display_frame_tap_ =
          std::make_unique<FinalDisplayFrameTapSink>(viewer_, scope_analyzer_);
      if (scope_panel_) {
        scope_panel_->SetAnalyzer(scope_analyzer_);
        scope_panel_->SetRequestChangedCallback([this](const ScopeRequest& request) {
          if (final_display_frame_tap_) {
            final_display_frame_tap_->SetScopeRequest(request);
          }
        });
        final_display_frame_tap_->SetScopeRequest(scope_panel_->CurrentRequest());
      }
    }
    controllers::AttachExecutionStages(exec, CurrentFrameSink());
    SyncViewerDisplayEncoding();

    // Inject pre-extracted raw metadata from the Image so downstream operators
    // (ColorTemp, LensCalib) resolve eagerly.
    try {
      auto img = image_pool_->Read<std::shared_ptr<Image>>(
          image_id_, [](const std::shared_ptr<Image>& i) { return i; });
      if (img && img->HasRawColorContext()) {
        exec->InjectRawMetadata(img->GetRawColorContext());
        WarmAsShotColorTempCacheFromRawMetadata();
      }
    } catch (...) {
      // Non-fatal: metadata injection is best-effort.
    }

    // Cached pipelines can clear transient GPU resources when returned to the service.
    // PipelineMgmtService now resyncs global params on load, so we no longer need a
    // per-dialog LMT rebind hack here.
    last_applied_lut_path_.clear();
  }

  void ApplyStateToPipeline(const AdjustmentState& render_state) {
    AdjustmentState render_state_sanitized = render_state;
    SanitizeOdtStateForUi(render_state_sanitized.odt_);

    auto  exec          = pipeline_guard_->pipeline_;
    auto& global_params = exec->GetGlobalParams();
    auto& loading       = exec->GetStage(PipelineStageName::Image_Loading);
    auto& geometry      = exec->GetStage(PipelineStageName::Geometry_Adjustment);
    auto& to_ws         = exec->GetStage(PipelineStageName::To_WorkingSpace);
    auto& output        = exec->GetStage(PipelineStageName::Output_Transform);

    loading.SetOperator(OperatorType::RAW_DECODE,
                        ParamsForField(AdjustmentField::RawDecode, render_state_sanitized));
    loading.SetOperator(OperatorType::LENS_CALIBRATION,
                        ParamsForField(AdjustmentField::LensCalib, render_state_sanitized),
                        global_params);
    loading.EnableOperator(OperatorType::LENS_CALIBRATION,
                           render_state_sanitized.lens_calib_enabled_, global_params);

    const auto color_temp_request = BuildColorTempRequest(render_state_sanitized);
    const bool color_temp_missing = !to_ws.GetOperator(OperatorType::COLOR_TEMP).has_value();
    if (color_temp_missing || !last_submitted_color_temp_request_.has_value() ||
        !ColorTempRequestEqual(*last_submitted_color_temp_request_, color_temp_request)) {
      to_ws.SetOperator(OperatorType::COLOR_TEMP,
                        ParamsForField(AdjustmentField::ColorTemp, render_state_sanitized),
                        global_params);
      to_ws.EnableOperator(OperatorType::COLOR_TEMP, true, global_params);
      last_submitted_color_temp_request_ = color_temp_request;
    } else {
      to_ws.EnableOperator(OperatorType::COLOR_TEMP, true, global_params);
    }

    output.SetOperator(OperatorType::ODT,
                       ParamsForField(AdjustmentField::Odt, render_state_sanitized),
                       global_params);
    output.EnableOperator(OperatorType::ODT, true, global_params);

    // Geometry editing is overlay-only. While the geometry panel is active,
    // render the full pre-geometry frame so recropping can always expand back
    // to the original image bounds.
    nlohmann::json crop_rotate_params;
    bool           apply_crop = committed_state_.crop_enabled_;
    if (active_panel_ == ControlPanelKind::Geometry) {
      crop_rotate_params = pipeline_defaults::MakeDefaultCropRotateParams();
      crop_rotate_params["crop_rotate"]["enable_crop"]    = false;
      crop_rotate_params["crop_rotate"]["expand_to_fit"]  = committed_state_.crop_expand_to_fit_;
      apply_crop = false;
    } else {
      crop_rotate_params = ParamsForField(AdjustmentField::CropRotate, committed_state_);
    }

    crop_rotate_params["crop_rotate"]["enable_crop"] = apply_crop;
    const bool geometry_enabled = crop_rotate_params["crop_rotate"].value("enabled", false);
    geometry.SetOperator(OperatorType::CROP_ROTATE, crop_rotate_params, global_params);
    geometry.EnableOperator(OperatorType::CROP_ROTATE, geometry_enabled, global_params);

    auto& basic         = exec->GetStage(PipelineStageName::Basic_Adjustment);
    basic.SetOperator(OperatorType::EXPOSURE, {{"exposure", render_state_sanitized.exposure_}},
                      global_params);
    basic.SetOperator(OperatorType::CONTRAST, {{"contrast", render_state_sanitized.contrast_}},
                      global_params);
    basic.SetOperator(OperatorType::BLACK, {{"black", render_state_sanitized.blacks_}},
                      global_params);
    basic.SetOperator(OperatorType::WHITE, {{"white", render_state_sanitized.whites_}},
                      global_params);
    basic.SetOperator(OperatorType::SHADOWS, {{"shadows", render_state_sanitized.shadows_}},
                      global_params);
    basic.SetOperator(OperatorType::HIGHLIGHTS,
                      {{"highlights", render_state_sanitized.highlights_}}, global_params);
    basic.SetOperator(OperatorType::CURVE,
                      CurveControlPointsToParams(render_state_sanitized.curve_points_),
                      global_params);

    auto& color = exec->GetStage(PipelineStageName::Color_Adjustment);
    color.SetOperator(OperatorType::SATURATION,
                      {{"saturation", render_state_sanitized.saturation_}}, global_params);
    color.EnableOperator(OperatorType::TINT, false, global_params);
    color.SetOperator(OperatorType::COLOR_WHEEL,
                      ParamsForField(AdjustmentField::ColorWheel, render_state_sanitized),
                      global_params);
    color.EnableOperator(OperatorType::COLOR_WHEEL, true, global_params);
    color.SetOperator(OperatorType::HLS,
                      ParamsForField(AdjustmentField::Hls, render_state_sanitized), global_params);
    color.EnableOperator(OperatorType::HLS, true, global_params);

    // LUT (LMT): rebind only when the path changes. The operator's SetGlobalParams now
    // derives lmt_enabled_/dirty state from the path, and PipelineMgmtService resyncs on load.
    if (render_state_sanitized.lut_path_ != last_applied_lut_path_) {
      color.SetOperator(OperatorType::LMT, {{"ocio_lmt", render_state_sanitized.lut_path_}},
                        global_params);
      last_applied_lut_path_ = render_state_sanitized.lut_path_;
    }

    auto& detail = exec->GetStage(PipelineStageName::Detail_Adjustment);
    detail.SetOperator(
        OperatorType::SHARPEN, {{"sharpen", {{"offset", render_state_sanitized.sharpen_}}}},
        global_params);
    detail.SetOperator(OperatorType::CLARITY, {{"clarity", render_state_sanitized.clarity_}},
                       global_params);
  }

  static constexpr std::chrono::milliseconds kFastPreviewMinSubmitInterval =
      controllers::render::kFastPreviewMinSubmitInterval;
  static constexpr std::chrono::milliseconds kQualityPreviewDebounceInterval =
      controllers::render::kQualityPreviewDebounceInterval;

  void EnsureQualityPreviewTimer() {
    if (quality_preview_timer_) {
      return;
    }
    quality_preview_timer_ = new QTimer(this);
    quality_preview_timer_->setSingleShot(true);
    QObject::connect(quality_preview_timer_, &QTimer::timeout, this,
                     [this]() { TriggerQualityPreviewRenderFromPipeline(); });
  }

  void ScheduleQualityPreviewRenderFromPipeline() {
    EnsureQualityPreviewTimer();
    quality_preview_timer_->start(static_cast<int>(kQualityPreviewDebounceInterval.count()));
  }

  auto CanSubmitFastPreviewNow() const -> bool {
    return controllers::render::CanSubmitFastPreviewNow(last_fast_preview_submit_time_,
                                                        std::chrono::steady_clock::now());
  }

  void EnsureFastPreviewSubmitTimer() {
    if (fast_preview_submit_timer_) {
      return;
    }
    fast_preview_submit_timer_ = new QTimer(this);
    fast_preview_submit_timer_->setSingleShot(true);
    QObject::connect(fast_preview_submit_timer_, &QTimer::timeout, this, [this]() {
      if (!inflight_) {
        StartNext();
      }
    });
  }

  void ArmFastPreviewSubmitTimer() {
    EnsureFastPreviewSubmitTimer();
    const int delay_ms = controllers::render::ComputeFastPreviewDelayMs(
        last_fast_preview_submit_time_, std::chrono::steady_clock::now());

    if (!fast_preview_submit_timer_->isActive()) {
      fast_preview_submit_timer_->start(delay_ms);
      return;
    }

    const int current_remaining = fast_preview_submit_timer_->remainingTime();
    if (current_remaining < 0 || delay_ms < current_remaining) {
      fast_preview_submit_timer_->start(delay_ms);
    }
  }

  void EnqueueRenderRequest(const AdjustmentState& snapshot, bool apply_state,
                            bool use_viewport_region = true) {
    PendingRenderRequest request{snapshot, apply_state, use_viewport_region};

    if (snapshot.type_ == RenderType::FAST_PREVIEW) {
      // Industry pattern for interactive rendering:
      // coalesce rapid slider updates and keep only the newest fast preview.
      if (quality_preview_timer_ && quality_preview_timer_->isActive()) {
        quality_preview_timer_->stop();
      }
      pending_fast_preview_request_ = std::move(request);
    } else {
      // Keep quality requests ordered and drop stale fast previews.
      pending_quality_render_requests_.push_back(std::move(request));
      pending_fast_preview_request_.reset();
      if (fast_preview_submit_timer_ && fast_preview_submit_timer_->isActive()) {
        fast_preview_submit_timer_->stop();
      }
    }

    if (!inflight_) {
      StartNext();
    }
  }

  void RequestRender(bool use_viewport_region = true) {
    EnqueueRenderRequest(state_, true, use_viewport_region);
  }

  void RequestRenderWithoutApplyingState(bool use_viewport_region = true) {
    EnqueueRenderRequest(state_, false, use_viewport_region);
  }

  void EnsurePollTimer() {
    if (poll_timer_) {
      return;
    }
    poll_timer_ = new QTimer(this);
    poll_timer_->setInterval(4);
    QObject::connect(poll_timer_, &QTimer::timeout, this, [this]() { PollInflight(); });
  }

  void PollInflight() {
    if (!inflight_future_.has_value()) {
      if (poll_timer_ && poll_timer_->isActive() && !inflight_) {
        poll_timer_->stop();
      }
      return;
    }

    if (inflight_future_->wait_for(0ms) != std::future_status::ready) {
      return;
    }

    try {
      (void)inflight_future_->get();
    } catch (...) {
    }
    inflight_future_.reset();
    OnRenderFinished();
  }

  void StartNext() {
    if (inflight_) {
      return;
    }

    std::optional<PendingRenderRequest> request;
    if (!pending_quality_render_requests_.empty()) {
      request = pending_quality_render_requests_.front();
      pending_quality_render_requests_.pop_front();
    } else if (pending_fast_preview_request_.has_value()) {
      if (!CanSubmitFastPreviewNow()) {
        ArmFastPreviewSubmitTimer();
        return;
      }
      request = pending_fast_preview_request_;
      pending_fast_preview_request_.reset();
      last_fast_preview_submit_time_ = std::chrono::steady_clock::now();
      if (fast_preview_submit_timer_ && fast_preview_submit_timer_->isActive()) {
        fast_preview_submit_timer_->stop();
      }
    }

    if (!request.has_value()) {
      return;
    }
    const PendingRenderRequest next_request = *request;

    if (spinner_) {
      spinner_->Start();
    }

    if (next_request.apply_state_) {
      ApplyStateToPipeline(next_request.state_);
      pipeline_guard_->dirty_ = true;
    }

    PipelineTask task                       = base_task_;
    task.options_.render_desc_.render_type_ = next_request.state_.type_;
    task.options_.render_desc_.use_viewport_region_ = next_request.use_viewport_region_;
    task.options_.is_callback_              = false;
    task.options_.is_seq_callback_          = false;
    task.options_.is_blocking_              = true;

    auto promise = std::make_shared<std::promise<std::shared_ptr<ImageBuffer>>>();
    auto fut     = promise->get_future();
    task.result_ = promise;

    inflight_    = true;
    scheduler_->ScheduleTask(std::move(task));

    inflight_future_ = std::move(fut);
    EnsurePollTimer();
    if (poll_timer_ && !poll_timer_->isActive()) {
      poll_timer_->start();
    }
  }

  void OnRenderFinished() {
    inflight_ = false;

    if (spinner_) {
      spinner_->Stop();
    }

    if (RefreshColorTempRuntimeStateFromGlobalParams()) {
      SyncColorTempControlsFromState();
    }

    if (!pending_quality_render_requests_.empty() || pending_fast_preview_request_.has_value()) {
      StartNext();
    } else if (poll_timer_ && poll_timer_->isActive()) {
      poll_timer_->stop();
    }
  }

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
  QWidget*                                                 versioning_panel_content_ = nullptr;
  QWidget*                                                 versioning_collapsed_nav_ = nullptr;
  QWidget*                                                 versioning_collapsed_gap_ = nullptr;
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
  QPushButton*                                             versioning_nav_btn_           = nullptr;
  std::unique_ptr<ShortcutRegistry>                        shortcut_registry_{};
  QComboBox*                                               working_mode_combo_           = nullptr;
  QPushButton*                                             new_working_btn_              = nullptr;
  QListWidget*                                             version_log_                  = nullptr;
  QListWidget*                                             tx_stack_                     = nullptr;
  QTimer*                                                  poll_timer_                   = nullptr;
  std::optional<std::future<std::shared_ptr<ImageBuffer>>> inflight_future_{};

  controllers::LutController                               lut_controller_{};
  LensCatalog                                              lens_catalog_{};

  std::string                                              last_applied_lut_path_{};
  std::optional<ColorTempRequestSnapshot>                  last_submitted_color_temp_request_{};
  AdjustmentState                                          state_{};
  AdjustmentState                                          committed_state_{};
  Version                                                  working_version_{};
  std::deque<PendingRenderRequest>                         pending_quality_render_requests_{};
  std::optional<PendingRenderRequest>                      pending_fast_preview_request_{};
  ControlPanelKind                                         active_panel_               = ControlPanelKind::Tone;
  bool                                                     pipeline_initialized_       = false;
  bool                                                     inflight_                   = false;
  QTimer*                                                  quality_preview_timer_      = nullptr;
  QTimer*                                                  fast_preview_submit_timer_  = nullptr;
  std::chrono::steady_clock::time_point                    last_fast_preview_submit_time_{};
  bool                                                     syncing_controls_           = false;
  bool                                                     force_next_full_frame_preview_ = false;
  bool                                                     versioning_collapsed_       = true;
  bool                                                     versioning_nav_hovered_     = false;
  qreal                                                    versioning_panel_progress_  = 0.0;
  int                                                      versioning_collapsed_gap_base_width_ = 0;
  int                                                      versioning_expanded_width_  = 224;
  float                                                    last_known_as_shot_cct_     = 6500.0f;
  float                                                    last_known_as_shot_tint_    = 0.0f;
  bool                                                     has_last_known_as_shot_color_temp_ = false;
  std::shared_ptr<IScopeAnalyzer>                          scope_analyzer_         = {};
  std::unique_ptr<FinalDisplayFrameTapSink>                final_display_frame_tap_ = {};
  std::map<QSlider*, std::function<void()>>                slider_reset_callbacks_{};
  std::function<void()>                                    curve_reset_callback_{};
};
}  // namespace

auto RunEditorDialog(std::shared_ptr<ImagePoolService>       image_pool,
                     std::shared_ptr<PipelineGuard>          pipeline_guard,
                     std::shared_ptr<EditHistoryMgmtService> history_service,
                     std::shared_ptr<EditHistoryGuard> history_guard, sl_element_id_t element_id,
                     image_id_t image_id, QWidget* parent) -> bool {
  EditorDialog dlg(std::move(image_pool), std::move(pipeline_guard), std::move(history_service),
                   std::move(history_guard), element_id, image_id, parent);
  dlg.exec();
  return true;
}

}  // namespace puerhlab::ui
