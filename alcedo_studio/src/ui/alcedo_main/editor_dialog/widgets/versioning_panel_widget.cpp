//  Copyright 2026 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#include "ui/alcedo_main/editor_dialog/widgets/versioning_panel_widget.hpp"

#include <QAbstractItemView>
#include <QByteArray>
#include <QColor>
#include <QEasingCurve>
#include <QEvent>
#include <QFile>
#include <QFrame>
#include <QHBoxLayout>
#include <QIcon>
#include <QIODevice>
#include <QListWidgetItem>
#include <QPainter>
#include <QPainterPath>
#include <QPixmap>
#include <QRect>
#include <QRegion>
#include <QSize>
#include <QSizePolicy>
#include <QStyle>
#include <QSvgRenderer>
#include <QTimer>
#include <Qt>
#include <algorithm>
#include <cmath>
#include <utility>

#include "ui/alcedo_main/app_theme.hpp"
#include "ui/alcedo_main/editor_dialog/widgets/history_cards.hpp"
#include "ui/alcedo_main/i18n.hpp"

namespace alcedo::ui {
namespace {

constexpr int kRailButtonSize       = 46;
constexpr int kExpandedMinWidth     = 320;
constexpr int kExpandedMaxWidth     = 420;
constexpr int kExpandedMinHeight    = 300;
constexpr int kExpandedMaxHeight    = 460;
constexpr int kAnimationMs          = 250;
constexpr int kEditorOuterMargin    = 14;
const QSize   kRailIconSize(18, 18);

constexpr char kLocalizedTextProperty[]      = "puerhlabI18nText";
constexpr char kLocalizedTextUpperProperty[] = "puerhlabI18nTextUpper";
constexpr char kLocalizedToolTipProperty[]   = "puerhlabI18nToolTip";

void SetLocalizedText(QObject* object, const char* source, bool uppercase = false) {
  if (!object || source == nullptr) {
    return;
  }
  object->setProperty(kLocalizedTextProperty, source);
  object->setProperty(kLocalizedTextUpperProperty, uppercase);
  QString text = Tr(source);
  if (uppercase) {
    text = text.toUpper();
  }
  if (auto* label = qobject_cast<QLabel*>(object)) {
    label->setText(text);
  } else if (auto* button = qobject_cast<QPushButton*>(object)) {
    button->setText(text);
  }
}

void SetLocalizedToolTip(QWidget* widget, const char* source) {
  if (!widget || source == nullptr) {
    return;
  }
  widget->setProperty(kLocalizedToolTipProperty, source);
  widget->setToolTip(Tr(source));
  widget->setAccessibleName(Tr(source));
}

auto NewLocalizedLabel(const char* source, QWidget* parent) -> QLabel* {
  auto* label = new QLabel(parent);
  SetLocalizedText(label, source);
  return label;
}

auto NewLocalizedButton(const char* source, QWidget* parent) -> QPushButton* {
  auto* button = new QPushButton(parent);
  SetLocalizedText(button, source);
  return button;
}

auto RenderRailToggleIcon(const QString& resource_path, const QColor& color, const QSize& size,
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

  const qreal scale =
      std::max<qreal>(2.0, std::ceil(std::max<qreal>(1.0, device_pixel_ratio) * 2.0));
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

}  // namespace

VersioningPanelWidget::VersioningPanelWidget(QWidget* parent) : QWidget(parent) {}

void VersioningPanelWidget::Configure(QWidget* flyout_parent, Callbacks callbacks) {
  flyout_parent_ = flyout_parent;
  callbacks_     = std::move(callbacks);
}

auto VersioningPanelWidget::MakeUiContext() const -> versioning::VersionUiContext {
  return versioning::VersionUiContext{
      .version_status     = version_status_,
      .commit_version_btn = commit_version_btn_,
      .undo_tx_btn        = undo_tx_btn_,
      .working_mode_combo = working_mode_combo_,
      .version_log        = version_log_,
      .tx_stack           = tx_stack_,
  };
}

auto VersioningPanelWidget::CurrentWorkingMode() const -> WorkingMode {
  return IsPlainWorkingMode() ? WorkingMode::Plain : WorkingMode::Incremental;
}

auto VersioningPanelWidget::IsPlainWorkingMode() const -> bool {
  return versioning::IsPlainModeSelected(working_mode_combo_);
}

auto VersioningPanelWidget::IsFlyoutVisible() const -> bool {
  return flyout_ != nullptr && flyout_->isVisible();
}

void VersioningPanelWidget::Build() {
  if (built_) {
    return;
  }
  setFixedWidth(kCollapsedWidth);
  setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Expanding);
  auto* host_layout = new QHBoxLayout(this);
  host_layout->setContentsMargins(0, 0, 0, 0);
  host_layout->setSpacing(0);

  BuildRail();
  host_layout->addWidget(rail_, 0);

  BuildFlyout();

  built_ = true;
  // Apply initial collapsed state without animation.
  SetCollapsed(true, /*animate=*/false);
}

void VersioningPanelWidget::BuildRail() {
  const auto&   theme           = AppTheme::Instance();
  const QColor  rail_tile       = theme.bgPanelColor();
  const QString rail_style      = QStringLiteral(
      "#EditorVersioningRail {"
      "  background: %1;"
      "  border: none;"
      "  border-radius: 14px;"
      "}"
      "#EditorVersioningRail QPushButton {"
      "  min-width: 40px;"
      "  min-height: 40px;"
      "  border-radius: 10px;"
      "  border: none;"
      "  background: %2;"
      "  padding: 0px;"
      "}"
      "#EditorVersioningRail QPushButton:hover {"
      "  background: %3;"
      "}"
      "#EditorVersioningRail QPushButton:pressed {"
      "  background: %4;"
      "}"
      "#EditorVersioningRail QPushButton[versioningActive=\"true\"] {"
      "  background: %5;"
      "}"
      "#EditorVersioningRail QPushButton[versioningActive=\"true\"]:hover {"
      "  background: %6;"
      "}")
      .arg(theme.glassPanelColor().name(QColor::HexArgb),
           QColor(rail_tile.red(), rail_tile.green(), rail_tile.blue(), 210)
               .name(QColor::HexArgb),
           QColor(rail_tile.red(), rail_tile.green(), rail_tile.blue(), 238)
               .name(QColor::HexArgb),
           QColor(rail_tile.red(), rail_tile.green(), rail_tile.blue(), 170)
               .name(QColor::HexArgb),
           QColor(rail_tile.red(), rail_tile.green(), rail_tile.blue(), 255)
               .name(QColor::HexArgb),
           QColor(rail_tile.red(), rail_tile.green(), rail_tile.blue(), 255)
               .name(QColor::HexArgb));

  rail_ = new QWidget(this);
  rail_->setObjectName("EditorVersioningRail");
  rail_->setAttribute(Qt::WA_StyledBackground, true);
  rail_->setStyleSheet(rail_style);
  rail_->setFixedWidth(kCollapsedWidth);
  auto* rail_layout = new QVBoxLayout(rail_);
  rail_layout->setContentsMargins(7, 12, 7, 12);
  rail_layout->setSpacing(8);
  rail_layout->setAlignment(Qt::AlignTop);

  const auto configure_rail_button = [](QPushButton* button) {
    if (!button) {
      return;
    }
    button->setCursor(Qt::PointingHandCursor);
    button->setAutoDefault(false);
    button->setDefault(false);
    button->setFocusPolicy(Qt::NoFocus);
    button->setProperty("versioningActive", false);
    button->setFixedSize(kRailButtonSize, kRailButtonSize);
    button->setIconSize(kRailIconSize);
  };

  history_btn_ = new QPushButton(rail_);
  configure_rail_button(history_btn_);
  rail_layout->addWidget(history_btn_, 0, Qt::AlignTop | Qt::AlignHCenter);

  versions_btn_ = new QPushButton(rail_);
  configure_rail_button(versions_btn_);
  rail_layout->addWidget(versions_btn_, 0, Qt::AlignTop | Qt::AlignHCenter);

  rail_layout->addSpacing(6);
  auto* nav_divider = new QFrame(rail_);
  nav_divider->setFrameShape(QFrame::HLine);
  nav_divider->setFixedWidth(kRailButtonSize - 10);
  nav_divider->setStyleSheet(
      QStringLiteral("color: %1; background: %1; border: none;")
          .arg(QColor(AppTheme::Instance().dividerColor().red(),
                      AppTheme::Instance().dividerColor().green(),
                      AppTheme::Instance().dividerColor().blue(), 96)
                   .name(QColor::HexArgb)));
  rail_layout->addWidget(nav_divider, 0, Qt::AlignHCenter);
  rail_layout->addStretch();

  QObject::connect(history_btn_, &QPushButton::clicked, this,
                   [this]() { HandleHistoryButtonClicked(); });
  QObject::connect(versions_btn_, &QPushButton::clicked, this,
                   [this]() { HandleVersionsButtonClicked(); });
}

void VersioningPanelWidget::BuildFlyout() {
  const auto&   theme              = AppTheme::Instance();
  const QString version_panel_style = QStringLiteral(
      "#EditorVersioningPanel {"
      "  background: %1;"
      "  border: none;"
      "  border-radius: 14px;"
      "}"
      "#EditorVersioningPanel QFrame#EditorSection {"
      "  background: transparent;"
      "  border: none;"
      "  border-radius: 0px;"
      "}"
      "#EditorVersioningPanel QLabel#EditorSectionTitle {"
      "  color: %2;"
      "}"
      "#EditorVersioningPanel QLabel#EditorSectionSub {"
      "  color: %3;"
      "}")
      .arg(QColor(theme.bgDeepColor().red(), theme.bgDeepColor().green(),
                  theme.bgDeepColor().blue(), 246)
               .name(QColor::HexArgb),
           theme.textColor().name(QColor::HexRgb),
           theme.textMutedColor().name(QColor::HexRgb));

  QWidget* const flyout_parent = flyout_parent_ != nullptr ? flyout_parent_ : this;
  flyout_                      = new QWidget(flyout_parent);
  flyout_->setObjectName("EditorVersioningFlyout");
  flyout_->setAttribute(Qt::WA_StyledBackground, false);
  flyout_->setAttribute(Qt::WA_TranslucentBackground, true);
  flyout_->hide();
  flyout_->installEventFilter(this);

  auto* flyout_layout = new QVBoxLayout(flyout_);
  flyout_layout->setContentsMargins(0, 0, 0, 0);
  flyout_layout->setSpacing(0);

  flyout_root_ = new QWidget(flyout_);
  flyout_root_->setObjectName("EditorVersioningPanel");
  flyout_root_->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
  flyout_root_->setAttribute(Qt::WA_StyledBackground, true);
  flyout_root_->setStyleSheet(version_panel_style);
  flyout_layout->addWidget(flyout_root_, 1);

  flyout_opacity_effect_ = new QGraphicsOpacityEffect(flyout_root_);
  flyout_root_->setGraphicsEffect(flyout_opacity_effect_);
  flyout_opacity_effect_->setOpacity(0.0);

  auto* root_outer_layout = new QVBoxLayout(flyout_root_);
  root_outer_layout->setContentsMargins(0, 0, 0, 0);
  root_outer_layout->setSpacing(0);
  auto* content = new QWidget(flyout_root_);
  content->setAttribute(Qt::WA_StyledBackground, false);
  content->setStyleSheet("background: transparent;");
  shared_layout_ = new QVBoxLayout(content);
  shared_layout_->setContentsMargins(20, 20, 20, 20);
  shared_layout_->setSpacing(14);
  root_outer_layout->addWidget(content, 1);

  // Pages stack and inner content (was BuildVersioningPanel).
  const auto& version_theme = theme;

  const QString version_primary_btn_style =
      QStringLiteral("QPushButton {"
                     "  color: %1;"
                     "  background: %2;"
                     "  border: none;"
                     "  border-radius: 10px;"
                     "  font-size: 13px;"
                     "  font-weight: 600;"
                     "  padding: 8px 14px;"
                     "}"
                     "QPushButton:hover { background: %3; }"
                     "QPushButton:pressed { background: %4; }"
                     "QPushButton:disabled { color: %5; background: %6; }")
          .arg(version_theme.bgCanvasColor().name(QColor::HexRgb),
               version_theme.accentColor().name(QColor::HexArgb),
               QColor(version_theme.accentColor().red(), version_theme.accentColor().green(),
                      version_theme.accentColor().blue(), 232)
                   .name(QColor::HexArgb),
               version_theme.accentSecondaryColor().name(QColor::HexArgb),
               version_theme.textMutedColor().name(QColor::HexRgb),
               QColor(version_theme.bgPanelColor().red(), version_theme.bgPanelColor().green(),
                      version_theme.bgPanelColor().blue(), 160)
                   .name(QColor::HexArgb));

  const QString version_secondary_btn_style =
      QStringLiteral("QPushButton {"
                     "  color: %1;"
                     "  background: %2;"
                     "  border: none;"
                     "  border-radius: 10px;"
                     "  font-size: 13px;"
                     "  font-weight: 600;"
                     "  padding: 8px 14px;"
                     "}"
                     "QPushButton:hover { background: %3; }"
                     "QPushButton:pressed { background: %4; }"
                     "QPushButton:disabled { color: %5; background: %2; }")
          .arg(version_theme.textColor().name(QColor::HexRgb),
               QColor(version_theme.bgPanelColor().red(), version_theme.bgPanelColor().green(),
                      version_theme.bgPanelColor().blue(), 200)
                   .name(QColor::HexArgb),
               QColor(version_theme.bgPanelColor().red(), version_theme.bgPanelColor().green(),
                      version_theme.bgPanelColor().blue(), 238)
                   .name(QColor::HexArgb),
               QColor(version_theme.bgPanelColor().red(), version_theme.bgPanelColor().green(),
                      version_theme.bgPanelColor().blue(), 160)
                   .name(QColor::HexArgb),
               version_theme.textMutedColor().name(QColor::HexRgb));

  const QString compact_combo_font_style =
      QStringLiteral("QComboBox {"
                     "  font-size: 12px;"
                     "  background: %1;"
                     "  color: %2;"
                     "  border: none;"
                     "  border-radius: 8px;"
                     "  padding: 4px 10px;"
                     "}"
                     "QComboBox::drop-down { border: 0px; width: 20px; }"
                     "QComboBox QAbstractItemView {"
                     "  background: %1;"
                     "  color: %2;"
                     "  border: none;"
                     "  selection-background-color: %3;"
                     "  selection-color: %4;"
                     "}")
          .arg(QColor(version_theme.bgPanelColor().red(), version_theme.bgPanelColor().green(),
                      version_theme.bgPanelColor().blue(), 210)
                   .name(QColor::HexArgb),
               version_theme.textColor().name(QColor::HexRgb),
               QColor(version_theme.accentSecondaryColor().red(),
                      version_theme.accentSecondaryColor().green(),
                      version_theme.accentSecondaryColor().blue(), 224)
                   .name(QColor::HexArgb),
               version_theme.bgCanvasColor().name(QColor::HexRgb));

  const QString divider_style =
      QStringLiteral("QFrame#VersioningDivider {"
                     "  background: %1;"
                     "  border: none;"
                     "  min-height: 1px;"
                     "  max-height: 1px;"
                     "}")
          .arg(QColor(version_theme.dividerColor().red(), version_theme.dividerColor().green(),
                      version_theme.dividerColor().blue(), 72)
                   .name(QColor::HexArgb));

  const QString embedded_list_style =
      QStringLiteral("QListWidget {"
                     "  background: transparent;"
                     "  border: none;"
                     "  border-radius: 0px;"
                     "  padding: 0px;"
                     "}"
                     "QListWidget::item {"
                     "  background: transparent;"
                     "  border: none;"
                     "  padding: 0px;"
                     "}"
                     "QListWidget::item:selected { background: transparent; }");

  shared_layout_->setContentsMargins(0, 0, 0, 0);
  shared_layout_->setSpacing(0);

  pages_stack_ = new QStackedWidget(content);
  pages_stack_->setObjectName("EditorVersioningPages");
  pages_stack_->setStyleSheet(QStringLiteral(
      "QStackedWidget#EditorVersioningPages { background: transparent; }"));
  shared_layout_->addWidget(pages_stack_, 1);

  const auto build_header_row = [&](const char* title_source, const char* pill_source) {
    auto* row = new QWidget(content);
    auto* layout = new QHBoxLayout(row);
    layout->setContentsMargins(0, 0, 0, 0);
    layout->setSpacing(8);

    auto* title_label = NewLocalizedLabel(title_source, row);
    title_label->setStyleSheet(
        QStringLiteral(
            "QLabel { color: %1; font-size: 15px; font-weight: 700; background: transparent; }")
            .arg(version_theme.textColor().name(QColor::HexRgb)));
    AppTheme::MarkFontRole(title_label, AppTheme::FontRole::UiBodyStrong);
    layout->addWidget(title_label, 0);
    layout->addStretch(1);

    if (pill_source != nullptr && pill_source[0] != '\0') {
      auto* pill = NewLocalizedLabel(pill_source, row);
      pill->setStyleSheet(QStringLiteral("QLabel {"
                                         "  color: %1;"
                                         "  background: %2;"
                                         "  border: none;"
                                         "  border-radius: 10px;"
                                         "  font-size: 11px;"
                                         "  font-weight: 600;"
                                         "  padding: 4px 12px;"
                                         "}")
                              .arg(version_theme.textColor().name(QColor::HexRgb),
                                   QColor(version_theme.bgPanelColor().red(),
                                          version_theme.bgPanelColor().green(),
                                          version_theme.bgPanelColor().blue(), 220)
                                       .name(QColor::HexArgb)));
      AppTheme::MarkFontRole(pill, AppTheme::FontRole::UiCaption);
      layout->addWidget(pill, 0);
    }

    return row;
  };

  const auto build_divider = [&]() {
    auto* divider = new QFrame(content);
    divider->setObjectName("VersioningDivider");
    divider->setStyleSheet(divider_style);
    return divider;
  };

  const auto build_section_divider = [&](const char* label_source) {
    auto* row    = new QWidget(content);
    auto* layout = new QHBoxLayout(row);
    layout->setContentsMargins(0, 4, 0, 4);
    layout->setSpacing(10);

    const QColor line_color = QColor(version_theme.dividerColor().red(),
                                     version_theme.dividerColor().green(),
                                     version_theme.dividerColor().blue(), 92);
    const QString line_style =
        QStringLiteral(
            "QFrame { background: %1; border: none; min-height: 1px; max-height: 1px; }")
            .arg(line_color.name(QColor::HexArgb));

    auto* line_left = new QFrame(row);
    line_left->setStyleSheet(line_style);
    line_left->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
    line_left->setFixedHeight(1);

    auto* label = NewLocalizedLabel(label_source, row);
    label->setStyleSheet(QStringLiteral("QLabel {"
                                        "  color: %1;"
                                        "  background: transparent;"
                                        "  font-size: 10px;"
                                        "  font-weight: 700;"
                                        "  letter-spacing: 2px;"
                                        "}")
                             .arg(version_theme.textMutedColor().name(QColor::HexRgb)));
    AppTheme::MarkFontRole(label, AppTheme::FontRole::UiCaption);

    auto* line_right = new QFrame(row);
    line_right->setStyleSheet(line_style);
    line_right->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
    line_right->setFixedHeight(1);

    layout->addWidget(line_left, 1);
    layout->addWidget(label, 0);
    layout->addWidget(line_right, 1);
    return row;
  };

  // History page.
  {
    auto* history_page = new QWidget(pages_stack_);
    history_page->setAttribute(Qt::WA_StyledBackground, false);
    history_page->setStyleSheet("background: transparent;");
    auto* page_layout = new QVBoxLayout(history_page);
    page_layout->setContentsMargins(18, 18, 18, 18);
    page_layout->setSpacing(10);

    page_layout->addWidget(build_header_row("Edit History", "Uncommitted"), 0);
    page_layout->addWidget(build_divider(), 0);

    tx_stack_ = new QListWidget(history_page);
    tx_stack_->setSelectionMode(QAbstractItemView::NoSelection);
    tx_stack_->setSpacing(4);
    tx_stack_->setVerticalScrollMode(QAbstractItemView::ScrollPerPixel);
    tx_stack_->setMinimumHeight(170);
    tx_stack_->setStyleSheet(embedded_list_style);
    tx_stack_->setFrameShape(QFrame::NoFrame);
    page_layout->addWidget(tx_stack_, 1);

    page_layout->addWidget(build_section_divider("COMMITTED STATE"), 0);

    auto* baseline_row    = new QWidget(history_page);
    auto* baseline_layout = new QHBoxLayout(baseline_row);
    baseline_layout->setContentsMargins(4, 4, 4, 4);
    baseline_layout->setSpacing(10);

    auto* baseline_label = NewLocalizedLabel("Baseline", baseline_row);
    baseline_label->setStyleSheet(
        QStringLiteral(
            "QLabel { color: %1; background: transparent; font-size: 13px; font-weight: 600; }")
            .arg(version_theme.textColor().name(QColor::HexRgb)));
    AppTheme::MarkFontRole(baseline_label, AppTheme::FontRole::UiBody);

    version_status_ = new QLabel(baseline_row);
    version_status_->setStyleSheet(
        QStringLiteral(
            "QLabel { color: %1; background: transparent; font-size: 11px; font-weight: 500; }")
            .arg(version_theme.textMutedColor().name(QColor::HexRgb)));
    AppTheme::MarkFontRole(version_status_, AppTheme::FontRole::DataCaption);

    baseline_layout->addWidget(baseline_label, 0);
    baseline_layout->addStretch(1);
    baseline_layout->addWidget(version_status_, 0);
    page_layout->addWidget(baseline_row, 0);

    auto* action_row    = new QWidget(history_page);
    auto* action_layout = new QHBoxLayout(action_row);
    action_layout->setContentsMargins(0, 8, 0, 0);
    action_layout->setSpacing(10);

    undo_tx_btn_ = NewLocalizedButton("Undo Last", action_row);
    undo_tx_btn_->setMinimumHeight(38);
    undo_tx_btn_->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
    undo_tx_btn_->setStyleSheet(version_secondary_btn_style);

    commit_version_btn_ = NewLocalizedButton("Commit All", action_row);
    commit_version_btn_->setMinimumHeight(38);
    commit_version_btn_->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
    commit_version_btn_->setStyleSheet(version_primary_btn_style);

    action_layout->addWidget(undo_tx_btn_, 1);
    action_layout->addWidget(commit_version_btn_, 1);
    page_layout->addWidget(action_row, 0);

    QObject::connect(undo_tx_btn_, &QPushButton::clicked, this, [this]() {
      if (callbacks_.undo_last_transaction) {
        callbacks_.undo_last_transaction();
      }
    });
    QObject::connect(commit_version_btn_, &QPushButton::clicked, this, [this]() {
      if (callbacks_.commit_working_version) {
        callbacks_.commit_working_version();
      }
    });

    pages_stack_->addWidget(history_page);
  }

  // Versions page.
  {
    auto* versions_page = new QWidget(pages_stack_);
    versions_page->setAttribute(Qt::WA_StyledBackground, false);
    versions_page->setStyleSheet("background: transparent;");
    auto* page_layout = new QVBoxLayout(versions_page);
    page_layout->setContentsMargins(18, 18, 18, 18);
    page_layout->setSpacing(10);

    page_layout->addWidget(build_header_row("Version Tree", ""), 0);
    page_layout->addWidget(build_divider(), 0);

    auto* mode_row    = new QWidget(versions_page);
    auto* mode_layout = new QHBoxLayout(mode_row);
    mode_layout->setContentsMargins(0, 2, 0, 2);
    mode_layout->setSpacing(10);

    auto* mode_label = NewLocalizedLabel("Working mode", mode_row);
    mode_label->setStyleSheet(
        QStringLiteral(
            "QLabel { color: %1; background: transparent; font-size: 11px; font-weight: 600; "
            "letter-spacing: 1px; }")
            .arg(version_theme.textMutedColor().name(QColor::HexRgb)));
    AppTheme::MarkFontRole(mode_label, AppTheme::FontRole::UiCaption);

    working_mode_combo_ = new QComboBox(mode_row);
    working_mode_combo_->addItem(Tr("Plain"), static_cast<int>(WorkingMode::Plain));
    working_mode_combo_->addItem(Tr("Incremental"), static_cast<int>(WorkingMode::Incremental));
    working_mode_combo_->setMinimumHeight(32);
    working_mode_combo_->setMinimumWidth(118);
    working_mode_combo_->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
    working_mode_combo_->setStyleSheet(compact_combo_font_style);

    new_working_btn_ = NewLocalizedButton("New Working", mode_row);
    new_working_btn_->setMinimumHeight(32);
    new_working_btn_->setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Fixed);
    new_working_btn_->setStyleSheet(version_secondary_btn_style);

    mode_layout->addWidget(mode_label, 0);
    mode_layout->addWidget(working_mode_combo_, 1);
    mode_layout->addWidget(new_working_btn_, 0);
    page_layout->addWidget(mode_row, 0);

    page_layout->addWidget(build_section_divider("COMMITTED STATE"), 0);

    version_log_ = new QListWidget(versions_page);
    version_log_->setSelectionMode(QAbstractItemView::SingleSelection);
    version_log_->setSpacing(2);
    version_log_->setVerticalScrollMode(QAbstractItemView::ScrollPerPixel);
    version_log_->setMinimumHeight(170);
    version_log_->setStyleSheet(embedded_list_style);
    version_log_->setFrameShape(QFrame::NoFrame);
    page_layout->addWidget(version_log_, 1);

    QObject::connect(new_working_btn_, &QPushButton::clicked, this, [this]() {
      if (callbacks_.start_new_working_version) {
        callbacks_.start_new_working_version();
      }
    });
    QObject::connect(working_mode_combo_, QOverload<int>::of(&QComboBox::currentIndexChanged), this,
                     [this](int) {
                       if (callbacks_.on_working_mode_changed) {
                         callbacks_.on_working_mode_changed();
                       }
                     });
    QObject::connect(version_log_, &QListWidget::itemSelectionChanged, this,
                     [this]() { RefreshVersionLogSelectionStyles(); });
    QObject::connect(version_log_, &QListWidget::itemClicked, this,
                     [this](QListWidgetItem* item) {
                       if (!item) {
                         return;
                       }
                       const QString version_id = item->data(Qt::UserRole).toString();
                       QTimer::singleShot(0, this, [this, version_id]() {
                         if (callbacks_.checkout_version_by_id) {
                           callbacks_.checkout_version_by_id(version_id);
                         }
                       });
                     });

    pages_stack_->addWidget(versions_page);
  }

  pages_stack_->setCurrentIndex(static_cast<int>(active_page_));
}

void VersioningPanelWidget::HandleHistoryButtonClicked() {
  const bool same_page_open = !collapsed_ && active_page_ == FlyoutPage::History;
  active_page_              = FlyoutPage::History;
  SetCollapsed(same_page_open ? true : false, /*animate=*/false);
}

void VersioningPanelWidget::HandleVersionsButtonClicked() {
  const bool same_page_open = !collapsed_ && active_page_ == FlyoutPage::Versions;
  active_page_              = FlyoutPage::Versions;
  SetCollapsed(same_page_open ? true : false, /*animate=*/false);
}

void VersioningPanelWidget::SetCollapsed(bool collapsed, bool animate) {
  const qreal target_progress = collapsed ? 0.0 : 1.0;
  if (!animate || std::abs(progress_ - target_progress) < 0.001) {
    if (flyout_anim_) {
      flyout_anim_->stop();
    }
    progress_   = target_progress;
    collapsed_  = collapsed;
    if (!collapsed && pages_stack_) {
      pages_stack_->setCurrentIndex(static_cast<int>(active_page_));
    }
    RefreshCollapseUi();
    return;
  }

  if (!collapsed && pages_stack_) {
    pages_stack_->setCurrentIndex(static_cast<int>(active_page_));
  }
  if (!collapsed && flyout_ && !flyout_->isVisible()) {
    flyout_->show();
    flyout_->raise();
    RepositionFlyout();
    QTimer::singleShot(0, this, [this]() {
      if (!flyout_ || progress_ <= 0.0) {
        return;
      }
      RepositionFlyout();
    });
  }

  if (!flyout_anim_) {
    flyout_anim_ = new QVariantAnimation(this);
    flyout_anim_->setDuration(kAnimationMs);
    flyout_anim_->setEasingCurve(QEasingCurve::OutCubic);
    QObject::connect(flyout_anim_, &QVariantAnimation::valueChanged, this,
                     [this](const QVariant& value) {
                       progress_ = value.toReal();
                       RefreshCollapseUi();
                     });
    QObject::connect(flyout_anim_, &QVariantAnimation::finished, this, [this]() {
      if (!flyout_anim_) {
        return;
      }
      progress_  = flyout_anim_->endValue().toReal();
      collapsed_ = progress_ < 0.5;
      RefreshCollapseUi();
    });
  }

  flyout_anim_->stop();
  flyout_anim_->setStartValue(progress_);
  flyout_anim_->setEndValue(target_progress);
  flyout_anim_->start();
}

void VersioningPanelWidget::RefreshCollapseUi() {
  if (!rail_) {
    return;
  }

  const auto& theme    = AppTheme::Instance();
  const qreal progress = std::clamp(progress_, static_cast<qreal>(0.0), static_cast<qreal>(1.0));
  if (pages_stack_) {
    pages_stack_->setCurrentIndex(static_cast<int>(active_page_));
  }

  if (flyout_) {
    if (progress > 0.0) {
      if (!flyout_->isVisible()) {
        flyout_->show();
        flyout_->raise();
        RepositionFlyout();
        QTimer::singleShot(0, this, [this]() {
          if (!flyout_ || progress_ <= 0.0) {
            return;
          }
          RepositionFlyout();
        });
      }
    } else if (flyout_->isVisible()) {
      flyout_->hide();
    }
  }

  if (flyout_opacity_effect_) {
    flyout_opacity_effect_->setOpacity(progress);
  }

  const bool panel_expanded = progress >= 0.5;
  collapsed_                = !panel_expanded;

  const auto update_nav_button = [&](QPushButton* button, const QString& icon_path,
                                     const QString& label, FlyoutPage page) {
    if (!button) {
      return;
    }
    const bool active = panel_expanded && active_page_ == page;
    button->setProperty("versioningActive", active);
    button->style()->unpolish(button);
    button->style()->polish(button);
    button->setIcon(RenderRailToggleIcon(icon_path,
                                         active ? theme.textColor() : theme.textMutedColor(),
                                         kRailIconSize, button->devicePixelRatioF()));
    const QString tooltip = active ? Tr("Hide %1").arg(label) : Tr("Show %1").arg(label);
    button->setToolTip(tooltip);
    button->setAccessibleName(tooltip);
  };

  update_nav_button(history_btn_,
                    QStringLiteral(":/history_icons/git-commit-horizontal.svg"),
                    Tr("Edit History"), FlyoutPage::History);
  update_nav_button(versions_btn_, QStringLiteral(":/panel_icons/git-branch.svg"),
                    Tr("Version Tree"), FlyoutPage::Versions);
}

void VersioningPanelWidget::RepositionFlyout() {
  if (!flyout_) {
    return;
  }
  if (!callbacks_.viewer_geometry) {
    return;
  }

  flyout_->ensurePolished();
  if (auto* layout = flyout_->layout()) {
    layout->activate();
  }
  if (pages_stack_) {
    pages_stack_->ensurePolished();
    if (auto* layout = pages_stack_->layout()) {
      layout->activate();
    }
    if (auto* page = pages_stack_->currentWidget()) {
      page->ensurePolished();
      if (auto* layout = page->layout()) {
        layout->activate();
      }
    }
  }
  if (flyout_root_) {
    flyout_root_->ensurePolished();
    if (auto* layout = flyout_root_->layout()) {
      layout->activate();
    }
  }

  const QRect viewer_rect = callbacks_.viewer_geometry().adjusted(14, 14, -14, -14);
  const int   gap         = 14;
  int         flyout_x    = viewer_rect.left() + 4;
  flyout_x                = std::max(flyout_x, geometry().right() + gap);
  flyout_x                = std::max(flyout_x, kEditorOuterMargin + 4);

  const int available_width = std::max(0, viewer_rect.right() - flyout_x - 12);
  const int desired_width =
      std::clamp(static_cast<int>(std::lround(static_cast<double>(viewer_rect.width()) * 0.30)),
                 kExpandedMinWidth, kExpandedMaxWidth);
  const int flyout_y = viewer_rect.top() + 2;
  const int flyout_w =
      std::clamp(desired_width, std::min(kExpandedMinWidth, std::max(220, available_width)),
                 std::max(220, available_width));
  int content_height = kExpandedMinHeight;
  if (pages_stack_) {
    if (auto* page = pages_stack_->currentWidget()) {
      page->ensurePolished();
      content_height = std::max(content_height, page->sizeHint().height() + 32);
    } else {
      content_height = std::max(content_height, pages_stack_->sizeHint().height() + 32);
    }
  } else if (flyout_root_) {
    content_height = std::max(content_height, flyout_root_->sizeHint().height());
  }
  const int flyout_h =
      std::clamp(content_height, kExpandedMinHeight,
                 std::min(kExpandedMaxHeight, std::max(220, viewer_rect.height() - 20)));

  flyout_->setGeometry(flyout_x, flyout_y, flyout_w, flyout_h);
  if (auto* layout = flyout_->layout()) {
    layout->activate();
  }
  if (flyout_root_) {
    if (auto* layout = flyout_root_->layout()) {
      layout->activate();
    }
    const QRect rect = flyout_root_->rect();
    if (rect.isValid() && rect.width() > 0 && rect.height() > 0) {
      QPainterPath path;
      path.addRoundedRect(QRectF(rect), 14.0, 14.0);
      flyout_root_->setMask(QRegion(path.toFillPolygon().toPolygon()));
    } else {
      flyout_root_->clearMask();
    }
  }
}

void VersioningPanelWidget::OnDialogResized() {
  if (flyout_ && flyout_->isVisible()) {
    RepositionFlyout();
  }
}

void VersioningPanelWidget::RefreshVersionLogSelectionStyles() {
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

void VersioningPanelWidget::RetranslateUi() {
  if (undo_tx_btn_) {
    undo_tx_btn_->setText(Tr("Undo Last"));
  }
  if (commit_version_btn_) {
    commit_version_btn_->setText(Tr("Commit All"));
  }
  if (new_working_btn_) {
    new_working_btn_->setText(Tr("New Working"));
  }
  if (working_mode_combo_) {
    const QSignalBlocker block(working_mode_combo_);
    const int            current_value = working_mode_combo_->currentData().toInt();
    working_mode_combo_->clear();
    working_mode_combo_->addItem(Tr("Plain"), static_cast<int>(WorkingMode::Plain));
    working_mode_combo_->addItem(Tr("Incremental"), static_cast<int>(WorkingMode::Incremental));
    const int index = working_mode_combo_->findData(current_value);
    working_mode_combo_->setCurrentIndex(std::max(0, index));
  }
  RefreshCollapseUi();
}

bool VersioningPanelWidget::eventFilter(QObject* obj, QEvent* event) {
  if (obj == flyout_ && event && event->type() == QEvent::Hide) {
    if (!collapsed_) {
      if (flyout_anim_) {
        flyout_anim_->stop();
      }
      progress_  = 0.0;
      collapsed_ = true;
      if (flyout_opacity_effect_) {
        flyout_opacity_effect_->setOpacity(0.0);
      }
      RefreshCollapseUi();
    }
  }
  return QWidget::eventFilter(obj, event);
}

}  // namespace alcedo::ui
