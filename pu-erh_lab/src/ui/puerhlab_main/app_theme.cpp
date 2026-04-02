//  Copyright 2026 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#include "ui/puerhlab_main/app_theme.hpp"

#include <QApplication>
#include <QCheckBox>
#include <QComboBox>
#include <QCoreApplication>
#include <QFontDatabase>
#include <QListWidget>
#include <QPushButton>
#include <QSpinBox>
#include <QVariantMap>
#include <QWidget>

#include <algorithm>

namespace puerhlab::ui {
namespace {

constexpr char kThemeFontRoleProperty[] = "puerhlabFontRole";

struct ThemeColors {
  QColor tone_gold;
  QColor tone_wine;
  QColor tone_steel;
  QColor tone_graphite;
  QColor tone_mist;
  QColor bg_canvas;
  QColor bg_deep;
  QColor bg_base;
  QColor bg_panel;
  QColor text;
  QColor text_muted;
  QColor accent_secondary;
  QColor danger_tint;
  QColor selected_tint;
  QColor hover;
  QColor divider;
  QColor glass_panel;
  QColor glass_stroke;
  QColor overlay;
  int panel_radius;
};

// Theme 0: Pu-erh (current warm theme)
auto MakePuerhTheme() -> ThemeColors {
  return ThemeColors{
      .tone_gold = QColor(0xB9, 0x8A, 0x4A),
      .tone_wine = QColor(0x7A, 0x3E, 0x47),
      .tone_steel = QColor(0x46, 0x51, 0x5E),
      .tone_graphite = QColor(0x15, 0x1A, 0x20),
      .tone_mist = QColor(0xE8, 0xE1, 0xD6),
      .bg_canvas = QColor(0x11, 0x14, 0x19),
      .bg_deep = QColor(0x15, 0x1A, 0x20),
      .bg_base = QColor(0x1A, 0x20, 0x27),
      .bg_panel = QColor(0x1D, 0x24, 0x2C),
      .text = QColor(0xE8, 0xE1, 0xD6),
      .text_muted = QColor(0xA7, 0x9F, 0x93),
      .accent_secondary = QColor(0xD2, 0xA5, 0x66),
      .danger_tint = QColor(122, 62, 71, 92),
      .selected_tint = QColor(185, 138, 74, 46),
      .hover = QColor(0x25, 0x2D, 0x36),
      .divider = QColor(233, 227, 216, 20),
      .glass_panel = QColor(29, 36, 44, 224),
      .glass_stroke = QColor(233, 227, 216, 28),
      .overlay = QColor(0x11, 0x14, 0x19, 0xC0),
      .panel_radius = 10,
  };
}

// Theme 1: Classic (previous neutral theme)
auto MakeClassicTheme() -> ThemeColors {
  return ThemeColors{
      .tone_gold = QColor(0xFC, 0xC7, 0x04),
      .tone_wine = QColor(0x8A, 0x05, 0x26),
      .tone_steel = QColor(0x4A, 0x4A, 0x4A),
      .tone_graphite = QColor(0x1A, 0x1A, 0x1A),
      .tone_mist = QColor(0xE6, 0xE6, 0xE6),
      .bg_canvas = QColor(0x11, 0x11, 0x11),
      .bg_deep = QColor(0x14, 0x14, 0x14),
      .bg_base = QColor(0x1F, 0x1F, 0x1F),
      .bg_panel = QColor(0x2B, 0x2B, 0x2B),
      .text = QColor(0xD0, 0xD0, 0xD0),
      .text_muted = QColor(0x88, 0x88, 0x88),
      .accent_secondary = QColor(0xFC, 0xC7, 0x04),
      .danger_tint = QColor(138, 5, 38, 82),
      .selected_tint = QColor(252, 199, 4, 46),
      .hover = QColor(0x33, 0x33, 0x33),
      .divider = QColor(230, 230, 230, 20),
      .glass_panel = QColor(43, 43, 43, 224),
      .glass_stroke = QColor(230, 230, 230, 28),
      .overlay = QColor(0x12, 0x12, 0x12, 0xC0),
      .panel_radius = 8,
  };
}

auto GetTheme(int index) -> const ThemeColors& {
  static const ThemeColors kPuerhTheme = MakePuerhTheme();
  static const ThemeColors kClassicTheme = MakeClassicTheme();
  return index == 1 ? kClassicTheme : kPuerhTheme;
}

struct FontFamilies {
  QString ui_latin = QStringLiteral("Inter");
  QString ui_zh = QStringLiteral("Noto Sans SC");
  QString ui_override;
  QString effective_language_code = QStringLiteral("en");
  QString data = QStringLiteral("IBM Plex Sans");
  QString mono;
};

auto FontState() -> FontFamilies& {
  static FontFamilies families;
  return families;
}

auto FontsRegisteredFlag() -> bool& {
  static bool loaded = false;
  return loaded;
}

auto IsChineseLanguageCode(const QString& code) -> bool {
  return code.startsWith(QStringLiteral("zh"), Qt::CaseInsensitive);
}

auto ActiveUiFamily(const FontFamilies& families) -> const QString& {
  if (!families.ui_override.isEmpty()) {
    return families.ui_override;
  }
  if (IsChineseLanguageCode(families.effective_language_code)) {
    return families.ui_zh;
  }
  return families.ui_latin;
}

void RefreshTopLevelWidgetFonts() {
  const auto widgets = QApplication::topLevelWidgets();
  for (QWidget* widget : widgets) {
    AppTheme::ApplyFontsRecursively(widget);
  }
}

auto RegisterFontResource(const QString& path, const QString& preferred_family) -> QString {
  const int font_id = QFontDatabase::addApplicationFont(path);
  if (font_id < 0) {
    return preferred_family;
  }

  const QStringList families = QFontDatabase::applicationFontFamilies(font_id);
  if (families.isEmpty()) {
    return preferred_family;
  }

  if (preferred_family.isEmpty()) {
    return families.front();
  }

  for (const QString& family : families) {
    if (family.compare(preferred_family, Qt::CaseInsensitive) == 0 ||
        family.startsWith(preferred_family, Qt::CaseInsensitive)) {
      return family;
    }
  }
  return families.front();
}

auto MakeFont(const QString& family, qreal point_size, QFont::Weight weight, bool italic = false)
    -> QFont {
  QFont font(family);
  font.setPointSizeF(point_size);
  font.setWeight(weight);
  font.setItalic(italic);
  font.setStyleStrategy(QFont::PreferAntialias);
  return font;
}

auto Hex(const QColor& color) -> QString {
  return color.name(QColor::HexRgb);
}

auto Rgba(const QColor& color) -> QString {
  return color.name(QColor::HexArgb);
}

auto WithAlpha(const QColor& color, int alpha) -> QColor {
  QColor tinted(color);
  tinted.setAlpha(std::clamp(alpha, 0, 255));
  return tinted;
}

}  // namespace

AppTheme::AppTheme(QObject* parent) : QObject(parent) {}

auto AppTheme::Instance() -> AppTheme& {
  static AppTheme instance;
  return instance;
}

void AppTheme::RegisterFonts() {
  if (FontsRegisteredFlag()) {
    return;
  }

  auto& families = FontState();
  families.ui_latin =
      RegisterFontResource(QStringLiteral(":/fonts/main_Inter.ttf"), QStringLiteral("Inter"));
  RegisterFontResource(QStringLiteral(":/fonts/main_Inter_italic.ttf"), families.ui_latin);
  families.ui_zh = RegisterFontResource(QStringLiteral(":/fonts/main_NotoSans_zh.ttf"),
                                        QStringLiteral("Noto Sans SC"));
  families.data = RegisterFontResource(QStringLiteral(":/fonts/main_IBM.ttf"),
                                       QStringLiteral("IBM Plex Sans"));
  families.mono = QFontDatabase::systemFont(QFontDatabase::FixedFont).family();

  FontsRegisteredFlag() = true;
}

void AppTheme::SetEffectiveLanguageCode(const QString& code) {
  RegisterFonts();

  auto& families = FontState();
  const QString previous_family = ActiveUiFamily(families);
  const QString normalized_code =
      code.trimmed().isEmpty() ? QStringLiteral("en") : code.trimmed();
  families.effective_language_code = normalized_code;

  if (ActiveUiFamily(families) == previous_family) {
    return;
  }

  RefreshTopLevelWidgetFonts();
  emit Instance().UiFontFamilyChanged();
}

auto AppTheme::TryRegisterUiFontOverride(const QString& path) -> bool {
  if (path.isEmpty()) {
    return false;
  }

  RegisterFonts();

  const int font_id = QFontDatabase::addApplicationFont(path);
  if (font_id < 0) {
    return false;
  }

  const QStringList families = QFontDatabase::applicationFontFamilies(font_id);
  if (families.isEmpty()) {
    return false;
  }

  auto& state = FontState();
  const QString previous_family = ActiveUiFamily(state);
  state.ui_override = families.front();
  if (ActiveUiFamily(state) != previous_family) {
    RefreshTopLevelWidgetFonts();
    emit Instance().UiFontFamilyChanged();
  }
  return true;
}

void AppTheme::ApplyApplicationFont(QApplication& app) {
  RegisterFonts();

  QFont app_font = app.font();
  app_font.setFamily(ActiveUiFamily(FontState()));
  app_font.setStyleStrategy(QFont::PreferAntialias);
  app.setFont(app_font);
  RefreshTopLevelWidgetFonts();
}

void AppTheme::ApplyApplicationFont() {
  if (auto* app = qobject_cast<QApplication*>(QCoreApplication::instance())) {
    ApplyApplicationFont(*app);
  }
}

auto AppTheme::Font(FontRole role) -> QFont {
  RegisterFonts();
  const auto& families = FontState();
  const QString ui_family = ActiveUiFamily(families);

  switch (role) {
    case FontRole::UiBody:
      return MakeFont(ui_family, 11.0, QFont::Medium);
    case FontRole::UiBodyStrong:
      return MakeFont(ui_family, 11.0, QFont::DemiBold);
    case FontRole::UiCaption:
      return MakeFont(ui_family, 10.0, QFont::Medium);
    case FontRole::UiCaptionStrong:
      return MakeFont(ui_family, 10.0, QFont::DemiBold);
    case FontRole::UiTitle:
      return MakeFont(ui_family, 11.0, QFont::DemiBold);
    case FontRole::UiHeadline:
      return MakeFont(ui_family, 14.0, QFont::Bold);
    case FontRole::UiOverline: {
      QFont font = MakeFont(ui_family, 9.0, QFont::DemiBold);
      font.setCapitalization(QFont::AllUppercase);
      font.setLetterSpacing(QFont::AbsoluteSpacing, 0.8);
      return font;
    }
    case FontRole::UiHint:
      return MakeFont(ui_family, 9.0, QFont::Medium);
    case FontRole::DataBody:
      return MakeFont(families.data, 11.0, QFont::Medium);
    case FontRole::DataBodyStrong:
      return MakeFont(families.data, 11.0, QFont::DemiBold);
    case FontRole::DataCaption:
      return MakeFont(families.data, 10.0, QFont::Medium);
    case FontRole::DataNumeric:
      return MakeFont(families.data, 12.0, QFont::DemiBold);
    case FontRole::DataOverlay:
      return MakeFont(families.data, 10.0, QFont::DemiBold);
    case FontRole::MonoBody:
      return MakeFont(families.mono, 10.0, QFont::DemiBold);
    case FontRole::MonoCaption:
      return MakeFont(families.mono, 9.0, QFont::Normal);
  }

  return MakeFont(ui_family, 11.0, QFont::Normal);
}

void AppTheme::ApplyFont(QWidget* widget, FontRole role) {
  if (!widget) {
    return;
  }
  widget->setFont(Font(role));
}

void AppTheme::MarkFontRole(QObject* object, FontRole role) {
  if (!object) {
    return;
  }
  object->setProperty(kThemeFontRoleProperty, static_cast<int>(role));
  if (auto* widget = qobject_cast<QWidget*>(object)) {
    ApplyFont(widget, role);
  }
}

void AppTheme::ApplyFontsRecursively(QWidget* root) {
  if (!root) {
    return;
  }

  ApplyFont(root, ResolveRole(root));

  const auto widgets = root->findChildren<QWidget*>();
  for (QWidget* widget : widgets) {
    ApplyFont(widget, ResolveRole(widget));
  }
}

auto AppTheme::EditorLabelStyle(const QColor& color) -> QString {
  return QStringLiteral("QLabel { color: %1; }").arg(color.name(QColor::HexRgb));
}

auto AppTheme::EditorPrimaryButtonStyle(bool include_disabled) -> QString {
  const auto&  theme         = AppTheme::Instance();
  const QColor accent        = theme.accentColor();
  const QColor accent_hover  = theme.accentSecondaryColor();
  const QColor dark_text     = theme.bgCanvasColor();
  const QColor disabled_text = theme.textMutedColor();
  const QColor disabled_bg   = theme.bgBaseColor();
  QString style =
      QStringLiteral("QPushButton {"
                     "  color: %1;"
                     "  background: %2;"
                     "  border: none;"
                     "  border-radius: 10px;"
                     "  padding: 0 14px;"
                     "}"
                     "QPushButton:hover {"
                     "  background: %3;"
                     "}"
                     "QPushButton:pressed {"
                     "  background: %2;"
                     "}")
          .arg(Hex(dark_text), Hex(accent), Hex(accent_hover));
  if (include_disabled) {
    style += QStringLiteral("QPushButton:disabled {"
                            "  color: %1;"
                            "  background: %2;"
                            "}")
                 .arg(Hex(disabled_text), Hex(disabled_bg));
  }
  return style;
}

auto AppTheme::EditorSecondaryButtonStyle() -> QString {
  const auto&  theme   = AppTheme::Instance();
  const QColor text    = theme.textColor();
  const QColor bg      = theme.bgPanelColor();
  const QColor hover   = theme.hoverColor();
  const QColor border  = theme.glassStrokeColor();
  return QStringLiteral("QPushButton {"
                        "  color: %1;"
                        "  background: %2;"
                        "  border: 1px solid %3;"
                        "  border-radius: 10px;"
                        "  padding: 0 14px;"
                        "}"
                        "QPushButton:hover {"
                        "  background: %4;"
                        "  border-color: %5;"
                        "}"
                        "QPushButton:pressed {"
                        "  background: %6;"
                        "}")
      .arg(Hex(text), Rgba(WithAlpha(bg, 224)), Rgba(border), Rgba(WithAlpha(hover, 245)),
           Rgba(WithAlpha(border, 176)), Rgba(WithAlpha(hover, 255)));
}

auto AppTheme::EditorPanelToggleStyle(bool active, bool is_first, bool is_last) -> QString {
  const auto&  theme               = AppTheme::Instance();
  const QString top_left_radius    = is_first ? QStringLiteral("10px") : QStringLiteral("0px");
  const QString bottom_left_radius = is_first ? QStringLiteral("10px") : QStringLiteral("0px");
  const QString top_right_radius   = is_last ? QStringLiteral("10px") : QStringLiteral("0px");
  const QString bottom_right_radius = is_last ? QStringLiteral("10px") : QStringLiteral("0px");

  if (active) {
    return QStringLiteral("QPushButton {"
                          "  color: %1;"
                          "  background: %2;"
                          "  border: 1px solid %3;"
                          "  padding: 0px;"
                          "  border-top-left-radius: %4;"
                          "  border-bottom-left-radius: %5;"
                          "  border-top-right-radius: %6;"
                          "  border-bottom-right-radius: %7;"
                          "}"
                          "QPushButton:hover {"
                          "  background: %8;"
                          "}")
        .arg(Hex(theme.bgCanvasColor()), Rgba(WithAlpha(theme.accentColor(), 224)),
             Rgba(WithAlpha(theme.accentSecondaryColor(), 112)), top_left_radius,
             bottom_left_radius, top_right_radius, bottom_right_radius,
             Rgba(WithAlpha(theme.accentSecondaryColor(), 255)));
  }

  return QStringLiteral("QPushButton {"
                        "  color: %1;"
                        "  background: transparent;"
                        "  border: 1px solid transparent;"
                        "  padding: 0px;"
                        "  border-top-left-radius: %2;"
                        "  border-bottom-left-radius: %3;"
                        "  border-top-right-radius: %4;"
                        "  border-bottom-right-radius: %5;"
                        "}"
                        "QPushButton:hover {"
                        "  background: %6;"
                        "}"
                        "QPushButton:pressed {"
                        "  background: %7;"
                        "}")
      .arg(Hex(theme.textColor()), top_left_radius, bottom_left_radius, top_right_radius,
           bottom_right_radius, Rgba(WithAlpha(theme.hoverColor(), 210)),
           Rgba(WithAlpha(theme.hoverColor(), 255)));
}

auto AppTheme::EditorMethodCardStyle(bool active) -> QString {
  const auto&  theme   = AppTheme::Instance();
  const QColor accent  = theme.accentColor();
  const QColor bg      = theme.bgBaseColor();
  const QColor hover   = theme.hoverColor();
  const QColor muted   = theme.textMutedColor();
  const QColor text    = theme.textColor();
  const QColor divider = theme.dividerColor();
  if (active) {
    return QStringLiteral("QPushButton {"
                          "  color: %1;"
                          "  background: %2;"
                          "  border: 1px solid %3;"
                          "  border-radius: 12px;"
                          "  padding: 16px 20px;"
                          "  text-align: left;"
                          "}"
                          "QPushButton:hover {"
                          "  background: %4;"
                          "}")
        .arg(Hex(accent), Rgba(WithAlpha(bg, 240)), Rgba(WithAlpha(accent, 136)),
             Rgba(WithAlpha(hover, 255)));
  }

  return QStringLiteral("QPushButton {"
                        "  color: %1;"
                        "  background: %2;"
                        "  border: 1px solid %3;"
                        "  border-radius: 12px;"
                        "  padding: 16px 20px;"
                        "  text-align: left;"
                        "}"
                        "QPushButton:hover {"
                        "  border: 1px solid %4;"
                        "  background: %5;"
                        "  color: %6;"
                        "}")
      .arg(Hex(muted), Rgba(WithAlpha(bg, 224)), Rgba(divider),
           Rgba(WithAlpha(theme.glassStrokeColor(), 176)), Rgba(WithAlpha(hover, 245)),
           Hex(text));
}

auto AppTheme::EditorComboBoxStyle() -> QString {
  const auto&  theme  = AppTheme::Instance();
  const QColor bg     = theme.bgBaseColor();
  const QColor text   = theme.textColor();
  const QColor accent = theme.accentColor();
  const QColor hover  = theme.hoverColor();
  const QColor border = theme.glassStrokeColor();
  return QStringLiteral("QComboBox {"
                        "  background: %1;"
                        "  color: %2;"
                        "  border: 1px solid %3;"
                        "  border-radius: 10px;"
                        "  padding: 4px 8px;"
                        "}"
                        "QComboBox:hover {"
                        "  border-color: %4;"
                        "}"
                        "QComboBox::drop-down {"
                        "  border: 0px;"
                        "  width: 24px;"
                        "}"
                        "QComboBox QAbstractItemView {"
                        "  background: %1;"
                        "  color: %2;"
                        "  border: 1px solid %3;"
                        "  selection-background-color: %5;"
                        "  selection-color: %6;"
                        "}"
                        "QComboBox QAbstractItemView::item:hover {"
                        "  background: %7;"
                        "  color: %2;"
                        "}"
                        "QComboBox QAbstractItemView::item:selected {"
                        "  background: %5;"
                        "  color: %6;"
                        "}")
      .arg(Rgba(WithAlpha(bg, 240)), Hex(text), Rgba(border), Rgba(WithAlpha(border, 196)),
           Hex(accent), Hex(theme.bgCanvasColor()), Rgba(WithAlpha(hover, 255)));
}

auto AppTheme::EditorSpinBoxStyle() -> QString {
  const auto&  theme  = AppTheme::Instance();
  const QColor bg     = theme.bgBaseColor();
  const QColor text   = theme.textColor();
  const QColor border = theme.glassStrokeColor();
  const QColor accent = theme.accentColor();
  return QStringLiteral("QSpinBox {"
                        "  background: %1;"
                        "  color: %2;"
                        "  border: 1px solid %3;"
                        "  border-radius: 8px;"
                        "  padding: 4px 8px;"
                        "}"
                        "QSpinBox:hover {"
                        "  border-color: %4;"
                        "}"
                        "QSpinBox:focus {"
                        "  border: 1px solid %5;"
                        "}"
                        "QSpinBox::up-button, QSpinBox::down-button {"
                        "  width: 0px;"
                        "}")
      .arg(Rgba(WithAlpha(bg, 240)), Hex(text), Rgba(border), Rgba(WithAlpha(border, 196)),
           Rgba(WithAlpha(accent, 224)));
}

auto AppTheme::EditorCheckBoxStyle() -> QString {
  const auto&  theme  = AppTheme::Instance();
  const QColor text   = theme.textColor();
  const QColor base   = theme.bgDeepColor();
  const QColor stroke = theme.glassStrokeColor();
  const QColor accent = theme.accentColor();
  return QStringLiteral("QCheckBox {"
                        "  color: %1;"
                        "  spacing: 8px;"
                        "}"
                        "QCheckBox::indicator {"
                        "  width: 16px;"
                        "  height: 16px;"
                        "}"
                        "QCheckBox::indicator:unchecked {"
                        "  background: %2;"
                        "  border: 1px solid %3;"
                        "  border-radius: 4px;"
                        "}"
                        "QCheckBox::indicator:checked {"
                        "  background: %4;"
                        "  border: 1px solid %4;"
                        "  border-radius: 4px;"
                        "}")
      .arg(Hex(text), Rgba(WithAlpha(base, 232)), Rgba(stroke), Hex(accent));
}

auto AppTheme::EditorScrollAreaStyle() -> QString {
  const auto& theme       = AppTheme::Instance();
  const QString bg_base   = Rgba(WithAlpha(theme.bgBaseColor(), 236));
  const QString bg_canvas = Rgba(WithAlpha(theme.bgCanvasColor(), 170));
  const QString accent    = Hex(theme.accentColor());

  return QStringLiteral("QScrollArea { background: %1; border: none; }"
                        "QScrollArea > QWidget, QScrollArea > QWidget > QWidget {"
                        "  background: %1;"
                        "}"
                        "QScrollBar:vertical {"
                        "  background: %2;"
                        "  width: 10px;"
                        "  margin: 2px;"
                        "  border-radius: 5px;"
                        "}"
                        "QScrollBar::handle:vertical {"
                        "  background: %3;"
                        "  border-radius: 5px;"
                        "}"
                        "QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0px; }"
                        "QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background: transparent; }")
      .arg(bg_base, bg_canvas, accent);
}

auto AppTheme::EditorListWidgetStyle() -> QString {
  const auto&  theme  = AppTheme::Instance();
  const QColor bg     = theme.bgDeepColor();
  const QColor border = theme.dividerColor();
  return QStringLiteral("QListWidget {"
                        "  background: %1;"
                        "  border: 1px solid %2;"
                        "  border-radius: 10px;"
                        "  padding: 8px;"
                        "}"
                        "QListWidget::item {"
                        "  padding: 2px;"
                        "}"
                        "QListWidget::item:selected {"
                        "  background: transparent;"
                        "}"
                        "QScrollBar:vertical {"
                        "  background: transparent;"
                        "  width: 6px;"
                        "  margin: 4px 2px 4px 0;"
                        "}"
                        "QScrollBar::handle:vertical {"
                        "  background: %3;"
                        "  border-radius: 3px;"
                        "  min-height: 24px;"
                        "}"
                        "QScrollBar::handle:vertical:hover,"
                        "QScrollBar::handle:vertical:pressed {"
                        "  background: %4;"
                        "}"
                        "QScrollBar::add-line:vertical,"
                        "QScrollBar::sub-line:vertical {"
                        "  height: 0px;"
                        "}"
                        "QScrollBar::add-page:vertical,"
                        "QScrollBar::sub-page:vertical {"
                        "  background: transparent;"
                        "}")
      .arg(Rgba(WithAlpha(bg, 216)), Rgba(border), Rgba(WithAlpha(theme.textMutedColor(), 132)),
           Hex(theme.accentColor()));
}

auto AppTheme::EditorHistoryCardStyle() -> QString {
  const auto& theme = AppTheme::Instance();
  return QStringLiteral("QFrame#HistoryCard {"
                        "  background: %1;"
                        "  border: 1px solid %2;"
                        "  border-radius: 10px;"
                        "}"
                        "QFrame#HistoryCard:hover {"
                        "  background: %3;"
                        "  border-color: %4;"
                        "}"
                        "QFrame#HistoryCard[selected=\"true\"] {"
                        "  background: %5;"
                        "  border: 1px solid %6;"
                        "}"
                        "QFrame#HistoryCard[selected=\"true\"]:hover {"
                        "  background: %7;"
                        "}")
      .arg(Rgba(WithAlpha(theme.bgPanelColor(), 224)), Rgba(theme.dividerColor()),
           Rgba(WithAlpha(theme.hoverColor(), 245)),
           Rgba(WithAlpha(theme.glassStrokeColor(), 180)), Rgba(theme.selectedTintColor()),
           Rgba(WithAlpha(theme.accentColor(), 144)),
           Rgba(WithAlpha(theme.selectedTintColor(), 68)));
}

auto AppTheme::EditorTransparentFrameStyle() -> QString {
  return QStringLiteral("QFrame {"
                        "  background: transparent;"
                        "  border: none;"
                        "  border-radius: 12px;"
                        "}");
}

auto AppTheme::uiFontFamily() const -> QString {
  RegisterFonts();
  return ActiveUiFamily(FontState());
}
auto AppTheme::dataFontFamily() const -> QString {
  RegisterFonts();
  return FontState().data;
}
auto AppTheme::monoFontFamily() const -> QString {
  RegisterFonts();
  return FontState().mono;
}

auto AppTheme::toneGold() const -> QColor { return GetTheme(current_theme_index_).tone_gold; }
auto AppTheme::toneWine() const -> QColor { return GetTheme(current_theme_index_).tone_wine; }
auto AppTheme::toneSteel() const -> QColor { return GetTheme(current_theme_index_).tone_steel; }
auto AppTheme::toneGraphite() const -> QColor { return GetTheme(current_theme_index_).tone_graphite; }
auto AppTheme::toneMist() const -> QColor { return GetTheme(current_theme_index_).tone_mist; }

auto AppTheme::bgCanvasColor() const -> QColor { return GetTheme(current_theme_index_).bg_canvas; }
auto AppTheme::bgDeepColor() const -> QColor { return GetTheme(current_theme_index_).bg_deep; }
auto AppTheme::bgBaseColor() const -> QColor { return GetTheme(current_theme_index_).bg_base; }
auto AppTheme::bgPanelColor() const -> QColor { return GetTheme(current_theme_index_).bg_panel; }
auto AppTheme::textColor() const -> QColor { return GetTheme(current_theme_index_).text; }
auto AppTheme::textMutedColor() const -> QColor { return GetTheme(current_theme_index_).text_muted; }
auto AppTheme::accentColor() const -> QColor { return toneGold(); }
auto AppTheme::accentSecondaryColor() const -> QColor { return GetTheme(current_theme_index_).accent_secondary; }
auto AppTheme::dangerColor() const -> QColor { return toneWine(); }
auto AppTheme::dangerTintColor() const -> QColor { return GetTheme(current_theme_index_).danger_tint; }
auto AppTheme::selectedTintColor() const -> QColor { return GetTheme(current_theme_index_).selected_tint; }
auto AppTheme::hoverColor() const -> QColor { return GetTheme(current_theme_index_).hover; }
auto AppTheme::dividerColor() const -> QColor { return GetTheme(current_theme_index_).divider; }
auto AppTheme::glassPanelColor() const -> QColor { return GetTheme(current_theme_index_).glass_panel; }
auto AppTheme::glassStrokeColor() const -> QColor { return GetTheme(current_theme_index_).glass_stroke; }
auto AppTheme::overlayColor() const -> QColor { return GetTheme(current_theme_index_).overlay; }
auto AppTheme::panelRadius() const -> int { return GetTheme(current_theme_index_).panel_radius; }

auto AppTheme::currentThemeIndex() const -> int { return current_theme_index_; }

void AppTheme::setCurrentThemeIndex(int index) {
  if (index < 0 || index > 1) {
    return;
  }
  if (current_theme_index_ == index) {
    return;
  }
  current_theme_index_ = index;
  emit ThemeChanged();
}

auto AppTheme::availableThemes() const -> QVariantList {
  return QVariantList{
      QVariantMap{{QStringLiteral("label"), tr("Pu-erh")}, {QStringLiteral("index"), 0}},
      QVariantMap{{QStringLiteral("label"), tr("Classic")}, {QStringLiteral("index"), 1}},
  };
}

auto AppTheme::ResolveRole(QWidget* widget) -> FontRole {
  if (!widget) {
    return FontRole::UiBody;
  }

  const QVariant tagged_role = widget->property(kThemeFontRoleProperty);
  if (tagged_role.isValid()) {
    return static_cast<FontRole>(tagged_role.toInt());
  }

  const QString object_name = widget->objectName();
  if (object_name == QLatin1String("EditorSectionTitle")) {
    return FontRole::UiTitle;
  }
  if (object_name == QLatin1String("EditorSectionSub")) {
    return FontRole::UiCaption;
  }

  if (qobject_cast<QPushButton*>(widget)) {
    return FontRole::UiBodyStrong;
  }
  if (qobject_cast<QCheckBox*>(widget) || qobject_cast<QComboBox*>(widget) ||
      qobject_cast<QSpinBox*>(widget) || qobject_cast<QListWidget*>(widget)) {
    return FontRole::UiBody;
  }

  return FontRole::UiBody;
}

}  // namespace puerhlab::ui
