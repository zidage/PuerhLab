//  Copyright 2026 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#include "ui/puerhlab_main/app_theme.hpp"

#include <QApplication>
#include <QCheckBox>
#include <QComboBox>
#include <QFontDatabase>
#include <QListWidget>
#include <QPushButton>
#include <QSpinBox>
#include <QWidget>

namespace puerhlab::ui {
namespace {

constexpr char kThemeFontRoleProperty[] = "puerhlabFontRole";

struct FontFamilies {
  QString ui = QStringLiteral("Inter");
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
  font.setHintingPreference(QFont::PreferFullHinting);
  return font;
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
  families.ui =
      RegisterFontResource(QStringLiteral(":/fonts/main_Inter.ttf"), QStringLiteral("Inter"));
  RegisterFontResource(QStringLiteral(":/fonts/main_Inter_italic.ttf"), families.ui);
  families.data = RegisterFontResource(QStringLiteral(":/fonts/main_IBM.ttf"),
                                       QStringLiteral("IBM Plex Sans"));
  families.mono = QFontDatabase::systemFont(QFontDatabase::FixedFont).family();

  FontsRegisteredFlag() = true;
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

  FontState().ui = families.front();
  return true;
}

void AppTheme::ApplyApplicationFont(QApplication& app) {
  RegisterFonts();

  QFont app_font = app.font();
  app_font.setFamily(FontState().ui);
  app_font.setHintingPreference(QFont::PreferFullHinting);
  app.setFont(app_font);
}

auto AppTheme::Font(FontRole role) -> QFont {
  RegisterFonts();
  const auto& families = FontState();

  switch (role) {
    case FontRole::UiBody:
      return MakeFont(families.ui, 10.5, QFont::Normal);
    case FontRole::UiBodyStrong:
      return MakeFont(families.ui, 10.5, QFont::DemiBold);
    case FontRole::UiCaption:
      return MakeFont(families.ui, 9.5, QFont::Normal);
    case FontRole::UiCaptionStrong:
      return MakeFont(families.ui, 9.5, QFont::DemiBold);
    case FontRole::UiTitle:
      return MakeFont(families.ui, 11.0, QFont::DemiBold);
    case FontRole::UiHeadline:
      return MakeFont(families.ui, 13.5, QFont::Bold);
    case FontRole::UiOverline: {
      QFont font = MakeFont(families.ui, 8.5, QFont::DemiBold);
      font.setCapitalization(QFont::AllUppercase);
      font.setLetterSpacing(QFont::AbsoluteSpacing, 0.8);
      return font;
    }
    case FontRole::UiHint:
      return MakeFont(families.ui, 9.0, QFont::Normal);
    case FontRole::DataBody:
      return MakeFont(families.data, 10.5, QFont::Normal);
    case FontRole::DataBodyStrong:
      return MakeFont(families.data, 10.5, QFont::DemiBold);
    case FontRole::DataCaption:
      return MakeFont(families.data, 9.5, QFont::Normal);
    case FontRole::DataNumeric:
      return MakeFont(families.data, 12.0, QFont::DemiBold);
    case FontRole::DataOverlay:
      return MakeFont(families.data, 9.5, QFont::DemiBold);
    case FontRole::MonoBody:
      return MakeFont(families.mono, 10.0, QFont::DemiBold);
    case FontRole::MonoCaption:
      return MakeFont(families.mono, 9.0, QFont::Normal);
  }

  return MakeFont(families.ui, 10.5, QFont::Normal);
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
  QString style =
      QStringLiteral("QPushButton {"
                     "  color: #121212;"
                     "  background: #FCC704;"
                     "  border: none;"
                     "  border-radius: 8px;"
                     "}"
                     "QPushButton:hover {"
                     "  background: #F5C200;"
                     "}");
  if (include_disabled) {
    style += QStringLiteral("QPushButton:disabled {"
                            "  color: #6A6A6A;"
                            "  background: #1A1A1A;"
                            "}");
  }
  return style;
}

auto AppTheme::EditorSecondaryButtonStyle() -> QString {
  return QStringLiteral("QPushButton {"
                        "  color: #E6E6E6;"
                        "  background: #3A3A3A;"
                        "  border: none;"
                        "  border-radius: 8px;"
                        "}"
                        "QPushButton:hover {"
                        "  background: #505050;"
                        "}");
}

auto AppTheme::EditorPanelToggleStyle(bool active) -> QString {
  if (active) {
    return QStringLiteral("QPushButton {"
                          "  color: #121212;"
                          "  background: #FCC704;"
                          "  border: none;"
                          "  border-radius: 8px;"
                          "}"
                          "QPushButton:hover {"
                          "  background: #F5C200;"
                          "}");
  }

  return QStringLiteral("QPushButton {"
                        "  color: #E6E6E6;"
                        "  background: #121212;"
                        "  border: 1px solid #2A2A2A;"
                        "  border-radius: 8px;"
                        "}"
                        "QPushButton:hover {"
                        "  border-color: #FCC704;"
                        "}");
}

auto AppTheme::EditorMethodCardStyle(bool active) -> QString {
  if (active) {
    return QStringLiteral("QPushButton {"
                          "  color: #FCC704;"
                          "  background: #1A1A1A;"
                          "  border: 2px solid #FCC704;"
                          "  border-radius: 12px;"
                          "  padding: 16px 20px;"
                          "  text-align: left;"
                          "}"
                          "QPushButton:hover {"
                          "  background: #242424;"
                          "}");
  }

  return QStringLiteral("QPushButton {"
                        "  color: #A0A0A0;"
                        "  background: #1A1A1A;"
                        "  border: 1px solid #333333;"
                        "  border-radius: 12px;"
                        "  padding: 16px 20px;"
                        "  text-align: left;"
                        "}"
                        "QPushButton:hover {"
                        "  border: 1px solid #666666;"
                        "  background: #242424;"
                        "  color: #E6E6E6;"
                        "}");
}

auto AppTheme::EditorComboBoxStyle() -> QString {
  return QStringLiteral("QComboBox {"
                        "  background: #1A1A1A;"
                        "  border: none;"
                        "  border-radius: 8px;"
                        "  padding: 4px 8px;"
                        "}"
                        "QComboBox::drop-down {"
                        "  border: 0px;"
                        "  width: 24px;"
                        "}"
                        "QComboBox QAbstractItemView {"
                        "  background: #1A1A1A;"
                        "  border: none;"
                        "  selection-background-color: #FCC704;"
                        "  selection-color: #121212;"
                        "}"
                        "QComboBox QAbstractItemView::item:hover {"
                        "  background: #202020;"
                        "  color: #E6E6E6;"
                        "}"
                        "QComboBox QAbstractItemView::item:selected {"
                        "  background: #FCC704;"
                        "  color: #121212;"
                        "}");
}

auto AppTheme::EditorSpinBoxStyle() -> QString {
  return QStringLiteral("QSpinBox {"
                        "  background: #1A1A1A;"
                        "  color: #E6E6E6;"
                        "  border: 1px solid #333333;"
                        "  border-radius: 6px;"
                        "  padding: 4px 8px;"
                        "}"
                        "QSpinBox:focus {"
                        "  border: 1px solid #FCC704;"
                        "}"
                        "QSpinBox::up-button, QSpinBox::down-button {"
                        "  width: 0px;"
                        "}");
}

auto AppTheme::EditorCheckBoxStyle() -> QString {
  return QStringLiteral("QCheckBox {"
                        "  color: #E6E6E6;"
                        "  spacing: 8px;"
                        "}"
                        "QCheckBox::indicator {"
                        "  width: 16px;"
                        "  height: 16px;"
                        "}"
                        "QCheckBox::indicator:unchecked {"
                        "  background: #121212;"
                        "  border: 1px solid #2A2A2A;"
                        "  border-radius: 3px;"
                        "}"
                        "QCheckBox::indicator:checked {"
                        "  background: #FCC704;"
                        "  border: 1px solid #FCC704;"
                        "  border-radius: 3px;"
                        "}");
}

auto AppTheme::EditorScrollAreaStyle() -> QString {
  return QStringLiteral("QScrollArea { background: transparent; border: none; }"
                        "QScrollBar:vertical {"
                        "  background: #121212;"
                        "  width: 10px;"
                        "  margin: 2px;"
                        "  border-radius: 5px;"
                        "}"
                        "QScrollBar::handle:vertical {"
                        "  background: #FCC704;"
                        "  border-radius: 5px;"
                        "}"
                        "QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0px; }"
                        "QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background: transparent; }");
}

auto AppTheme::EditorListWidgetStyle() -> QString {
  return QStringLiteral("QListWidget {"
                        "  background: #121212;"
                        "  border: none;"
                        "  border-radius: 10px;"
                        "  padding: 6px;"
                        "}"
                        "QListWidget::item {"
                        "  padding: 2px;"
                        "}"
                        "QListWidget::item:selected {"
                        "  background: transparent;"
                        "}");
}

auto AppTheme::EditorHistoryCardStyle() -> QString {
  return QStringLiteral("QFrame#HistoryCard {"
                        "  background: #1A1A1A;"
                        "  border: none;"
                        "  border-radius: 10px;"
                        "}"
                        "QFrame#HistoryCard:hover {"
                        "  background: #202020;"
                        "}"
                        "QFrame#HistoryCard[selected=\"true\"] {"
                        "  background: rgba(252, 199, 4, 0.20);"
                        "  border: 2px solid #FCC704;"
                        "}");
}

auto AppTheme::EditorTransparentFrameStyle() -> QString {
  return QStringLiteral("QFrame {"
                        "  background: transparent;"
                        "  border: none;"
                        "  border-radius: 12px;"
                        "}");
}

auto AppTheme::uiFontFamily() const -> QString { return FontState().ui; }
auto AppTheme::dataFontFamily() const -> QString { return FontState().data; }
auto AppTheme::monoFontFamily() const -> QString { return FontState().mono; }

auto AppTheme::toneGold() const -> QColor { return QColor(QStringLiteral("#FCC704")); }
auto AppTheme::toneWine() const -> QColor { return QColor(QStringLiteral("#8A0526")); }
auto AppTheme::toneSteel() const -> QColor { return QColor(QStringLiteral("#4A4A4A")); }
auto AppTheme::toneGraphite() const -> QColor { return QColor(QStringLiteral("#1A1A1A")); }
auto AppTheme::toneMist() const -> QColor { return QColor(QStringLiteral("#E6E6E6")); }

auto AppTheme::bgCanvasColor() const -> QColor { return QColor(QStringLiteral("#111111")); }
auto AppTheme::bgDeepColor() const -> QColor { return QColor(QStringLiteral("#141414")); }
auto AppTheme::bgBaseColor() const -> QColor { return QColor(QStringLiteral("#1F1F1F")); }
auto AppTheme::bgPanelColor() const -> QColor { return QColor(QStringLiteral("#2B2B2B")); }
auto AppTheme::textColor() const -> QColor { return toneMist(); }
auto AppTheme::textMutedColor() const -> QColor { return QColor(QStringLiteral("#888888")); }
auto AppTheme::accentColor() const -> QColor { return toneGold(); }
auto AppTheme::accentSecondaryColor() const -> QColor { return toneGold(); }
auto AppTheme::dangerColor() const -> QColor { return toneWine(); }
auto AppTheme::dangerTintColor() const -> QColor { return QColor(138, 5, 38, 82); }
auto AppTheme::selectedTintColor() const -> QColor { return QColor(252, 199, 4, 46); }
auto AppTheme::hoverColor() const -> QColor { return QColor(QStringLiteral("#333333")); }
auto AppTheme::overlayColor() const -> QColor { return QColor(QStringLiteral("#C0121212")); }
auto AppTheme::panelRadius() const -> int { return 8; }

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
