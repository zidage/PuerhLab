//  Copyright 2026 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#pragma once

#include <QObject>
#include <QColor>
#include <QFont>
#include <QString>
#include <QVariantList>

class QApplication;
class QWidget;

namespace puerhlab::ui {

class AppTheme final : public QObject {
  Q_OBJECT
  Q_PROPERTY(QString uiFontFamily READ uiFontFamily NOTIFY UiFontFamilyChanged)
  Q_PROPERTY(QString headlineFontFamily READ headlineFontFamily NOTIFY UiFontFamilyChanged)
  Q_PROPERTY(QString dataFontFamily READ dataFontFamily CONSTANT)
  Q_PROPERTY(QString monoFontFamily READ monoFontFamily CONSTANT)
  Q_PROPERTY(QColor toneGold READ toneGold NOTIFY ThemeChanged)
  Q_PROPERTY(QColor toneWine READ toneWine NOTIFY ThemeChanged)
  Q_PROPERTY(QColor toneSteel READ toneSteel NOTIFY ThemeChanged)
  Q_PROPERTY(QColor toneGraphite READ toneGraphite NOTIFY ThemeChanged)
  Q_PROPERTY(QColor toneMist READ toneMist NOTIFY ThemeChanged)
  Q_PROPERTY(QColor bgCanvasColor READ bgCanvasColor NOTIFY ThemeChanged)
  Q_PROPERTY(QColor bgDeepColor READ bgDeepColor NOTIFY ThemeChanged)
  Q_PROPERTY(QColor bgBaseColor READ bgBaseColor NOTIFY ThemeChanged)
  Q_PROPERTY(QColor bgPanelColor READ bgPanelColor NOTIFY ThemeChanged)
  Q_PROPERTY(QColor textColor READ textColor NOTIFY ThemeChanged)
  Q_PROPERTY(QColor textMutedColor READ textMutedColor NOTIFY ThemeChanged)
  Q_PROPERTY(QColor accentColor READ accentColor NOTIFY ThemeChanged)
  Q_PROPERTY(QColor accentSecondaryColor READ accentSecondaryColor NOTIFY ThemeChanged)
  Q_PROPERTY(QColor dangerColor READ dangerColor NOTIFY ThemeChanged)
  Q_PROPERTY(QColor dangerTintColor READ dangerTintColor NOTIFY ThemeChanged)
  Q_PROPERTY(QColor selectedTintColor READ selectedTintColor NOTIFY ThemeChanged)
  Q_PROPERTY(QColor hoverColor READ hoverColor NOTIFY ThemeChanged)
  Q_PROPERTY(QColor dividerColor READ dividerColor NOTIFY ThemeChanged)
  Q_PROPERTY(QColor glassPanelColor READ glassPanelColor NOTIFY ThemeChanged)
  Q_PROPERTY(QColor glassStrokeColor READ glassStrokeColor NOTIFY ThemeChanged)
  Q_PROPERTY(QColor overlayColor READ overlayColor NOTIFY ThemeChanged)
  Q_PROPERTY(int panelRadius READ panelRadius NOTIFY ThemeChanged)
  Q_PROPERTY(int currentThemeIndex READ currentThemeIndex WRITE setCurrentThemeIndex NOTIFY ThemeChanged)
  Q_PROPERTY(QVariantList availableThemes READ availableThemes CONSTANT)

 public:
  enum class FontRole : int {
    UiBody = 0,
    UiBodyStrong,
    UiCaption,
    UiCaptionStrong,
    UiTitle,
    UiHeadline,
    UiOverline,
    UiHint,
    DataBody,
    DataBodyStrong,
    DataCaption,
    DataNumeric,
    DataOverlay,
    MonoBody,
    MonoCaption
  };
  Q_ENUM(FontRole)

  static auto Instance() -> AppTheme&;

  static void RegisterFonts();
  static void SetEffectiveLanguageCode(const QString& code);
  static void ApplyApplicationFont();
  static auto TryRegisterUiFontOverride(const QString& path) -> bool;
  static void ApplyApplicationFont(QApplication& app);

  static auto Font(FontRole role) -> QFont;
  static void ApplyFont(QWidget* widget, FontRole role);
  static void MarkFontRole(QObject* object, FontRole role);
  static void ApplyFontsRecursively(QWidget* root);

  static auto EditorLabelStyle(const QColor& color) -> QString;
  static auto EditorPrimaryButtonStyle(bool include_disabled = false) -> QString;
  static auto EditorSecondaryButtonStyle() -> QString;
  static auto EditorPanelToggleStyle(bool active, bool is_first = false,
                                     bool is_last = false) -> QString;
  static auto EditorMethodCardStyle(bool active) -> QString;
  static auto EditorComboBoxStyle() -> QString;
  static auto EditorSpinBoxStyle() -> QString;
  static auto EditorCheckBoxStyle() -> QString;
  static auto EditorScrollAreaStyle() -> QString;
  static auto EditorListWidgetStyle() -> QString;
  static auto EditorHistoryCardStyle() -> QString;
  static auto EditorTransparentFrameStyle() -> QString;

  auto uiFontFamily() const -> QString;
  auto headlineFontFamily() const -> QString;
  auto dataFontFamily() const -> QString;
  auto monoFontFamily() const -> QString;

  auto toneGold() const -> QColor;
  auto toneWine() const -> QColor;
  auto toneSteel() const -> QColor;
  auto toneGraphite() const -> QColor;
  auto toneMist() const -> QColor;

  auto bgCanvasColor() const -> QColor;
  auto bgDeepColor() const -> QColor;
  auto bgBaseColor() const -> QColor;
  auto bgPanelColor() const -> QColor;
  auto textColor() const -> QColor;
  auto textMutedColor() const -> QColor;
  auto accentColor() const -> QColor;
  auto accentSecondaryColor() const -> QColor;
  auto dangerColor() const -> QColor;
  auto dangerTintColor() const -> QColor;
  auto selectedTintColor() const -> QColor;
  auto hoverColor() const -> QColor;
  auto dividerColor() const -> QColor;
  auto glassPanelColor() const -> QColor;
  auto glassStrokeColor() const -> QColor;
  auto overlayColor() const -> QColor;
  auto panelRadius() const -> int;

  auto currentThemeIndex() const -> int;
  void setCurrentThemeIndex(int index);
  auto availableThemes() const -> QVariantList;

 signals:
  void UiFontFamilyChanged();
  void ThemeChanged();

 private:
  explicit AppTheme(QObject* parent = nullptr);

  static auto ResolveRole(QWidget* widget) -> FontRole;

  int current_theme_index_ = 0;
};

}  // namespace puerhlab::ui
