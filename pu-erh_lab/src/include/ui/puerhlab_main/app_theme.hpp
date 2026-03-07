//  Copyright 2026 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#pragma once

#include <QObject>
#include <QColor>
#include <QFont>
#include <QString>

class QApplication;
class QWidget;

namespace puerhlab::ui {

class AppTheme final : public QObject {
  Q_OBJECT
  Q_PROPERTY(QString uiFontFamily READ uiFontFamily CONSTANT)
  Q_PROPERTY(QString dataFontFamily READ dataFontFamily CONSTANT)
  Q_PROPERTY(QString monoFontFamily READ monoFontFamily CONSTANT)
  Q_PROPERTY(QColor toneGold READ toneGold CONSTANT)
  Q_PROPERTY(QColor toneWine READ toneWine CONSTANT)
  Q_PROPERTY(QColor toneSteel READ toneSteel CONSTANT)
  Q_PROPERTY(QColor toneGraphite READ toneGraphite CONSTANT)
  Q_PROPERTY(QColor toneMist READ toneMist CONSTANT)
  Q_PROPERTY(QColor bgCanvasColor READ bgCanvasColor CONSTANT)
  Q_PROPERTY(QColor bgDeepColor READ bgDeepColor CONSTANT)
  Q_PROPERTY(QColor bgBaseColor READ bgBaseColor CONSTANT)
  Q_PROPERTY(QColor bgPanelColor READ bgPanelColor CONSTANT)
  Q_PROPERTY(QColor textColor READ textColor CONSTANT)
  Q_PROPERTY(QColor textMutedColor READ textMutedColor CONSTANT)
  Q_PROPERTY(QColor accentColor READ accentColor CONSTANT)
  Q_PROPERTY(QColor accentSecondaryColor READ accentSecondaryColor CONSTANT)
  Q_PROPERTY(QColor dangerColor READ dangerColor CONSTANT)
  Q_PROPERTY(QColor dangerTintColor READ dangerTintColor CONSTANT)
  Q_PROPERTY(QColor selectedTintColor READ selectedTintColor CONSTANT)
  Q_PROPERTY(QColor hoverColor READ hoverColor CONSTANT)
  Q_PROPERTY(QColor overlayColor READ overlayColor CONSTANT)
  Q_PROPERTY(int panelRadius READ panelRadius CONSTANT)

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
  static auto TryRegisterUiFontOverride(const QString& path) -> bool;
  static void ApplyApplicationFont(QApplication& app);

  static auto Font(FontRole role) -> QFont;
  static void ApplyFont(QWidget* widget, FontRole role);
  static void MarkFontRole(QObject* object, FontRole role);
  static void ApplyFontsRecursively(QWidget* root);

  static auto EditorLabelStyle(const QColor& color) -> QString;
  static auto EditorPrimaryButtonStyle(bool include_disabled = false) -> QString;
  static auto EditorSecondaryButtonStyle() -> QString;
  static auto EditorPanelToggleStyle(bool active) -> QString;
  static auto EditorMethodCardStyle(bool active) -> QString;
  static auto EditorComboBoxStyle() -> QString;
  static auto EditorSpinBoxStyle() -> QString;
  static auto EditorCheckBoxStyle() -> QString;
  static auto EditorScrollAreaStyle() -> QString;
  static auto EditorListWidgetStyle() -> QString;
  static auto EditorHistoryCardStyle() -> QString;
  static auto EditorTransparentFrameStyle() -> QString;

  auto uiFontFamily() const -> QString;
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
  auto overlayColor() const -> QColor;
  auto panelRadius() const -> int;

 private:
  explicit AppTheme(QObject* parent = nullptr);

  static auto ResolveRole(QWidget* widget) -> FontRole;
};

}  // namespace puerhlab::ui
