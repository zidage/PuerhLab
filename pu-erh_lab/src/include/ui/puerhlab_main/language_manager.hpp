//  Copyright 2026 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#pragma once

#include <QObject>
#include <QCoreApplication>
#include <QPointer>
#include <QTranslator>
#include <QVariantList>

#include <memory>

class QQmlEngine;
class QLocale;

namespace puerhlab::ui {

class LanguageManager final : public QObject {
  Q_OBJECT
  Q_PROPERTY(QVariantList availableLanguages READ AvailableLanguages NOTIFY AvailableLanguagesChanged)
  Q_PROPERTY(QString currentLanguageCode READ CurrentLanguageCode NOTIFY CurrentLanguageCodeChanged)
  Q_PROPERTY(QString effectiveLanguageCode READ EffectiveLanguageCode
             NOTIFY EffectiveLanguageCodeChanged)

 public:
  explicit LanguageManager(QCoreApplication* app, QObject* parent = nullptr);

  [[nodiscard]] auto AvailableLanguages() const -> QVariantList;
  [[nodiscard]] auto CurrentLanguageCode() const -> const QString& { return current_language_code_; }
  [[nodiscard]] auto EffectiveLanguageCode() const -> const QString& {
    return effective_language_code_;
  }

  static auto ResolveSystemLanguageCode(const QLocale& locale) -> QString;

  void AttachEngine(QQmlEngine* engine);

  Q_INVOKABLE void setLanguage(const QString& code);

 signals:
  void AvailableLanguagesChanged();
  void CurrentLanguageCodeChanged();
  void EffectiveLanguageCodeChanged();
  void LanguageChanged();

 private:
  [[nodiscard]] static auto NormalizeLanguageCode(const QString& code) -> QString;
  [[nodiscard]] auto EffectiveCodeForCurrentSelection() const -> QString;
  [[nodiscard]] auto TranslationResourcePathForCode(const QString& code) const -> QString;
  void LoadPersistedLanguage();
  void ApplyLanguage(bool emitSignals);

  QCoreApplication*          app_                    = nullptr;
  QPointer<QQmlEngine>       qml_engine_{};
  std::unique_ptr<QTranslator> translator_{};
  QString                    current_language_code_  = QStringLiteral("system");
  QString                    effective_language_code_ = QStringLiteral("en");
};

}  // namespace puerhlab::ui
