//  Copyright 2026 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#include "ui/puerhlab_main/language_manager.hpp"

#include <QCoreApplication>
#include <QLocale>
#include <QSettings>
#include <QQmlEngine>

#include "ui/puerhlab_main/i18n.hpp"

namespace {

constexpr auto kLanguageSettingKey = "ui/language";

auto DisplayLabelForCode(const QString& code) -> QString {
  if (code == QStringLiteral("system")) {
    return QCoreApplication::translate("LanguageManager", "Follow System");
  }
  if (code == QStringLiteral("zh-CN")) {
    return QCoreApplication::translate("LanguageManager", "Simplified Chinese");
  }
  return QCoreApplication::translate("LanguageManager", "English");
}

}  // namespace

namespace puerhlab::ui {

LanguageManager::LanguageManager(QCoreApplication* app, QObject* parent)
    : QObject(parent), app_(app), translator_(std::make_unique<QTranslator>()) {
  LoadPersistedLanguage();
  ApplyLanguage(false);
}

auto LanguageManager::AvailableLanguages() const -> QVariantList {
  return {
      QVariantMap{{"code", QStringLiteral("system")},
                  {"label", DisplayLabelForCode(QStringLiteral("system"))}},
      QVariantMap{{"code", QStringLiteral("en")},
                  {"label", DisplayLabelForCode(QStringLiteral("en"))}},
      QVariantMap{{"code", QStringLiteral("zh-CN")},
                  {"label", DisplayLabelForCode(QStringLiteral("zh-CN"))}},
  };
}

auto LanguageManager::ResolveSystemLanguageCode(const QLocale& locale) -> QString {
  return locale.bcp47Name().startsWith(QStringLiteral("zh"), Qt::CaseInsensitive)
             ? QStringLiteral("zh-CN")
             : QStringLiteral("en");
}

void LanguageManager::AttachEngine(QQmlEngine* engine) {
  qml_engine_ = engine;
  if (qml_engine_) {
    qml_engine_->retranslate();
  }
}

void LanguageManager::setLanguage(const QString& code) {
  const QString normalized = NormalizeLanguageCode(code);
  if (normalized == current_language_code_) {
    return;
  }

  current_language_code_ = normalized;
  QSettings{}.setValue(QLatin1String(kLanguageSettingKey), current_language_code_);
  ApplyLanguage(true);
}

auto LanguageManager::NormalizeLanguageCode(const QString& code) -> QString {
  const QString normalized = code.trimmed();
  if (normalized.compare(QStringLiteral("system"), Qt::CaseInsensitive) == 0) {
    return QStringLiteral("system");
  }
  if (normalized.compare(QStringLiteral("zh-CN"), Qt::CaseInsensitive) == 0 ||
      normalized.compare(QStringLiteral("zh_CN"), Qt::CaseInsensitive) == 0 ||
      normalized.compare(QStringLiteral("zh"), Qt::CaseInsensitive) == 0) {
    return QStringLiteral("zh-CN");
  }
  return QStringLiteral("en");
}

auto LanguageManager::EffectiveCodeForCurrentSelection() const -> QString {
  if (current_language_code_ == QStringLiteral("system")) {
    return ResolveSystemLanguageCode(QLocale::system());
  }
  return current_language_code_;
}

auto LanguageManager::TranslationResourcePathForCode(const QString& code) const -> QString {
  if (code == QStringLiteral("zh-CN")) {
    return QStringLiteral(":/i18n/puerhlab_main_zh_CN.qm");
  }
  return QStringLiteral(":/i18n/puerhlab_main_en.qm");
}

void LanguageManager::LoadPersistedLanguage() {
  const QString stored = QSettings{}.value(QLatin1String(kLanguageSettingKey),
                                           QStringLiteral("system"))
                             .toString();
  current_language_code_ = NormalizeLanguageCode(stored);
}

void LanguageManager::ApplyLanguage(bool emitSignals) {
  if (!app_) {
    return;
  }

  app_->removeTranslator(translator_.get());

  const QString next_effective = EffectiveCodeForCurrentSelection();
  const QString qm_path        = TranslationResourcePathForCode(next_effective);
  const bool translator_loaded = translator_->load(qm_path);
  if (translator_loaded) {
    app_->installTranslator(translator_.get());
  }

  const bool effective_changed = effective_language_code_ != next_effective;
  effective_language_code_     = next_effective;

  if (qml_engine_) {
    qml_engine_->retranslate();
  }

  i18n::TranslationNotifier::Instance().NotifyLanguageChanged();

  if (!emitSignals) {
    return;
  }

  emit CurrentLanguageCodeChanged();
  if (effective_changed) {
    emit EffectiveLanguageCodeChanged();
  }
  emit AvailableLanguagesChanged();
  emit LanguageChanged();
}

}  // namespace puerhlab::ui
