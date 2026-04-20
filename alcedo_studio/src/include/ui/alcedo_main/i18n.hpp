//  Copyright 2026 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#pragma once

#include <QObject>
#include <QCoreApplication>
#include <QString>
#include <QStringList>

#include <filesystem>
#include <string>
#include <type_traits>
#include <utility>

#define ALCEDO_I18N_CONTEXT "Alcedo"

namespace alcedo::ui {

inline auto Tr(const char* text) -> QString {
  return QCoreApplication::translate(ALCEDO_I18N_CONTEXT, text);
}

}  // namespace alcedo::ui

namespace alcedo::ui::i18n {

struct LocalizedText {
  const char* context_ = ALCEDO_I18N_CONTEXT;
  const char* source_  = "";
  QStringList args_{};

  [[nodiscard]] auto Render() const -> QString;
  [[nodiscard]] auto IsEmpty() const -> bool;
};

auto PathToQString(const std::filesystem::path& path) -> QString;
auto ToArgumentString(const QString& value) -> QString;
auto ToArgumentString(QStringView value) -> QString;
auto ToArgumentString(const char* value) -> QString;
auto ToArgumentString(const std::string& value) -> QString;
auto ToArgumentString(const std::filesystem::path& value) -> QString;
auto ToArgumentString(bool value) -> QString;

template <typename T>
auto ToArgumentString(const T& value) -> QString {
  if constexpr (std::is_integral_v<T> && !std::is_same_v<T, bool>) {
    return QString::number(value);
  } else if constexpr (std::is_floating_point_v<T>) {
    return QString::number(value, 'f', 2);
  } else {
    return QString::fromUtf8(value);
  }
}

template <typename... Args>
auto MakeLocalizedText(const char* context, const char* source, Args&&... args)
    -> LocalizedText {
  LocalizedText text;
  text.context_ = context;
  text.source_  = source;
  (text.args_.push_back(ToArgumentString(std::forward<Args>(args))), ...);
  return text;
}

class TranslationNotifier final : public QObject {
  Q_OBJECT

 public:
  static auto Instance() -> TranslationNotifier&;

  void NotifyLanguageChanged();

 signals:
  void LanguageChanged();

 private:
  explicit TranslationNotifier(QObject* parent = nullptr);
};

}  // namespace alcedo::ui::i18n
