//  Copyright 2026 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#include "ui/puerhlab_main/i18n.hpp"

namespace puerhlab::ui::i18n {

auto LocalizedText::Render() const -> QString {
  if (IsEmpty()) {
    return {};
  }

  QString translated =
      QCoreApplication::translate(context_ ? context_ : PUERHLAB_I18N_CONTEXT, source_);
  for (const QString& arg : args_) {
    translated = translated.arg(arg);
  }
  return translated;
}

auto LocalizedText::IsEmpty() const -> bool {
  return source_ == nullptr || source_[0] == '\0';
}

auto PathToQString(const std::filesystem::path& path) -> QString {
#ifdef _WIN32
  return QString::fromStdWString(path.wstring());
#else
  return QString::fromStdString(path.string());
#endif
}

auto ToArgumentString(const QString& value) -> QString { return value; }
auto ToArgumentString(QStringView value) -> QString { return value.toString(); }
auto ToArgumentString(const char* value) -> QString {
  return value ? QString::fromUtf8(value) : QString{};
}
auto ToArgumentString(const std::string& value) -> QString {
  return QString::fromUtf8(value.c_str(), static_cast<qsizetype>(value.size()));
}
auto ToArgumentString(const std::filesystem::path& value) -> QString {
  return PathToQString(value);
}
auto ToArgumentString(bool value) -> QString { return value ? QStringLiteral("true")
                                                            : QStringLiteral("false"); }

TranslationNotifier::TranslationNotifier(QObject* parent) : QObject(parent) {}

auto TranslationNotifier::Instance() -> TranslationNotifier& {
  static TranslationNotifier notifier;
  return notifier;
}

void TranslationNotifier::NotifyLanguageChanged() {
  emit LanguageChanged();
}

}  // namespace puerhlab::ui::i18n
