//  Copyright 2026 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#include <QApplication>
#include <QCoreApplication>
#include <QFont>
#include <QFontDatabase>
#include <QGuiApplication>
#include <QSettings>
#include <QQmlApplicationEngine>
#include <QQmlContext>
#include <QQuickStyle>
#include <QtGlobal>

#include <exiv2/error.hpp>
#include <optional>
#include <string>
#include <string_view>

#include "ui/puerhlab_main/album_backend/album_backend.hpp"
#include "ui/puerhlab_main/app_theme.hpp"
#include "ui/puerhlab_main/language_manager.hpp"
#include "edit/operators/operator_registeration.hpp"
#include "utils/clock/time_provider.hpp"

namespace {

auto FindArgValue(int argc, char** argv, std::string_view option_name)
    -> std::optional<std::string_view> {
  const std::string opt_eq = std::string(option_name) + "=";
  for (int i = 1; i < argc; ++i) {
    const std::string_view arg(argv[i] ? argv[i] : "");
    if (arg == option_name) {
      if (i + 1 < argc && argv[i + 1]) {
        return std::string_view(argv[i + 1]);
      }
      return std::nullopt;
    }
    if (arg.rfind(opt_eq, 0) == 0) {
      return arg.substr(opt_eq.size());
    }
  }
  return std::nullopt;
}

}  // namespace

int main(int argc, char* argv[]) {
#if QT_VERSION < QT_VERSION_CHECK(6, 0, 0)
  QCoreApplication::setAttribute(Qt::AA_EnableHighDpiScaling);
  QCoreApplication::setAttribute(Qt::AA_UseHighDpiPixmaps);
#else
  QGuiApplication::setHighDpiScaleFactorRoundingPolicy(
      Qt::HighDpiScaleFactorRoundingPolicy::PassThrough);
#endif

  puerhlab::TimeProvider::Refresh();
  puerhlab::RegisterAllOperators();
  Exiv2::LogMsg::setLevel(Exiv2::LogMsg::Level::error);

  QApplication app(argc, argv);
  {
    QFont default_font = app.font();
    default_font.setStyleStrategy(QFont::PreferAntialias);
    app.setFont(default_font);
  }
  QCoreApplication::setOrganizationName(QStringLiteral("PuerhLab"));
  QCoreApplication::setOrganizationDomain(QStringLiteral("puerhlab.app"));
  QCoreApplication::setApplicationName(QStringLiteral("PuerhLab"));
  puerhlab::ui::AppTheme::RegisterFonts();
  if (const auto arg = FindArgValue(argc, argv, "--font"); arg.has_value()) {
    puerhlab::ui::AppTheme::TryRegisterUiFontOverride(QString::fromUtf8(arg->data(), arg->size()));
  } else if (const auto env = qEnvironmentVariable("PUERHLAB_FONT_PATH"); !env.isEmpty()) {
    puerhlab::ui::AppTheme::TryRegisterUiFontOverride(env);
  }
  puerhlab::ui::LanguageManager language_manager(&app);
  puerhlab::ui::AppTheme::SetEffectiveLanguageCode(language_manager.EffectiveLanguageCode());
  puerhlab::ui::AppTheme::ApplyApplicationFont(app);
  QObject::connect(&language_manager, &puerhlab::ui::LanguageManager::EffectiveLanguageCodeChanged,
                   &app, [&app, &language_manager]() {
                     puerhlab::ui::AppTheme::SetEffectiveLanguageCode(
                         language_manager.EffectiveLanguageCode());
                     puerhlab::ui::AppTheme::ApplyApplicationFont(app);
                   });
  QQuickStyle::setStyle("Material");

  puerhlab::ui::AlbumBackend backend;

  QQmlApplicationEngine engine;
  engine.addImportPath("qrc:/");
  language_manager.AttachEngine(&engine);
  engine.rootContext()->setContextProperty("albumBackend", &backend);
  engine.rootContext()->setContextProperty("appTheme", &puerhlab::ui::AppTheme::Instance());
  engine.rootContext()->setContextProperty("languageManager", &language_manager);

  QObject::connect(&engine, &QQmlApplicationEngine::objectCreationFailed, &app,
                   []() { QCoreApplication::exit(-1); }, Qt::QueuedConnection);

  engine.loadFromModule("PuerhLab.Main", "Main");

  return app.exec();
}
