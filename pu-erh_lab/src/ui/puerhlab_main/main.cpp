//  Copyright 2026 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#include <QApplication>
#include <QCoreApplication>
#include <QFont>
#include <QFontDatabase>
#include <QGuiApplication>
#include <QIcon>
#include <QSettings>
#include <QQmlApplicationEngine>
#include <QQmlContext>
#include <QQuickStyle>
#include <QString>
#include <QtGlobal>

#include <exiv2/error.hpp>
#include <optional>
#include <string>
#include <string_view>

#include "ui/puerhlab_main/album_backend/album_backend.hpp"
#include "ui/puerhlab_main/app_theme.hpp"
#include "ui/puerhlab_main/language_manager.hpp"
#include "edit/operators/operator_registeration.hpp"
#include "utils/cuda/cuda_driver_requirements.hpp"
#include "utils/clock/time_provider.hpp"

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <Windows.h>
#endif

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

#if defined(_WIN32)
void ShowStartupErrorDialog(const QString& message) {
  ::MessageBoxW(nullptr, reinterpret_cast<LPCWSTR>(message.utf16()), L"PuerhLab",
                MB_OK | MB_ICONERROR | MB_TOPMOST);
}

auto BuildCudaDriverUpdateMessage(const puerhlab::cuda::DriverSupportInfo& support_info) -> QString {
  const QString required_version = QString::fromStdString(
      puerhlab::cuda::FormatCudaVersion(puerhlab::cuda::kMinimumSupportedCudaDriverVersion));
  const QString detected_version = QString::fromStdString(
      puerhlab::cuda::FormatCudaVersion(support_info.detected_cuda_driver_version));

  QString detail_line;
  switch (support_info.status) {
    case puerhlab::cuda::DriverSupportStatus::kDriverTooOld:
      detail_line = QStringLiteral("Detected CUDA driver compatibility: %1.\n")
                        .arg(detected_version);
      break;
    case puerhlab::cuda::DriverSupportStatus::kDriverUnavailable:
      detail_line = QStringLiteral("No usable NVIDIA CUDA driver was detected.\n");
      break;
    case puerhlab::cuda::DriverSupportStatus::kQueryFailed:
      detail_line = QStringLiteral("Failed to query the installed NVIDIA CUDA driver.\n");
      break;
    case puerhlab::cuda::DriverSupportStatus::kSupported:
      break;
  }

  QString message =
      QStringLiteral("PuerhLab requires an NVIDIA graphics driver with CUDA %1 or newer on "
                     "Windows.\n\n%2Please update your graphics driver to the latest version "
                     "and launch the app again.")
          .arg(required_version, detail_line);
  if (!support_info.detail.empty()) {
    message += QStringLiteral("\n\nDetails: %1").arg(QString::fromStdString(support_info.detail));
  }
  return message;
}
#endif

}  // namespace

int main(int argc, char* argv[]) {
#if QT_VERSION < QT_VERSION_CHECK(6, 0, 0)
  QCoreApplication::setAttribute(Qt::AA_EnableHighDpiScaling);
  QCoreApplication::setAttribute(Qt::AA_UseHighDpiPixmaps);
#else
  QGuiApplication::setHighDpiScaleFactorRoundingPolicy(
      Qt::HighDpiScaleFactorRoundingPolicy::PassThrough);
#endif

#if defined(_WIN32)
  const auto cuda_driver_support = puerhlab::cuda::CheckDriverSupport();
  if (!cuda_driver_support.IsSupported()) {
    ShowStartupErrorDialog(BuildCudaDriverUpdateMessage(cuda_driver_support));
    return -1;
  }
#endif

  puerhlab::TimeProvider::Refresh();
  puerhlab::RegisterAllOperators();
  Exiv2::LogMsg::setLevel(Exiv2::LogMsg::Level::error);

  QApplication app(argc, argv);
  app.setWindowIcon(QIcon(QStringLiteral(":/ICON/unnamed.png")));
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
