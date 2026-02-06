#include <QApplication>
#include <QCoreApplication>
#include <QFont>
#include <QFontDatabase>
#include <QQmlApplicationEngine>
#include <QQmlContext>
#include <QQuickStyle>

#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "AlbumBackend.h"
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

auto FsPathToQString(const std::filesystem::path& path) -> QString {
#if defined(_WIN32)
  return QString::fromStdWString(path.wstring());
#else
  return QString::fromUtf8(path.string().c_str());
#endif
}

void ApplyExternalAppFont(QApplication& app, int argc, char** argv) {
  std::vector<std::filesystem::path> candidates;

  if (const auto arg = FindArgValue(argc, argv, "--font"); arg.has_value()) {
    candidates.emplace_back(std::string(arg.value()));
  }
  if (const auto env = qEnvironmentVariable("PUERHLAB_FONT_PATH"); !env.isEmpty()) {
    candidates.emplace_back(env.toStdString());
  }

  const auto app_dir =
      std::filesystem::path(QCoreApplication::applicationDirPath().toStdWString());
  candidates.emplace_back(app_dir / "fonts" / "main_IBM.ttf");

#if defined(PUERHLAB_SOURCE_DIR)
  candidates.emplace_back(std::filesystem::path(PUERHLAB_SOURCE_DIR) / "pu-erh_lab" / "src" /
                          "config" / "fonts" / "main_IBM.ttf");
#endif

  for (const auto& path : candidates) {
    std::error_code ec;
    if (!std::filesystem::exists(path, ec) || ec) {
      continue;
    }
    const int font_id = QFontDatabase::addApplicationFont(FsPathToQString(path));
    if (font_id < 0) {
      continue;
    }
    const auto families = QFontDatabase::applicationFontFamilies(font_id);
    if (families.isEmpty()) {
      continue;
    }
    app.setFont(QFont(families.front()));
    return;
  }
}

}  // namespace

int main(int argc, char* argv[]) {
  puerhlab::TimeProvider::Refresh();
  puerhlab::RegisterAllOperators();

  QCoreApplication::setAttribute(Qt::AA_UseDesktopOpenGL);
  QCoreApplication::setAttribute(Qt::AA_EnableHighDpiScaling);
  QCoreApplication::setAttribute(Qt::AA_UseHighDpiPixmaps);

  QApplication app(argc, argv);
  ApplyExternalAppFont(app, argc, argv);
  QQuickStyle::setStyle("Material");

  puerhlab::demo::AlbumBackend backend;

  QQmlApplicationEngine engine;
  engine.rootContext()->setContextProperty("albumBackend", &backend);

  QObject::connect(&engine, &QQmlApplicationEngine::objectCreationFailed, &app,
                   []() { QCoreApplication::exit(-1); }, Qt::QueuedConnection);

  engine.loadFromModule("PuerhLab.AlbumEditorQml", "Main");

  return app.exec();
}
