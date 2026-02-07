#include <opencv2/core/hal/interface.h>
#include <qlogging.h>

#include <QApplication>
#include <QBoxLayout>
#include <QComboBox>
#include <QCoreApplication>
#include <QFontDatabase>
#include <QImage>
#include <QLabel>
#include <QSlider>
#include <QStyleFactory>
#include <QTimer>
#include <algorithm>
#include <cctype>
#include <filesystem>
#include <future>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>

#include "app/import_service.hpp"
#include "app/project_service.hpp"
#include "edit/pipeline/pipeline_cpu.hpp"
#include "image/image_buffer.hpp"
#include "renderer/pipeline_scheduler.hpp"
#include "renderer/pipeline_task.hpp"
#include "sleeve/sleeve_manager.hpp"
#include "type/supported_file_type.hpp"
#include "ui/edit_viewer/edit_viewer.hpp"
#include "ui_test_fixation.hpp"

using namespace puerhlab;

static QString FsPathToQString(const std::filesystem::path& path) {
#if defined(_WIN32)
  return QString::fromStdWString(path.wstring());
#else
  return QString::fromUtf8(path.string().c_str());
#endif
}

static std::optional<std::string_view> FindArgValue(int argc, char** argv,
                                                    std::string_view opt_name) {
  const std::string opt_eq = std::string(opt_name) + "=";
  for (int i = 1; i < argc; ++i) {
    const std::string_view arg(argv[i] ? argv[i] : "");
    if (arg == opt_name) {
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

static std::vector<std::filesystem::path> ListCubeLutsInDir(const std::filesystem::path& dir) {
  std::vector<std::filesystem::path> results;
  std::error_code                    ec;
  if (!std::filesystem::exists(dir, ec) || ec || !std::filesystem::is_directory(dir, ec)) {
    return results;
  }

  for (const auto& entry : std::filesystem::directory_iterator(dir, ec)) {
    if (ec) {
      break;
    }
    if (!entry.is_regular_file(ec) || ec) {
      ec.clear();
      continue;
    }

    const auto ext = entry.path().extension().string();
    std::string ext_lower;
    ext_lower.reserve(ext.size());
    for (const char c : ext) {
      ext_lower.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
    }
    if (ext_lower == ".cube") {
      results.push_back(entry.path());
    }
  }

  std::sort(results.begin(), results.end(),
            [](const std::filesystem::path& a, const std::filesystem::path& b) {
              return a.filename().string() < b.filename().string();
            });

  return results;
}

static void ApplyExternalAppFont(QApplication& app, int argc, char** argv) {
  std::vector<std::filesystem::path> candidates;

  if (const auto arg = FindArgValue(argc, argv, "--font"); arg.has_value()) {
    candidates.emplace_back(std::string(arg.value()));
  }
  if (const auto env = qEnvironmentVariable("PUERHLAB_FONT_PATH"); !env.isEmpty()) {
    candidates.emplace_back(env.toStdString());
  }

  const auto app_dir = std::filesystem::path(QCoreApplication::applicationDirPath().toStdWString());
  candidates.emplace_back(app_dir / "fonts" / "main_IBM.ttf");

#if defined(PUERHLAB_SOURCE_DIR)
  candidates.emplace_back(std::filesystem::path(PUERHLAB_SOURCE_DIR) / "pu-erh_lab" / "src" / "config" /
                          "fonts" / "main_IBM.ttf");
#endif

  for (const auto& path : candidates) {
    std::error_code ec;
    if (!std::filesystem::exists(path, ec) || ec) {
      continue;
    }

    const int font_id = QFontDatabase::addApplicationFont(FsPathToQString(path));
    if (font_id < 0) {
      qWarning() << "[ImagePreview] Failed to load font:" << FsPathToQString(path);
      continue;
    }

    const auto families = QFontDatabase::applicationFontFamilies(font_id);
    if (families.isEmpty()) {
      qWarning() << "[ImagePreview] Loaded font but no families reported:" << FsPathToQString(path);
      continue;
    }

    QFont f(families.front());
    app.setFont(f);
    return;
  }

  qWarning() << "[ImagePreview] No external font applied. Provide `--font <path>` or set "
                "`PUERHLAB_FONT_PATH`.";
}

void SetPipelineTemplate(std::shared_ptr<PipelineExecutor> executor) {
  auto&          raw_stage     = executor->GetStage(PipelineStageName::Image_Loading);
  auto&          global_params = executor->GetGlobalParams();
  nlohmann::json decode_params;
#ifdef HAVE_CUDA
  decode_params["raw"]["cuda"] = true;
#else
  decode_params["raw"]["cuda"] = false;
#endif
  decode_params["raw"]["highlights_reconstruct"] = false;
  decode_params["raw"]["use_camera_wb"]          = true;
  decode_params["raw"]["user_wb"]                = 7500.f;
  decode_params["raw"]["backend"]                = "puerh";
  raw_stage.SetOperator(OperatorType::RAW_DECODE, decode_params);
  // raw_stage.SetOperator(OperatorType::CST, to_ws_params);

  // auto& to_ws          = executor->GetStage(PipelineStageName::To_WorkingSpace);
  // to_ws.SetOperator(OperatorType::CST, to_ws_params);
  nlohmann::json to_ws_params;
  to_ws_params["ocio"] = {
      {"src", "ACES2065-1"}, {"dst", "ACEScc"}, {"normalize", true}, {"transform_type", 0}};
  auto& input_stage = executor->GetStage(PipelineStageName::Basic_Adjustment);
  input_stage.SetOperator(OperatorType::TO_WS, to_ws_params, global_params);

  nlohmann::json output_params;
  auto&          output_stage = executor->GetStage(PipelineStageName::Output_Transform);
  output_params["ocio"]       = {
      {"src", "ACEScc"}, {"dst", "Camera Rec.709"}, {"limit", true}, {"transform_type", 1}};
  output_params["aces_odt"] = {{"encoding_space", "rec709"},
                               {"encoding_etof", "gamma_2_2"},
                               {"limiting_space", "rec709"},
                               {"peak_luminance", 100.0f}};
  output_stage.SetOperator(OperatorType::ODT, output_params, global_params);
}

static QImage cvMatToQImage(const cv::Mat& mat) {
  // if (mat.type() == CV_8UC3) {
  //   QImage image(mat.data, mat.cols, mat.rows, static_cast<int>(mat.step),
  //   QImage::Format_RGB888); return image;
  // } else if (mat.type() == CV_8UC1) {
  //   QImage image(mat.data, mat.cols, mat.rows, static_cast<int>(mat.step),
  //                QImage::Format_Grayscale8);
  //   return image;
  // } else {
  //   throw std::runtime_error("Unsupported image format for display");
  // }
  return QImage{mat.data, mat.cols, mat.rows, static_cast<int>(mat.step),
                QImage::Format_RGBA32FPx4};
}

static void ApplyMaterialLikeTheme(QApplication& app) {
  // Use a consistent cross-platform base style.
  app.setStyle(QStyleFactory::create("Fusion"));

  // Material-like dark palette.
  QPalette p;
  p.setColor(QPalette::Window, QColor(0x12, 0x12, 0x12));
  p.setColor(QPalette::WindowText, QColor(0xE8, 0xEA, 0xED));
  p.setColor(QPalette::Base, QColor(0x1E, 0x1E, 0x1E));
  p.setColor(QPalette::AlternateBase, QColor(0x20, 0x20, 0x20));
  p.setColor(QPalette::ToolTipBase, QColor(0x20, 0x20, 0x20));
  p.setColor(QPalette::ToolTipText, QColor(0xE8, 0xEA, 0xED));
  p.setColor(QPalette::Text, QColor(0xE8, 0xEA, 0xED));
  p.setColor(QPalette::Button, QColor(0x1E, 0x1E, 0x1E));
  p.setColor(QPalette::ButtonText, QColor(0xE8, 0xEA, 0xED));
  p.setColor(QPalette::BrightText, Qt::white);
  p.setColor(QPalette::Link, QColor(0x5F, 0xA2, 0xFF));
  p.setColor(QPalette::Highlight, QColor(0x5F, 0xA2, 0xFF));
  p.setColor(QPalette::HighlightedText, QColor(0x08, 0x0A, 0x0C));
  app.setPalette(p);

  // Global QSS: keep it simple, flat, and with comfortable spacing.
  app.setStyleSheet(
      "QWidget {"
      "  color: #E8EAED;"
      "  font-size: 12px;"
      "}"
      "QSlider {"
      "  font-size: 14px;"
      "}"
      "QLabel {"
      "  color: #E8EAED;"
      "}"
      "QToolTip {"
      "  background-color: #202124;"
      "  color: #E8EAED;"
      "  border: 1px solid #303134;"
      "  padding: 6px 8px;"
      "  border-radius: 8px;"
      "}"
      // Material-like slider (horizontal)
      "QSlider::groove:horizontal {"
      "  height: 4px;"
      "  background: #3C4043;"
      "  border-radius: 2px;"
      "}"
      "QSlider::sub-page:horizontal {"
      "  background: #8ab4f8;"
      "  border-radius: 2px;"
      "}"
      "QSlider::add-page:horizontal {"
      "  background: #3C4043;"
      "  border-radius: 2px;"
      "}"
      "QSlider::handle:horizontal {"
      "  background: #8ab4f8;"
      "  width: 18px;"
      "  height: 18px;"
      "  margin: -7px 0;"  // centers handle on 4px groove
      "  border-radius: 9px;"
      "}"
      "QSlider::handle:horizontal:hover {"
      "  background: #8ab4f8;"
      "}"
      "QSlider::handle:horizontal:pressed {"
      "  background: #8ab4f8;"
      "}"
      "QSlider::tick-mark {"
      "  background: transparent;"
      "}");
}

int main(int argc, char* argv[]) {
  struct AdjustmentState {
    float      exposure_   = 1.0f;
    float      contrast_   = 1.0f;
    float      saturation_ = 0.0f;
    float      tint_       = 0.0f;
    float      blacks_     = 0.0f;
    float      whites_     = 0.0f;
    float      shadows_    = 0.0f;
    float      highlights_ = 0.0f;
    float      sharpen_    = 0.0f;
    float      clarity_    = 0.0f;

    std::string lut_path_;
    RenderType type_       = RenderType::FAST_PREVIEW;
  };

  QCoreApplication::setAttribute(Qt::AA_UseDesktopOpenGL);

  QApplication app(argc, argv);
  ApplyExternalAppFont(app, argc, argv);
  ApplyMaterialLikeTheme(app);

  QWidget window;
  auto*   root   = new QHBoxLayout(&window);

  auto*   viewer = new QtEditViewer(&window);
  viewer->setMinimumSize(800, 600);
  viewer->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
  viewer->setStyleSheet(
      "QOpenGLWidget {"
      "  background: #121212;"
      "  border: 1px solid #303134;"
      "  border-radius: 12px;"
      "}");

  QWidget* controls       = new QWidget(&window);
  auto*    controlsLayout = new QVBoxLayout(controls);

  controls->setStyleSheet(
      "QWidget {"
      "  background: #1E1E1E;"
      "  border: 1px solid #303134;"
      "  border-radius: 12px;"
      "}");

  controlsLayout->setContentsMargins(16, 16, 16, 16);
  controlsLayout->setSpacing(12);

  controlsLayout->addStretch();

  root->addWidget(viewer, /*stretch*/ 1);
  root->addWidget(controls);

  window.setWindowTitle("Qt Image Preview");
  window.resize(1500, 1000);
  window.show();

  UIHistoryTests tests;
  tests.SetUp();

  // SleeveManager             manager{db_path};

  std::filesystem::path  img_root_path = std::string(TEST_IMG_PATH) + "/raw/camera/sony/a1";
  std::shared_ptr<Image> img_ptr;
  for (const auto& entry : std::filesystem::directory_iterator(img_root_path)) {
    // Load the first file
    if (is_supported_file(entry)) {
      img_ptr = std::make_shared<Image>(0, entry.path().string(), ImageType::DNG);
      break;
    }
  }

  if (!img_ptr) {
    qCritical() << "No supported image found in" << QString::fromStdString(img_root_path.string());
    return -1;
  }

  PipelineScheduler scheduler{};
  PipelineTask      base_task;
  auto              buffer = ByteBufferLoader::LoadFromImage(img_ptr);
  base_task.input_         = buffer ? std::make_shared<ImageBuffer>(std::move(*buffer)) : nullptr;

  auto base_executor       = std::make_shared<CPUPipelineExecutor>(true);
  base_executor->SetPreviewMode(true);
  base_task.pipeline_executor_ = base_executor;
  SetPipelineTemplate(base_task.pipeline_executor_);
  base_task.options_.render_desc_.render_type_ = RenderType::FULL_RES_PREVIEW;

  
  auto& global_params                          = base_task.pipeline_executor_->GetGlobalParams();
  global_params.lmt_enabled_                   = true;
  // global_params.color_wheel_enabled_           = false;
  // global_params.hls_enabled_                   = false;
  // global_params.vibrance_enabled_              = false;
  // global_params.contrast_enabled_              = false;
  // global_params.curve_enabled_                 = false;
  // global_params.saturation_enabled_            = false;

  // Register a default exposure
  auto& basic_stage = base_task.pipeline_executor_->GetStage(PipelineStageName::Basic_Adjustment);
  basic_stage.SetOperator(OperatorType::EXPOSURE, {{"exposure", 0.0f}}, global_params);
  basic_stage.SetOperator(OperatorType::CONTRAST, {{"contrast", 1.0f}}, global_params);
  basic_stage.SetOperator(OperatorType::BLACK, {{"black", 0.0f}}, global_params);
  basic_stage.SetOperator(OperatorType::WHITE, {{"white", 0.0f}}, global_params);
  basic_stage.SetOperator(OperatorType::SHADOWS, {{"shadows", 0.0f}}, global_params);
  basic_stage.SetOperator(OperatorType::HIGHLIGHTS, {{"highlights", 0.0f}}, global_params);

  auto& color_stage = base_task.pipeline_executor_->GetStage(PipelineStageName::Color_Adjustment);
  color_stage.SetOperator(OperatorType::SATURATION, {{"saturation", 0.0f}}, global_params);
  color_stage.SetOperator(OperatorType::TINT, {{"tint", 0.0f}}, global_params);

  // Prefer LUTs next to the executable (installed layout), fall back to source tree.
  const auto app_luts_dir = std::filesystem::path(
      QCoreApplication::applicationDirPath().toStdWString()) / "LUTs";
  const auto src_luts_dir = std::filesystem::path(CONFIG_PATH) / "LUTs";
  const auto luts_dir     = std::filesystem::is_directory(app_luts_dir) ? app_luts_dir : src_luts_dir;
  const auto lut_files    = ListCubeLutsInDir(luts_dir);

  std::vector<std::string> lut_paths_by_index;
  lut_paths_by_index.emplace_back("");  // index 0 => None

  QStringList lut_display_names;
  lut_display_names.push_back("None");
  for (const auto& p : lut_files) {
    lut_paths_by_index.push_back(p.generic_string());
    lut_display_names.push_back(QString::fromStdString(p.filename().string()));
  }

  int default_lut_index = 0;
  for (int i = 1; i < static_cast<int>(lut_paths_by_index.size()); ++i) {
    if (std::filesystem::path(lut_paths_by_index[i]).filename() == "5207.cube") {
      default_lut_index = i;
      break;
    }
  }

  if (default_lut_index > 0) {
    global_params.lmt_enabled_ = true;
    color_stage.SetOperator(OperatorType::LMT, {{"ocio_lmt", lut_paths_by_index[default_lut_index]}},
                            global_params);
  } else {
    global_params.lmt_enabled_ = false;
  }

  auto& detail_stage = base_task.pipeline_executor_->GetStage(PipelineStageName::Detail_Adjustment);
  detail_stage.SetOperator(OperatorType::SHARPEN, {{"sharpen", {{"offset", 0.0f}}}}, global_params);
  detail_stage.SetOperator(OperatorType::CLARITY, {{"clarity", 0.0f}}, global_params);
  // Set execution stages
  base_executor->SetExecutionStages(viewer);

  AdjustmentState adjustments{};
  adjustments.lut_path_ = lut_paths_by_index[default_lut_index];

  std::string last_applied_lut_path = adjustments.lut_path_;
  bool        last_lmt_enabled      = !adjustments.lut_path_.empty();
  auto            scheduleAdjustments = [&](const AdjustmentState& state) {
    PipelineTask task                       = base_task;

    task.options_.render_desc_.render_type_ = state.type_;
    auto& basic = task.pipeline_executor_->GetStage(PipelineStageName::Basic_Adjustment);
    basic.SetOperator(OperatorType::EXPOSURE, {{"exposure", state.exposure_}}, global_params);
    basic.SetOperator(OperatorType::CONTRAST, {{"contrast", state.contrast_}}, global_params);
    basic.SetOperator(OperatorType::BLACK, {{"black", state.blacks_}}, global_params);
    basic.SetOperator(OperatorType::WHITE, {{"white", state.whites_}}, global_params);
    basic.SetOperator(OperatorType::SHADOWS, {{"shadows", state.shadows_}}, global_params);
    basic.SetOperator(OperatorType::HIGHLIGHTS, {{"highlights", state.highlights_}}, global_params);

    auto& color = task.pipeline_executor_->GetStage(PipelineStageName::Color_Adjustment);
    color.SetOperator(OperatorType::SATURATION, {{"saturation", state.saturation_}}, global_params);
    color.SetOperator(OperatorType::TINT, {{"tint", state.tint_}}, global_params);

    // Avoid re-applying the LUT (LMT) on every slider change; only update when it changes.
    const bool want_lmt_enabled = !state.lut_path_.empty();
    if (state.lut_path_ != last_applied_lut_path || want_lmt_enabled != last_lmt_enabled) {
      if (want_lmt_enabled) {
        global_params.lmt_enabled_ = true;
        color.SetOperator(OperatorType::LMT, {{"ocio_lmt", state.lut_path_}}, global_params);
      } else {
        global_params.lmt_enabled_ = false;
      }
      last_applied_lut_path = state.lut_path_;
      last_lmt_enabled      = want_lmt_enabled;
    }

    auto& detail = task.pipeline_executor_->GetStage(PipelineStageName::Detail_Adjustment);

    detail.SetOperator(OperatorType::SHARPEN, {{"sharpen", {{"offset", state.sharpen_}}}},
                                  global_params);
    detail.SetOperator(OperatorType::CLARITY, {{"clarity", state.clarity_}}, global_params);

    task.options_.is_blocking_ = false;
    task.options_.is_callback_ = true;

    task.callback_             = [viewer](ImageBuffer&) {
      // viewer->NotifyFrameReady();
    };

    scheduler.ScheduleTask(std::move(task));
  };

  auto addComboBox = [&](const QString& name, const QStringList& items, int initial_index,
                         auto&& onChange) {
    auto* label = new QLabel(name, controls);
    label->setStyleSheet(
        "QLabel {"
        "  color: #E8EAED;"
        "  font-size: 14px;"
        "  font-weight: 400;"
        "}");

    auto* combo = new QComboBox(controls);
    combo->addItems(items);
    combo->setCurrentIndex(initial_index);
    combo->setMinimumWidth(240);
    combo->setFixedHeight(32);
    combo->setStyleSheet(
        "QComboBox {"
        "  background: #202124;"
        "  border: 1px solid #303134;"
        "  border-radius: 8px;"
        "  padding: 4px 8px;"
        "}"
        "QComboBox::drop-down {"
        "  border: 0px;"
        "  width: 24px;"
        "}"
        "QComboBox QAbstractItemView {"
        "  background: #202124;"
        "  border: 1px solid #303134;"
      "  selection-background-color: #8ab4f8;"
      "  selection-color: #080A0C;"
      "}"
      "QComboBox QAbstractItemView::item:hover {"
      "  background: #2B2F33;"
      "  color: #E8EAED;"
      "}"
      "QComboBox QAbstractItemView::item:selected {"
      "  background: #8ab4f8;"
      "  color: #080A0C;"
        "}");

    QObject::connect(combo, QOverload<int>::of(&QComboBox::currentIndexChanged), controls,
                     [onChange = std::forward<decltype(onChange)>(onChange)](int idx) {
                       onChange(idx);
                     });

    auto* row       = new QWidget(controls);
    auto* rowLayout = new QHBoxLayout(row);
    rowLayout->setContentsMargins(0, 0, 0, 0);
    rowLayout->addWidget(label, /*stretch*/ 1);
    rowLayout->addWidget(combo);

    controlsLayout->insertWidget(controlsLayout->count() - 1, row);
    return combo;
  };

  addComboBox(
      "LUT", lut_display_names, default_lut_index,
      [&](int idx) {
        if (idx < 0 || idx >= static_cast<int>(lut_paths_by_index.size())) {
          return;
        }
        adjustments.lut_path_ = lut_paths_by_index[idx];
        scheduleAdjustments(adjustments);
      });

  auto addSlider = [&](const QString& name, int min, int max, int value, auto&& onChange,
                       auto&& formatter) {
    auto* info = new QLabel(QString("%1: %2").arg(name).arg(formatter(value)), controls);
    info->setStyleSheet(
        "QLabel {"
        "  color: #E8EAED;"
        "  font-size: 14px;"
        "  font-weight: 400;"
        "}");

    auto* slider = new QSlider(Qt::Horizontal, controls);
    slider->setRange(min, max);
    slider->setValue(value);
    slider->setSingleStep(1);
    slider->setPageStep(std::max(1, (max - min) / 20));
    slider->setMinimumWidth(240);
    // Larger hit area without changing the Material-like track thickness.
    slider->setFixedHeight(32);

    QObject::connect(
        slider, &QSlider::valueChanged, controls,
        [info, name, formatter, onChange = std::forward<decltype(onChange)>(onChange)](int v) {
          info->setText(QString("%1: %2").arg(name).arg(formatter(v)));
          onChange(v);
        });

    auto* row       = new QWidget(controls);
    auto* rowLayout = new QHBoxLayout(row);
    rowLayout->setContentsMargins(0, 0, 0, 0);
    rowLayout->addWidget(info, /*stretch*/ 1);
    rowLayout->addWidget(slider);

    controlsLayout->insertWidget(controlsLayout->count() - 1, row);
  };

  addSlider(
      "Exposure", -1000, 1000, 100,
      [&](int v) {
        adjustments.exposure_ = static_cast<float>(v) / 100.0f;
        scheduleAdjustments(adjustments);
      },
      [](int v) { return QString::number(v / 100.0, 'f', 2); });

  addSlider(
      "Contrast", -100, 100, 0,
      [&](int v) {
        adjustments.contrast_ = static_cast<float>(v);
        scheduleAdjustments(adjustments);
      },
      [](int v) { return QString::number(v, 'f', 2); });

  addSlider(
      "Saturation", -100, 100, 0,
      [&](int v) {
        adjustments.saturation_ = static_cast<float>(v);
        scheduleAdjustments(adjustments);
      },
      [](int v) { return QString::number(v, 'f', 2); });

  addSlider(
      "Tint", -100, 100, 0,
      [&](int v) {
        adjustments.tint_ = static_cast<float>(v);
        scheduleAdjustments(adjustments);
      },
      [](int v) { return QString::number(v, 'f', 2); });

  addSlider(
      "Blacks", -100, 100, 0,
      [&](int v) {
        adjustments.blacks_ = static_cast<float>(v);
        scheduleAdjustments(adjustments);
      },
      [](int v) { return QString::number(v, 'f', 2); });

  addSlider(
      "Whites", -100, 100, 0,
      [&](int v) {
        adjustments.whites_ = static_cast<float>(v);
        scheduleAdjustments(adjustments);
      },
      [](int v) { return QString::number(v, 'f', 2); });

  addSlider(
      "Shadows", -100, 100, 0,
      [&](int v) {
        adjustments.shadows_ = static_cast<float>(v);
        scheduleAdjustments(adjustments);
      },
      [](int v) { return QString::number(v, 'f', 2); });

  addSlider(
      "Highlights", -100, 100, 0,
      [&](int v) {
        adjustments.highlights_ = static_cast<float>(v);
        scheduleAdjustments(adjustments);
      },
      [](int v) { return QString::number(v, 'f', 2); });

  addSlider(
      "Sharpen", -100, 100, 0,
      [&](int v) {
        adjustments.sharpen_ = static_cast<float>(v);
        scheduleAdjustments(adjustments);
      },
      [](int v) { return QString::number(v, 'f', 2); });

  addSlider(
      "Clarity", -100, 100, 0,
      [&](int v) {
        adjustments.clarity_ = static_cast<float>(v);
        scheduleAdjustments(adjustments);
      },
      [](int v) { return QString::number(v, 'f', 2); });

  // Defer initial scheduling until the event loop starts so the QOpenGLWidget
  // has a valid OpenGL context before any frame sink resize/map.
  AdjustmentState init_state = adjustments;
  init_state.type_           = RenderType::FULL_RES_PREVIEW; // TODO: For benchmarking only
  QTimer::singleShot(0, &window, [&, init_state]() { scheduleAdjustments(init_state); });

  int ret = app.exec();
  tests.TearDown();
  return ret;
}
