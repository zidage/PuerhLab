#include <qfontdatabase.h>

#include <QApplication>
#include <QComboBox>
#include <QCoreApplication>
#include <QFileDialog>
#include <QFontDatabase>
#include <QHBoxLayout>
#include <QIcon>
#include <QLabel>
#include <QListWidget>
#include <QMessageBox>
#include <QPointer>
#include <QProgressDialog>
#include <QPushButton>
#include <QPixmap>
#include <QSlider>
#include <QStyleFactory>
#include <QTimer>
#include <QVBoxLayout>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <exiv2/exiv2.hpp>
#include <filesystem>
#include <future>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include <json.hpp>
#include <opencv2/opencv.hpp>

#include "app/import_service.hpp"
#include "app/pipeline_service.hpp"
#include "app/project_service.hpp"
#include "app/render_service.hpp"
#include "app/thumbnail_service.hpp"
#include "edit/operators/operator_registeration.hpp"
#include "image/image_buffer.hpp"
#include "io/image/image_loader.hpp"
#include "renderer/pipeline_task.hpp"
#include "type/supported_file_type.hpp"
#include "ui/edit_viewer/edit_viewer.hpp"

namespace puerhlab {
namespace {
using namespace std::chrono_literals;

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

static void ApplyMaterialLikeTheme(QApplication& app) {
  app.setStyle(QStyleFactory::create("Fusion"));

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
  p.setColor(QPalette::Link, QColor(0x5F, 0xA2, 0xFF));
  p.setColor(QPalette::Highlight, QColor(0x5F, 0xA2, 0xFF));
  p.setColor(QPalette::HighlightedText, QColor(0x08, 0x0A, 0x0C));
  app.setPalette(p);
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
    if (entry.path().extension() == ".cube") {
      results.push_back(entry.path());
    }
  }

  std::sort(results.begin(), results.end(),
            [](const std::filesystem::path& a, const std::filesystem::path& b) {
              return a.filename().string() < b.filename().string();
            });

  return results;
}

static QImage MatRgba32fToQImageCopy(const cv::Mat& rgba32f_or_u8) {
  if (rgba32f_or_u8.empty()) {
    return {};
  }

  cv::Mat rgba8;
  if (rgba32f_or_u8.type() == CV_32FC4) {
    rgba32f_or_u8.convertTo(rgba8, CV_8UC4, 255.0);
  } else if (rgba32f_or_u8.type() == CV_8UC4) {
    rgba8 = rgba32f_or_u8;
  } else {
    cv::Mat tmp;
    rgba32f_or_u8.convertTo(tmp, CV_8UC4);
    rgba8 = tmp;
  }

  if (!rgba8.isContinuous()) {
    rgba8 = rgba8.clone();
  }

  QImage img(rgba8.data, rgba8.cols, rgba8.rows, static_cast<int>(rgba8.step),
             QImage::Format_RGBA8888);
  return img.copy();
}

static void SetPipelineTemplate(std::shared_ptr<PipelineExecutor> executor) {
  auto&          raw_stage     = executor->GetStage(PipelineStageName::Image_Loading);
  auto&          global_params = executor->GetGlobalParams();
  nlohmann::json decode_params;
  decode_params["raw"]["cuda"]                  = false;
  decode_params["raw"]["highlights_reconstruct"]= false;
  decode_params["raw"]["use_camera_wb"]         = true;
  decode_params["raw"]["user_wb"]               = 7500.f;
  decode_params["raw"]["backend"]               = "puerh";
  raw_stage.SetOperator(OperatorType::RAW_DECODE, decode_params);

  nlohmann::json to_ws_params;
  to_ws_params["ocio"] = {
      {"src", "ACES2065-1"}, {"dst", "ACEScc"}, {"normalize", true}, {"transform_type", 0}};
  auto& basic = executor->GetStage(PipelineStageName::Basic_Adjustment);
  basic.SetOperator(OperatorType::TO_WS, to_ws_params, global_params);

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

class EditorDialog final : public QDialog {
 public:
  EditorDialog(std::shared_ptr<ImagePoolService>    image_pool,
               std::shared_ptr<PipelineGuard>       pipeline_guard,
               sl_element_id_t                      element_id,
               image_id_t                           image_id,
               QWidget*                             parent = nullptr)
      : QDialog(parent),
        image_pool_(std::move(image_pool)),
        pipeline_guard_(std::move(pipeline_guard)),
        element_id_(element_id),
        image_id_(image_id),
        scheduler_(RenderService::GetPreviewScheduler()) {
    if (!image_pool_ || !pipeline_guard_ || !pipeline_guard_->pipeline_ || !scheduler_) {
      throw std::runtime_error("EditorDialog: missing services");
    }

    setModal(true);
    setWindowTitle(QString("Editor - element #%1").arg(static_cast<qulonglong>(element_id_)));
    resize(1500, 1000);

    auto* root = new QHBoxLayout(this);

    viewer_ = new QtEditViewer(this);
    viewer_->setMinimumSize(800, 600);
    viewer_->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);

    controls_ = new QWidget(this);
    auto* controls_layout = new QVBoxLayout(controls_);
    controls_layout->setContentsMargins(16, 16, 16, 16);
    controls_layout->setSpacing(12);
    controls_->setFixedWidth(420);

    root->addWidget(viewer_, 1);
    root->addWidget(controls_, 0);

    poll_timer_ = new QTimer(this);
    poll_timer_->setInterval(16);
    connect(poll_timer_, &QTimer::timeout, this, [this]() { PollRenderFuture(); });

    const auto luts_dir  = std::filesystem::path(CONFIG_PATH) / "LUTs";
    const auto lut_files = ListCubeLutsInDir(luts_dir);

    lut_paths_.push_back("");
    lut_names_.push_back("None");
    for (const auto& p : lut_files) {
      lut_paths_.push_back(p.generic_string());
      lut_names_.push_back(QString::fromStdString(p.filename().string()));
    }

    auto* lut_combo = new QComboBox(controls_);
    lut_combo->addItems(lut_names_);
    controls_layout->addWidget(new QLabel("LUT", controls_));
    controls_layout->addWidget(lut_combo);

    auto add_slider = [&](const QString& name, int min, int max, int value, auto&& on_change) {
      auto* label = new QLabel(name, controls_);
      auto* s     = new QSlider(Qt::Horizontal, controls_);
      s->setRange(min, max);
      s->setValue(value);
      controls_layout->addWidget(label);
      controls_layout->addWidget(s);
      connect(s, &QSlider::valueChanged, controls_,
              [on_change = std::forward<decltype(on_change)>(on_change)](int v) { on_change(v); });
      return s;
    };

    add_slider("Exposure (x100)", -200, 200, 100, [&](int v) {
      state_.exposure_ = static_cast<float>(v) / 100.0f;
      RequestRender();
    });
    add_slider("Contrast (x100)", 0, 300, 100, [&](int v) {
      state_.contrast_ = static_cast<float>(v) / 100.0f;
      RequestRender();
    });
    add_slider("Saturation", -100, 100, 0, [&](int v) {
      state_.saturation_ = static_cast<float>(v);
      RequestRender();
    });
    add_slider("Tint", -100, 100, 0, [&](int v) {
      state_.tint_ = static_cast<float>(v);
      RequestRender();
    });
    add_slider("Blacks", -100, 100, 0, [&](int v) {
      state_.blacks_ = static_cast<float>(v);
      RequestRender();
    });
    add_slider("Whites", -100, 100, 0, [&](int v) {
      state_.whites_ = static_cast<float>(v);
      RequestRender();
    });
    add_slider("Shadows", -100, 100, 0, [&](int v) {
      state_.shadows_ = static_cast<float>(v);
      RequestRender();
    });
    add_slider("Highlights", -100, 100, 0, [&](int v) {
      state_.highlights_ = static_cast<float>(v);
      RequestRender();
    });
    add_slider("Sharpen", -100, 100, 0, [&](int v) {
      state_.sharpen_ = static_cast<float>(v);
      RequestRender();
    });
    add_slider("Clarity", -100, 100, 0, [&](int v) {
      state_.clarity_ = static_cast<float>(v);
      RequestRender();
    });

    connect(lut_combo, QOverload<int>::of(&QComboBox::currentIndexChanged), controls_,
            [this](int idx) {
              if (idx < 0 || idx >= static_cast<int>(lut_paths_.size())) {
                return;
              }
              state_.lut_path_ = lut_paths_[idx];
              RequestRender();
            });

    controls_layout->addStretch();

    SetupPipeline();

    // Requirement (3): init state render via singleShot.
    QTimer::singleShot(0, this, [this]() {
      first_frame_ = true;
      state_.type_ = RenderType::FULL_RES_PREVIEW;
      RequestRender();
    });
  }

 private:
  struct AdjustmentState {
    float       exposure_   = 1.0f;
    float       contrast_   = 1.0f;
    float       saturation_ = 0.0f;
    float       tint_       = 0.0f;
    float       blacks_     = 0.0f;
    float       whites_     = 0.0f;
    float       shadows_    = 0.0f;
    float       highlights_ = 0.0f;
    float       sharpen_    = 0.0f;
    float       clarity_    = 0.0f;
    std::string lut_path_;
    RenderType  type_       = RenderType::FAST_PREVIEW;
  };

  void SetupPipeline() {
    auto img_desc = image_pool_->Read<std::shared_ptr<Image>>(
        image_id_, [](const std::shared_ptr<Image>& img) { return img; });
    auto bytes = ByteBufferLoader::LoadFromImage(img_desc);
    if (!bytes) {
      throw std::runtime_error("EditorDialog: failed to load image bytes");
    }

    base_task_.input_             = std::make_shared<ImageBuffer>(std::move(*bytes));
    base_task_.pipeline_executor_ = pipeline_guard_->pipeline_;

    auto exec = pipeline_guard_->pipeline_;
    exec->SetPreviewMode(true);

    // Requirement (1/3): use preview CPU demo's operator setup.
    SetPipelineTemplate(exec);

    auto& global_params = exec->GetGlobalParams();
    auto& basic         = exec->GetStage(PipelineStageName::Basic_Adjustment);
    basic.SetOperator(OperatorType::EXPOSURE, {{"exposure", 0.0f}}, global_params);
    basic.SetOperator(OperatorType::CONTRAST, {{"contrast", 1.0f}}, global_params);
    basic.SetOperator(OperatorType::BLACK, {{"black", 0.0f}}, global_params);
    basic.SetOperator(OperatorType::WHITE, {{"white", 0.0f}}, global_params);
    basic.SetOperator(OperatorType::SHADOWS, {{"shadows", 0.0f}}, global_params);
    basic.SetOperator(OperatorType::HIGHLIGHTS, {{"highlights", 0.0f}}, global_params);

    auto& color = exec->GetStage(PipelineStageName::Color_Adjustment);
    color.SetOperator(OperatorType::SATURATION, {{"saturation", 0.0f}}, global_params);
    color.SetOperator(OperatorType::TINT, {{"tint", 0.0f}}, global_params);

    auto& detail = exec->GetStage(PipelineStageName::Detail_Adjustment);
    detail.SetOperator(OperatorType::SHARPEN, {{"sharpen", {{"offset", 0.0f}}}}, global_params);
    detail.SetOperator(OperatorType::CLARITY, {{"clarity", 0.0f}}, global_params);

    exec->SetExecutionStages(viewer_);

    last_applied_lut_path_.clear();
    last_lmt_enabled_ = false;
  }

  void ApplyStateToPipeline() {
    auto exec = pipeline_guard_->pipeline_;
    auto& global_params = exec->GetGlobalParams();

    auto& basic = exec->GetStage(PipelineStageName::Basic_Adjustment);
    basic.SetOperator(OperatorType::EXPOSURE, {{"exposure", state_.exposure_}}, global_params);
    basic.SetOperator(OperatorType::CONTRAST, {{"contrast", state_.contrast_}}, global_params);
    basic.SetOperator(OperatorType::BLACK, {{"black", state_.blacks_}}, global_params);
    basic.SetOperator(OperatorType::WHITE, {{"white", state_.whites_}}, global_params);
    basic.SetOperator(OperatorType::SHADOWS, {{"shadows", state_.shadows_}}, global_params);
    basic.SetOperator(OperatorType::HIGHLIGHTS, {{"highlights", state_.highlights_}}, global_params);

    auto& color = exec->GetStage(PipelineStageName::Color_Adjustment);
    color.SetOperator(OperatorType::SATURATION, {{"saturation", state_.saturation_}}, global_params);
    color.SetOperator(OperatorType::TINT, {{"tint", state_.tint_}}, global_params);

    const bool want_lmt_enabled = !state_.lut_path_.empty();
    if (state_.lut_path_ != last_applied_lut_path_ || want_lmt_enabled != last_lmt_enabled_) {
      if (want_lmt_enabled) {
        global_params.lmt_enabled_ = true;
        color.SetOperator(OperatorType::LMT, {{"ocio_lmt", state_.lut_path_}}, global_params);
      } else {
        global_params.lmt_enabled_ = false;
      }
      last_applied_lut_path_ = state_.lut_path_;
      last_lmt_enabled_      = want_lmt_enabled;
    }

    auto& detail = exec->GetStage(PipelineStageName::Detail_Adjustment);
    detail.SetOperator(OperatorType::SHARPEN, {{"sharpen", {{"offset", state_.sharpen_}}}},
                       global_params);
    detail.SetOperator(OperatorType::CLARITY, {{"clarity", state_.clarity_}}, global_params);
  }

  void RequestRender() {
    pending_ = state_;
    has_pending_ = true;
    if (!inflight_) {
      StartNext();
    }
  }

  void StartNext() {
    if (!has_pending_) {
      return;
    }

    state_       = pending_;
    has_pending_ = false;

    if (first_frame_) {
      controls_->setEnabled(false);
      waiting_ = new QProgressDialog("Rendering first frame…", QString(), 0, 0, this);
      waiting_->setWindowModality(Qt::ApplicationModal);
      waiting_->setCancelButton(nullptr);
      waiting_->setMinimumDuration(0);
      waiting_->show();
      QCoreApplication::processEvents();
    }

    ApplyStateToPipeline();
    pipeline_guard_->dirty_ = true;

    PipelineTask task = base_task_;
    task.options_.render_desc_.render_type_ = state_.type_;
    task.options_.is_callback_ = false;
    task.options_.is_seq_callback_ = false;
    task.options_.is_blocking_ = true;

    auto promise = std::make_shared<std::promise<std::shared_ptr<ImageBuffer>>>();
    future_.emplace(promise->get_future());
    task.result_ = promise;

    inflight_ = true;
    scheduler_->ScheduleTask(std::move(task));
    poll_timer_->start();
  }

  void PollRenderFuture() {
    if (!inflight_ || !future_.has_value()) {
      poll_timer_->stop();
      return;
    }

    if (future_->wait_for(0ms) != std::future_status::ready) {
      return;
    }

    try {
      (void)future_->get();
    } catch (...) {
    }
    future_.reset();
    inflight_ = false;

    if (first_frame_) {
      first_frame_ = false;
      if (waiting_) {
        waiting_->close();
        waiting_->deleteLater();
        waiting_ = nullptr;
      }
      controls_->setEnabled(true);
      state_.type_ = RenderType::FAST_PREVIEW;
    }

    if (has_pending_) {
      StartNext();
    } else {
      poll_timer_->stop();
    }
  }

  std::shared_ptr<ImagePoolService>  image_pool_;
  std::shared_ptr<PipelineGuard>     pipeline_guard_;
  sl_element_id_t                    element_id_ = 0;
  image_id_t                         image_id_   = 0;

  std::shared_ptr<PipelineScheduler> scheduler_;
  PipelineTask                        base_task_{};

  QtEditViewer*                       viewer_      = nullptr;
  QWidget*                            controls_    = nullptr;
  QTimer*                             poll_timer_  = nullptr;
  QProgressDialog*                    waiting_     = nullptr;

  std::vector<std::string>            lut_paths_{};
  QStringList                         lut_names_{};

  std::string                         last_applied_lut_path_{};
  bool                                last_lmt_enabled_ = false;

  AdjustmentState                     state_{};
  AdjustmentState                     pending_{};
  bool                                first_frame_  = true;
  bool                                inflight_     = false;
  bool                                has_pending_  = false;
  std::optional<std::future<std::shared_ptr<ImageBuffer>>> future_{};
};

class AlbumWindow final : public QWidget {
 public:
  AlbumWindow(std::shared_ptr<ProjectService>      project,
              std::filesystem::path                meta_path,
              std::shared_ptr<ThumbnailService>    thumbnail_service,
              std::shared_ptr<ImagePoolService>    image_pool,
              std::shared_ptr<PipelineMgmtService> pipeline_service,
              QWidget*                             parent = nullptr)
      : QWidget(parent),
        project_(std::move(project)),
        meta_path_(std::move(meta_path)),
        thumbnails_(std::move(thumbnail_service)),
        image_pool_(std::move(image_pool)),
        pipeline_service_(std::move(pipeline_service)),
        import_service_(project_ ? project_->GetSleeveService() : nullptr,
                        project_ ? project_->GetImagePoolService() : nullptr) {
    if (!project_ || !thumbnails_ || !image_pool_ || !pipeline_service_) {
      throw std::runtime_error("AlbumWindow: missing services");
    }

    auto* root = new QVBoxLayout(this);
    root->setContentsMargins(10, 10, 10, 10);
    root->setSpacing(8);

    auto* top = new QHBoxLayout();
    import_btn_ = new QPushButton("Import…", this);
    status_     = new QLabel("No images. Click Import…", this);
    top->addWidget(import_btn_, 0);
    top->addWidget(status_, 1);
    root->addLayout(top);

    list_ = new QListWidget(this);
    list_->setViewMode(QListView::IconMode);
    list_->setResizeMode(QListWidget::Adjust);
    list_->setMovement(QListView::Static);
    list_->setIconSize(QSize(220, 160));
    list_->setGridSize(QSize(240, 190));
    list_->setSpacing(8);
    root->addWidget(list_, 1);

    connect(import_btn_, &QPushButton::clicked, this, [this]() { BeginImport(); });
    connect(list_, &QListWidget::itemClicked, this, [this](QListWidgetItem* item) {
      if (!item) {
        return;
      }
      const auto element_id = static_cast<sl_element_id_t>(item->data(Qt::UserRole + 1).toUInt());
      const auto image_id   = static_cast<image_id_t>(item->data(Qt::UserRole + 2).toUInt());
      OpenEditor(element_id, image_id);
    });
  }

 private:
  void SetBusyUi(bool busy, const QString& label) {
    import_btn_->setEnabled(!busy);
    list_->setEnabled(!busy);
    if (busy) {
      if (!busy_) {
        busy_ = new QProgressDialog(label, QString(), 0, 0, this);
        busy_->setWindowModality(Qt::ApplicationModal);
        busy_->setCancelButton(nullptr);
        busy_->setMinimumDuration(0);
        busy_->show();
      } else {
        busy_->setLabelText(label);
        busy_->show();
      }
    } else {
      if (busy_) {
        busy_->close();
        busy_->deleteLater();
        busy_ = nullptr;
      }
    }
  }

  void BeginImport() {
    if (import_inflight_) {
      return;
    }

    const QStringList files = QFileDialog::getOpenFileNames(
        this, "Import images", QString(),
        "Images (*.dng *.nef *.cr2 *.cr3 *.arw *.rw2 *.raf *.tif *.tiff *.jpg *.jpeg *.png);;All "
        "Files (*)");
    if (files.isEmpty()) {
      return;
    }

    std::vector<image_path_t> paths;
    paths.reserve(static_cast<size_t>(files.size()));
    for (const auto& f : files) {
      std::filesystem::path p = std::filesystem::path(f.toStdWString());
      if (is_supported_file(p)) {
        paths.push_back(std::move(p));
      }
    }

    if (paths.empty()) {
      QMessageBox::information(this, "Import", "No supported images selected.");
      return;
    }

    import_inflight_ = true;
    SetBusyUi(true, "Importing…");

    auto job = std::make_shared<ImportJob>();
    current_import_job_ = job;

    QPointer<AlbumWindow> self(this);
    job->on_progress_ = [self](const ImportProgress& p) {
      if (!self) {
        return;
      }
      const uint32_t total        = p.total_;
      const uint32_t placeholders = p.placeholders_created_.load();
      const uint32_t meta_done    = p.metadata_done_.load();
      const uint32_t failed       = p.failed_.load();

      QMetaObject::invokeMethod(self, [self, total, placeholders, meta_done, failed]() {
        if (!self) {
          return;
        }
        self->SetBusyUi(
            true,
            QString("Importing… %1/%2 (meta %3, failed %4)")
                .arg(placeholders)
                .arg(total)
                .arg(meta_done)
                .arg(failed));
      }, Qt::QueuedConnection);
    };

    job->on_finished_ = [self](const ImportResult& r) {
      if (!self) {
        return;
      }
      QMetaObject::invokeMethod(self, [self, r]() {
        if (!self) {
          return;
        }
        self->FinishImport(r);
      }, Qt::QueuedConnection);
    };

    current_import_job_ = import_service_.ImportToFolder(paths, image_path_t{}, {}, job);
  }

  void FinishImport(const ImportResult& result) {
    import_inflight_ = false;
    SetBusyUi(false, "");

    if (!current_import_job_ || !current_import_job_->import_log_) {
      QMessageBox::warning(this, "Import", "Import finished but no log snapshot available.");
      return;
    }

    const auto snapshot = current_import_job_->import_log_->Snapshot();

    import_service_.SyncImports(snapshot, image_path_t{});
    project_->GetSleeveService()->Sync();
    project_->GetImagePoolService()->SyncWithStorage();
    project_->SaveProject(meta_path_);

    if (result.failed_ != 0) {
      QMessageBox::warning(this, "Import",
                           QString("Import finished: %1 imported, %2 failed.")
                               .arg(result.imported_)
                               .arg(result.failed_));
    }

    for (const auto& c : snapshot.created_) {
      AddAlbumItem(c.element_id_, c.image_id_);
    }

    status_->setText(QString("Showing %1 images").arg(list_->count()));
  }

  void AddAlbumItem(sl_element_id_t element_id, image_id_t image_id) {
    auto* item = new QListWidgetItem();
    item->setText(QString("#%1").arg(static_cast<qulonglong>(element_id)));
    item->setData(Qt::UserRole + 1, static_cast<uint32_t>(element_id));
    item->setData(Qt::UserRole + 2, static_cast<uint32_t>(image_id));
    item->setSizeHint(QSize(240, 190));
    list_->addItem(item);

    // Requirement (1): clickable cells; thumbnails load async.
    CallbackDispatcher ui_dispatcher = [](std::function<void()> fn) {
      auto* obj = QCoreApplication::instance();
      if (!obj) {
        fn();
        return;
      }
      QMetaObject::invokeMethod(obj, std::move(fn), Qt::QueuedConnection);
    };

    auto                  svc  = thumbnails_;
    QPointer<AlbumWindow> self(this);
    thumbnails_->GetThumbnail(
        element_id, image_id,
        [self, svc, element_id, item, ui_dispatcher](std::shared_ptr<ThumbnailGuard> guard) {
          if (!guard || !guard->thumbnail_buffer_) {
            return;
          }

          std::thread([self, svc, element_id, item, ui_dispatcher, guard = std::move(guard)]() mutable {
            QImage scaled;
            try {
              auto* buf = guard->thumbnail_buffer_.get();
              if (!buf) {
                throw std::runtime_error("null buffer");
              }
              if (!buf->cpu_data_valid_ && buf->gpu_data_valid_) {
                buf->SyncToCPU();
              }
              if (buf->cpu_data_valid_) {
                QImage img = MatRgba32fToQImageCopy(buf->GetCPUData());
                scaled = img.scaled(220, 160, Qt::KeepAspectRatio, Qt::SmoothTransformation);
              }
            } catch (...) {
            }

            ui_dispatcher([self, element_id, item, scaled]() mutable {
              if (!self || !item) {
                return;
              }
              if (!self->list_ || self->list_->row(item) < 0) {
                return;
              }
              if (!scaled.isNull()) {
                item->setIcon(QIcon(QPixmap::fromImage(scaled)));
              }
            });

            try {
              if (svc) {
                svc->ReleaseThumbnail(element_id);
              }
            } catch (...) {
            }
          }).detach();
        },
        /*pin_if_found=*/true, ui_dispatcher);
  }

  void OpenEditor(sl_element_id_t element_id, image_id_t image_id) {
    if (element_id == 0) {
      return;
    }

    // Requirement (2): create a new preview scheduler (in dialog ctor) and pass pipeline guard.
    std::shared_ptr<PipelineGuard> guard;
    try {
      guard = pipeline_service_->LoadPipeline(element_id);
    } catch (const std::exception& e) {
      QMessageBox::warning(this, "Editor", QString("LoadPipeline failed: %1").arg(e.what()));
      return;
    }

    try {
      EditorDialog dlg(image_pool_, guard, element_id, image_id, this);
      dlg.exec();
    } catch (const std::exception& e) {
      QMessageBox::warning(this, "Editor", QString("Editor failed: %1").arg(e.what()));
    }

    // Requirement (4): return PipelineGuard back to cell / persist to DB.
    pipeline_service_->SavePipeline(guard);
    pipeline_service_->Sync();
    project_->SaveProject(meta_path_);
  }

  std::shared_ptr<ProjectService>      project_;
  std::filesystem::path                meta_path_;
  std::shared_ptr<ThumbnailService>    thumbnails_;
  std::shared_ptr<ImagePoolService>    image_pool_;
  std::shared_ptr<PipelineMgmtService> pipeline_service_;
  ImportServiceImpl                    import_service_;

  QPushButton*                         import_btn_ = nullptr;
  QLabel*                              status_     = nullptr;
  QListWidget*                         list_       = nullptr;

  bool                                 import_inflight_ = false;
  std::shared_ptr<ImportJob>           current_import_job_{};
  QProgressDialog*                     busy_ = nullptr;
};

}  // namespace
}  // namespace puerhlab

int main(int argc, char* argv[]) {
  Exiv2::LogMsg::setLevel(Exiv2::LogMsg::Level::mute);
  puerhlab::RegisterAllOperators();

  QCoreApplication::setAttribute(Qt::AA_UseDesktopOpenGL);

  QApplication app(argc, argv);
  puerhlab::ApplyExternalAppFont(app, argc, argv);
  puerhlab::ApplyMaterialLikeTheme(app);

  try {
    const auto db_path   = std::filesystem::temp_directory_path() / "album_editor_demo.db";
    const auto meta_path = std::filesystem::temp_directory_path() / "album_editor_demo.json";

    auto project = std::make_shared<puerhlab::ProjectService>(db_path, meta_path);
    auto pipeline_service =
        std::make_shared<puerhlab::PipelineMgmtService>(project->GetStorageService());
    auto thumb_service = std::make_shared<puerhlab::ThumbnailService>(
        project->GetSleeveService(), project->GetImagePoolService(), pipeline_service);

    auto* w = new puerhlab::AlbumWindow(project, meta_path, thumb_service,
                                        project->GetImagePoolService(), pipeline_service);
    w->setWindowTitle("pu-erh_lab - Album + Editor (Qt Demo)");
    w->resize(1400, 900);
    w->show();

    const int rc = app.exec();

    pipeline_service->Sync();
    project->GetImagePoolService()->SyncWithStorage();
    project->SaveProject(meta_path);
    return rc;
  } catch (const std::exception& e) {
    std::cerr << "[AlbumEditorQtDemo] Fatal: " << e.what() << std::endl;
    return 1;
  }
}
