#include "EditorDialog.h"

#include <QAbstractItemView>
#include <QComboBox>
#include <QCoreApplication>
#include <QDateTime>
#include <QDialog>
#include <QEventLoop>
#include <QFontDatabase>
#include <QFrame>
#include <QGridLayout>
#include <QHBoxLayout>
#include <QLabel>
#include <QListWidget>
#include <QMessageBox>
#include <QPainter>
#include <QPushButton>
#include <QScrollArea>
#include <QSlider>
#include <QStyle>
#include <QTimer>
#include <QVBoxLayout>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <future>
#include <json.hpp>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "app/render_service.hpp"
#include "edit/history/edit_transaction.hpp"
#include "edit/history/version.hpp"
#include "image/image.hpp"
#include "io/image/image_loader.hpp"
#include "renderer/pipeline_task.hpp"
#include "ui/edit_viewer/edit_viewer.hpp"

namespace puerhlab::demo {
namespace {

using namespace std::chrono_literals;

auto ListCubeLutsInDir(const std::filesystem::path& dir) -> std::vector<std::filesystem::path> {
  std::vector<std::filesystem::path> files;
  std::error_code                    ec;
  if (!std::filesystem::exists(dir, ec) || ec) {
    return files;
  }

  for (const auto& entry : std::filesystem::directory_iterator(dir, ec)) {
    if (ec) {
      break;
    }
    if (!entry.is_regular_file()) {
      continue;
    }
    std::wstring ext = entry.path().extension().wstring();
    std::transform(ext.begin(), ext.end(), ext.begin(), ::towlower);
    if (ext == L".cube") {
      files.push_back(entry.path());
    }
  }

  std::sort(files.begin(), files.end(),
            [](const std::filesystem::path& a, const std::filesystem::path& b) {
              return a.filename().wstring() < b.filename().wstring();
            });
  return files;
}
class SpinnerWidget final : public QWidget {
 public:
  explicit SpinnerWidget(QWidget* parent = nullptr) : QWidget(parent) {
    setFixedSize(22, 22);
    setAttribute(Qt::WA_TransparentForMouseEvents);
    setAttribute(Qt::WA_TranslucentBackground);

    timer_ = new QTimer(this);
    timer_->setInterval(16);
    QObject::connect(timer_, &QTimer::timeout, this, [this]() {
      angle_deg_ = (angle_deg_ + 18) % 360;
      update();
    });
    hide();
  }

  void Start() {
    show();
    raise();
    if (!timer_->isActive()) {
      timer_->start();
    }
  }

  void Stop() {
    if (timer_->isActive()) {
      timer_->stop();
    }
    hide();
  }

 protected:
  void paintEvent(QPaintEvent*) override {
    QPainter painter(this);
    painter.setRenderHint(QPainter::Antialiasing, true);

    const QRectF r = QRectF(2.5, 2.5, width() - 5.0, height() - 5.0);

    // Subtle background ring.
    {
      QPen pen(QColor(0x30, 0x31, 0x34, 200));
      pen.setWidthF(2.0);
      pen.setCapStyle(Qt::RoundCap);
      painter.setPen(pen);
      painter.drawArc(r, 0 * 16, 360 * 16);
    }

    // Foreground arc.
    {
      QPen pen(QColor(0x8a, 0xb4, 0xf8, 230));
      pen.setWidthF(2.2);
      pen.setCapStyle(Qt::RoundCap);
      painter.setPen(pen);
      painter.drawArc(r, (90 - angle_deg_) * 16, 100 * 16);
    }
  }

 private:
  QTimer* timer_     = nullptr;
  int     angle_deg_ = 0;
};

class HistoryLaneWidget final : public QWidget {
 public:
  HistoryLaneWidget(QColor dot, QColor line, bool draw_top, bool draw_bottom,
                    QWidget* parent = nullptr)
      : QWidget(parent),
        dot_(std::move(dot)),
        line_(std::move(line)),
        draw_top_(draw_top),
        draw_bottom_(draw_bottom) {
    setFixedWidth(18);
    setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Expanding);
    setAttribute(Qt::WA_TransparentForMouseEvents);
  }

  void SetConnectors(bool draw_top, bool draw_bottom) {
    draw_top_    = draw_top;
    draw_bottom_ = draw_bottom;
    update();
  }

 protected:
  void paintEvent(QPaintEvent*) override {
    QPainter p(this);
    p.setRenderHint(QPainter::Antialiasing, true);

    const int cx = width() / 2;
    const int cy = height() / 2;

    // Vertical lane.
    {
      QPen pen(line_);
      pen.setWidthF(2.0);
      pen.setCapStyle(Qt::RoundCap);
      p.setPen(pen);

      if (draw_top_) {
        p.drawLine(QPointF(cx, 2.0), QPointF(cx, cy - 6.0));
      }
      if (draw_bottom_) {
        p.drawLine(QPointF(cx, cy + 6.0), QPointF(cx, height() - 2.0));
      }
    }

    // Node.
    {
      p.setPen(Qt::NoPen);
      p.setBrush(dot_);
      p.drawEllipse(QPointF(cx, cy), 4.4, 4.4);
      p.setBrush(QColor(0x12, 0x12, 0x12));
      p.drawEllipse(QPointF(cx, cy), 2.0, 2.0);
    }
  }

 private:
  QColor dot_;
  QColor line_;
  bool   draw_top_    = false;
  bool   draw_bottom_ = false;
};

class HistoryCardWidget final : public QFrame {
 public:
  explicit HistoryCardWidget(QWidget* parent = nullptr) : QFrame(parent) {
    setObjectName("HistoryCard");
    setAttribute(Qt::WA_StyledBackground, true);
    setAttribute(Qt::WA_Hover, true);
    setProperty("selected", false);

    setStyleSheet(
        "QFrame#HistoryCard {"
        "  background: #16181A;"
        "  border: 1px solid #303134;"
        "  border-radius: 10px;"
        "}"
        "QFrame#HistoryCard:hover {"
        "  background: #1E2124;"
        "}"
        "QFrame#HistoryCard[selected=\"true\"] {"
        "  background: rgba(138, 180, 248, 0.14);"
        "  border: 1px solid rgba(138, 180, 248, 0.55);"
        "}");
  }

  void SetSelected(bool selected) {
    if (property("selected").toBool() == selected) {
      return;
    }
    setProperty("selected", selected);
    style()->unpolish(this);
    style()->polish(this);
    update();
  }
};

static QLabel* MakePillLabel(const QString& text, const QString& fg, const QString& bg,
                             const QString& border, QWidget* parent) {
  auto* l = new QLabel(text, parent);
  l->setStyleSheet(QString("QLabel {"
                           "  color: %1;"
                           "  background: %2;"
                           "  border: 1px solid %3;"
                           "  border-radius: 10px;"
                           "  padding: 1px 7px;"
                           "  font-size: 11px;"
                           "}")
                       .arg(fg, bg, border));
  return l;
}

class EditorDialog final : public QDialog {
 public:
  enum class WorkingMode : int { Incremental = 0, Plain = 1 };

  EditorDialog(std::shared_ptr<ImagePoolService>       image_pool,
               std::shared_ptr<PipelineGuard>          pipeline_guard,
               std::shared_ptr<EditHistoryMgmtService> history_service,
               std::shared_ptr<EditHistoryGuard> history_guard, sl_element_id_t element_id,
               image_id_t image_id, QWidget* parent = nullptr)
      : QDialog(parent),
        image_pool_(std::move(image_pool)),
        pipeline_guard_(std::move(pipeline_guard)),
        history_service_(std::move(history_service)),
        history_guard_(std::move(history_guard)),
        element_id_(element_id),
        image_id_(image_id),
        scheduler_(RenderService::GetPreviewScheduler()) {
    if (!image_pool_ || !pipeline_guard_ || !pipeline_guard_->pipeline_ || !history_service_ ||
        !history_guard_ || !history_guard_->history_ || !scheduler_) {
      throw std::runtime_error("EditorDialog: missing services");
    }

    setModal(true);
    setSizeGripEnabled(true);
    setWindowFlag(Qt::WindowMinMaxButtonsHint, true);
    setWindowFlag(Qt::MSWindowsFixedSizeDialogHint, false);
    setWindowTitle(QString("Editor - element #%1").arg(static_cast<qulonglong>(element_id_)));
    setMinimumSize(1080, 680);
    resize(1500, 1000);

    auto* root = new QHBoxLayout(this);
    root->setContentsMargins(10, 10, 10, 10);
    root->setSpacing(12);

    viewer_    = new QtEditViewer(this);
    viewer_->setMinimumSize(560, 360);
    viewer_->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    viewer_->setStyleSheet(
        "QOpenGLWidget {"
        "  background: #121212;"
        "  border: 1px solid #303134;"
        "  border-radius: 12px;"
        "}");

    // Viewer container allows a small overlay spinner during rendering.
    viewer_container_ = new QWidget(this);
    auto* viewer_grid = new QGridLayout(viewer_container_);
    viewer_grid->setContentsMargins(0, 0, 0, 0);
    viewer_grid->setSpacing(0);
    viewer_grid->addWidget(viewer_, 0, 0);

    spinner_ = new SpinnerWidget(viewer_container_);
    viewer_grid->addWidget(spinner_, 0, 0, Qt::AlignRight | Qt::AlignBottom);
    viewer_grid->setRowStretch(0, 1);
    viewer_grid->setColumnStretch(0, 1);

    controls_scroll_ = new QScrollArea(this);
    controls_scroll_->setFrameShape(QFrame::NoFrame);
    controls_scroll_->setWidgetResizable(true);
    controls_scroll_->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    controls_scroll_->setMinimumWidth(380);
    controls_scroll_->setMaximumWidth(540);
    controls_scroll_->setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Expanding);
    controls_scroll_->setStyleSheet(
        "QScrollArea { background: transparent; border: none; }"
        "QScrollBar:vertical {"
        "  background: #101822;"
        "  width: 10px;"
        "  margin: 2px;"
        "  border-radius: 5px;"
        "}"
        "QScrollBar::handle:vertical {"
        "  background: #2E415A;"
        "  border-radius: 5px;"
        "}"
        "QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0px; }"
        "QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background: transparent; }");

    controls_             = new QWidget(controls_scroll_);
    auto* controls_layout = new QVBoxLayout(controls_);
    controls_layout->setContentsMargins(16, 16, 16, 16);
    controls_layout->setSpacing(12);
    controls_->setMinimumWidth(380);
    controls_->setObjectName("EditorControlsPanel");
    controls_->setStyleSheet(
        "#EditorControlsPanel {"
        "  background: #161E2A;"
        "  border: 1px solid #2C3D55;"
        "  border-radius: 14px;"
        "}"
        "#EditorControlsPanel QFrame#EditorSection {"
        "  background: #111923;"
        "  border: 1px solid #2B3B51;"
        "  border-radius: 12px;"
        "}"
        "#EditorControlsPanel QLabel#EditorSectionTitle {"
        "  color: #EAF2FF;"
        "  font-size: 13px;"
        "  font-weight: 620;"
        "}"
        "#EditorControlsPanel QLabel#EditorSectionSub {"
        "  color: #9DB0D0;"
        "  font-size: 11px;"
        "}");
    controls_scroll_->setWidget(controls_);

    root->addWidget(viewer_container_, 1);
    root->addWidget(controls_scroll_, 0);

    auto* controls_header = new QLabel("Adjustments", controls_);
    controls_header->setObjectName("SectionTitle");
    auto* controls_sub = new QLabel("Editing controls are scoped to the active image.", controls_);
    controls_sub->setObjectName("MetaText");
    controls_sub->setWordWrap(true);
    controls_layout->addWidget(controls_header, 0);
    controls_layout->addWidget(controls_sub, 0);

    // Prefer LUTs next to the executable (installed layout), fall back to source tree.
    const auto app_luts_dir = std::filesystem::path(
        QCoreApplication::applicationDirPath().toStdWString()) / "LUTs";
    const auto src_luts_dir = std::filesystem::path(CONFIG_PATH) / "LUTs";
    const auto luts_dir     = std::filesystem::is_directory(app_luts_dir) ? app_luts_dir : src_luts_dir;
    const auto lut_files    = ListCubeLutsInDir(luts_dir);

    lut_paths_.push_back("");  // index 0 => None
    lut_names_.push_back("None");
    for (const auto& p : lut_files) {
      lut_paths_.push_back(p.generic_string());
      lut_names_.push_back(QString::fromStdString(p.filename().string()));
    }

    auto addSection = [&](const QString& title, const QString& subtitle) {
      auto* frame = new QFrame(controls_);
      frame->setObjectName("EditorSection");
      auto* v = new QVBoxLayout(frame);
      v->setContentsMargins(12, 10, 12, 10);
      v->setSpacing(2);

      auto* t = new QLabel(title, frame);
      t->setObjectName("EditorSectionTitle");
      auto* s = new QLabel(subtitle, frame);
      s->setObjectName("EditorSectionSub");
      s->setWordWrap(true);
      v->addWidget(t, 0);
      v->addWidget(s, 0);
      controls_layout->addWidget(frame, 0);
    };

    controls_layout->addStretch();

    int default_lut_index = 0;
    for (int i = 1; i < static_cast<int>(lut_paths_.size()); ++i) {
      if (std::filesystem::path(lut_paths_[i]).filename() == "5207.cube") {
        default_lut_index = i;
        break;
      }
    }

    // If the pipeline already has operator params (loaded from PipelineService/storage),
    // initialize UI state from those params rather than overwriting them.
    const bool loaded_state_from_pipeline = LoadStateFromPipelineIfPresent();
    if (!loaded_state_from_pipeline) {
      // Demo-friendly default: apply a LUT only for brand-new pipelines with no saved params.
      state_.lut_path_ = lut_paths_[default_lut_index];
    }
    committed_state_ = state_;

    // Seed a working version from the latest committed one (if any).
    try {
      const auto parent_id = history_guard_->history_->GetLatestVersion().ver_ref_.GetVersionID();
      working_version_     = Version(element_id_, parent_id);
    } catch (...) {
      working_version_ = Version(element_id_);
    }
    working_version_.SetBasePipelineExecutor(pipeline_guard_->pipeline_);

    int initial_lut_index = 0;
    if (!state_.lut_path_.empty()) {
      auto it = std::find(lut_paths_.begin(), lut_paths_.end(), state_.lut_path_);
      if (it != lut_paths_.end()) {
        initial_lut_index = static_cast<int>(std::distance(lut_paths_.begin(), it));
      } else {
        // Keep UI consistent even if LUT path comes from an external/custom location.
        lut_paths_.push_back(state_.lut_path_);
        lut_names_.push_back(
            QString::fromStdString(std::filesystem::path(state_.lut_path_).filename().string()));
        initial_lut_index = static_cast<int>(lut_paths_.size() - 1);
      }
    }

    auto addComboBox = [&](const QString& name, const QStringList& items, int initial_index,
                           auto&& onChange) {
      auto* label = new QLabel(name, controls_);
      label->setStyleSheet(
          "QLabel {"
          "  color: #E8EAED;"
          "  font-size: 14px;"
          "  font-weight: 400;"
          "}");

      auto* combo = new QComboBox(controls_);
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

      QObject::connect(
          combo, QOverload<int>::of(&QComboBox::currentIndexChanged), controls_,
          [onChange = std::forward<decltype(onChange)>(onChange)](int idx) { onChange(idx); });

      auto* row       = new QWidget(controls_);
      auto* rowLayout = new QHBoxLayout(row);
      rowLayout->setContentsMargins(0, 0, 0, 0);
      rowLayout->addWidget(label, /*stretch*/ 1);
      rowLayout->addWidget(combo);

      controls_layout->insertWidget(controls_layout->count() - 1, row);
      return combo;
    };

    auto addSlider = [&](const QString& name, int min, int max, int value, auto&& onChange,
                         auto&& onRelease, auto&& formatter) {
      auto* info = new QLabel(QString("%1: %2").arg(name).arg(formatter(value)), controls_);
      info->setStyleSheet(
          "QLabel {"
          "  color: #E8EAED;"
          "  font-size: 14px;"
          "  font-weight: 400;"
          "}");

      auto* slider = new QSlider(Qt::Horizontal, controls_);
      slider->setRange(min, max);
      slider->setValue(value);
      slider->setSingleStep(1);
      slider->setPageStep(std::max(1, (max - min) / 20));
      slider->setMinimumWidth(240);
      slider->setFixedHeight(32);

      QObject::connect(
          slider, &QSlider::valueChanged, controls_,
          [info, name, formatter, onChange = std::forward<decltype(onChange)>(onChange)](int v) {
            info->setText(QString("%1: %2").arg(name).arg(formatter(v)));
            onChange(v);
          });

      QObject::connect(
          slider, &QSlider::sliderReleased, controls_,
          [onRelease = std::forward<decltype(onRelease)>(onRelease)]() { onRelease(); });

      auto* row       = new QWidget(controls_);
      auto* rowLayout = new QHBoxLayout(row);
      rowLayout->setContentsMargins(0, 0, 0, 0);
      rowLayout->addWidget(info, /*stretch*/ 1);
      rowLayout->addWidget(slider);

      controls_layout->insertWidget(controls_layout->count() - 1, row);
      return slider;
    };

    addSection("Pipeline", "Choose LUT and render behavior.");

    lut_combo_ = addComboBox("LUT", lut_names_, initial_lut_index, [&](int idx) {
      if (idx < 0 || idx >= static_cast<int>(lut_paths_.size())) {
        return;
      }
      state_.lut_path_ = lut_paths_[idx];
      CommitAdjustment(AdjustmentField::Lut);
    });

    addSection("Tone", "Primary tonal shaping controls.");

    exposure_slider_ = addSlider(
        "Exposure", -1000, 1000, static_cast<int>(std::lround(state_.exposure_ * 100.0f)),
        [&](int v) {
          state_.exposure_ = static_cast<float>(v) / 100.0f;
          RequestRender();
        },
        [this]() { CommitAdjustment(AdjustmentField::Exposure); },
        [](int v) { return QString::number(v / 100.0, 'f', 2); });

    contrast_slider_ = addSlider(
        "Contrast", -100, 100, static_cast<int>(std::lround(state_.contrast_)),
        [&](int v) {
          state_.contrast_ = static_cast<float>(v);
          RequestRender();
        },
        [this]() { CommitAdjustment(AdjustmentField::Contrast); },
        [](int v) { return QString::number(v, 'f', 2); });

    addSection("Color", "Color balance and saturation.");

    saturation_slider_ = addSlider(
        "Saturation", -100, 100, static_cast<int>(std::lround(state_.saturation_)),
        [&](int v) {
          state_.saturation_ = static_cast<float>(v);
          RequestRender();
        },
        [this]() { CommitAdjustment(AdjustmentField::Saturation); },
        [](int v) { return QString::number(v, 'f', 2); });

    tint_slider_ = addSlider(
        "Tint", -100, 100, static_cast<int>(std::lround(state_.tint_)),
        [&](int v) {
          state_.tint_ = static_cast<float>(v);
          RequestRender();
        },
        [this]() { CommitAdjustment(AdjustmentField::Tint); },
        [](int v) { return QString::number(v, 'f', 2); });

    blacks_slider_ = addSlider(
        "Blacks", -100, 100, static_cast<int>(std::lround(state_.blacks_)),
        [&](int v) {
          state_.blacks_ = static_cast<float>(v);
          RequestRender();
        },
        [this]() { CommitAdjustment(AdjustmentField::Blacks); },
        [](int v) { return QString::number(v, 'f', 2); });

    whites_slider_ = addSlider(
        "Whites", -100, 100, static_cast<int>(std::lround(state_.whites_)),
        [&](int v) {
          state_.whites_ = static_cast<float>(v);
          RequestRender();
        },
        [this]() { CommitAdjustment(AdjustmentField::Whites); },
        [](int v) { return QString::number(v, 'f', 2); });

    shadows_slider_ = addSlider(
        "Shadows", -100, 100, static_cast<int>(std::lround(state_.shadows_)),
        [&](int v) {
          state_.shadows_ = static_cast<float>(v);
          RequestRender();
        },
        [this]() { CommitAdjustment(AdjustmentField::Shadows); },
        [](int v) { return QString::number(v, 'f', 2); });

    highlights_slider_ = addSlider(
        "Highlights", -100, 100, static_cast<int>(std::lround(state_.highlights_)),
        [&](int v) {
          state_.highlights_ = static_cast<float>(v);
          RequestRender();
        },
        [this]() { CommitAdjustment(AdjustmentField::Highlights); },
        [](int v) { return QString::number(v, 'f', 2); });

    addSection("Detail", "Micro-contrast and sharpen controls.");

    sharpen_slider_ = addSlider(
        "Sharpen", -100, 100, static_cast<int>(std::lround(state_.sharpen_)),
        [&](int v) {
          state_.sharpen_ = static_cast<float>(v);
          RequestRender();
        },
        [this]() { CommitAdjustment(AdjustmentField::Sharpen); },
        [](int v) { return QString::number(v, 'f', 2); });

    clarity_slider_ = addSlider(
        "Clarity", -100, 100, static_cast<int>(std::lround(state_.clarity_)),
        [&](int v) {
          state_.clarity_ = static_cast<float>(v);
          RequestRender();
        },
        [this]() { CommitAdjustment(AdjustmentField::Clarity); },
        [](int v) { return QString::number(v, 'f', 2); });

    addSection("Versioning", "Commit and inspect edit history.");

    // Edit-history commit controls.
    {
      auto* row       = new QWidget(controls_);
      auto* rowLayout = new QHBoxLayout(row);
      rowLayout->setContentsMargins(0, 0, 0, 0);
      rowLayout->setSpacing(10);

      version_status_ = new QLabel(row);
      version_status_->setStyleSheet(
          "QLabel {"
          "  color: #AAB0B6;"
          "  font-size: 12px;"
          "}");

      commit_version_btn_ = new QPushButton("Commit Version", row);
      commit_version_btn_->setFixedHeight(32);
      commit_version_btn_->setStyleSheet(
          "QPushButton {"
          "  background: #2B2F33;"
          "  border: 1px solid #303134;"
          "  border-radius: 8px;"
          "  padding: 6px 10px;"
          "}"
          "QPushButton:hover {"
          "  background: #34383C;"
          "}"
          "QPushButton:disabled {"
          "  color: #6B7075;"
          "  background: #1E1E1E;"
          "}");

      rowLayout->addWidget(version_status_, /*stretch*/ 1);
      rowLayout->addWidget(commit_version_btn_, /*stretch*/ 0);
      controls_layout->insertWidget(controls_layout->count() - 1, row);

      QObject::connect(commit_version_btn_, &QPushButton::clicked, this,
                       [this]() { CommitWorkingVersion(); });
    }

    // Edit-history visualization ("git log"-like) + working version mode.
    {
      auto* frame = new QFrame(controls_);
      frame->setStyleSheet(
          "QFrame {"
          "  background: transparent;"
          "  border: 1px solid #303134;"
          "  border-radius: 12px;"
          "}");

      auto* layout = new QVBoxLayout(frame);
      layout->setContentsMargins(10, 10, 10, 10);
      layout->setSpacing(8);

      auto* mode_row    = new QWidget(frame);
      auto* mode_layout = new QHBoxLayout(mode_row);
      mode_layout->setContentsMargins(0, 0, 0, 0);
      mode_layout->setSpacing(8);

      auto* mode_label = new QLabel("Working version:", mode_row);
      mode_label->setStyleSheet(
          "QLabel {"
          "  color: #AAB0B6;"
          "  font-size: 12px;"
          "}");

      working_mode_combo_ = new QComboBox(mode_row);
      working_mode_combo_->addItem("Incremental (from latest)",
                                   static_cast<int>(WorkingMode::Incremental));
      working_mode_combo_->addItem("Plain (no parent)", static_cast<int>(WorkingMode::Plain));
      working_mode_combo_->setFixedHeight(28);
      working_mode_combo_->setStyleSheet(
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
          "}");

      new_working_btn_ = new QPushButton("New", mode_row);
      new_working_btn_->setFixedHeight(28);
      new_working_btn_->setStyleSheet(
          "QPushButton {"
          "  background: #2B2F33;"
          "  border: 1px solid #303134;"
          "  border-radius: 8px;"
          "  padding: 4px 10px;"
          "}"
          "QPushButton:hover {"
          "  background: #34383C;"
          "}");

      mode_layout->addWidget(mode_label, /*stretch*/ 0);
      mode_layout->addWidget(working_mode_combo_, /*stretch*/ 1);
      mode_layout->addWidget(new_working_btn_, /*stretch*/ 0);

      layout->addWidget(mode_row);

      auto* versions_label = new QLabel("Versions", frame);
      versions_label->setStyleSheet(
          "QLabel {"
          "  color: #E8EAED;"
          "  font-size: 12px;"
          "  font-weight: 500;"
          "}");
      layout->addWidget(versions_label);

      version_log_ = new QListWidget(frame);
      version_log_->setSelectionMode(QAbstractItemView::SingleSelection);
      version_log_->setSpacing(6);
      version_log_->setVerticalScrollMode(QAbstractItemView::ScrollPerPixel);
      version_log_->setMinimumHeight(150);
      version_log_->setStyleSheet(
          "QListWidget {"
          "  background: #121212;"
          "  border: 1px solid #303134;"
          "  border-radius: 10px;"
          "  padding: 6px;"
          "}"
          "QListWidget::item {"
          "  padding: 2px;"
          "}"
          "QListWidget::item:selected {"
          "  background: transparent;"
          "}");
      layout->addWidget(version_log_);

      QObject::connect(version_log_, &QListWidget::itemSelectionChanged, this,
                       [this]() { RefreshVersionLogSelectionStyles(); });

      auto* tx_label = new QLabel("Uncommitted transactions (stack)", frame);
      tx_label->setStyleSheet(
          "QLabel {"
          "  color: #E8EAED;"
          "  font-size: 12px;"
          "  font-weight: 500;"
          "}");
      layout->addWidget(tx_label);

      tx_stack_ = new QListWidget(frame);
      tx_stack_->setSelectionMode(QAbstractItemView::NoSelection);
      tx_stack_->setSpacing(6);
      tx_stack_->setVerticalScrollMode(QAbstractItemView::ScrollPerPixel);
      tx_stack_->setMinimumHeight(170);
      tx_stack_->setStyleSheet(
          "QListWidget {"
          "  background: #121212;"
          "  border: 1px solid #303134;"
          "  border-radius: 10px;"
          "  padding: 6px;"
          "}"
          "QListWidget::item {"
          "  padding: 2px;"
          "}");
      layout->addWidget(tx_stack_, /*stretch*/ 1);

      controls_layout->insertWidget(controls_layout->count() - 1, frame);

      QObject::connect(new_working_btn_, &QPushButton::clicked, this,
                       [this]() { StartNewWorkingVersionFromUi(); });
      QObject::connect(working_mode_combo_, QOverload<int>::of(&QComboBox::currentIndexChanged),
                       this, [this](int) { UpdateVersionUi(); });
    }

    UpdateVersionUi();

    SetupPipeline();

    // Requirement (3): init state render via singleShot.
    QTimer::singleShot(0, this, [this]() {
      state_.type_ = RenderType::FULL_RES_PREVIEW;
      RequestRender();
      state_.type_ = RenderType::FAST_PREVIEW;
    });
  }

 private:
  enum class AdjustmentField {
    Exposure,
    Contrast,
    Saturation,
    Tint,
    Blacks,
    Whites,
    Shadows,
    Highlights,
    Sharpen,
    Clarity,
    Lut,
  };

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
    RenderType  type_ = RenderType::FAST_PREVIEW;
  };

  static bool NearlyEqual(float a, float b) { return std::abs(a - b) <= 1e-6f; }

  void        RefreshVersionLogSelectionStyles() {
    if (!version_log_) {
      return;
    }
    for (int i = 0; i < version_log_->count(); ++i) {
      auto* item = version_log_->item(i);
      if (!item) {
        continue;
      }
      auto* w = version_log_->itemWidget(item);
      if (!w) {
        continue;
      }
      if (auto* card = dynamic_cast<HistoryCardWidget*>(w)) {
        card->SetSelected(item->isSelected());
      }
    }
  }

  void UpdateVersionUi() {
    if (!version_status_ || !commit_version_btn_) {
      return;
    }

    const size_t tx_count = working_version_.GetAllEditTransactions().size();
    QString      label    = QString("Uncommitted: %1 tx").arg(static_cast<qulonglong>(tx_count));

    if (working_version_.HasParentVersion()) {
      label += QString(" | parent: %1")
                   .arg(QString::fromStdString(
                       working_version_.GetParentVersionID().ToString().substr(0, 8)));
    } else {
      label += " | plain";
    }

    if (working_mode_combo_) {
      const auto mode = static_cast<WorkingMode>(working_mode_combo_->currentData().toInt());
      label += (mode == WorkingMode::Plain) ? " | mode: plain" : " | mode: incremental";
    }

    if (history_guard_ && history_guard_->history_) {
      try {
        const auto latest_id = history_guard_->history_->GetLatestVersion().ver_ref_.GetVersionID();
        label += QString(" | Latest: %1")
                     .arg(QString::fromStdString(latest_id.ToString().substr(0, 8)));
      } catch (...) {
      }
    }

    version_status_->setText(label);
    commit_version_btn_->setEnabled(tx_count > 0);

    if (tx_stack_) {
      tx_stack_->clear();
      const auto&  txs   = working_version_.GetAllEditTransactions();
      const size_t total = txs.size();
      size_t       i     = 0;
      for (const auto& tx : txs) {
        const QString title = QString::fromStdString(tx.Describe(true, 110));

        auto*         item  = new QListWidgetItem(tx_stack_);
        item->setToolTip(QString::fromStdString(tx.ToJSON().dump(2)));
        item->setSizeHint(QSize(0, 58));

        auto* card = new HistoryCardWidget(tx_stack_);
        auto* row  = new QHBoxLayout(card);
        row->setContentsMargins(10, 8, 10, 8);
        row->setSpacing(10);

        const QColor dot  = QColor(0xF2, 0xC0, 0x5C);
        const QColor line = QColor(0x30, 0x31, 0x34);
        auto*        lane = new HistoryLaneWidget(dot, line, /*draw_top*/ i > 0,
                                                  /*draw_bottom*/ (i + 1) < total, card);
        row->addWidget(lane, 0);

        auto* body = new QVBoxLayout();
        body->setContentsMargins(0, 0, 0, 0);
        body->setSpacing(2);

        auto* title_l = new QLabel(title, card);
        title_l->setWordWrap(true);
        title_l->setStyleSheet(
            "QLabel {"
            "  color: #E8EAED;"
            "  font-size: 12px;"
            "  font-weight: 500;"
            "}");

        auto* meta_l =
            new QLabel(QString("uncommitted | #%1").arg(static_cast<qulonglong>(i + 1)), card);
        meta_l->setStyleSheet(
            "QLabel {"
            "  color: #AAB0B6;"
            "  font-size: 11px;"
            "}");

        body->addWidget(title_l);
        body->addWidget(meta_l);
        row->addLayout(body, 1);

        tx_stack_->setItemWidget(item, card);
        ++i;
      }
    }

    if (version_log_) {
      QString prev_selected_id;
      if (auto* cur = version_log_->currentItem()) {
        prev_selected_id = cur->data(Qt::UserRole).toString();
      }

      version_log_->clear();
      if (history_guard_ && history_guard_->history_) {
        const auto& tree = history_guard_->history_->GetCommitTree();
        Hash128     latest_id{};
        try {
          latest_id = history_guard_->history_->GetLatestVersion().ver_ref_.GetVersionID();
        } catch (...) {
        }

        const Hash128 base_parent = working_version_.GetParentVersionID();

        int           row_index   = 0;
        const int     total_rows  = static_cast<int>(tree.size());

        for (auto it = tree.rbegin(); it != tree.rend(); ++it, ++row_index) {
          const auto& ver      = it->ver_ref_;
          const auto  ver_id   = ver.GetVersionID();
          const auto  short_id = QString::fromStdString(ver_id.ToString().substr(0, 8));
          const auto  when =
              QDateTime::fromSecsSinceEpoch(static_cast<qint64>(ver.GetLastModifiedTime()))
                  .toString("yyyy-MM-dd HH:mm:ss");
          const auto committed_tx_count =
              static_cast<qulonglong>(ver.GetAllEditTransactions().size());

          QString     msg;
          const auto& txs = ver.GetAllEditTransactions();
          if (!txs.empty()) {
            msg = QString::fromStdString(txs.front().Describe(true, 70));
          } else {
            msg = "(empty)";
          }

          const bool is_head  = (ver_id == latest_id);
          const bool is_base  = (base_parent == ver_id && working_version_.HasParentVersion());
          const bool is_plain = !ver.HasParentVersion();

          auto*      item     = new QListWidgetItem(version_log_);
          item->setData(Qt::UserRole, QString::fromStdString(ver_id.ToString()));
          item->setToolTip(QString("version=%1\nparent=%2\ntx=%3")
                               .arg(QString::fromStdString(ver_id.ToString()))
                               .arg(QString::fromStdString(ver.GetParentVersionID().ToString()))
                               .arg(committed_tx_count));
          item->setSizeHint(QSize(0, 74));

          auto* card = new HistoryCardWidget(version_log_);
          auto* row  = new QHBoxLayout(card);
          row->setContentsMargins(10, 9, 10, 9);
          row->setSpacing(10);

          const QColor dot  = is_head
                                  ? QColor(0x8a, 0xb4, 0xf8)
                                  : (is_base ? QColor(0x81, 0xC9, 0x95) : QColor(0x9A, 0x9E, 0xA3));
          const QColor line = QColor(0x30, 0x31, 0x34);
          auto*        lane = new HistoryLaneWidget(dot, line, /*draw_top*/ row_index > 0,
                                                    /*draw_bottom*/ (row_index + 1) < total_rows, card);
          row->addWidget(lane, 0);

          auto* body = new QVBoxLayout();
          body->setContentsMargins(0, 0, 0, 0);
          body->setSpacing(4);

          auto* top = new QHBoxLayout();
          top->setContentsMargins(0, 0, 0, 0);
          top->setSpacing(8);

          const QFont mono   = QFontDatabase::systemFont(QFontDatabase::FixedFont);
          auto*       hash_l = new QLabel(short_id, card);
          hash_l->setFont(mono);
          hash_l->setStyleSheet(
              "QLabel {"
              "  color: #E8EAED;"
              "  font-size: 12px;"
              "  font-weight: 600;"
              "}");

          top->addWidget(hash_l, 0);

          if (is_head) {
            top->addWidget(MakePillLabel("HEAD", "#0B1A2B", "rgba(138, 180, 248, 0.95)",
                                         "rgba(138, 180, 248, 0.95)", card),
                           0);
          }
          if (is_base) {
            top->addWidget(MakePillLabel("BASE", "#071C12", "rgba(129, 201, 149, 0.90)",
                                         "rgba(129, 201, 149, 0.90)", card),
                           0);
          }
          if (is_plain) {
            top->addWidget(MakePillLabel("PLAIN", "#202124", "rgba(170, 176, 182, 0.20)",
                                         "rgba(170, 176, 182, 0.30)", card),
                           0);
          } else {
            const auto parent_short =
                QString::fromStdString(ver.GetParentVersionID().ToString().substr(0, 8));
            top->addWidget(
                MakePillLabel(QString("PARENT %1").arg(parent_short), "#AAB0B6",
                              "rgba(170, 176, 182, 0.08)", "rgba(170, 176, 182, 0.18)", card),
                0);
          }

          top->addStretch(1);

          auto* tx_pill =
              MakePillLabel(QString("tx %1").arg(committed_tx_count), "#AAB0B6",
                            "rgba(170, 176, 182, 0.08)", "rgba(170, 176, 182, 0.18)", card);
          top->addWidget(tx_pill, 0);

          auto* msg_l = new QLabel(msg, card);
          msg_l->setWordWrap(true);
          msg_l->setStyleSheet(
              "QLabel {"
              "  color: #E8EAED;"
              "  font-size: 12px;"
              "}");

          auto* meta_l = new QLabel(when, card);
          meta_l->setStyleSheet(
              "QLabel {"
              "  color: #AAB0B6;"
              "  font-size: 11px;"
              "}");

          body->addLayout(top);
          body->addWidget(msg_l);
          body->addWidget(meta_l);
          row->addLayout(body, 1);

          version_log_->setItemWidget(item, card);

          const QString ver_id_str = QString::fromStdString(ver_id.ToString());
          if (!prev_selected_id.isEmpty() && ver_id_str == prev_selected_id) {
            version_log_->setCurrentItem(item);
            item->setSelected(true);
          } else if (prev_selected_id.isEmpty() && is_head) {
            version_log_->setCurrentItem(item);
            item->setSelected(true);
          }
        }
      }
      RefreshVersionLogSelectionStyles();
    }
  }

  void CommitWorkingVersion() {
    if (!history_service_ || !history_guard_ || !history_guard_->history_) {
      QMessageBox::warning(this, "History", "Edit history service not available.");
      return;
    }

    const size_t tx_count = working_version_.GetAllEditTransactions().size();
    if (tx_count == 0) {
      QMessageBox::information(this, "History", "No uncommitted transactions.");
      return;
    }

    history_id_t committed_id{};
    try {
      committed_id = history_service_->CommitVersion(history_guard_, std::move(working_version_));
    } catch (const std::exception& e) {
      QMessageBox::warning(this, "History", QString("Commit failed: %1").arg(e.what()));
      working_version_ = Version(element_id_);
      working_version_.SetBasePipelineExecutor(pipeline_guard_->pipeline_);
      UpdateVersionUi();
      return;
    }

    // Start a fresh working version chained from the committed one.
    StartNewWorkingVersionFromCommit(committed_id);
    UpdateVersionUi();
  }

  auto CurrentWorkingMode() const -> WorkingMode {
    if (!working_mode_combo_) {
      return WorkingMode::Incremental;
    }
    return static_cast<WorkingMode>(working_mode_combo_->currentData().toInt());
  }

  void StartNewWorkingVersionFromUi() {
    if (CurrentWorkingMode() == WorkingMode::Plain) {
      working_version_ = Version(element_id_);
      working_version_.SetBasePipelineExecutor(pipeline_guard_->pipeline_);
      UpdateVersionUi();
      return;
    }

    // Incremental: seed from latest committed version (if any).
    try {
      if (history_guard_ && history_guard_->history_) {
        const auto parent_id = history_guard_->history_->GetLatestVersion().ver_ref_.GetVersionID();
        working_version_     = Version(element_id_, parent_id);
      } else {
        working_version_ = Version(element_id_);
      }
    } catch (...) {
      working_version_ = Version(element_id_);
    }
    working_version_.SetBasePipelineExecutor(pipeline_guard_->pipeline_);
    UpdateVersionUi();
  }

  void StartNewWorkingVersionFromCommit(const Hash128& committed_id) {
    if (CurrentWorkingMode() == WorkingMode::Plain) {
      working_version_ = Version(element_id_);
    } else {
      working_version_ = Version(element_id_, committed_id);
    }
    working_version_.SetBasePipelineExecutor(pipeline_guard_->pipeline_);
  }

  std::pair<PipelineStageName, OperatorType> FieldSpec(AdjustmentField field) const {
    switch (field) {
      case AdjustmentField::Exposure:
        return {PipelineStageName::Basic_Adjustment, OperatorType::EXPOSURE};
      case AdjustmentField::Contrast:
        return {PipelineStageName::Basic_Adjustment, OperatorType::CONTRAST};
      case AdjustmentField::Saturation:
        return {PipelineStageName::Color_Adjustment, OperatorType::SATURATION};
      case AdjustmentField::Tint:
        return {PipelineStageName::Color_Adjustment, OperatorType::TINT};
      case AdjustmentField::Blacks:
        return {PipelineStageName::Basic_Adjustment, OperatorType::BLACK};
      case AdjustmentField::Whites:
        return {PipelineStageName::Basic_Adjustment, OperatorType::WHITE};
      case AdjustmentField::Shadows:
        return {PipelineStageName::Basic_Adjustment, OperatorType::SHADOWS};
      case AdjustmentField::Highlights:
        return {PipelineStageName::Basic_Adjustment, OperatorType::HIGHLIGHTS};
      case AdjustmentField::Sharpen:
        return {PipelineStageName::Detail_Adjustment, OperatorType::SHARPEN};
      case AdjustmentField::Clarity:
        return {PipelineStageName::Detail_Adjustment, OperatorType::CLARITY};
      case AdjustmentField::Lut:
        return {PipelineStageName::Color_Adjustment, OperatorType::LMT};
    }
    return {PipelineStageName::Basic_Adjustment, OperatorType::EXPOSURE};
  }

  nlohmann::json ParamsForField(AdjustmentField field, const AdjustmentState& s) const {
    switch (field) {
      case AdjustmentField::Exposure:
        return {{"exposure", s.exposure_}};
      case AdjustmentField::Contrast:
        return {{"contrast", s.contrast_}};
      case AdjustmentField::Saturation:
        return {{"saturation", s.saturation_}};
      case AdjustmentField::Tint:
        return {{"tint", s.tint_}};
      case AdjustmentField::Blacks:
        return {{"black", s.blacks_}};
      case AdjustmentField::Whites:
        return {{"white", s.whites_}};
      case AdjustmentField::Shadows:
        return {{"shadows", s.shadows_}};
      case AdjustmentField::Highlights:
        return {{"highlights", s.highlights_}};
      case AdjustmentField::Sharpen:
        return {{"sharpen", {{"offset", s.sharpen_}}}};
      case AdjustmentField::Clarity:
        return {{"clarity", s.clarity_}};
      case AdjustmentField::Lut:
        return {{"ocio_lmt", s.lut_path_}};
    }
    return {};
  }

  bool FieldChanged(AdjustmentField field) const {
    switch (field) {
      case AdjustmentField::Exposure:
        return !NearlyEqual(state_.exposure_, committed_state_.exposure_);
      case AdjustmentField::Contrast:
        return !NearlyEqual(state_.contrast_, committed_state_.contrast_);
      case AdjustmentField::Saturation:
        return !NearlyEqual(state_.saturation_, committed_state_.saturation_);
      case AdjustmentField::Tint:
        return !NearlyEqual(state_.tint_, committed_state_.tint_);
      case AdjustmentField::Blacks:
        return !NearlyEqual(state_.blacks_, committed_state_.blacks_);
      case AdjustmentField::Whites:
        return !NearlyEqual(state_.whites_, committed_state_.whites_);
      case AdjustmentField::Shadows:
        return !NearlyEqual(state_.shadows_, committed_state_.shadows_);
      case AdjustmentField::Highlights:
        return !NearlyEqual(state_.highlights_, committed_state_.highlights_);
      case AdjustmentField::Sharpen:
        return !NearlyEqual(state_.sharpen_, committed_state_.sharpen_);
      case AdjustmentField::Clarity:
        return !NearlyEqual(state_.clarity_, committed_state_.clarity_);
      case AdjustmentField::Lut:
        return state_.lut_path_ != committed_state_.lut_path_;
    }
    return false;
  }

  void CommitAdjustment(AdjustmentField field) {
    if (!FieldChanged(field) || !pipeline_guard_ || !pipeline_guard_->pipeline_) {
      // Still fulfill the "full res on release/change" behavior.
      state_.type_ = RenderType::FULL_RES_PREVIEW;
      RequestRender();
      state_.type_ = RenderType::FAST_PREVIEW;
      return;
    }

    const auto [stage_name, op_type] = FieldSpec(field);
    const auto            old_params = ParamsForField(field, committed_state_);
    const auto            new_params = ParamsForField(field, state_);

    auto                  exec       = pipeline_guard_->pipeline_;
    auto&                 stage      = exec->GetStage(stage_name);
    const auto            op         = stage.GetOperator(op_type);
    const TransactionType tx_type =
        (op.has_value() && op.value() != nullptr) ? TransactionType::_EDIT : TransactionType::_ADD;

    EditTransaction tx{tx_type, op_type, stage_name, new_params};
    tx.SetLastOperatorParams(old_params);
    (void)tx.ApplyTransaction(*exec);

    working_version_.AppendEditTransaction(std::move(tx));
    pipeline_guard_->dirty_ = true;

    committed_state_        = state_;
    UpdateVersionUi();

    state_.type_ = RenderType::FULL_RES_PREVIEW;
    RequestRender();
    state_.type_ = RenderType::FAST_PREVIEW;
  }

  bool LoadStateFromPipelineIfPresent() {
    auto exec = pipeline_guard_ ? pipeline_guard_->pipeline_ : nullptr;
    if (!exec) {
      return false;
    }

    auto ReadFloat = [](const PipelineStage& stage, OperatorType type,
                        const char* key) -> std::optional<float> {
      const auto op = stage.GetOperator(type);
      if (!op.has_value() || op.value() == nullptr) {
        return std::nullopt;
      }
      const auto j = op.value()->ExportOperatorParams();
      if (!j.contains("params")) {
        return std::nullopt;
      }
      const auto& params = j["params"];
      if (!params.contains(key)) {
        return std::nullopt;
      }
      try {
        return params[key].get<float>();
      } catch (...) {
        return std::nullopt;
      }
    };

    auto ReadNestedFloat = [](const PipelineStage& stage, OperatorType type, const char* key1,
                              const char* key2) -> std::optional<float> {
      const auto op = stage.GetOperator(type);
      if (!op.has_value() || op.value() == nullptr) {
        return std::nullopt;
      }
      const auto j = op.value()->ExportOperatorParams();
      if (!j.contains("params")) {
        return std::nullopt;
      }
      const auto& params = j["params"];
      if (!params.contains(key1)) {
        return std::nullopt;
      }
      const auto& inner = params[key1];
      if (!inner.contains(key2)) {
        return std::nullopt;
      }
      try {
        return inner[key2].get<float>();
      } catch (...) {
        return std::nullopt;
      }
    };

    auto ReadString = [](const PipelineStage& stage, OperatorType type,
                         const char* key) -> std::optional<std::string> {
      const auto op = stage.GetOperator(type);
      if (!op.has_value() || op.value() == nullptr) {
        return std::nullopt;
      }
      const auto j = op.value()->ExportOperatorParams();
      if (!j.contains("params")) {
        return std::nullopt;
      }
      const auto& params = j["params"];
      if (!params.contains(key)) {
        return std::nullopt;
      }
      try {
        return params[key].get<std::string>();
      } catch (...) {
        return std::nullopt;
      }
    };

    const auto& basic  = exec->GetStage(PipelineStageName::Basic_Adjustment);
    const auto& color  = exec->GetStage(PipelineStageName::Color_Adjustment);
    const auto& detail = exec->GetStage(PipelineStageName::Detail_Adjustment);

    // If there is no exposure operator, treat this as a fresh pipeline without saved params.
    if (!basic.GetOperator(OperatorType::EXPOSURE).has_value()) {
      return false;
    }

    if (const auto v = ReadFloat(basic, OperatorType::EXPOSURE, "exposure"); v.has_value()) {
      state_.exposure_ = v.value();
    }
    if (const auto v = ReadFloat(basic, OperatorType::CONTRAST, "contrast"); v.has_value()) {
      state_.contrast_ = v.value();
    }
    if (const auto v = ReadFloat(basic, OperatorType::BLACK, "black"); v.has_value()) {
      state_.blacks_ = v.value();
    }
    if (const auto v = ReadFloat(basic, OperatorType::WHITE, "white"); v.has_value()) {
      state_.whites_ = v.value();
    }
    if (const auto v = ReadFloat(basic, OperatorType::SHADOWS, "shadows"); v.has_value()) {
      state_.shadows_ = v.value();
    }
    if (const auto v = ReadFloat(basic, OperatorType::HIGHLIGHTS, "highlights"); v.has_value()) {
      state_.highlights_ = v.value();
    }

    if (const auto v = ReadFloat(color, OperatorType::SATURATION, "saturation"); v.has_value()) {
      state_.saturation_ = v.value();
    }
    if (const auto v = ReadFloat(color, OperatorType::TINT, "tint"); v.has_value()) {
      state_.tint_ = v.value();
    }

    if (const auto v = ReadNestedFloat(detail, OperatorType::SHARPEN, "sharpen", "offset");
        v.has_value()) {
      state_.sharpen_ = v.value();
    }
    if (const auto v = ReadFloat(detail, OperatorType::CLARITY, "clarity"); v.has_value()) {
      state_.clarity_ = v.value();
    }

    // LUT (LMT): if a path exists, treat it as enabled.
    const auto lut = ReadString(color, OperatorType::LMT, "ocio_lmt");
    if (lut.has_value() && !lut->empty()) {
      state_.lut_path_ = *lut;
    } else {
      // Fall back to global flag, but keep empty path if none is specified.
      state_.lut_path_.clear();
    }

    return true;
  }

  void SetupPipeline() {
    auto img_desc = image_pool_->Read<std::shared_ptr<Image>>(
        image_id_, [](const std::shared_ptr<Image>& img) { return img; });
    auto bytes = ByteBufferLoader::LoadFromImage(img_desc);
    if (!bytes) {
      throw std::runtime_error("EditorDialog: failed to load image bytes");
    }

    base_task_.input_             = std::make_shared<ImageBuffer>(std::move(*bytes));
    base_task_.pipeline_executor_ = pipeline_guard_->pipeline_;

    auto           exec           = pipeline_guard_->pipeline_;
    // exec->SetPreviewMode(true);

    // auto& global_params = exec->GetGlobalParams();
    auto&          loading        = exec->GetStage(PipelineStageName::Image_Loading);
    nlohmann::json decode_params;
#ifdef HAVE_CUDA
    decode_params["raw"]["cuda"] = true;
#else
    decode_params["raw"]["cuda"] = false;
#endif
    decode_params["raw"]["highlights_reconstruct"] = true;
    decode_params["raw"]["use_camera_wb"]          = true;
    decode_params["raw"]["user_wb"]                = 7600.f;
    decode_params["raw"]["backend"]                = "puerh";
    loading.SetOperator(OperatorType::RAW_DECODE, decode_params);

    // auto& basic         = exec->GetStage(PipelineStageName::Basic_Adjustment);
    // basic.SetOperator(OperatorType::EXPOSURE, {{"exposure", 0.0f}}, global_params);
    // basic.SetOperator(OperatorType::CONTRAST, {{"contrast", 1.0f}}, global_params);
    // basic.SetOperator(OperatorType::BLACK, {{"black", 0.0f}}, global_params);
    // basic.SetOperator(OperatorType::WHITE, {{"white", 0.0f}}, global_params);
    // basic.SetOperator(OperatorType::SHADOWS, {{"shadows", 0.0f}}, global_params);
    // basic.SetOperator(OperatorType::HIGHLIGHTS, {{"highlights", 0.0f}}, global_params);

    // auto& color = exec->GetStage(PipelineStageName::Color_Adjustment);
    // color.SetOperator(OperatorType::SATURATION, {{"saturation", 0.0f}}, global_params);
    // color.SetOperator(OperatorType::TINT, {{"tint", 0.0f}}, global_params);

    // auto& detail = exec->GetStage(PipelineStageName::Detail_Adjustment);
    // detail.SetOperator(OperatorType::SHARPEN, {{"sharpen", {{"offset", 0.0f}}}}, global_params);
    // detail.SetOperator(OperatorType::CLARITY, {{"clarity", 0.0f}}, global_params);

    exec->SetExecutionStages(viewer_);

    // Cached pipelines can clear transient GPU resources when returned to the service.
    // PipelineMgmtService now resyncs global params on load, so we no longer need a
    // per-dialog LMT rebind hack here.
    last_applied_lut_path_.clear();
  }

  void ApplyStateToPipeline() {
    auto  exec          = pipeline_guard_->pipeline_;
    auto& global_params = exec->GetGlobalParams();

    auto& basic         = exec->GetStage(PipelineStageName::Basic_Adjustment);
    basic.SetOperator(OperatorType::EXPOSURE, {{"exposure", state_.exposure_}}, global_params);
    basic.SetOperator(OperatorType::CONTRAST, {{"contrast", state_.contrast_}}, global_params);
    basic.SetOperator(OperatorType::BLACK, {{"black", state_.blacks_}}, global_params);
    basic.SetOperator(OperatorType::WHITE, {{"white", state_.whites_}}, global_params);
    basic.SetOperator(OperatorType::SHADOWS, {{"shadows", state_.shadows_}}, global_params);
    basic.SetOperator(OperatorType::HIGHLIGHTS, {{"highlights", state_.highlights_}},
                      global_params);

    auto& color = exec->GetStage(PipelineStageName::Color_Adjustment);
    color.SetOperator(OperatorType::SATURATION, {{"saturation", state_.saturation_}},
                      global_params);
    color.SetOperator(OperatorType::TINT, {{"tint", state_.tint_}}, global_params);

    // LUT (LMT): rebind only when the path changes. The operator's SetGlobalParams now
    // derives lmt_enabled_/dirty state from the path, and PipelineMgmtService resyncs on load.
    if (state_.lut_path_ != last_applied_lut_path_) {
      color.SetOperator(OperatorType::LMT, {{"ocio_lmt", state_.lut_path_}}, global_params);
      last_applied_lut_path_ = state_.lut_path_;
    }

    auto& detail = exec->GetStage(PipelineStageName::Detail_Adjustment);
    detail.SetOperator(OperatorType::SHARPEN, {{"sharpen", {{"offset", state_.sharpen_}}}},
                       global_params);
    detail.SetOperator(OperatorType::CLARITY, {{"clarity", state_.clarity_}}, global_params);
  }

  void RequestRender() {
    pending_     = state_;
    has_pending_ = true;
    if (!inflight_) {
      StartNext();
    }
  }

  void EnsurePollTimer() {
    if (poll_timer_) {
      return;
    }
    poll_timer_ = new QTimer(this);
    poll_timer_->setInterval(16);
    QObject::connect(poll_timer_, &QTimer::timeout, this, [this]() { PollInflight(); });
  }

  void PollInflight() {
    if (!inflight_future_.has_value()) {
      if (poll_timer_ && poll_timer_->isActive() && !inflight_) {
        poll_timer_->stop();
      }
      return;
    }

    if (inflight_future_->wait_for(0ms) != std::future_status::ready) {
      return;
    }

    try {
      (void)inflight_future_->get();
    } catch (...) {
    }
    inflight_future_.reset();
    OnRenderFinished();
  }

  void StartNext() {
    if (!has_pending_) {
      return;
    }

    state_       = pending_;
    has_pending_ = false;

    if (spinner_) {
      spinner_->Start();
      // Ensure the spinner paints before any potentially blocking work.
      QCoreApplication::processEvents(QEventLoop::ExcludeUserInputEvents);
    }

    ApplyStateToPipeline();
    pipeline_guard_->dirty_                 = true;

    PipelineTask task                       = base_task_;
    task.options_.render_desc_.render_type_ = state_.type_;
    task.options_.is_callback_              = false;
    task.options_.is_seq_callback_          = false;
    task.options_.is_blocking_              = true;

    auto promise = std::make_shared<std::promise<std::shared_ptr<ImageBuffer>>>();
    auto fut     = promise->get_future();
    task.result_ = promise;

    inflight_    = true;
    scheduler_->ScheduleTask(std::move(task));

    inflight_future_ = std::move(fut);
    EnsurePollTimer();
    if (poll_timer_ && !poll_timer_->isActive()) {
      poll_timer_->start();
    }
  }

  void OnRenderFinished() {
    inflight_ = false;

    if (spinner_) {
      spinner_->Stop();
    }

    if (has_pending_) {
      StartNext();
    } else if (poll_timer_ && poll_timer_->isActive()) {
      poll_timer_->stop();
    }
  }

  std::shared_ptr<ImagePoolService>                        image_pool_;
  std::shared_ptr<PipelineGuard>                           pipeline_guard_;
  std::shared_ptr<EditHistoryMgmtService>                  history_service_;
  std::shared_ptr<EditHistoryGuard>                        history_guard_;
  sl_element_id_t                                          element_id_ = 0;
  image_id_t                                               image_id_   = 0;

  std::shared_ptr<PipelineScheduler>                       scheduler_;
  PipelineTask                                             base_task_{};

  QtEditViewer*                                            viewer_             = nullptr;
  QWidget*                                                 viewer_container_   = nullptr;
  QScrollArea*                                             controls_scroll_    = nullptr;
  SpinnerWidget*                                           spinner_            = nullptr;
  QWidget*                                                 controls_           = nullptr;
  QComboBox*                                               lut_combo_          = nullptr;
  QSlider*                                                 exposure_slider_    = nullptr;
  QSlider*                                                 contrast_slider_    = nullptr;
  QSlider*                                                 saturation_slider_  = nullptr;
  QSlider*                                                 tint_slider_        = nullptr;
  QSlider*                                                 blacks_slider_      = nullptr;
  QSlider*                                                 whites_slider_      = nullptr;
  QSlider*                                                 shadows_slider_     = nullptr;
  QSlider*                                                 highlights_slider_  = nullptr;
  QSlider*                                                 sharpen_slider_     = nullptr;
  QSlider*                                                 clarity_slider_     = nullptr;
  QLabel*                                                  version_status_     = nullptr;
  QPushButton*                                             commit_version_btn_ = nullptr;
  QComboBox*                                               working_mode_combo_ = nullptr;
  QPushButton*                                             new_working_btn_    = nullptr;
  QListWidget*                                             version_log_        = nullptr;
  QListWidget*                                             tx_stack_           = nullptr;
  QTimer*                                                  poll_timer_         = nullptr;
  std::optional<std::future<std::shared_ptr<ImageBuffer>>> inflight_future_{};

  std::vector<std::string>                                 lut_paths_{};
  QStringList                                              lut_names_{};

  std::string                                              last_applied_lut_path_{};
  AdjustmentState                                          state_{};
  AdjustmentState                                          committed_state_{};
  Version                                                  working_version_{};
  AdjustmentState                                          pending_{};
  bool                                                     inflight_    = false;
  bool                                                     has_pending_ = false;
};
}  // namespace

auto OpenEditorDialog(std::shared_ptr<ImagePoolService>       image_pool,
                      std::shared_ptr<PipelineGuard>          pipeline_guard,
                      std::shared_ptr<EditHistoryMgmtService> history_service,
                      std::shared_ptr<EditHistoryGuard>       history_guard,
                      sl_element_id_t                         element_id,
                      image_id_t                              image_id,
                      QWidget*                                parent) -> bool {
  EditorDialog dlg(std::move(image_pool), std::move(pipeline_guard), std::move(history_service),
                   std::move(history_guard), element_id, image_id, parent);
  dlg.exec();
  return true;
}

}  // namespace puerhlab::demo
