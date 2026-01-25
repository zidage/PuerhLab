#include <QApplication>
#include <QCoreApplication>
#include <QGridLayout>
#include <QHBoxLayout>
#include <QImage>
#include <QLabel>
#include <QMetaObject>
#include <QPointer>
#include <QScrollBar>
#include <QVBoxLayout>
#include <QWheelEvent>
#include <QWidget>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <future>
#include <iostream>
#include <memory>
#include <opencv2/opencv.hpp>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "app/import_service.hpp"
#include "app/pipeline_service.hpp"
#include "app/project_service.hpp"
#include "app/thumbnail_service.hpp"
#include "edit/operators/operator_registeration.hpp"


namespace puerhlab {
namespace {
using namespace std::chrono_literals;

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
    // Best-effort conversion.
    cv::Mat tmp;
    rgba32f_or_u8.convertTo(tmp, CV_8UC4);
    rgba8 = tmp;
  }

  if (!rgba8.isContinuous()) {
    rgba8 = rgba8.clone();
  }

  // Assume RGBA byte order.
  QImage img(rgba8.data, rgba8.cols, rgba8.rows, static_cast<int>(rgba8.step),
             QImage::Format_RGBA8888);
  return img.copy();
}

struct AlbumIds {
  std::vector<std::pair<sl_element_id_t, image_id_t>> ids;
};

static AlbumIds ImportBatchToTempProject(const std::filesystem::path& db_path,
                                         const std::filesystem::path& meta_path) {
  ProjectService            project(db_path, meta_path);
  auto                      fs_service = project.GetSleeveService();
  auto                      img_pool   = project.GetImagePoolService();
  ImportServiceImpl         import_service(fs_service, img_pool);

  std::filesystem::path     img_root_path = {TEST_IMG_PATH "/raw/batch_import"};
  std::vector<image_path_t> paths;
  for (const auto& entry : std::filesystem::directory_iterator(img_root_path)) {
    if (entry.is_regular_file()) {
      paths.push_back(entry.path());
    }
  }
  if (paths.empty()) {
    throw std::runtime_error("No images found under TEST_IMG_PATH/raw/batch_import");
  }

  auto                       import_job = std::make_shared<ImportJob>();
  std::promise<ImportResult> done;
  auto                       fut = done.get_future();
  import_job->on_finished_       = [&done](const ImportResult& r) { done.set_value(r); };

  import_job                     = import_service.ImportToFolder(paths, L"", {}, import_job);
  if (!import_job) {
    throw std::runtime_error("ImportToFolder returned null job");
  }
  if (fut.wait_for(120s) != std::future_status::ready) {
    throw std::runtime_error("Import did not finish in time");
  }

  const auto result = fut.get();
  if (result.failed_ != 0u) {
    throw std::runtime_error("Import failed for some images");
  }

  if (!import_job->import_log_) {
    throw std::runtime_error("Import log is null");
  }
  auto snapshot = import_job->import_log_->Snapshot();
  if (snapshot.created_.empty()) {
    throw std::runtime_error("No created entries in import snapshot");
  }

  AlbumIds     out;
  const size_t count = std::min<size_t>(256, snapshot.created_.size());
  out.ids.reserve(count);
  for (size_t i = 0; i < count; ++i) {
    out.ids.push_back({snapshot.created_[i].element_id_, snapshot.created_[i].image_id_});
  }

  import_service.SyncImports(snapshot, L"");
  project.GetSleeveService()->Sync();
  project.GetImagePoolService()->SyncWithStorage();
  project.SaveProject(meta_path);

  return out;
}

class AlbumWidget final : public QWidget {
 public:
  AlbumWidget(std::shared_ptr<ThumbnailService>                   thumbnail_service,
              std::vector<std::pair<sl_element_id_t, image_id_t>> ids, QWidget* parent = nullptr)
      : QWidget(parent), service_(std::move(thumbnail_service)), ids_(std::move(ids)) {
    if (!service_) {
      throw std::runtime_error("ThumbnailService is null");
    }
    if (ids_.empty()) {
      throw std::runtime_error("No ids to display");
    }

    auto* root = new QVBoxLayout(this);

    scrollbar_ = new QScrollBar(Qt::Vertical, this);
    connect(scrollbar_, &QScrollBar::valueChanged, this, [this](int v) {
      start_ = static_cast<size_t>(std::max(0, v));
      RebindAllCells();
      Prefetch();
    });

    auto* content = new QWidget(this);
    grid_         = new QGridLayout(content);
    grid_->setSpacing(6);
    grid_->setContentsMargins(6, 6, 6, 6);

    auto* h = new QHBoxLayout();
    h->addWidget(content, 1);
    h->addWidget(scrollbar_, 0);
    root->addLayout(h, 1);

    InitCells();
    UpdateScrollbar();
    RebindAllCells();
    Prefetch();
  }

  ~AlbumWidget() override { ReleaseAllVisible(); }

 protected:
  void wheelEvent(QWheelEvent* event) override {
    const int steps = event->angleDelta().y() / 120;
    if (steps != 0) {
      const int delta = -steps * static_cast<int>(columns_);  // scroll by a row
      const int next =
          std::clamp(scrollbar_->value() + delta, scrollbar_->minimum(), scrollbar_->maximum());
      scrollbar_->setValue(next);
    }
    event->accept();
  }

 private:
  struct Cell {
    QLabel*                         label            = nullptr;
    size_t                          bound_idx        = static_cast<size_t>(-1);
    sl_element_id_t                 bound_element_id = 0;
    uint64_t                        generation       = 0;
    std::shared_ptr<ThumbnailGuard> guard;
  };

  void InitCells() {
    cells_.resize(view_size_);

    for (size_t i = 0; i < view_size_; ++i) {
      auto* label = new QLabel(this);
      label->setMinimumSize(cell_w_, cell_h_);
      label->setAlignment(Qt::AlignCenter);
      label->setStyleSheet(
          "QLabel { background: #202124; color: #bdbdbd; border: 1px solid #3c4043; }");
      label->setText("(empty)");
      cells_[i].label = label;

      const int r     = static_cast<int>(i / columns_);
      const int c     = static_cast<int>(i % columns_);
      grid_->addWidget(label, r, c);
    }
  }

  void UpdateScrollbar() {
    const size_t window    = std::min(view_size_, ids_.size());
    const size_t max_start = (ids_.size() > window) ? (ids_.size() - window) : 0;
    scrollbar_->setRange(0, static_cast<int>(max_start));
    scrollbar_->setPageStep(static_cast<int>(columns_));
    scrollbar_->setSingleStep(1);
  }

  void ReleaseAllVisible() {
    for (auto& cell : cells_) {
      ReleaseCell(cell);
    }
  }

  void ReleaseCell(Cell& cell) {
    if (cell.guard && cell.bound_element_id != 0) {
      try {
        service_->ReleaseThumbnail(cell.bound_element_id);
      } catch (...) {
      }
    }
    cell.guard.reset();
    cell.bound_idx        = static_cast<size_t>(-1);
    cell.bound_element_id = 0;
    cell.generation++;
  }

  void RebindAllCells() {
    const size_t window    = std::min(view_size_, ids_.size());
    const size_t max_start = (ids_.size() > window) ? (ids_.size() - window) : 0;
    start_                 = std::min(start_, max_start);

    for (size_t pos = 0; pos < window; ++pos) {
      auto&        cell     = cells_[pos];
      const size_t want_idx = start_ + pos;

      if (cell.bound_idx == want_idx) {
        continue;
      }

      ReleaseCell(cell);
      cell.bound_idx        = want_idx;
      cell.bound_element_id = ids_[want_idx].first;
      cell.label->setText(QString("Loading: #%1").arg(static_cast<qulonglong>(want_idx)));
      cell.label->setPixmap({});

      RequestForCell(pos, want_idx, /*pin=*/true);
    }

    // If ids_ smaller than view, clear remaining.
    for (size_t pos = window; pos < view_size_; ++pos) {
      auto& cell = cells_[pos];
      ReleaseCell(cell);
      cell.label->setText("(empty)");
      cell.label->setPixmap({});
    }
  }

  void RequestForCell(size_t cell_pos, size_t idx, bool pin) {
    if (cell_pos >= cells_.size()) {
      return;
    }

    auto                  svc = service_;
    QPointer<AlbumWidget> self(this);

    auto&                 cell       = cells_[cell_pos];
    const auto            element_id = ids_[idx].first;
    const auto            image_id   = ids_[idx].second;
    const uint64_t        gen        = ++cell.generation;

    // Marshal callbacks to the Qt UI thread.
    CallbackDispatcher    dispatcher = [](std::function<void()> fn) {
      auto* obj = QCoreApplication::instance();
      if (!obj) {
        fn();
        return;
      }
      QMetaObject::invokeMethod(obj, std::move(fn), Qt::QueuedConnection);
    };

    svc->GetThumbnail(
        element_id, image_id,
        [self, svc, cell_pos, idx, element_id, gen](std::shared_ptr<ThumbnailGuard> guard) {
          // If the cell got rebound, release immediately.
          if (!guard) {
            return;
          }

          if (!self) {
            try {
              svc->ReleaseThumbnail(element_id);
            } catch (...) {
            }
            return;
          }

          if (cell_pos >= self->cells_.size()) {
            try {
              svc->ReleaseThumbnail(element_id);
            } catch (...) {
            }
            return;
          }

          auto&      cell       = self->cells_[cell_pos];

          const bool still_same = (cell.bound_idx == idx) &&
                                  (cell.bound_element_id == element_id) && (cell.generation == gen);
          if (!still_same) {
            try {
              svc->ReleaseThumbnail(element_id);
            } catch (...) {
            }
            return;
          }

          cell.guard = guard;
          if (!guard->thumbnail_buffer_) {
            cell.label->setText("(no buffer)");
            return;
          }

          auto* buffer = guard->thumbnail_buffer_.get();
          if (!buffer->cpu_data_valid_ && buffer->gpu_data_valid_) {
            try {
              buffer->SyncToCPU();
            } catch (...) {
              cell.label->setText("(SyncToCPU failed)");
              return;
            }
          }

          if (!buffer->cpu_data_valid_) {
            cell.label->setText("(no CPU data)");
            return;
          }

          auto&  mat = buffer->GetCPUData();
          QImage img = MatRgba32fToQImageCopy(mat);
          if (img.isNull()) {
            cell.label->setText("(empty image)");
            return;
          }

          QPixmap px = QPixmap::fromImage(img).scaled(
              self->cell_w_, self->cell_h_, Qt::KeepAspectRatio, Qt::SmoothTransformation);
          cell.label->setPixmap(px);
        //   cell.label->setText(QString("#%1").arg(static_cast<qulonglong>(idx)));
        },
        pin, dispatcher);
  }

  void Prefetch() {
    const size_t window    = std::min(view_size_, ids_.size());
    const size_t max_start = (ids_.size() > window) ? (ids_.size() - window) : 0;
    start_                 = std::min(start_, max_start);

    const size_t begin     = (start_ > prefetch_each_side_) ? (start_ - prefetch_each_side_) : 0;
    const size_t end       = std::min(ids_.size(), start_ + window + prefetch_each_side_);

    // Prefetch: request non-visible indices with pin=false.
    // IMPORTANT: do NOT blindly ReleaseThumbnail() on completion, because the same in-flight
    // request can later become visible and be joined via pending_ (sharing the guard).
    for (size_t idx = begin; idx < end; ++idx) {
      if (idx >= start_ && idx < (start_ + window)) {
        continue;
      }

      if (prefetch_inflight_idx_.size() >= max_prefetch_inflight_) {
        break;
      }

      if (prefetch_inflight_idx_.contains(idx)) {
        continue;
      }

      const auto            element_id = ids_[idx].first;
      const auto            image_id   = ids_[idx].second;

      auto                  svc        = service_;
      QPointer<AlbumWidget> self(this);

      prefetch_inflight_idx_.insert(idx);

      CallbackDispatcher dispatcher = [](std::function<void()> fn) {
        auto* obj = QCoreApplication::instance();
        if (!obj) {
          fn();
          return;
        }
        QMetaObject::invokeMethod(obj, std::move(fn), Qt::QueuedConnection);
      };

      svc->GetThumbnail(
          element_id, image_id,
          [self, svc, idx, element_id](std::shared_ptr<ThumbnailGuard> guard) {
            (void)guard;
            if (!self) {
              try {
                svc->ReleaseThumbnail(element_id);
              } catch (...) {
              }
              return;
            }

            self->prefetch_inflight_idx_.erase(idx);

            const size_t window      = std::min(self->view_size_, self->ids_.size());
            const bool   visible_now = idx >= self->start_ && idx < (self->start_ + window);
            if (!visible_now) {
              try {
                svc->ReleaseThumbnail(element_id);
              } catch (...) {
              }
            }
          },
          /*pin_if_found=*/false, dispatcher);
    }
  }

  std::shared_ptr<ThumbnailService>                   service_;
  std::vector<std::pair<sl_element_id_t, image_id_t>> ids_;

  QGridLayout*                                        grid_      = nullptr;
  QScrollBar*                                         scrollbar_ = nullptr;
  std::vector<Cell>                                   cells_;

  size_t                                              start_     = 0;

  const size_t                                        columns_   = 5;
  const size_t                                        view_size_ = 50;  // 10x5
  const size_t               prefetch_each_side_                 = 7;   // match fuzz test idea

  const size_t               max_prefetch_inflight_              = 12;
  std::unordered_set<size_t> prefetch_inflight_idx_;

  const int                  cell_w_ = 220;
  const int                  cell_h_ = 160;
};

}  // namespace
}  // namespace puerhlab

int main(int argc, char** argv) {
  Exiv2::LogMsg::setLevel(Exiv2::LogMsg::Level::mute);
  puerhlab::RegisterAllOperators();
  QApplication app(argc, argv);

  try {
    const auto db_path   = std::filesystem::temp_directory_path() / "thumbnail_album_demo.db";
    const auto meta_path = std::filesystem::temp_directory_path() / "thumbnail_album_demo.json";

    // Fresh temp project each run.
    if (std::filesystem::exists(db_path)) {
      std::filesystem::remove(db_path);
    }
    if (std::filesystem::exists(meta_path)) {
      std::filesystem::remove(meta_path);
    }

    auto                     imported = puerhlab::ImportBatchToTempProject(db_path, meta_path);

    puerhlab::ProjectService project(db_path, meta_path);
    auto                     img_pool = project.GetImagePoolService();
    auto                     pipeline_service =
        std::make_shared<puerhlab::PipelineMgmtService>(project.GetStorageService());

    auto thumbnail_service = std::make_shared<puerhlab::ThumbnailService>(
        project.GetSleeveService(), img_pool, pipeline_service);

    auto* w = new puerhlab::AlbumWidget(thumbnail_service, std::move(imported.ids));
    w->setWindowTitle("pu-erh_lab - Thumbnail Album Qt Demo");
    w->resize(1400, 900);
    w->show();

    const int rc = app.exec();

    pipeline_service->Sync();
    img_pool->SyncWithStorage();
    project.SaveProject(meta_path);

    return rc;
  } catch (const std::exception& e) {
    std::cerr << "[ThumbnailAlbumQtDemo] Fatal: " << e.what() << std::endl;
    return 1;
  }
}
