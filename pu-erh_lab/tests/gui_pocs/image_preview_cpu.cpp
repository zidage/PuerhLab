#include <opencv2/core/hal/interface.h>

#include <QApplication>
#include <QBoxLayout>
#include <QImage>
#include <QLabel>
#include <QScrollBar>
#include <QTimer>
#include <future>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>

#include "edit/pipeline/pipeline_cpu.hpp"
#include "edit/scheduler/pipeline_scheduler.hpp"
#include "sleeve/sleeve_manager.hpp"
#include "type/supported_file_type.hpp"
#include "ui_test_fixation.hpp"

using namespace puerhlab;

void SetPipelineTemplate(std::shared_ptr<PipelineExecutor> executor) {
  auto&          raw_stage     = executor->GetStage(PipelineStageName::Image_Loading);
  auto&          global_params = executor->GetGlobalParams();
  nlohmann::json decode_params;
#ifdef HAVE_CUDA
  decode_params["raw"]["cuda"] = false;
#else
  decode_params["raw"]["cuda"] = false;
#endif
  decode_params["raw"]["highlights_reconstruct"] = true;
  decode_params["raw"]["use_camera_wb"]          = true;
  decode_params["raw"]["user_wb"]                = 5500;
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
  output_params["aces_odt"] = {
      {"encoding_space", "rec709"},
      {"encoding_etof", "gamma_1_8"},
      {"limiting_space", "rec709"},
      {"peak_luminance", 100.0f}};
  output_stage.SetOperator(OperatorType::ODT, output_params, global_params);
}

static QImage cvMatToQImage(const cv::Mat& mat) {
  // if (mat.type() == CV_8UC3) {
  //   QImage image(mat.data, mat.cols, mat.rows, static_cast<int>(mat.step), QImage::Format_RGB888);
  //   return image;
  // } else if (mat.type() == CV_8UC1) {
  //   QImage image(mat.data, mat.cols, mat.rows, static_cast<int>(mat.step),
  //                QImage::Format_Grayscale8);
  //   return image;
  // } else {
  //   throw std::runtime_error("Unsupported image format for display");
  // }
  return QImage{mat.data, mat.cols, mat.rows, static_cast<int>(mat.step), QImage::Format_RGBA32FPx4};
}

int main(int argc, char* argv[]) {
  struct AdjustmentState {
    float exposure_   = 0.0f;
    float contrast_   = 1.0f;
    float saturation_ = 0.0f;
    float tint_       = 0.0f;
    float blacks_     = 0.0f;
    float whites_     = 0.0f;
    float shadows_    = 0.0f;
    float highlights_ = 0.0f;
    float sharpen_    = 0.0f;
    float clarity_    = 0.0f;
  };

  QApplication app(argc, argv);

  QWidget      window;
  auto*        root  = new QHBoxLayout(&window);

  QLabel*      label = new QLabel(&window);
  label->setMinimumSize(800, 600);
  label->setAlignment(Qt::AlignCenter);

  QWidget* controls       = new QWidget(&window);
  auto*    controlsLayout = new QVBoxLayout(controls);

  controlsLayout->addStretch();

  root->addWidget(label, /*stretch*/ 1);
  root->addWidget(controls);

  window.setWindowTitle("Qt Image Preview");
  window.resize(1500, 1000);
  window.show();

  UIHistoryTests tests;
  tests.SetUp();
  auto                      db_path = tests.GetDBPath();
  SleeveManager             manager{db_path};
  ImageLoader               image_loader(128, 1, 0);

  image_path_t              path = std::string(TEST_IMG_PATH) + "/raw/camera/sony/a7iv";
  std::vector<image_path_t> imgs;
  for (const auto& img : std::filesystem::directory_iterator(path)) {
    if (!img.is_directory() && is_supported_file(img.path())) imgs.push_back(img.path());
  }
  manager.LoadToPath(imgs, L"");
  auto              img_pool = manager.GetPool()->GetPool();
  auto              img_ptr  = img_pool.begin()->second;

  PipelineScheduler scheduler{};
  PipelineTask      base_task;
  auto              buffer = ByteBufferLoader::LoadFromImage(img_ptr);
  base_task.input_         = buffer ? std::make_shared<ImageBuffer>(std::move(*buffer)) : nullptr;

  auto base_executor       = std::make_shared<CPUPipelineExecutor>(true);
  base_executor->SetPreviewMode(true);
  base_task.pipeline_executor_ = base_executor;
  SetPipelineTemplate(base_task.pipeline_executor_);
  auto& global_params = base_task.pipeline_executor_->GetGlobalParams();
  // Register a default exposure
  auto& basic_stage   = base_task.pipeline_executor_->GetStage(PipelineStageName::Basic_Adjustment);
  basic_stage.SetOperator(OperatorType::EXPOSURE, {{"exposure", 0.0f}}, global_params);
  basic_stage.SetOperator(OperatorType::CONTRAST, {{"contrast", 1.0f}}, global_params);
  basic_stage.SetOperator(OperatorType::BLACK, {{"black", 0.0f}}, global_params);
  basic_stage.SetOperator(OperatorType::WHITE, {{"white", 0.0f}}, global_params);
  basic_stage.SetOperator(OperatorType::SHADOWS, {{"shadows", 0.0f}}, global_params);
  basic_stage.SetOperator(OperatorType::HIGHLIGHTS, {{"highlights", 0.0f}}, global_params);

  auto& color_stage = base_task.pipeline_executor_->GetStage(PipelineStageName::Color_Adjustment);
  color_stage.SetOperator(OperatorType::SATURATION, {{"saturation", 0.0f}}, global_params);
  color_stage.SetOperator(OperatorType::TINT, {{"tint", 0.0f}}, global_params);

  std::string LUT_PATH = std::string(CONFIG_PATH) + "LUTs/ACES CCT 2383 D65.cube";
  color_stage.SetOperator(OperatorType::LMT, {{"ocio_lmt", LUT_PATH}}, global_params);

  auto& detail_stage = base_task.pipeline_executor_->GetStage(PipelineStageName::Detail_Adjustment);
  detail_stage.SetOperator(OperatorType::SHARPEN, {{"sharpen", {{"offset", 0.0f}}}}, global_params);
  detail_stage.SetOperator(OperatorType::CLARITY, {{"clarity", 0.0f}}, global_params);
  // Set execution stages
  base_executor->SetExecutionStages();

  const qreal     dpr = app.devicePixelRatio();

  AdjustmentState adjustments{};
  auto            scheduleAdjustments = [&](const AdjustmentState& state) {
    PipelineTask task = base_task;

    auto&        basic = task.pipeline_executor_->GetStage(PipelineStageName::Basic_Adjustment);
    basic.SetOperator(OperatorType::EXPOSURE, {{"exposure", state.exposure_}}, global_params);
    basic.SetOperator(OperatorType::CONTRAST, {{"contrast", state.contrast_}}, global_params);
    basic.SetOperator(OperatorType::BLACK, {{"black", state.blacks_}}, global_params);
    basic.SetOperator(OperatorType::WHITE, {{"white", state.whites_}}, global_params);
    basic.SetOperator(OperatorType::SHADOWS, {{"shadows", state.shadows_}}, global_params);
    basic.SetOperator(OperatorType::HIGHLIGHTS, {{"highlights", state.highlights_}}, global_params);

    auto& color = task.pipeline_executor_->GetStage(PipelineStageName::Color_Adjustment);
    color.SetOperator(OperatorType::SATURATION, {{"saturation", state.saturation_}}, global_params);
    color.SetOperator(OperatorType::TINT, {{"tint", state.tint_}}, global_params);
    auto& detail = task.pipeline_executor_->GetStage(PipelineStageName::Detail_Adjustment);

    detail.SetOperator(OperatorType::SHARPEN, {{"sharpen", {{"offset", state.sharpen_}}}},
                                  global_params);
    detail.SetOperator(OperatorType::CLARITY, {{"clarity", state.clarity_}}, global_params);

    task.options_.is_blocking_ = false;
    task.options_.is_callback_ = true;

    task.callback_             = [label, dpr](ImageBuffer& output) {
      try {
      cv::Mat img = output.GetCPUData();
      // img.convertTo(img, CV_8UC3, 255.0);

      QImage qimg = cvMatToQImage(img);

      qimg.setDevicePixelRatio(dpr);
      label->setPixmap(QPixmap::fromImage(qimg));
      } catch (const std::exception& e) {
        qDebug("Failed to get image from pipeline: %s", e.what());
      }
    };

    scheduler.ScheduleTask(std::move(task));
  };

  auto addSlider = [&](const QString& name, int min, int max, int value, auto&& onChange,
                       auto&& formatter) {
    auto* info   = new QLabel(QString("%1: %2").arg(name).arg(formatter(value)), controls);
    auto* bar    = new QScrollBar(Qt::Horizontal, controls);
    bar->setRange(min, max);
    bar->setValue(value);
    bar->setSingleStep(1);
    bar->setPageStep(std::max(1, (max - min) / 20));
    bar->setMinimumWidth(220);
    bar->setStyleSheet(
      "QScrollBar:horizontal {"
      "  background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #1f1f1f, stop:1 #262626);"
      "  border: 1px solid #3a3a3a;"
      "  height: 18px;"
      "  margin: 6px 14px 6px 14px;"
      "  border-radius: 9px;"
      "}"
      "QScrollBar::handle:horizontal {"
      "  background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #5fa2ff, stop:1 #3f7de0);"
      "  border: 1px solid #2f63b8;"
      "  border-radius: 8px;"
      "  min-width: 32px;"
      "  margin: 1px;"
      "}"
      "QScrollBar::handle:horizontal:hover {"
      "  background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #76b4ff, stop:1 #4f8cf0);"
      "  border-color: #3874cc;"
      "}"
      "QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {"
      "  border: none;"
      "  background: transparent;"
      "  width: 0px;"
      "}"
      "QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {"
      "  background: transparent;"
      "}");

    QObject::connect(
        bar, &QScrollBar::valueChanged, controls,
        [info, name, formatter, onChange = std::forward<decltype(onChange)>(onChange)](int v) {
          info->setText(QString("%1: %2").arg(name).arg(formatter(v)));
          onChange(v);
        });

    auto* row       = new QWidget(controls);
    auto* rowLayout = new QHBoxLayout(row);
    rowLayout->setContentsMargins(0, 0, 0, 0);
    rowLayout->addWidget(info, /*stretch*/ 1);
    rowLayout->addWidget(bar);

    controlsLayout->insertWidget(controlsLayout->count() - 1, row);
  };

  addSlider(
      "Exposure", -500, 500, 0,
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
  scheduleAdjustments(adjustments);

  int ret = app.exec();
  tests.TearDown();
  return ret;
}