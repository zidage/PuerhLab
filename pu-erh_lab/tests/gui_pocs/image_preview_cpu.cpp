#include <opencv2/core/hal/interface.h>

#include <QApplication>
#include <QBoxLayout>
#include <QImage>
#include <QLabel>
#include <QSlider>
#include <QTimer>
#include <future>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>

#include "edit/operators/op_base.hpp"
#include "edit/pipeline/pipeline_cpu.hpp"
#include "edit/scheduler/pipeline_scheduler.hpp"
#include "sleeve/sleeve_manager.hpp"
#include "type/supported_file_type.hpp"
#include "ui_test_fixation.hpp"


using namespace puerhlab;

void SetPipelineTemplate(std::shared_ptr<PipelineExecutor> executor) {
  auto&          raw_stage = executor->GetStage(PipelineStageName::Image_Loading);
  nlohmann::json decode_params;
#ifdef HAVE_CUDA
  decode_params["raw"]["cuda"] = false;
#else
  decode_params["raw"]["cuda"] = false;
#endif
  decode_params["raw"]["highlights_reconstruct"] = false;
  decode_params["raw"]["use_camera_wb"]          = true;
  decode_params["raw"]["user_wb"]                = 5500;
  decode_params["raw"]["backend"]                = "puerh";
  nlohmann::json to_ws_params;
  to_ws_params["ocio"] = {{"src", ""}, {"dst", "ACEScct"}, {"normalize", true}};
  raw_stage.SetOperator(OperatorType::RAW_DECODE, decode_params);
  raw_stage.SetOperator(OperatorType::CST, to_ws_params);

  // auto& to_ws          = executor->GetStage(PipelineStageName::To_WorkingSpace);
  // to_ws.SetOperator(OperatorType::CST, to_ws_params);

  nlohmann::json output_params;
  auto&          output_stage = executor->GetStage(PipelineStageName::Output_Transform);
  output_params["ocio"]       = {{"src", "ACEScct"}, {"dst", "Camera Rec.709"}, {"limit", true}};
  output_stage.SetOperator(OperatorType::CST, output_params);
}

static QImage cvMatToQImage(const cv::Mat& mat) {
  if (mat.type() == CV_8UC3) {
    QImage image(mat.data, mat.cols, mat.rows, static_cast<int>(mat.step), QImage::Format_RGB888);
    return image;
  } else if (mat.type() == CV_8UC1) {
    QImage image(mat.data, mat.cols, mat.rows, static_cast<int>(mat.step),
                 QImage::Format_Grayscale8);
    return image;
  } else {
    throw std::runtime_error("Unsupported image format for display");
  }
}

int main(int argc, char* argv[]) {
  struct AdjustmentState {
    float exposure   = 0.0f;
    float contrast   = 1.0f;
    float saturation = 0.0f;
    float blacks     = 0.0f;
    float whites     = 0.0f;
    float shadows    = 0.0f;
    float highlights = 0.0f;
    float sharpen    = 0.0f;
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

  image_path_t              path = std::string(TEST_IMG_PATH) + "/raw/showcase";
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
  base_task._input         = buffer ? std::make_shared<ImageBuffer>(std::move(*buffer)) : nullptr;

  auto base_executor       = std::make_shared<CPUPipelineExecutor>(true);
  base_executor->SetPreviewMode(true);
  base_task._pipeline_executor = base_executor;

  SetPipelineTemplate(base_task._pipeline_executor);
  // Register a default exposure
  auto& basic_stage = base_task._pipeline_executor->GetStage(PipelineStageName::Basic_Adjustment);
  basic_stage.SetOperator(OperatorType::EXPOSURE, {{"exposure", 0.0f}});
  basic_stage.SetOperator(OperatorType::CONTRAST, {{"contrast", 1.0f}});
  basic_stage.SetOperator(OperatorType::BLACK, {{"black", 0.0f}});
  basic_stage.SetOperator(OperatorType::WHITE, {{"white", 0.0f}});
  basic_stage.SetOperator(OperatorType::SHADOWS, {{"shadows", 0.0f}});
  basic_stage.SetOperator(OperatorType::HIGHLIGHTS, {{"highlights", 0.0f}});

  auto& color_stage = base_task._pipeline_executor->GetStage(PipelineStageName::Color_Adjustment);
  color_stage.SetOperator(OperatorType::SATURATION, {{"saturation", 0.0f}});

  std::string LUT_PATH = std::string(CONFIG_PATH) + "LUTs/ACES CCT 2383 D65.cube";
  color_stage.SetOperator(OperatorType::LMT, {{"ocio_lmt", LUT_PATH}});

  auto& detail_stage = base_task._pipeline_executor->GetStage(PipelineStageName::Detail_Adjustment);
  detail_stage.SetOperator(OperatorType::SHARPEN, {{"sharpen", {"offset", 0.0f}}});

  // Set execution stages
  base_executor->SetExecutionStages();

  const qreal     dpr = app.devicePixelRatio();

  AdjustmentState adjustments{};
  auto            scheduleAdjustments = [&](const AdjustmentState& state) {
    PipelineTask task = base_task;

    auto&        basic = task._pipeline_executor->GetStage(PipelineStageName::Basic_Adjustment);
    basic.SetOperator(OperatorType::EXPOSURE, {{"exposure", state.exposure}});
    basic.SetOperator(OperatorType::CONTRAST, {{"contrast", state.contrast}});
    basic.SetOperator(OperatorType::BLACK, {{"black", state.blacks}});
    basic.SetOperator(OperatorType::WHITE, {{"white", state.whites}});
    basic.SetOperator(OperatorType::SHADOWS, {{"shadows", state.shadows}});
    basic.SetOperator(OperatorType::HIGHLIGHTS, {{"highlights", state.highlights}});

    auto& color = task._pipeline_executor->GetStage(PipelineStageName::Color_Adjustment);
    color.SetOperator(OperatorType::SATURATION, {{"saturation", state.saturation}});

    auto& detail = task._pipeline_executor->GetStage(PipelineStageName::Detail_Adjustment);

    detail.SetOperator(OperatorType::SHARPEN, {{"sharpen", {{"offset", state.sharpen}}}});

    task._options._is_blocking = false;
    task._options._is_callback = true;

    task._callback             = [label, dpr](ImageBuffer& output) {
      cv::Mat img = output.GetCPUData();
      img.convertTo(img, CV_8U, 255.0);

      QImage qimg = cvMatToQImage(img);

      qimg.setDevicePixelRatio(dpr);
      label->setPixmap(QPixmap::fromImage(qimg));
    };

    scheduler.ScheduleTask(std::move(task));
  };

  auto defaultFormatter = [](int v) { return QString::number(v); };
  auto addSlider        = [&](const QString& name, int min, int max, int value, auto&& onChange,
                       auto&& formatter) {
    auto* info   = new QLabel(QString("%1: %2").arg(name).arg(formatter(value)), controls);
    auto* slider = new QSlider(Qt::Horizontal, controls);
    slider->setRange(min, max);
    slider->setValue(value);
    slider->setMinimumWidth(200);

    QObject::connect(
        slider, &QSlider::valueChanged, controls,
        [info, name, formatter, onChange = std::forward<decltype(onChange)>(onChange)](int v) {
          info->setText(QString("%1: %2").arg(name).arg(formatter(v)));
          onChange(v);
        });

    controlsLayout->insertWidget(controlsLayout->count() - 1, info);
    controlsLayout->insertWidget(controlsLayout->count() - 1, slider);
  };

  addSlider(
      "Exposure", -500, 500, 0,
      [&](int v) {
        adjustments.exposure = static_cast<float>(v) / 100.0f;
        scheduleAdjustments(adjustments);
      },
      [](int v) { return QString::number(v / 100.0, 'f', 2); });

  addSlider(
      "Contrast", -100, 100, 0,
      [&](int v) {
        adjustments.contrast = static_cast<float>(v);
        scheduleAdjustments(adjustments);
      },
      [](int v) { return QString::number(v, 'f', 2); });

  addSlider(
      "Saturation", -100, 100, 0,
      [&](int v) {
        adjustments.saturation = static_cast<float>(v);
        scheduleAdjustments(adjustments);
      },
      [](int v) { return QString::number(v, 'f', 2); });

  addSlider(
      "Blacks", -100, 100, 0,
      [&](int v) {
        adjustments.blacks = static_cast<float>(v);
        scheduleAdjustments(adjustments);
      },
      [](int v) { return QString::number(v, 'f', 2); });

  addSlider(
      "Whites", -100, 100, 0,
      [&](int v) {
        adjustments.whites = static_cast<float>(v);
        scheduleAdjustments(adjustments);
      },
      [](int v) { return QString::number(v, 'f', 2); });

  addSlider(
      "Shadows", -100, 100, 0,
      [&](int v) {
        adjustments.shadows = static_cast<float>(v);
        scheduleAdjustments(adjustments);
      },
      [](int v) { return QString::number(v, 'f', 2); });

  addSlider(
      "Highlights", -100, 100, 0,
      [&](int v) {
        adjustments.highlights = static_cast<float>(v);
        scheduleAdjustments(adjustments);
      },
      [](int v) { return QString::number(v, 'f', 2); });

  addSlider(
      "Sharpen", -100, 100, 0,
      [&](int v) {
        adjustments.sharpen = static_cast<float>(v);
        scheduleAdjustments(adjustments);
      },
      [](int v) { return QString::number(v, 'f', 2); });

  scheduleAdjustments(adjustments);

  int ret = app.exec();
  tests.TearDown();
  return ret;
}