#include <QApplication>
#include <QImage>
#include <QLabel>

#include <QBoxLayout>
#include <QSlider>
#include <QTimer>


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
  decode_params["raw"]["highlights_reconstruct"] = true;
  decode_params["raw"]["use_camera_wb"]          = true;
  decode_params["raw"]["backend"]                = "puerh";
  nlohmann::json to_ws_params;
  to_ws_params["ocio"] = {{"src", "Linear Rec.709 (sRGB)"}, {"dst", "ACEScc"}, {"normalize", true}};
  raw_stage.SetOperator(OperatorType::RAW_DECODE, decode_params);
  raw_stage.SetOperator(OperatorType::CST, to_ws_params);

  // auto& to_ws          = executor->GetStage(PipelineStageName::To_WorkingSpace);
  // to_ws.SetOperator(OperatorType::CST, to_ws_params);

  nlohmann::json output_params;
  auto&          output_stage = executor->GetStage(PipelineStageName::Output_Transform);
  output_params["ocio"]       = {{"src", "ACEScc"}, {"dst", "Camera Rec.709"}, {"limit", true}};
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
  QApplication app(argc, argv);

  QWidget      window;
  auto*        root   = new QHBoxLayout(&window);

  QLabel*      label  = new QLabel(&window);
  label->setMinimumSize(800, 600);
  label->setAlignment(Qt::AlignCenter);

  QWidget*     controls = new QWidget(&window);
  auto*        controlsLayout = new QVBoxLayout(controls);
  QLabel*      sliderInfo = new QLabel("Highlights: 0", controls);
  auto*        slider = new QSlider(Qt::Horizontal, controls); 
  slider->setRange(-100, 100);
  slider->setValue(0);
  slider->setMinimumWidth(200);

  controlsLayout->addWidget(sliderInfo);
  controlsLayout->addWidget(slider);
  controlsLayout->addStretch();

  root->addWidget(label, /*stretch*/1);
  root->addWidget(controls);

  window.setWindowTitle("Qt Image Preview");
  window.resize(1500, 1000);
  window.show();

  UIHistoryTests tests;
  tests.SetUp();
  auto                      db_path = tests.GetDBPath();
  SleeveManager             manager{db_path};
  ImageLoader               image_loader(128, 1, 0);

  image_path_t              path = std::string(TEST_IMG_PATH) + "/raw/still_life";
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
  nlohmann::json params;
  params["highlights"] = 0.0f;
  basic_stage.SetOperator(OperatorType::HIGHLIGHTS, params);

  // Set execution stages
  base_executor->SetExecutionStages();

  const qreal dpr                  = app.devicePixelRatio();

  auto        scheduleWithExposure = [&](float offset) {
    PipelineTask task = base_task;

    auto& basic_stage = task._pipeline_executor->GetStage(PipelineStageName::Basic_Adjustment);
    params["highlights"] = offset;

    basic_stage.SetOperator(OperatorType::HIGHLIGHTS, params);

    task._options._is_blocking = false;
    task._options._is_callback = true;

    task._callback             = [label, dpr](ImageBuffer& output) {
      cv::Mat img = output.GetCPUData().clone();
      img.convertTo(img, CV_8U, 255.0);

      QImage qimg = cvMatToQImage(img);

      qimg.setDevicePixelRatio(dpr);
      label->setPixmap(QPixmap::fromImage(qimg));
    };

    scheduler.ScheduleTask(std::move(task));
  };

  scheduleWithExposure(0.0f);

  QObject::connect(slider, &QSlider::valueChanged, [&](int v) {
    sliderInfo->setText(QString("Highlights: %1").arg(v));
    float val = static_cast<float>(v);
    scheduleWithExposure(val);
  });

  int ret = app.exec();
  tests.TearDown();
  return ret;
}