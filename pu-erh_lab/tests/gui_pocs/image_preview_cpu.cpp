#include <qapplication.h>
#include <qimage.h>
#include <qlabel.h>
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
  decode_params["raw"]["highlights_reconstruct"] = false;
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
    QLabel label;
    label.setWindowTitle("Qt Image Preview");
    label.resize(2000, 1500);
    label.show();

    UIHistoryTests tests;
    tests.SetUp();
    auto db_path = tests.GetDBPath();
    SleeveManager manager{db_path};
    ImageLoader image_loader(128, 1, 0);

    image_path_t path = std::string(TEST_IMG_PATH) + "/raw/camera/sony/a1";
    std::vector<image_path_t> imgs;
    for (const auto& img : std::filesystem::directory_iterator(path)) {
        if (!img.is_directory() && is_supported_file(img.path()))
            imgs.push_back(img.path());
    }
    manager.LoadToPath(imgs, L"");
    auto img_pool = manager.GetPool()->GetPool();
    auto img_ptr = img_pool.begin()->second;

    PipelineScheduler scheduler{};
    PipelineTask base_task;
    auto buffer = ByteBufferLoader::LoadFromImage(img_ptr);
    base_task._input = buffer ? std::make_shared<ImageBuffer>(std::move(*buffer)) : nullptr;

    auto pipeline_executor = std::make_shared<CPUPipelineExecutor>(true);
    pipeline_executor->SetPreviewMode(true);
    base_task._pipeline_executor = pipeline_executor;
    SetPipelineTemplate(base_task._pipeline_executor);

    nlohmann::json exposure_params;
    exposure_params["exposure"] = 0.0f;
    auto& basic_stage = base_task._pipeline_executor->GetStage(PipelineStageName::Basic_Adjustment);
    basic_stage.SetOperator(OperatorType::EXPOSURE, exposure_params);
    pipeline_executor->SetExecutionStages();

    // Animation state
    float exposure = -5.0f;

    QTimer* timer = new QTimer();
    QObject::connect(timer, &QTimer::timeout, [&]() {
        if (exposure > 5.0f) {
            timer->stop();
            tests.TearDown();
            return;
        }

        PipelineTask task = base_task;
        auto& basic_stage = task._pipeline_executor->GetStage(PipelineStageName::Basic_Adjustment);
        exposure_params["exposure"] = exposure;
        basic_stage.SetOperator(OperatorType::EXPOSURE, exposure_params);

        task._options._is_blocking = true;
        task._result = std::make_shared<std::promise<std::shared_ptr<ImageBuffer>>>();
        auto future = task._result->get_future();

        scheduler.ScheduleTask(std::move(task));
        auto result = future.get();

        cv::Mat img = result->GetCPUData();
        img.convertTo(img, CV_8U, 255.0);
        // cv::cvtColor(img, img, cv::COLOR_RGB2BGR);
        QImage qimage(img.data, img.cols, img.rows, img.step, QImage::Format_RGB888);
        label.setPixmap(QPixmap::fromImage(qimage).scaled(label.size(), Qt::KeepAspectRatio));

        exposure += 0.05f;
    });

    timer->start(10); // 1 ms interval for smooth animation
    return app.exec();
}