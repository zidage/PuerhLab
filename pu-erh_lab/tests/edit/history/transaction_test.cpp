#include <future>
#include <memory>
#include <opencv2/core/mat.hpp>
#ifdef HAVE_CUDA
#include <opencv2/cudaimgproc.hpp>
#endif
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>

#include "edit/history/edit_transaction.hpp"
#include "edit/history/version.hpp"
#include "edit/pipeline/pipeline_cpu.hpp"
#include "edit/scheduler/pipeline_scheduler.hpp"
#include "edit/scheduler/pipeline_task.hpp"
#include "history_test_fixation.hpp"
#include "image/image_buffer.hpp"
#include "sleeve/sleeve_manager.hpp"
#include "type/supported_file_type.hpp"

using namespace puerhlab;

TEST_F(EditHistoryTests, DISABLED_ApplyAddTransaction) {
  CPUPipelineExecutor pipeline = CPUPipelineExecutor();

  Version             version(1);
  EditTransaction     transaction(1, TransactionType::_ADD, OperatorType::EXPOSURE,
                                  PipelineStageName::Basic_Adjustment, {{"exposure", 0.5}});

  EXPECT_TRUE(transaction.ApplyTransaction(pipeline));
}

TEST_F(EditHistoryTests, DISABLED_ApplyDeleteTransaction) {
  CPUPipelineExecutor pipeline = CPUPipelineExecutor();

  // First, add an operator to delete later.
  EditTransaction     add_transaction(1, TransactionType::_ADD, OperatorType::EXPOSURE,
                                      PipelineStageName::Basic_Adjustment, {{"exposure", 0.5}});
  EXPECT_TRUE(add_transaction.ApplyTransaction(pipeline));

  // Now, create a delete transaction for the same operator.
  EditTransaction delete_transaction(2, TransactionType::_DELETE, OperatorType::EXPOSURE,
                                     PipelineStageName::Basic_Adjustment, {}, &add_transaction);

  EXPECT_TRUE(delete_transaction.ApplyTransaction(pipeline));
}

TEST_F(EditHistoryTests, DISABLED_ApplyEditTransaction) {
  CPUPipelineExecutor pipeline = CPUPipelineExecutor();

  // First, add an operator to edit later.
  EditTransaction     add_transaction(1, TransactionType::_ADD, OperatorType::EXPOSURE,
                                      PipelineStageName::Basic_Adjustment, {{"exposure", 0.5}});
  EXPECT_TRUE(add_transaction.ApplyTransaction(pipeline));

  // Now, create an edit transaction for the same operator.
  EditTransaction edit_transaction(2, TransactionType::_EDIT, OperatorType::EXPOSURE,
                                   PipelineStageName::Basic_Adjustment, {{"exposure", 1.0}},
                                   &add_transaction);

  EXPECT_TRUE(edit_transaction.ApplyTransaction(pipeline));
}

TEST_F(EditHistoryTests, DISABLED_RedoTransaction) {
  CPUPipelineExecutor pipeline = CPUPipelineExecutor();

  // First, add an operator to edit later.
  EditTransaction     add_transaction(1, TransactionType::_ADD, OperatorType::EXPOSURE,
                                      PipelineStageName::Basic_Adjustment, {{"exposure", 0.5}});
  EXPECT_TRUE(add_transaction.ApplyTransaction(pipeline));

  // Now, create an edit transaction for the same operator.
  EditTransaction edit_transaction(2, TransactionType::_EDIT, OperatorType::EXPOSURE,
                                   PipelineStageName::Basic_Adjustment, {{"exposure", 1.0}},
                                   &add_transaction);

  EXPECT_TRUE(edit_transaction.ApplyTransaction(pipeline));

  // Redo the edit transaction
  EXPECT_TRUE(edit_transaction.RedoTransaction(pipeline));
}

TEST_F(EditHistoryTests, DISABLED_RedoWithoutParentTransaction) {
  CPUPipelineExecutor pipeline = CPUPipelineExecutor();

  // Create a transaction without a parent
  EditTransaction     transaction(1, TransactionType::_ADD, OperatorType::EXPOSURE,
                                  PipelineStageName::Basic_Adjustment, {{"exposure", 0.5}});

  // Attempt to redo the transaction, which should fail since there's no parent
  EXPECT_FALSE(transaction.RedoTransaction(pipeline));
}

void SetPipelineTemplate(std::shared_ptr<PipelineExecutor> executor) {
  auto&          raw_stage = executor->GetStage(PipelineStageName::Image_Loading);
  nlohmann::json decode_params;
  decode_params["raw"]["cuda"]                   = false;
  decode_params["raw"]["highlights_reconstruct"] = false;
  decode_params["raw"]["use_camera_wb"]          = true;
  decode_params["raw"]["backend"]                = "puerh";
  raw_stage.SetOperator(OperatorType::RAW_DECODE, decode_params);

  nlohmann::json to_ws_params;
  to_ws_params["ocio"] = {
      {"src", "Linear Rec.709 (sRGB)"}, {"dst", "ACEScct"}, {"normalize", true}};

  auto& to_ws = executor->GetStage(PipelineStageName::To_WorkingSpace);
  to_ws.SetOperator(OperatorType::CST, to_ws_params);

  nlohmann::json output_params;
  auto&          output_stage = executor->GetStage(PipelineStageName::Output_Transform);
  output_params["ocio"]       = {{"src", "ACEScct"}, {"dst", "Camera Rec.709"}, {"limit", true}};
  output_stage.SetOperator(OperatorType::CST, output_params);
}

TEST_F(EditHistoryTests, TestWithImage) {
  {
    SleeveManager             manager{db_path_};
    ImageLoader               image_loader(128, 8, 0);
    image_path_t              path = std::string(TEST_IMG_PATH) + "/raw/building";
    std::vector<image_path_t> imgs;
    for (const auto& img : std::filesystem::directory_iterator(path)) {
      if (!img.is_directory() && is_supported_file(img.path())) imgs.push_back(img.path());
    }

    manager.LoadToPath(imgs, L"");

    // Read image data
    auto              img_pool = manager.GetPool()->GetPool();

    PipelineScheduler scheduler{};

    for (const auto& img : img_pool) {
      auto         img_ptr = img.second;
      PipelineTask task1;

      auto         buffer    = ByteBufferLoader::LoadFromImage(img_ptr);
      task1._input           = buffer ? std::make_shared<ImageBuffer>(std::move(*buffer))
                                        : nullptr;

      auto pipeline_executor = std::make_shared<CPUPipelineExecutor>();
      pipeline_executor->SetThumbnailMode(true);

      task1._pipeline_executor = pipeline_executor;
      SetPipelineTemplate(task1._pipeline_executor);

      task1._options._is_blocking = false;
      task1._options._is_callback = true;
      auto save_callback          = [img_ptr](ImageBuffer&) {
        // gpu_data.convertTo(gpu_data, CV_16UC3, 65535.0f);
        // cv::cuda::cvtColor(gpu_data, gpu_data, cv::COLOR_RGB2BGR);

        // output.SyncToCPU();

        // std::string           file_name =
        // conv::ToBytes(img_ptr->_image_path.filename().wstring()); std::string           time =
        // TimeProvider::TimePointToString(TimeProvider::Now());

        // static constexpr auto save_path = TEST_IMG_PATH "/my_pipeline/batch_results/{}.tif";
        // std::string           save_name = file_name + "_" + time;
        // cv::imwrite(std::format(save_path, save_name), output.GetCPUData());
      };

      auto empty_callback         = [](ImageBuffer&) {};

      task1._callback             = empty_callback;
      task1._options._is_blocking = true;
      task1._result               = std::make_shared<std::promise<ImageBuffer>>();

      auto future_task1           = task1._result->get_future();

      auto task2                  = task1;  // Make a copy of task1 for task2
      task2._result               = std::make_shared<std::promise<ImageBuffer>>();
      ;  // Clear the promise for task2 to avoid issues
      auto future_task2           = task2._result->get_future();
      task2._options._is_blocking = true;  // Make task2 non-blocking

      scheduler.ScheduleTask(std::move(task1));

      future_task1.get();  // Wait for task1 to complete

      // Create and apply an edit transaction
      EditTransaction tx1(1, TransactionType::_ADD, OperatorType::EXPOSURE,
                          PipelineStageName::Basic_Adjustment, {{"exposure", 0.5}});
      tx1.ApplyTransaction(*pipeline_executor);

      auto task3                  = task2;
      task3._result               = std::make_shared<std::promise<ImageBuffer>>();
      auto future_task3           = task3._result->get_future();
      task3._options._is_blocking = true;  // Make task3 non-blocking

      scheduler.ScheduleTask(std::move(task2));
      future_task2.get();  // Wait for task2 to complete

      // Create and apply another edit transaction
      EditTransaction tx2(2, TransactionType::_ADD, OperatorType::CONTRAST,
                          PipelineStageName::Basic_Adjustment, {{"contrast", 50}}, &tx1);
      tx2.ApplyTransaction(*pipeline_executor);

      scheduler.ScheduleTask(std::move(task3));
      future_task3.get();  // Wait for task3 to complete
    }
  }
}

TEST_F(EditHistoryTests, DISABLED_TestWithImage_Animated) {
  {
    SleeveManager             manager{db_path_};
    ImageLoader               image_loader(128, 8, 0);
    image_path_t              path = std::string(TEST_IMG_PATH) + "/raw/building";
    std::vector<image_path_t> imgs;
    for (const auto& img : std::filesystem::directory_iterator(path)) {
      if (!img.is_directory() && is_supported_file(img.path())) imgs.push_back(img.path());
    }

    manager.LoadToPath(imgs, L"");

    // Read image data
    auto              img_pool = manager.GetPool()->GetPool();

    PipelineScheduler scheduler{};

    auto              img_ptr = img_pool.begin()->second;

    // Animation loop for the same image
    cv::namedWindow("preview");
    cv::resizeWindow("preview", 800, 600);

    PipelineTask task;
    auto         buffer    = ByteBufferLoader::LoadFromImage(img_ptr);
    task._input            = buffer ? std::make_shared<ImageBuffer>(std::move(*buffer)) : nullptr;

    auto pipeline_executor = std::make_shared<CPUPipelineExecutor>();
    pipeline_executor->SetThumbnailMode(true);

    task._pipeline_executor = pipeline_executor;
    SetPipelineTemplate(task._pipeline_executor);

    task._options._is_callback     = false;
    task._options._is_seq_callback = true;
    auto display_callback          = [](ImageBuffer&, uint32_t id) {
      auto time = TimeProvider::TimePointToString(TimeProvider::Now());
      std::cout << "New frame " << id << " rendered at " << time << "." << std::endl;

      // cv::cvtColor(output.GetCPUData(), output.GetCPUData(), cv::COLOR_RGB2BGR);
    };
    task._seq_callback         = display_callback;
    task._options._is_blocking = true;

    nlohmann::json exposure_params;
    exposure_params["exposure"] = 0.0f;
    for (float exposure = -2.0f; exposure <= 2.0f; exposure += 0.05f) {
      PipelineTask task1 = task;  // Make a copy of task for task1
      auto& basic_stage  = task1._pipeline_executor->GetStage(PipelineStageName::Basic_Adjustment);
      exposure_params["exposure"] = exposure;
      basic_stage.SetOperator(OperatorType::EXPOSURE, exposure_params);
      task1._options._is_blocking = true;
      task1._result               = std::make_shared<std::promise<ImageBuffer>>();
      auto future_task1           = task1._result->get_future();

      scheduler.ScheduleTask(std::move(task1));
      auto result = future_task1.get();  // Wait for task1 to complete
      cv::cvtColor(result.GetCPUData(), result.GetCPUData(), cv::COLOR_RGB2BGR);
      cv::imshow("preview", result.GetCPUData());
      cv::waitKey(1);
    }
  }
}

TEST_F(EditHistoryTests, DISABLED_TestWithPreviewPipeline) {
  {
    SleeveManager             manager{db_path_};
    ImageLoader               image_loader(128, 1, 0);
    image_path_t              path = std::string(TEST_IMG_PATH) + "/raw/building";
    std::vector<image_path_t> imgs;
    for (const auto& img : std::filesystem::directory_iterator(path)) {
      if (!img.is_directory() && is_supported_file(img.path())) imgs.push_back(img.path());
    }

    manager.LoadToPath(imgs, L"");

    // Read image data
    auto              img_pool = manager.GetPool()->GetPool();

    PipelineScheduler scheduler{};

    auto              img_ptr = img_pool.begin()->second;

    // Animation loop for the same image
    cv::namedWindow("preview");
    cv::resizeWindow("preview", 800, 600);

    PipelineTask task;
    auto         buffer    = ByteBufferLoader::LoadFromImage(img_ptr);
    task._input            = buffer ? std::make_shared<ImageBuffer>(std::move(*buffer)) : nullptr;

    auto pipeline_executor = std::make_shared<CPUPipelineExecutor>(true);
    pipeline_executor->SetThumbnailMode(true);

    task._pipeline_executor = pipeline_executor;
    SetPipelineTemplate(task._pipeline_executor);

    task._options._is_callback     = false;
    task._options._is_seq_callback = true;
    auto display_callback          = [](ImageBuffer& output, uint32_t id) {
      std::cout << "New frame " << id << std::endl;
    };
    task._seq_callback         = display_callback;
    task._options._is_blocking = true;

    nlohmann::json exposure_params;
    exposure_params["exposure"] = 0.0f;
    for (float exposure = -2.0f; exposure <= 2.0f; exposure += 0.05f) {
      PipelineTask task1 = task;  // Make a copy of task for task1
      auto& basic_stage  = task1._pipeline_executor->GetStage(PipelineStageName::Basic_Adjustment);
      exposure_params["exposure"] = exposure;
      basic_stage.SetOperator(OperatorType::EXPOSURE, exposure_params);
      task1._options._is_blocking = true;
      task1._result               = std::make_shared<std::promise<ImageBuffer>>();
      auto future_task1           = task1._result->get_future();

      scheduler.ScheduleTask(std::move(task1));
      auto result = future_task1.get();  // Wait for task1 to complete
      cv::cvtColor(result.GetCPUData(), result.GetCPUData(), cv::COLOR_RGB2BGR);
      cv::imshow("preview", result.GetCPUData());
      cv::waitKey(1);
    }
  }
}