#include <future>
#include <opencv2/cudaimgproc.hpp>

#include "edit/history/edit_transaction.hpp"
#include "edit/history/version.hpp"
#include "edit/pipeline/pipeline_cpu.hpp"
#include "edit/scheduler/pipeline_scheduler.hpp"
#include "edit/scheduler/pipeline_task.hpp"
#include "history_test_fixation.hpp"
#include "image/image_buffer.hpp"
#include "sleeve/sleeve_manager.hpp"
#include "type/supported_file_type.hpp"
#include "utils/string/convert.hpp"

using namespace puerhlab;

TEST_F(EditHistoryTests, ApplyAddTransaction) {
  CPUPipelineExecutor pipeline = CPUPipelineExecutor();

  Version             version(1);
  EditTransaction     transaction(1, TransactionType::_ADD, OperatorType::EXPOSURE,
                                  PipelineStageName::Basic_Adjustment, {{"exposure", 0.5}});

  EXPECT_TRUE(transaction.ApplyTransaction(pipeline));
}

TEST_F(EditHistoryTests, ApplyDeleteTransaction) {
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

TEST_F(EditHistoryTests, ApplyEditTransaction) {
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

TEST_F(EditHistoryTests, RedoTransaction) {
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

TEST_F(EditHistoryTests, RedoWithoutParentTransaction) {
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
    SleeveManager manager{db_path_};
    ImageLoader   image_loader(128, 8, 0);
    image_path_t  path =
        L"D:\\Projects\\pu-erh_lab\\pu-erh_lab\\tests\\resources\\sample_"
        L"images\\raw\\building";
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

      auto         buffer      = ByteBufferLoader::LoadFromImage(img_ptr);
      task1._input             = std::make_shared<ImageBuffer>(std::move(buffer));

      auto pipeline_executor   = std::make_shared<CPUPipelineExecutor>();

      task1._pipeline_executor = pipeline_executor;
      SetPipelineTemplate(task1._pipeline_executor);

      task1._options._is_blocking = false;
      task1._options._is_callback = true;
      auto save_callback          = [img_ptr](ImageBuffer& output) {
        output.SyncToGPU();

        auto& gpu_data = output.GetGPUData();
        gpu_data.convertTo(gpu_data, CV_16UC3, 65535.0f);
        cv::cuda::cvtColor(gpu_data, gpu_data, cv::COLOR_RGB2BGR);

        output.SyncToCPU();

        std::string file_name = conv::ToBytes(img_ptr->_image_path.filename().wstring());
        std::string time      = TimeProvider::TimePointToString(TimeProvider::Now());

        std::string save_name = file_name + "_" + time;
        cv::imwrite(std::format("D:\\Projects\\pu-erh_lab\\pu-erh_lab\\tests\\resources\\sample_"
                                                  "images\\my_pipeline\\batch_results\\{}.tif",
                                         save_name),
                             output.GetCPUData());
      };
      task1._callback             = save_callback;
      task1._options._is_blocking = true;
      task1._result               = std::make_shared<std::promise<ImageBuffer>>();

      auto future_task1           = task1._result->get_future();

      auto task2                  = task1;    // Make a copy of task1 for task2
      task2._result               = nullptr;  // Clear the promise for task2 to avoid issues
      task2._options._is_blocking = false;    // Make task2 non-blocking

      scheduler.ScheduleTask(std::move(task1));

      future_task1.get();  // Wait for task1 to complete

      // Create and apply an edit transaction
      EditTransaction tx1(1, TransactionType::_ADD, OperatorType::EXPOSURE,
                          PipelineStageName::Basic_Adjustment, {{"exposure", 0.5}});
      tx1.ApplyTransaction(*pipeline_executor);
      scheduler.ScheduleTask(std::move(task2));
    }
  }
}