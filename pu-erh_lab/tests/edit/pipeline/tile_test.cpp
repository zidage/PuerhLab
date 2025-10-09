#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>

#include "edit/operators/basic/exposure_op.hpp"
#include "edit/operators/op_kernel.hpp"
#include "edit/pipeline/tile_scheduler.hpp"
#include "io/image/image_loader.hpp"
#include "pipeline_test_fixation.hpp"
#include "sleeve/sleeve_manager.hpp"
#include "type/supported_file_type.hpp"

namespace puerhlab {
TEST_F(PipelineTests, TileSchedulerBasic) {
  nlohmann::json params;
  params["exposure"] = 1.0f;
  ExposureOp   exp_op{params};

  KernelStream stream;
  stream.AddToStream(exp_op.ToKernel());

  cv::Mat bgr8 =
      cv::imread(std::string(TEST_IMG_PATH) + "/jpeg/tile_tests/test_img.jpg", cv::IMREAD_COLOR);
  ASSERT_FALSE(bgr8.empty()) << "Failed to read image";
  cv::resize(bgr8, bgr8, cv::Size(1200, 800));

  cv::Mat img;
  bgr8.convertTo(img, CV_32FC3, 1.0 / 255.0);
  cv::cvtColor(img, img, cv::COLOR_BGR2RGB);

  auto          img_buffer = std::make_shared<ImageBuffer>(std::move(img));

  TileScheduler scheduler{img_buffer, stream};

  auto          output = scheduler.ApplyOps();
  // cv::imshow("Output", output->GetCPUData());
  // cv::waitKey(0);
}
}  // namespace puerhlab