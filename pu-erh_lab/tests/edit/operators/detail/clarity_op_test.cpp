#include "edit/operators/detail/clarity_op.hpp"

#include "../op_test_fixation.hpp"
#include "sleeve/sleeve_manager.hpp"



using namespace puerhlab;
TEST_F(OperationTests, ClarityAdjustmentTest) {
  {
    SleeveManager manager{db_path_};
    ImageLoader   image_loader(128, 8, 0);
    image_path_t  path =
        L"D:\\Projects\\pu-erh_lab\\pu-erh_"
        L"lab\\tests\\resources\\sample_images\\jpg_1";
    std::vector<image_path_t> imgs;
    for (const auto& img : std::filesystem::directory_iterator(path)) {
      imgs.push_back(img.path());
    }

    auto display_callback = [](size_t idx, std::weak_ptr<Image> img) {};

    manager.LoadToPath(imgs, L"");
    auto view = manager.GetView();
    view->UpdateView(L"");
    // Load thumbnail for these images, do nothing with the callback
    view->LoadPreview(0, 10, display_callback);
    // For now, adjust the thumbnail only
    auto img = manager.GetPool()->AccessElement(0, AccessType::THUMB).value().lock();

    cv::namedWindow("Clarity Animation", cv::WINDOW_AUTOSIZE);
    ClarityOp   op{0.0f};
    auto        param = op.GetParams();
    ImageBuffer to_adjust{img->GetThumbnailData()};
    // Delay between two adjustments
    int         delay = 10;
    for (float value = -100.0f; value <= 100.0f; value += 1.0f) {
      param[op.GetScriptName()] = value;
      op.SetParams(param);

      ImageBuffer to_display = op.Apply(to_adjust);
      std::string info       = "Clarity: " + std::to_string(value);
      auto&       display    = to_display.GetCPUData();
      cv::putText(display, info, cv::Point(20, 40), cv::FONT_HERSHEY_SIMPLEX, 1.0,
                  cv::Scalar(255, 255, 255), 2);
      cv::imshow("Clarity Animation", display);
      to_adjust = {img->GetThumbnailData()};

      if (cv::waitKey(delay) == 27) break;  // Press ESC to exit
    }

    cv::waitKey(2000);
  }
}