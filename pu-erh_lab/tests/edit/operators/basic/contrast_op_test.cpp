#include "edit/operators/basic/contrast_op.hpp"

#include "../op_test_fixation.hpp"

#include <gtest/gtest.h>

#include <exiv2/error.hpp>
#include <opencv2/highgui.hpp>
#include <utility>

#include "image/image_buffer.hpp"
#include "sleeve/sleeve_manager.hpp"


namespace puerhlab {

TEST_F(OperationTests, DISABLED_AdjustmentTest1) {
  {
    SleeveManager manager{db_path_};
    ImageLoader   image_loader(128, 8, 0);
    image_path_t  path =
        TEST_IMG_PATH "/sample_images/jpg";
    std::vector<image_path_t> imgs;
    for (const auto& img : std::filesystem::directory_iterator(path)) {
      imgs.push_back(img.path());
    }

    auto display_callback = [](size_t idx, std::weak_ptr<Image> img) {
      auto  access_img = img.lock();
      auto& thumbnail  = access_img->GetThumbnailBuffer();
      cv::imshow(std::format("img_before: {}", idx), access_img->GetThumbnailData());
      cv::waitKey(1);
      ContrastOp op{15.0f};
      auto       adjusted = op.Apply(thumbnail);

      access_img->LoadThumbnail(std::move(adjusted));

      cv::imshow(std::format("img_after: {}", idx), access_img->GetThumbnailData());
      cv::waitKey(1);
    };

    manager.LoadToPath(imgs, L"");
    auto view = manager.GetView();
    view->UpdateView(L"");
    view->LoadPreview(0, 20, display_callback);
    cv::waitKey();
  }
}

TEST_F(OperationTests, AdjustmentTest2) {
  {
    SleeveManager manager{db_path_};
    ImageLoader   image_loader(128, 8, 0);
    image_path_t  path =
        TEST_IMG_PATH "/sample_images/jpg";
    std::vector<image_path_t> imgs;
    for (const auto& img : std::filesystem::directory_iterator(path)) {
      imgs.push_back(img.path());
    }

    auto display_callback = [](size_t, std::weak_ptr<Image>) {
    };

    manager.LoadToPath(imgs, L"");
    auto view = manager.GetView();
    view->UpdateView(L"");
    // Load thumbnail for these images, do nothing with the callback
    view->LoadPreview(0, 20, display_callback);
    // For now, adjust the thumbnail only
    auto img = manager.GetPool()->AccessElement(0, AccessType::META).value().lock();

    cv::namedWindow("Contrast Animation", cv::WINDOW_AUTOSIZE);
    ContrastOp op{0.0f};
    auto       param = op.GetParams();
    ImageBuffer to_adjust{img->GetThumbnailData()};
    int delay = 20;
    for (float value = -100.0f; value <= 100.0f; value += 1.0f) {
      param[op.GetScriptName()] = value;
      op.SetParams(param);
      // Update the exposure of the image

      ImageBuffer to_display = op.Apply(to_adjust);
      cv::imshow("Contrast Animation", to_display.GetCPUData());
      to_adjust = {img->GetThumbnailData()};

      if (cv::waitKey(delay) == 27) break; // Press ESC to exit
    }

    cv::waitKey(0);
  }
}
}  // namespace puerhlab