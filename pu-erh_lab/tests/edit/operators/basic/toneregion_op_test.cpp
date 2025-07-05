#include "edit/operators/basic/contrast_op.hpp"
#include "edit/operators/basic/tone_region_op.hpp"

#include "op_test_fixation.hpp"

#include <gtest/gtest.h>

#include <exiv2/error.hpp>
#include <opencv2/highgui.hpp>
#include <utility>

#include "image/image_buffer.hpp"
#include "sleeve/sleeve_manager.hpp"


namespace puerhlab {

TEST_F(OperationTests, DISABLED_BlackAdjustmentTest) {
  {
    SleeveManager manager{db_path_};
    ImageLoader   image_loader(128, 8, 0);
    image_path_t  path =
        L"D:\\Projects\\pu-erh_lab\\pu-erh_"
        L"lab\\tests\\resources\\sample_images\\jpg";
    std::vector<image_path_t> imgs;
    for (const auto& img : std::filesystem::directory_iterator(path)) {
      imgs.push_back(img.path());
    }

    auto display_callback = [](size_t idx, std::weak_ptr<Image> img) {
    };

    manager.LoadToPath(imgs, L"");
    auto view = manager.GetView();
    view->UpdateView(L"");
    // Load thumbnail for these images, do nothing with the callback
    view->LoadPreview(0, 20, display_callback);
    // For now, adjust the thumbnail only
    auto img = manager.GetPool()->AccessElement(0, AccessType::META).value().lock();

    cv::namedWindow("Black level Animation", cv::WINDOW_AUTOSIZE);
    ToneRegionOp op{0.0f, ToneRegion::BLACK};
    auto       param = op.GetParams();
    ImageBuffer to_adjust{img->GetThumbnailData()};
    int delay = 20;
    for (float value = -100.0f; value <= 100.0f; value += 1.0f) {
      param[op.GetScriptName()]["offset"] = value;
      op.SetParams(param);

      ImageBuffer to_display = op.Apply(to_adjust);
      cv::imshow("Black level Animation", to_display.GetCPUData());
      to_adjust = {img->GetThumbnailData()};

      if (cv::waitKey(delay) == 27) break; // Press ESC to exit
    }

    cv::waitKey(1);
  }
}

TEST_F(OperationTests, DISABLED_WhiteAdjustmentTest) {
  {
    SleeveManager manager{db_path_};
    ImageLoader   image_loader(128, 8, 0);
    image_path_t  path =
        L"D:\\Projects\\pu-erh_lab\\pu-erh_"
        L"lab\\tests\\resources\\sample_images\\jpg";
    std::vector<image_path_t> imgs;
    for (const auto& img : std::filesystem::directory_iterator(path)) {
      imgs.push_back(img.path());
    }

    auto display_callback = [](size_t idx, std::weak_ptr<Image> img) {
    };

    manager.LoadToPath(imgs, L"");
    auto view = manager.GetView();
    view->UpdateView(L"");
    // Load thumbnail for these images, do nothing with the callback
    view->LoadPreview(0, 20, display_callback);
    // For now, adjust the thumbnail only
    auto img = manager.GetPool()->AccessElement(0, AccessType::META).value().lock();

    cv::namedWindow("White level Animation", cv::WINDOW_AUTOSIZE);
    ToneRegionOp op{0.0f, ToneRegion::WHITE};
    auto       param = op.GetParams();
    ImageBuffer to_adjust{img->GetThumbnailData()};
    int delay = 20;
    for (float value = -100.0f; value <= 100.0f; value += 1.0f) {
      param[op.GetScriptName()]["offset"] = value;
      op.SetParams(param);

      ImageBuffer to_display = op.Apply(to_adjust);
      cv::imshow("White level Animation", to_display.GetCPUData());
      to_adjust = {img->GetThumbnailData()};

      if (cv::waitKey(delay) == 27) break; // Press ESC to exit
    }

    cv::waitKey(1);
  }
}

TEST_F(OperationTests, ShadowsAdjustmentTest) {
  {
    SleeveManager manager{db_path_};
    ImageLoader   image_loader(128, 8, 0);
    image_path_t  path =
        L"D:\\Projects\\pu-erh_lab\\pu-erh_"
        L"lab\\tests\\resources\\sample_images\\jpg";
    std::vector<image_path_t> imgs;
    for (const auto& img : std::filesystem::directory_iterator(path)) {
      imgs.push_back(img.path());
    }

    auto display_callback = [](size_t idx, std::weak_ptr<Image> img) {
    };

    manager.LoadToPath(imgs, L"");
    auto view = manager.GetView();
    view->UpdateView(L"");
    // Load thumbnail for these images, do nothing with the callback
    view->LoadPreview(0, 20, display_callback);
    // For now, adjust the thumbnail only
    auto img = manager.GetPool()->AccessElement(0, AccessType::META).value().lock();

    cv::namedWindow("Shadows Animation", cv::WINDOW_AUTOSIZE);
    ToneRegionOp op{0.0f, ToneRegion::SHADOWS};
    auto       param = op.GetParams();
    ImageBuffer to_adjust{img->GetThumbnailData()};
    int          delay = 20;
    ImageBuffer to_display;
    for (float value = -100.0f; value <= 100.0f; value += 1.0f) {
      param[op.GetScriptName()]["offset"] = value;
      op.SetParams(param);

      to_display = op.Apply(to_adjust);
      cv::imshow("Shadows Animation", to_display.GetCPUData());
      to_adjust = {img->GetThumbnailData()};

      if (cv::waitKey(delay) == 27) break; // Press ESC to exit
    }

    cv::waitKey(0);
  }
}

TEST_F(OperationTests, DISABLED_HighlightsAdjustmentTest) {
  {
    SleeveManager manager{db_path_};
    ImageLoader   image_loader(128, 8, 0);
    image_path_t  path =
        L"D:\\Projects\\pu-erh_lab\\pu-erh_"
        L"lab\\tests\\resources\\sample_images\\jpg";
    std::vector<image_path_t> imgs;
    for (const auto& img : std::filesystem::directory_iterator(path)) {
      imgs.push_back(img.path());
    }

    auto display_callback = [](size_t idx, std::weak_ptr<Image> img) {
    };

    manager.LoadToPath(imgs, L"");
    auto view = manager.GetView();
    view->UpdateView(L"");
    // Load thumbnail for these images, do nothing with the callback
    view->LoadPreview(0, 20, display_callback);
    // For now, adjust the thumbnail only
    auto img = manager.GetPool()->AccessElement(0, AccessType::META).value().lock();

    cv::namedWindow("Highlights Animation", cv::WINDOW_AUTOSIZE);
    ToneRegionOp op{0.0f, ToneRegion::HIGHLIGHTS};
    auto       param = op.GetParams();
    ImageBuffer to_adjust{img->GetThumbnailData()};
    int delay = 20;
    for (float value = -100.0f; value <= 100.0f; value += 1.0f) {
      param[op.GetScriptName()]["offset"] = value;
      op.SetParams(param);

      ImageBuffer to_display = op.Apply(to_adjust);
      cv::imshow("Highlights Animation", to_display.GetCPUData());
      to_adjust = {img->GetThumbnailData()};

      if (cv::waitKey(delay) == 27) break; // Press ESC to exit
    }

    cv::waitKey(1);
  }
}
}  // namespace puerhlab