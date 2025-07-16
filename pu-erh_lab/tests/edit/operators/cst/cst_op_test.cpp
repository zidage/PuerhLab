#include "edit/operators/cst/cst_op.hpp"

#include <opencv2/highgui.hpp>

#include "../op_test_fixation.hpp"
#include "sleeve/sleeve_manager.hpp"

using namespace puerhlab;
TEST_F(OperationTests, ColorSpaceTransformTest) {
  {
    SleeveManager manager{db_path_};
    ImageLoader   image_loader(128, 8, 0);
    image_path_t  path =
        L"D:\\Projects\\pu-erh_lab\\pu-erh_lab\\tests\\resources\\sample_images\\cst\\adobe_rgb";
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
    auto            img = manager.GetPool()->AccessElement(0, AccessType::THUMB).value().lock();

    OCIOTransformOp op{"Gamma 2.2 Encoded AdobeRGB", "Gamma 2.2 Encoded Rec.709",
                       "D:\\Projects\\pu-erh_lab\\pu-erh_lab\\src\\config\\OCIO\\config.ocio"};
    ImageBuffer     to_adjust{img->GetThumbnailData()};
    ImageBuffer     to_display = op.Apply(to_adjust);
    cv::namedWindow("Adobe RGB", cv::WINDOW_KEEPRATIO);
    cv::imshow("Adobe RGB", img->GetThumbnailData());
    cv::namedWindow("sRGB", cv::WINDOW_KEEPRATIO);
    cv::imshow("sRGB", to_display.GetCPUData());

    cv::waitKey(0);
  }
}