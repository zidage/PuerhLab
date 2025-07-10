#include "edit/operators/wheel/color_wheel_op.hpp"

#include <opencv2/highgui.hpp>

#include "../op_test_fixation.hpp"
#include "image/image_buffer.hpp"
#include "sleeve/sleeve_manager.hpp"

using namespace puerhlab;
TEST_F(OperationTests, SharpenAdjustmentTest) {
  {
    SleeveManager manager{db_path_};
    ImageLoader   image_loader(128, 8, 0);
    image_path_t  path =
        L"D:\\Projects\\pu-erh_lab\\pu-erh_"
        L"lab\\tests\\resources\\sample_images\\original_jpg";
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
    auto           img = manager.GetPool()->AccessElement(0, AccessType::THUMB).value().lock();

    ColorWheelOp   color_wheels;
    // A teal and orange look
    nlohmann::json teal_orange_look = {
        {"lift",
         {{"color_offset.x", -0.12},
          {"color_offset.y", 0.03},
          {"color_offset.z", 0.1},
          {"luminance_offset", 0.0}}},
        {"gamma", {{"color_offset.x", 1.0},
          {"color_offset.y", 1.00},
          {"color_offset.z", 1.00},
          {"luminance_offset", 0.0}}},
        {"gain",
         {{"color_offset.x", 1.35},
          {"color_offset.y", 0.93},
          {"color_offset.z", 0.68},
          {"luminance_offset", 0.0}}},
        {"crossovers", {{"lift", 0.2}, {"gain", 0.8}}}};
    nlohmann::json params;
    params[color_wheels._script_name] = teal_orange_look;
    color_wheels.SetParams(params);

    ImageBuffer orignal{img->GetThumbnailData().clone()};
    ImageBuffer input{img->GetThumbnailData()};
    ImageBuffer result = color_wheels.Apply(input);

    cv::imshow("Original", orignal.GetCPUData());
    cv::imshow("Teal and Orange Look", result.GetCPUData());
    cv::waitKey(0);
  }
}