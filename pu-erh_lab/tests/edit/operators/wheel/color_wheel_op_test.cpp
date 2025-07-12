#include "edit/operators/wheel/color_wheel_op.hpp"

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>

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
    nlohmann::json teal_orange_look = {{"lift",
                                        {{"color_offset.x", 0.0},
                                         {"color_offset.y", 0.0},
                                         {"color_offset.z", 0.0},
                                         {"luminance_offset", 0.0}}},
                                       {"gamma",
                                        {{"color_offset.x", 1.0},
                                         {"color_offset.y", 1.0},
                                         {"color_offset.z", 1.00},
                                         {"luminance_offset", -0.3}}},
                                       {"gain",
                                        {{"color_offset.x", 1.0},
                                         {"color_offset.y", 1.0},
                                         {"color_offset.z", 1.0},
                                         {"luminance_offset", 0.2}}},
                                       {"crossovers", {{"lift", 0.2}, {"gain", 0.8}}}};
    nlohmann::json params;
    params[color_wheels._script_name] = teal_orange_look;
    color_wheels.SetParams(params);

    ImageBuffer orignal{img->GetThumbnailData().clone()};
    cv::Mat     gradient_1d(1, 256, CV_32FC1);
    for (int i = 0; i < 256; ++i) gradient_1d.at<float>(0, i) = i / 256.0f;

    cv::Mat gradient_2d;
    cv::resize(gradient_1d, gradient_2d, cv::Size(256, 256), 0, 0, cv::INTER_LINEAR);
    cv::Mat gradient_2d_3c(256, 256, CV_32FC3);
    cv::cvtColor(gradient_2d, gradient_2d_3c, cv::COLOR_GRAY2BGR);

    cv::Mat waveform_input(256, 256, CV_32FC1, cv::Scalar(0.0f));
    cv::Mat waveform_output(256, 256, CV_32FC1, cv::Scalar(0.0f));
    
    ImageBuffer input{gradient_2d_3c};
    ImageBuffer result = color_wheels.Apply(input);

    gradient_2d.forEach<float>([&](float& pixel, const int* pos) {
      waveform_input.at<float>(static_cast<int>((1.0f - pixel) * 255.0f, pos[1])) = 1.0f;
    });

    result.GetCPUData().forEach<cv::Vec3f>([&](cv::Vec3f& pixel, const int* pos) {
      if (pixel[0] > 1.0f) {
        pixel[0] = 1.0f;
      }
      waveform_output.at<float>(static_cast<int>((1.0f - pixel[0]) * 255.0f), pos[1]) = 1.0f;
    });

    cv::imshow("Original", gradient_2d);
    cv::imshow("OriginalWave", waveform_input);
    cv::imshow("After", result.GetCPUData());
    cv::imshow("AfterWave", waveform_output);
    cv::waitKey(0);
  }
}