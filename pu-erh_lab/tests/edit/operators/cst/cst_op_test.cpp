#include "edit/operators/cst/cst_op.hpp"

#include <opencv2/core/hal/interface.h>

#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>

#include "../op_test_fixation.hpp"
#include "decoders/raw_decoder.hpp"
#include "edit/operators/basic/exposure_op.hpp"
#include "edit/operators/basic/tone_region_op.hpp"
#include "edit/operators/color/saturation_op.hpp"
#include "edit/operators/color/vibrance_op.hpp"
#include "edit/operators/detail/clarity_op.hpp"
#include "image/image_buffer.hpp"
#include "sleeve/sleeve_manager.hpp"
#include "utils/string/convert.hpp"

using namespace puerhlab;
TEST_F(OperationTests, DISABLED_ColorSpaceTransformTest) {
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
    // For now, adjust the thumbnails'RGBsRGB
    auto img = manager.GetPool()->AccessElement(0, AccessType::THUMB).value().lock();

    OCIO_ACES_Transform_Op op{
        "Gamma 2.4 Encoded Rec.709", "sRGB Encoded Rec.709 (sRGB)",
        "D:\\Projects\\pu-erh_lab\\pu-erh_lab\\src\\config\\OCIO\\config.ocio"};
    ImageBuffer to_adjust{img->GetThumbnailData()};
    ImageBuffer to_display = op.Apply(to_adjust);
    cv::namedWindow("Adobe RGB", cv::WINDOW_KEEPRATIO);
    cv::imshow("Adobe RGB", img->GetThumbnailData());
    cv::Mat display_8u;
    to_display.GetCPUData().convertTo(display_8u, CV_8UC3);
    cv::namedWindow("sRGB", cv::WINDOW_KEEPRATIO);
    cv::imshow("sRGB", display_8u);

    cv::waitKey(0);
  }
}

TEST_F(OperationTests, DISABLED_ACESWorkflowTest1) {
  {
    SleeveManager manager{db_path_};
    ImageLoader   image_loader(128, 8, 0);
    image_path_t  path =
        L"D:\\Projects\\pu-erh_lab\\pu-erh_lab\\tests\\resources\\sample_images\\raw\\building";
    std::vector<image_path_t> imgs;
    for (const auto& img : std::filesystem::directory_iterator(path)) {
      if (!img.is_directory()) imgs.push_back(img.path());
    }

    manager.LoadToPath(imgs, L"");

    // Read image data
    manager.GetPool()->RecordAccess(0, AccessType::FULL_IMG);
    auto            img = manager.GetPool()->AccessElement(0, AccessType::FULL_IMG).value().lock();
    RawDecoder      decoder;
    std::ifstream   file(img->_image_path, std::ios::binary | std::ios::ate);
    std::streamsize fileSize = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<char> buffer(fileSize);
    if (!file.read(buffer.data(), fileSize)) {
      FAIL();
    }
    file.close();

    decoder.Decode(std::move(buffer), img);

    OCIO_ACES_Transform_Op IDT{
        "", "ACEScct", "D:\\Projects\\pu-erh_lab\\pu-erh_lab\\src\\config\\OCIO\\config.ocio"};
    ImageBuffer source{img->GetImageData()};
    ImageBuffer to_adjust = IDT.Apply(source);
    // OCIO_ACES_Transform_Op     LMT{"ACEScct", OCIO::ROLE_SCENE_LINEAR};
    // ImageBuffer intermediate = LMT.Apply(to_adjust);
    // OCIO_ACES_Transform_Op ODT(OCIO::ROLE_SCENE_LINEAR, "Rec.1886 Rec.709 - Display");
    // ImageBuffer to_display = ODT.Apply(intermediate, "ACES 1.0 - SDR Video");
    // cv::namedWindow("ACES2065-1", cv::WINDOW_KEEPRATIO);
    // cv::imshow("ACES2065-1", img->GetImageData());
    // cv::namedWindow("sRGB", cv::WINDOW_KEEPRATIO);
    // cv::imshow("sRGB", to_display.GetCPUData());

    cv::Mat     to_save_acescct;
    to_adjust.GetCPUData().convertTo(to_save_acescct, CV_16UC3, 65535.0f);
    cv::imwrite(
        "D:\\Projects\\pu-erh_lab\\pu-erh_lab\\tests\\resources\\sample_images\\my_"
        "pipeline\\acescct.tiff",
        to_save_acescct);

    cv::Mat to_save_aces2065;
    img->GetImageData().convertTo(to_save_aces2065, CV_16UC3, 65535.0f);
    cv::imwrite(
        "D:\\Projects\\pu-erh_lab\\pu-erh_lab\\tests\\resources\\sample_images\\my_"
        "pipeline\\aces2065.tiff",
        to_save_aces2065);
    cv::waitKey(0);
  }
}

TEST_F(OperationTests, ACESWorkflowTest2) {
  {
    SleeveManager manager{db_path_};
    ImageLoader   image_loader(128, 8, 0);
    image_path_t  path =
        L"D:\\Projects\\pu-erh_lab\\pu-erh_lab\\tests\\resources\\sample_images\\raw\\lowlight";
    std::vector<image_path_t> imgs;
    for (const auto& img : std::filesystem::directory_iterator(path)) {
      if (!img.is_directory()) imgs.push_back(img.path());
    }

    manager.LoadToPath(imgs, L"");

    // Read image data
    manager.GetPool()->RecordAccess(0, AccessType::FULL_IMG);
    auto            img = manager.GetPool()->AccessElement(0, AccessType::FULL_IMG).value().lock();
    RawDecoder      decoder;
    std::ifstream   file(img->_image_path, std::ios::binary | std::ios::ate);
    std::streamsize fileSize = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<char> buffer(fileSize);
    if (!file.read(buffer.data(), fileSize)) {
      FAIL();
    }
    file.close();

    decoder.Decode(std::move(buffer), img);

    OCIO_ACES_Transform_Op IDT{
        "ACES - ACES2065-1", "ACEScct",
        "D:\\Projects\\pu-erh_lab\\pu-erh_lab\\src\\config\\OCIO\\config.ocio"};
    ImageBuffer            source{img->GetImageData()};
    ImageBuffer            to_adjust = IDT.Apply(source);

    ExposureOp             EV{0.15f};
    to_adjust = EV.Apply(to_adjust);
    ToneRegionOp           HLOP{-20.0f, ToneRegion::HIGHLIGHTS};
    to_adjust = HLOP.Apply(to_adjust);

    ToneRegionOp           SDOP{20.0f, ToneRegion::SHADOWS};
    to_adjust = SDOP.Apply(to_adjust);

    SaturationOp VIB{10.0f};
    to_adjust = VIB.Apply(to_adjust);

    ClarityOp CLRT{30.0f};
    to_adjust = CLRT.Apply(to_adjust);

    std::filesystem::path look_path{"D:\\Projects\\pu-erh_lab\\pu-erh_lab\\src\\config\\LUTs\\ACES CCT 2383 D65.cube"};
    OCIO_ACES_Transform_Op LMT{look_path};
    to_adjust = LMT.ApplyLMT(to_adjust);
    
    OCIO_ACES_Transform_Op pre_RRT{"ACEScct", ""};
    ImageBuffer            intermediate = pre_RRT.Apply(to_adjust);
    OCIO_ACES_Transform_Op ODT("", "sRGB - Display");
    ImageBuffer            to_display = ODT.Apply(intermediate);

    cv::Mat                to_save_rec709;
    to_display.GetCPUData().convertTo(to_save_rec709, CV_16UC3, 65535.0f);
    cv::cvtColor(to_save_rec709, to_save_rec709, cv::COLOR_RGB2BGR);
    cv::imwrite(std::format("D:\\Projects\\pu-erh_lab\\pu-erh_lab\\tests\\resources\\sample_"
                            "images\\my_pipeline\\{}.tiff",
                            conv::ToBytes(img->_image_name)),
                to_save_rec709);
  }
}