#include "edit/operators/cst/cst_op.hpp"

#include <opencv2/highgui.hpp>

#include "../op_test_fixation.hpp"
#include "decoders/raw_decoder.hpp"
#include "image/image_buffer.hpp"
#include "sleeve/sleeve_manager.hpp"

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
    auto            img = manager.GetPool()->AccessElement(0, AccessType::THUMB).value().lock();

    ACES_IDT_Op op{"Gamma 2.4 Encoded Rec.709", "sRGB Encoded Rec.709 (sRGB)",
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

TEST_F(OperationTests, ACESWorkflowTest) {
  {
    SleeveManager manager{db_path_};
    ImageLoader   image_loader(128, 8, 0);
    image_path_t  path =
        L"D:\\Projects\\pu-erh_lab\\pu-erh_lab\\tests\\resources\\sample_images\\raw";
    std::vector<image_path_t> imgs;
    for (const auto& img : std::filesystem::directory_iterator(path)) {
      if (!img.is_directory())
        imgs.push_back(img.path());
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

    ACES_IDT_Op IDT{"ACES2065-1", "Gamma 2.2 Encoded Rec.709",
                       "D:\\Projects\\pu-erh_lab\\pu-erh_lab\\src\\config\\OCIO\\config.ocio"};
    ImageBuffer     source{img->GetImageData()};
    ImageBuffer     to_adjust = IDT.Apply(source);

    // ACES_IDT_Op     INT{"ACEScct", "ACES2065-1"};
    // ImageBuffer intermediate = INT.Apply(to_adjust);
    // ACES_IDT_Op ODT("ACES2065-1", "Gamma 2.2 Encoded Rec.709");
    // ImageBuffer to_display = ODT.Apply(intermediate);
    cv::namedWindow("ACES2065-1", cv::WINDOW_KEEPRATIO);
    cv::imshow("ACES2065-1", img->GetImageData());
    cv::namedWindow("sRGB", cv::WINDOW_KEEPRATIO);
    cv::imshow("sRGB", to_adjust.GetCPUData());

    cv::waitKey(0);
  }
}