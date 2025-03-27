#include <gtest/gtest.h>
#include <libraw/libraw.h>
#include <libraw/libraw_types.h>

#include <vector>
#include <filesystem>
#include <opencv2/opencv.hpp>

#include "type/type.hpp"

TEST(SingleThumbnailLoad, BasicAssertions) {
  // std::cout << std::filesystem::current_path() << std::endl;
  // Set the path of the test image, which has to be an absolute path, didn't know why...
  image_path_t test_img = L"D:\\Projects\\pu-erh_lab\\pu-erh_lab\\tests\\resources\\sample_images\\raw\\_DSC0726.ARW";
  LibRaw raw_processor;

  // Try to open the file
  ASSERT_EQ(raw_processor.open_file(test_img.c_str()), LIBRAW_SUCCESS);

  // Try to unpack the image -> to metadata and
  ASSERT_EQ(raw_processor.unpack(), LIBRAW_SUCCESS);

  ASSERT_EQ(raw_processor.unpack_thumb(), LIBRAW_SUCCESS);

  libraw_processed_image_t *thumbnail = raw_processor.dcraw_make_mem_thumb();
  if (!thumbnail) {
    std::cerr << "Error generating RGB image!" << std::endl;
    ASSERT_TRUE(false);
  }

  // Decide the format of the thumbnail
  cv::Mat img;
  if (thumbnail->type == LIBRAW_IMAGE_JPEG) {
      // For JPEG format, decode directly
      std::vector<uchar> jpegData(thumbnail->data, thumbnail->data + thumbnail->data_size);
      img = cv::imdecode(jpegData, cv::IMREAD_COLOR);
  } else if (thumbnail->type == LIBRAW_IMAGE_BITMAP) {
      // For Bmp file, manually decode
      img = cv::Mat(thumbnail->height, thumbnail->width, CV_8UC3, thumbnail->data);
      cv::cvtColor(img, img, cv::COLOR_RGB2BGR);  // 转换颜色通道
  } else {
      std::cerr << "Unsupported thumbnail format." << std::endl;
      LibRaw::dcraw_clear_mem(thumbnail);
      return;
  }

  // if (!img.empty()) {
  //   imshow("RAW Thumbnail", img);
  //   cv::waitKey(0);
  // }


  LibRaw::dcraw_clear_mem(thumbnail);
  raw_processor.recycle();
}