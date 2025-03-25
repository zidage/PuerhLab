#include <gtest/gtest.h>
#include <libraw/libraw.h>
#include <libraw/libraw_types.h>

#include <filesystem>
#include <opencv2/opencv.hpp>

#include "type/type.hpp"

TEST(SingleRawLoad, BasicAssertions) {
  // std::cout << std::filesystem::current_path() << std::endl;
  // Set the path of the test image, which has to be an absolute path, didn't know why...
  image_path_t test_img = L"D:\\Projects\\pu-erh_lab\\pu-erh_lab\\tests\\resources\\sample_images\\raw\\_DSC1306.dng";
  LibRaw raw_processor;

  // Try to open the file
  ASSERT_EQ(raw_processor.open_file(test_img), LIBRAW_SUCCESS);

  // Try to unpack the image -> to metadata and 
  ASSERT_EQ(raw_processor.unpack(), LIBRAW_SUCCESS);

  ASSERT_EQ(raw_processor.dcraw_process(), LIBRAW_SUCCESS);

  libraw_processed_image_t *img = raw_processor.dcraw_make_mem_image();
  if (!img) {
    std::cerr << "Error generating RGB image!" << std::endl;
    ASSERT_TRUE(false);
  }

  cv::Mat image(img->height, img->width, CV_8UC3, img->data);
  cv::Size newSize(image.cols / 8, image.rows / 8);
  cv::Mat imageRGB;
  cv::cvtColor(image, imageRGB, cv::COLOR_BGR2RGB);
  cv::Mat resizedImage;
  cv::resize(imageRGB, resizedImage, newSize, 0, 0, cv::INTER_LINEAR);
  
  cv::imshow("RAW Image", resizedImage);
  cv::waitKey(0);


  LibRaw::dcraw_clear_mem(img);
  raw_processor.recycle();
}