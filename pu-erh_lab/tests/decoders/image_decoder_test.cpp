#include "decoders/image_decoder.hpp"
#include "type/type.hpp"

#include <gtest/gtest.h>

TEST(SingleImageDecoder, BasicAssertions) {
  // Test decode only one image
  int a = 0;
  puerhlab::ImageDecoder image_decoder(2, 4);

  auto decoder_future1 = image_decoder.ScheduleDecode(
      L"D:\\Projects\\pu-erh_lab\\pu-erh_lab\\tests\\resources\\sample_"
      L"images\\jpg\\_DSC0293.jpg");
  auto decoder_future2 = image_decoder.ScheduleDecode(
      L"D:\\Projects\\pu-erh_lab\\pu-erh_lab\\tests\\resources\\sample_"
      L"images\\jpg\\_DSC0301.jpg");
  auto decoder_future3 = image_decoder.ScheduleDecode(
    L"D:\\Projects\\pu-erh_lab\\pu-erh_lab\\tests\\resources\\sample_"
    L"images\\jpg\\_DSC0306.jpg");
  auto decoder_future4 = image_decoder.ScheduleDecode(
      L"D:\\Projects\\pu-erh_lab\\pu-erh_lab\\tests\\resources\\sample_"
      L"images\\jpg\\_DSC0310.jpg");
  // decoder_future1.get();
  // decoder_future2.get();
  std::cin >> a;
}