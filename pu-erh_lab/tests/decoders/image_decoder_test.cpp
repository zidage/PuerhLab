#include "decoders/image_decoder.hpp"
#include "type/type.hpp"

//#include "../leak_detector/memory_leak_detector.hpp"

#include <gtest/gtest.h>

TEST(MultipleImageDecoder, FORCE_LEAK) {
  // Test decode only one image
  //MemoryLeakDetector leakDetector;
  puerhlab::ImageDecoder image_decoder(8, 8);

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
  auto decoder_future5 = image_decoder.ScheduleDecode(
      L"D:\\Projects\\pu-erh_lab\\pu-erh_lab\\tests\\resources\\sample_"
      L"images\\jpg\\_DSC0312.jpg");
  auto decoder_future6 = image_decoder.ScheduleDecode(
      L"D:\\Projects\\pu-erh_lab\\pu-erh_lab\\tests\\resources\\sample_"
      L"images\\jpg\\_DSC0313.jpg");
  auto decoder_future7 = image_decoder.ScheduleDecode(
      L"D:\\Projects\\pu-erh_lab\\pu-erh_lab\\tests\\resources\\sample_"
      L"images\\jpg\\_DSC0317.jpg");
  auto decoder_future8 = image_decoder.ScheduleDecode(
      L"D:\\Projects\\pu-erh_lab\\pu-erh_lab\\tests\\resources\\sample_"
      L"images\\jpg\\_DSC0352.jpg");

  decoder_future1.get();
  decoder_future2.get();
  decoder_future3.get();
  decoder_future4.get();
  decoder_future5.get();
  decoder_future6.get();
  decoder_future7.get();
  decoder_future8.get();

  // decoder_future2.get();
  //std::cin >> a;
}