#include "decoders/decoder_scheduler.hpp"
#include "../leak_detector/memory_leak_detector.hpp"
#include "decoders/decoder_scheduler.hpp"

#include <cstdint>
#include <future>
#include <gtest/gtest.h>
#include <memory>

TEST(MultipleImageDecoder, FORCE_LEAK) {
  // MemoryLeakDetector leakDetector;
  //  Test decode only one image

  puerhlab::DecoderScheduler scheduler(8, 8);

  std::shared_ptr<std::promise<uint32_t>> decode_promise_1 =
      std::make_shared<std::promise<uint32_t>>();
  auto decoder_future1 = decode_promise_1->get_future();
  scheduler.ScheduleDecode(
      L"D:\\Projects\\pu-erh_lab\\pu-erh_lab\\tests\\resources\\sample_"
      L"images\\jpg\\_DSC0293.jpg", puerhlab::DecodeType::THUMB,
      decode_promise_1);

  std::shared_ptr<std::promise<uint32_t>> decode_promise_2 =
      std::make_shared<std::promise<uint32_t>>();
  auto decoder_future2 = decode_promise_2->get_future();
  scheduler.ScheduleDecode(
      L"D:\\Projects\\pu-erh_lab\\pu-erh_lab\\tests\\resources\\sample_"
      L"images\\jpg\\_DSC0301.jpg", puerhlab::DecodeType::THUMB,
      decode_promise_2);

  std::shared_ptr<std::promise<uint32_t>> decode_promise_3 =
      std::make_shared<std::promise<uint32_t>>();
  auto decoder_future3 = decode_promise_3->get_future();
  scheduler.ScheduleDecode(
      L"D:\\Projects\\pu-erh_lab\\pu-erh_lab\\tests\\resources\\sample_"
      L"images\\jpg\\_DSC0306.jpg", puerhlab::DecodeType::THUMB,
      decode_promise_3);

  std::shared_ptr<std::promise<uint32_t>> decode_promise_4 =
      std::make_shared<std::promise<uint32_t>>();
  auto decoder_future4 = decode_promise_4->get_future();
  scheduler.ScheduleDecode(
      L"D:\\Projects\\pu-erh_lab\\pu-erh_lab\\tests\\resources\\sample_"
      L"images\\jpg\\_DSC0310.jpg", puerhlab::DecodeType::THUMB,
      decode_promise_4);

  std::shared_ptr<std::promise<uint32_t>> decode_promise_5 =
      std::make_shared<std::promise<uint32_t>>();
  auto decoder_future5 = decode_promise_5->get_future();
  scheduler.ScheduleDecode(
      L"D:\\Projects\\pu-erh_lab\\pu-erh_lab\\tests\\resources\\sample_"
      L"images\\jpg\\_DSC0312.jpg", puerhlab::DecodeType::THUMB,
      decode_promise_5);

  std::shared_ptr<std::promise<uint32_t>> decode_promise_6 =
      std::make_shared<std::promise<uint32_t>>();
  auto decoder_future6 = decode_promise_6->get_future();
  scheduler.ScheduleDecode(
      L"D:\\Projects\\pu-erh_lab\\pu-erh_lab\\tests\\resources\\sample_"
      L"images\\jpg\\_DSC0313.jpg", puerhlab::DecodeType::THUMB,
      decode_promise_6);

  std::shared_ptr<std::promise<uint32_t>> decode_promise_7 =
      std::make_shared<std::promise<uint32_t>>();
  auto decoder_future7 = decode_promise_7->get_future();
  scheduler.ScheduleDecode(
      L"D:\\Projects\\pu-erh_lab\\pu-erh_lab\\tests\\resources\\sample_"
      L"images\\jpg\\_DSC0317.jpg", puerhlab::DecodeType::THUMB,
      decode_promise_7);

  std::shared_ptr<std::promise<uint32_t>> decode_promise_8 =
      std::make_shared<std::promise<uint32_t>>();
  auto decoder_future8 = decode_promise_8->get_future();
  scheduler.ScheduleDecode(
      L"D:\\Projects\\pu-erh_lab\\pu-erh_lab\\tests\\resources\\sample_"
      L"images\\jpg\\_DSC0352.jpg", puerhlab::DecodeType::THUMB,
      decode_promise_8);

  ASSERT_EQ(decoder_future1.get(), 0);
  ASSERT_EQ(decoder_future2.get(), 1);
  ASSERT_EQ(decoder_future3.get(), 2);
  ASSERT_EQ(decoder_future4.get(), 3);
  ASSERT_EQ(decoder_future5.get(), 4);
  ASSERT_EQ(decoder_future6.get(), 5);
  ASSERT_EQ(decoder_future7.get(), 6);
  ASSERT_EQ(decoder_future8.get(), 7);

  // decoder_future2.get();
  // std::cin >> a;
}