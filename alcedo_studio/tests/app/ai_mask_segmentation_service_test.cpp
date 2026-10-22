//  Copyright 2026 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

// SM3.A: the AI Mask tool lifecycle of AiMaskSegmentationService over a fake segmentation client.

#include "app/ai_mask_segmentation_service.hpp"

#include <gtest/gtest.h>

#include <chrono>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace alcedo {
namespace {

using CallLog = std::vector<std::string>;

auto Ok() -> SegmentationCallStatus {
  SegmentationCallStatus status;
  status.ok = true;
  return status;
}

auto Failed(std::string message) -> SegmentationCallStatus {
  SegmentationCallStatus status;
  status.ok    = false;
  status.error = std::move(message);
  return status;
}

/// Writes each call into a shared log; answers with configurable results.
class FakeSegmentationClient final : public sidecar_client::SegmentationClient {
 public:
  explicit FakeSegmentationClient(std::shared_ptr<CallLog> log) : log_(std::move(log)) {}

  auto LoadModels(const std::string& model_directory, const std::string& device,
                  std::chrono::milliseconds /*timeout*/) -> SegmentationCallStatus override {
    log_->push_back("load " + model_directory + " " + device);
    return Ok();
  }
  auto UnloadModels(std::chrono::milliseconds /*timeout*/) -> SegmentationCallStatus override {
    log_->push_back("unload");
    return Ok();
  }
  auto GetStatus(std::chrono::milliseconds /*timeout*/) -> SegmentationStatus override {
    log_->push_back("status");
    return SegmentationStatus{.status = Ok()};
  }
  auto StartSession(const SegmentationRgb8Image& image, std::chrono::milliseconds /*timeout*/)
      -> SegmentationSessionResult override {
    const auto session_id = "s" + std::to_string(++started_);
    log_->push_back("start " + session_id + " " + std::to_string(image.width));
    SegmentationSessionResult result;
    if (fail_start_) {
      result.status = Failed("encoder failed");
      return result;
    }
    result.status     = Ok();
    result.session_id = session_id;
    return result;
  }
  auto Decode(const std::string& session_id, const SegmentationPrompt& prompt,
              std::chrono::milliseconds /*timeout*/) -> SegmentationDecodeResult override {
    log_->push_back("decode " + session_id + " " + std::to_string(prompt.points.size()));
    return SegmentationDecodeResult{.status = Ok()};
  }
  auto EndSession(const std::string& session_id, std::chrono::milliseconds /*timeout*/)
      -> SegmentationCallStatus override {
    log_->push_back("end " + session_id);
    return fail_end_ ? Failed("transport closed") : Ok();
  }

  bool fail_start_ = false;
  bool fail_end_   = false;

 private:
  std::shared_ptr<CallLog> log_;
  int                      started_ = 0;
};

struct ServiceUnderTest {
  std::shared_ptr<CallLog>                   log    = std::make_shared<CallLog>();
  FakeSegmentationClient*                    client = nullptr;
  std::unique_ptr<AiMaskSegmentationService> service;

  ServiceUnderTest() {
    auto fake                           = std::make_unique<FakeSegmentationClient>(log);
    client                              = fake.get();
    // The lease deleter writes into the same log, so the test sees when the lease is released.
    auto                  log_for_lease = log;
    std::shared_ptr<void> lease(new int(0), [log_for_lease](int* token) {
      delete token;
      log_for_lease->push_back("lease released");
    });
    service = std::make_unique<AiMaskSegmentationService>(std::move(lease), std::move(fake));
  }
};

auto Image(uint32_t width) -> SegmentationRgb8Image {
  SegmentationRgb8Image image;
  image.width  = width;
  image.height = 4;
  image.pixels.assign(static_cast<size_t>(width) * 4 * 3, 0);
  return image;
}

TEST(AiMaskSegmentationServiceTest, StartImageEndsPreviousSession) {
  ServiceUnderTest under_test;
  ASSERT_TRUE(under_test.service->StartImage(Image(8)).status.ok);
  ASSERT_TRUE(under_test.service->StartImage(Image(16)).status.ok);

  EXPECT_EQ(*under_test.log, (CallLog{"start s1 8", "end s1", "start s2 16"}));
  EXPECT_TRUE(under_test.service->HasSession());
}

TEST(AiMaskSegmentationServiceTest, StartImageReportsEndFailureWithoutStarting) {
  ServiceUnderTest under_test;
  ASSERT_TRUE(under_test.service->StartImage(Image(8)).status.ok);
  under_test.client->fail_end_ = true;

  const auto result            = under_test.service->StartImage(Image(16));

  EXPECT_FALSE(result.status.ok);
  EXPECT_EQ(result.status.error, "transport closed");
  EXPECT_EQ(*under_test.log, (CallLog{"start s1 8", "end s1"}));
  EXPECT_TRUE(under_test.service->HasSession());
}

TEST(AiMaskSegmentationServiceTest, FailedStartKeepsNoSession) {
  ServiceUnderTest under_test;
  under_test.client->fail_start_ = true;

  EXPECT_FALSE(under_test.service->StartImage(Image(8)).status.ok);
  EXPECT_FALSE(under_test.service->HasSession());
  EXPECT_TRUE(under_test.service->EndImage().ok);
  EXPECT_EQ(*under_test.log, (CallLog{"start s1 8"}));
}

TEST(AiMaskSegmentationServiceTest, DecodeWithoutSessionFailsWithoutCall) {
  ServiceUnderTest   under_test;
  SegmentationPrompt prompt;
  prompt.points.push_back({.x = 0.5F, .y = 0.5F, .label = SegmentationPointLabel::kAdd});

  const auto result = under_test.service->Decode(prompt);

  EXPECT_FALSE(result.status.ok);
  EXPECT_TRUE(under_test.log->empty());
  ASSERT_TRUE(under_test.service->StartImage(Image(8)).status.ok);
  EXPECT_TRUE(under_test.service->Decode(prompt).status.ok);
  EXPECT_EQ(under_test.log->back(), "decode s1 1");
}

TEST(AiMaskSegmentationServiceTest, CloseEndsSessionUnloadsAndReleasesLease) {
  ServiceUnderTest under_test;
  ASSERT_TRUE(under_test.service->LoadModels("models", "directml").ok);
  ASSERT_TRUE(under_test.service->StartImage(Image(8)).status.ok);

  EXPECT_TRUE(under_test.service->Close().ok);
  EXPECT_EQ(*under_test.log,
            (CallLog{"load models directml", "start s1 8", "end s1", "unload", "lease released"}));
  EXPECT_TRUE(under_test.service->IsClosed());

  EXPECT_TRUE(under_test.service->Close().ok);
  EXPECT_FALSE(under_test.service->StartImage(Image(8)).status.ok);
  EXPECT_EQ(under_test.log->size(), 5U);
}

TEST(AiMaskSegmentationServiceTest, DestructorClosesService) {
  ServiceUnderTest under_test;
  ASSERT_TRUE(under_test.service->StartImage(Image(8)).status.ok);

  under_test.client = nullptr;
  under_test.service.reset();

  EXPECT_EQ(*under_test.log, (CallLog{"start s1 8", "end s1", "unload", "lease released"}));
}

}  // namespace
}  // namespace alcedo
