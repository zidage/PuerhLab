//  Copyright 2026 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

// SM3.A: the gRPC segmentation client against an in-process SegmentationService.

#include "sidecar_client/segmentation_client.hpp"

#include <grpcpp/security/server_credentials.h>
#include <grpcpp/server.h>
#include <grpcpp/server_builder.h>
#include <grpcpp/server_context.h>
#include <gtest/gtest.h>

#include <chrono>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <vector>

#include "ai_common.pb.h"
#include "segmentation.grpc.pb.h"

namespace alcedo::sidecar_client {
namespace {

constexpr int  kMaxMessageBytes = 64 * 1024 * 1024;
constexpr auto kTimeout         = std::chrono::milliseconds(10000);

void           SetOkHeader(alcedo::ai::AiResponseHeader* header) {
  header->set_status(alcedo::ai::AI_STATUS_OK);
  header->set_error_code(alcedo::ai::AI_ERROR_NONE);
}

/// Records what the client sends; answers with fixed values.
class RecordingSegmentationService final : public alcedo::ai::SegmentationService::Service {
 public:
  auto LoadSegmentationModels(grpc::ServerContext*                             context,
                              const alcedo::ai::LoadSegmentationModelsRequest* request,
                              alcedo::ai::LoadSegmentationModelsResponse*      response)
      -> grpc::Status override {
    Record(context);
    auto* header = response->mutable_header();
    header->set_status(alcedo::ai::AI_STATUS_INVALID_ARGUMENT);
    header->set_error_code(alcedo::ai::AI_ERROR_INVALID_INPUT);
    header->set_error_message("segmentation device \"" + request->device() + "\" is not supported");
    return grpc::Status::OK;
  }

  auto GetSegmentationStatus(grpc::ServerContext* context,
                             const alcedo::ai::GetSegmentationStatusRequest* /*request*/,
                             alcedo::ai::GetSegmentationStatusResponse* response)
      -> grpc::Status override {
    Record(context);
    SetOkHeader(response->mutable_header());
    response->set_loaded(true);
    response->set_model_set_id("alcedo-ai-mask-v1");
    response->set_backend("onnxruntime-directml");
    auto* model = response->add_models();
    model->set_role("encoder");
    model->set_device("directml:0");
    model->set_load_ms(2900);
    response->set_session_count(1);
    return grpc::Status::OK;
  }

  auto StartSegmentationSession(grpc::ServerContext*                               context,
                                const alcedo::ai::StartSegmentationSessionRequest* request,
                                alcedo::ai::StartSegmentationSessionResponse*      response)
      -> grpc::Status override {
    Record(context);
    {
      std::lock_guard lock(mutex_);
      received_width_       = request->width();
      received_height_      = request->height();
      received_pixel_bytes_ = request->rgb8_pixels().size();
    }
    SetOkHeader(response->mutable_header());
    response->set_session_id("session-1");
    response->set_encoder_ms(70);
    return grpc::Status::OK;
  }

  auto DecodeSegmentationPrompt(grpc::ServerContext*                               context,
                                const alcedo::ai::DecodeSegmentationPromptRequest* request,
                                alcedo::ai::DecodeSegmentationPromptResponse*      response)
      -> grpc::Status override {
    Record(context);
    {
      std::lock_guard lock(mutex_);
      decode_request_ = *request;
    }
    SetOkHeader(response->mutable_header());
    for (uint32_t token : {2U, 3U, 1U}) {
      auto* level = response->add_masks();
      level->mutable_logits()->Resize(256 * 256, static_cast<float>(token));
      level->set_score(0.25F * static_cast<float>(token));
      level->set_token(token);
      level->set_area(100 * token);
    }
    response->set_selected_index(2);
    response->set_logit_size(256);
    response->set_decoder_ms(12);
    return grpc::Status::OK;
  }

  [[nodiscard]] auto Peers() -> std::set<std::string> {
    std::lock_guard lock(mutex_);
    return peers_;
  }
  [[nodiscard]] auto ReceivedPixelBytes() -> size_t {
    std::lock_guard lock(mutex_);
    return received_pixel_bytes_;
  }
  [[nodiscard]] auto ReceivedSize() -> std::pair<uint32_t, uint32_t> {
    std::lock_guard lock(mutex_);
    return {received_width_, received_height_};
  }
  [[nodiscard]] auto DecodeRequest() -> alcedo::ai::DecodeSegmentationPromptRequest {
    std::lock_guard lock(mutex_);
    return decode_request_;
  }

 private:
  void Record(grpc::ServerContext* context) {
    std::lock_guard lock(mutex_);
    peers_.insert(context->peer());
  }

  std::mutex                                  mutex_;
  std::set<std::string>                       peers_;
  uint32_t                                    received_width_       = 0;
  uint32_t                                    received_height_      = 0;
  size_t                                      received_pixel_bytes_ = 0;
  alcedo::ai::DecodeSegmentationPromptRequest decode_request_;
};

class SegmentationClientTest : public ::testing::Test {
 protected:
  void SetUp() override {
    grpc::ServerBuilder builder;
    int                 port = 0;
    builder.AddListeningPort("127.0.0.1:0", grpc::InsecureServerCredentials(), &port);
    builder.SetMaxReceiveMessageSize(kMaxMessageBytes);
    builder.SetMaxSendMessageSize(kMaxMessageBytes);
    builder.RegisterService(&service_);
    server_ = builder.BuildAndStart();
    ASSERT_NE(server_, nullptr);
    ASSERT_GT(port, 0);
    client_ = MakeGrpcSegmentationClient("127.0.0.1:" + std::to_string(port));
  }

  void TearDown() override {
    client_.reset();
    if (server_) {
      server_->Shutdown();
    }
  }

  RecordingSegmentationService        service_;
  std::unique_ptr<grpc::Server>       server_;
  std::unique_ptr<SegmentationClient> client_;
};

TEST_F(SegmentationClientTest, StartSessionSendsImageLargerThanDefaultLimit) {
  SegmentationRgb8Image image;
  image.width  = 4096;
  image.height = 2731;
  image.pixels.assign(static_cast<size_t>(image.width) * image.height * 3, 127);

  const auto result = client_->StartSession(image, kTimeout);

  ASSERT_TRUE(result.status.ok) << result.status.error;
  EXPECT_EQ(result.session_id, "session-1");
  EXPECT_EQ(result.encoder_ms, 70);
  EXPECT_EQ(service_.ReceivedPixelBytes(), image.pixels.size());
  EXPECT_EQ(service_.ReceivedSize(), (std::pair<uint32_t, uint32_t>{4096, 2731}));
}

TEST_F(SegmentationClientTest, SequentialCallsUseOneConnection) {
  for (int call = 0; call < 20; ++call) {
    const auto status = client_->GetStatus(kTimeout);
    ASSERT_TRUE(status.status.ok) << status.status.error;
    EXPECT_TRUE(status.loaded);
    ASSERT_EQ(status.models.size(), 1U);
    EXPECT_EQ(status.models[0].device, "directml:0");
    EXPECT_EQ(status.models[0].load_ms, 2900);
  }
  EXPECT_EQ(service_.Peers().size(), 1U);
}

TEST_F(SegmentationClientTest, FailureHeaderMapsToFailedStatus) {
  const auto status = client_->LoadModels("D:/models/alcedo-ai-mask-v1", "cuda", kTimeout);

  EXPECT_FALSE(status.ok);
  EXPECT_EQ(status.status, static_cast<int>(alcedo::ai::AI_STATUS_INVALID_ARGUMENT));
  EXPECT_EQ(status.error_code, static_cast<int>(alcedo::ai::AI_ERROR_INVALID_INPUT));
  EXPECT_EQ(status.error, "segmentation device \"cuda\" is not supported");
}

TEST_F(SegmentationClientTest, UnreachableEndpointReportsTransportError) {
  // Port 1 has no listener on the test machine.
  auto       unreachable = MakeGrpcSegmentationClient("127.0.0.1:1");
  const auto status      = unreachable->UnloadModels(std::chrono::milliseconds(500));

  EXPECT_FALSE(status.ok);
  EXPECT_EQ(status.status, static_cast<int>(alcedo::ai::AI_STATUS_UNSPECIFIED));
  EXPECT_FALSE(status.error.empty());
}

TEST_F(SegmentationClientTest, DecodeRequestCarriesPromptAndMaskInput) {
  SegmentationPrompt prompt;
  prompt.points = {{.x = 0.25F, .y = 0.5F, .label = SegmentationPointLabel::kAdd},
                   {.x = 0.75F, .y = 0.5F, .label = SegmentationPointLabel::kSubtract}};
  prompt.box    = SegmentationBox{.x0 = 0.1F, .y0 = 0.2F, .x1 = 0.9F, .y1 = 0.8F};
  prompt.mask_input.assign(256 * 256, -3.0F);
  prompt.mask_input[17] = 5.0F;

  const auto result     = client_->Decode("session-1", prompt, kTimeout);

  ASSERT_TRUE(result.status.ok) << result.status.error;
  const auto request = service_.DecodeRequest();
  EXPECT_EQ(request.session_id(), "session-1");
  ASSERT_EQ(request.points_size(), 2);
  EXPECT_FLOAT_EQ(request.points(0).x(), 0.25F);
  EXPECT_EQ(request.points(0).label(), alcedo::ai::SEGMENTATION_POINT_LABEL_ADD);
  EXPECT_EQ(request.points(1).label(), alcedo::ai::SEGMENTATION_POINT_LABEL_SUBTRACT);
  ASSERT_TRUE(request.has_box());
  EXPECT_FLOAT_EQ(request.box().x1(), 0.9F);
  ASSERT_EQ(request.mask_input_size(), 256 * 256);
  EXPECT_FLOAT_EQ(request.mask_input(17), 5.0F);

  ASSERT_EQ(result.masks.size(), 3U);
  EXPECT_EQ(result.masks[0].token, 2U);
  EXPECT_EQ(result.masks[2].area, 100U);
  EXPECT_EQ(result.masks[1].logits.size(), 256U * 256U);
  EXPECT_FLOAT_EQ(result.masks[1].logits[0], 3.0F);
  EXPECT_EQ(result.selected_index, 2U);
  EXPECT_EQ(result.logit_size, 256U);
  EXPECT_EQ(result.decoder_ms, 12);
}

}  // namespace
}  // namespace alcedo::sidecar_client
