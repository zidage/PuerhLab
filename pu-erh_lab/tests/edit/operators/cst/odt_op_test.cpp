#include <gtest/gtest.h>

#include "edit/operators/cst/odt_op.hpp"
#include "edit/pipeline/default_pipeline_params.hpp"

namespace puerhlab {

TEST(ODTOpTests, DefaultRoundTripUsesOpenDRT) {
  const nlohmann::json params = pipeline_defaults::MakeDefaultODTParams();
  ODT_Op               op(params);

  const nlohmann::json exported = op.GetParams();
  ASSERT_TRUE(exported.contains("odt"));
  EXPECT_EQ(exported["odt"]["method"], "open_drt");
  EXPECT_EQ(exported["odt"]["encoding_space"], "rec709");
  EXPECT_EQ(exported["odt"]["encoding_etof"], "gamma_2_2");
  EXPECT_EQ(exported["odt"]["open_drt"]["look_preset"], "standard");
}

TEST(ODTOpTests, MethodSwitchesToACES2) {
  nlohmann::json params = pipeline_defaults::MakeDefaultODTParams();
  params["odt"]["method"] = "aces_2_0";
  params["odt"]["limiting_space"] = "rec709";

  ODT_Op         op(params);
  OperatorParams global_params;
  op.SetGlobalParams(global_params);

  EXPECT_EQ(global_params.to_output_params_.method_, ColorUtils::ODTMethod::ACES_2_0);
  EXPECT_EQ(op.GetParams()["odt"]["method"], "aces_2_0");
  EXPECT_EQ(op.GetParams()["odt"]["limiting_space"], "rec709");
}

TEST(ODTOpTests, UnsupportedOpenDRTOutputCombinationThrows) {
  nlohmann::json params = pipeline_defaults::MakeDefaultODTParams();
  params["odt"]["encoding_space"] = "prophoto";
  EXPECT_THROW({ ODT_Op op(params); }, std::runtime_error);
}

TEST(ODTOpTests, PresetExpansionProducesStableResolvedRuntimeValues) {
  nlohmann::json params = pipeline_defaults::MakeDefaultODTParams();
  params["odt"]["open_drt"]["tonescale_preset"] = "aces_2_0";

  ODT_Op         op(params);
  OperatorParams global_params;
  op.SetGlobalParams(global_params);

  const auto& runtime = global_params.to_output_params_.open_drt_params_;
  EXPECT_NEAR(runtime.tn_con_, 1.15f, 1e-6f);
  EXPECT_NEAR(runtime.tn_toe_, 0.04f, 1e-6f);
  EXPECT_NEAR(runtime.ts_x0_, 0.18f, 1e-6f);
  EXPECT_NEAR(runtime.ts_dsc_, 1.0f, 1e-6f);
}

}  // namespace puerhlab
