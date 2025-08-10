#include <easy/profiler.h>

#include <string>

#include "decoders/raw_decoder.hpp"
#include "edit/operators/detail/clarity_op.hpp"
#include "edit/operators/op_base.hpp"
#include "edit/operators/operator_factory.hpp"
#include "edit/pipeline/pipeline_cpu.hpp"
#include "pipeline_test_fixation.hpp"
#include "sleeve/sleeve_manager.hpp"
#include "utils/string/convert.hpp"


using namespace puerhlab;
TEST_F(PipelineTests, SimpleTest1) {
  {
    SleeveManager manager{db_path_};
    ImageLoader   image_loader(128, 8, 0);
    image_path_t  path =
        L"D:\\Projects\\pu-erh_lab\\pu-erh_lab\\tests\\resources\\sample_images\\raw\\street";
    std::vector<image_path_t> imgs;
    for (const auto& img : std::filesystem::directory_iterator(path)) {
      if (!img.is_directory()) imgs.push_back(img.path());
    }

    manager.LoadToPath(imgs, L"");

    // Read image data
    ThreadPool     thread_pool{8};
    auto           img_pool = manager.GetPool()->GetPool();

    nlohmann::json to_ws_params;
    to_ws_params["ocio"] = {{"src", ""}, {"dst", "ACEScct"}};

    nlohmann::json basic_params;
    basic_params["exposure"]   = 0.2f;
    basic_params["highlights"] = -85.0f;
    basic_params["shadows"]    = 55.0f;
    basic_params["white"]      = -25.0f;
    basic_params["black"]      = 40.0f;

    nlohmann::json color_wheel_params;
    color_wheel_params["color_wheel"] = {{"lift",
                                          {{"color_offset.x", 0.0},
                                           {"color_offset.y", 0.0},
                                           {"color_offset.z", 0.0},
                                           {"luminance_offset", 0.0}}},
                                         {"gamma",
                                          {{"color_offset.x", 1.0},
                                           {"color_offset.y", 1.0},
                                           {"color_offset.z", 1.00},
                                           {"luminance_offset", 0.0}}},
                                         {"gain",
                                          {{"color_offset.x", 1.0},
                                           {"color_offset.y", 1.0},
                                           {"color_offset.z", 1.0},
                                           {"luminance_offset", 0.0}}},
                                         {"crossovers", {{"lift", 0.2}, {"gain", 0.8}}}};

    nlohmann::json contrast_params;
    contrast_params["contrast"] = 5.0f;
    nlohmann::json lmt_params;
    lmt_params["ocio_lmt"] =
        "D:\\Projects\\pu-erh_lab\\pu-erh_lab\\src\\config\\LUTs\\ACES CCT 2383 D65.cube";

    nlohmann::json pre_output_params;
    pre_output_params["ocio"] = {{"src", "ACEScct"}, {"dst", ""}};

    nlohmann::json output_params;
    output_params["ocio"] = {{"src", ""}, {"dst", "Camera Rec.709"}};

    for (auto& pair : img_pool) {
      auto task = [pair, img_pool, to_ws_params, color_wheel_params, basic_params, contrast_params,
                   lmt_params, pre_output_params, output_params]() mutable {
        CPUPipeline pipeline{};
        auto&       to_ws = pipeline.GetStage(PipelineStageName::To_WorkingSpace);
        to_ws.SetOperator(OperatorType::CST, to_ws_params);

        auto& adj = pipeline.GetStage(PipelineStageName::Basic_Adjustment);
        adj.SetOperator(OperatorType::BLACK, basic_params);
        adj.SetOperator(OperatorType::WHITE, basic_params);

        auto& lmt = pipeline.GetStage(PipelineStageName::Color_Adjustment);
        // lmt.SetOperator(OperatorType::ACES_TONE_MAPPING, output_params);
        lmt.SetOperator(OperatorType::LMT, lmt_params);
        lmt.SetOperator(OperatorType::CST, pre_output_params);
        //

        auto& output_stage = pipeline.GetStage(PipelineStageName::Output_Transform);
        output_stage.SetOperator(OperatorType::CST, output_params);

        auto            img = pair.second;
        RawDecoder      decoder;
        std::ifstream   file(img->_image_path, std::ios::binary | std::ios::ate);
        std::streamsize fileSize = file.tellg();
        file.seekg(0, std::ios::beg);
        std::vector<char> buffer(fileSize);
        if (!file.read(buffer.data(), fileSize)) {
          FAIL();
        }
        file.close();
        EASY_BLOCK("Raw Decoding");
        decoder.Decode(std::move(buffer), img);
        EASY_END_BLOCK;

        ImageBuffer source{img->GetImageData()};
        EASY_BLOCK("Pipeline Processing");
        auto output = pipeline.Apply(source);
        EASY_END_BLOCK;

        EASY_BLOCK("Image Saving");
        cv::Mat to_save_rec709;
        output.GetCPUData().convertTo(to_save_rec709, CV_16UC3, 65535.0f);
        cv::cvtColor(to_save_rec709, to_save_rec709, cv::COLOR_RGB2BGR);
        cv::imwrite(std::format("D:\\Projects\\pu-erh_lab\\pu-erh_lab\\tests\\resources\\sample_"
                                "images\\my_pipeline\\batch_results\\{}.tiff",
                                conv::ToBytes(img->_image_name)),
                    to_save_rec709);
        img->ClearData();
        to_save_rec709.release();
        output.ReleaseCPUData();
        EASY_END_BLOCK;
      };
      thread_pool.Submit(task);
    }
  }
}