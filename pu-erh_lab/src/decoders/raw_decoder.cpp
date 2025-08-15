/*
 * @file        pu-erh_lab/src/include/decoders/raw_decoder.hpp
 * @brief       A decoder used to decode raw files, e.g. .ARW
 * @author      Yurun Zi
 * @date        2025-03-19
 * @license     MIT
 *
 * @copyright   Copyright (c) 2025 Yurun Zi
 */

// Copyright (c) 2025 Yurun Zi
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "decoders/raw_decoder.hpp"

#include <libraw/libraw_const.h>
#include <opencv2/core/hal/interface.h>

#include <cstdint>
#include <memory>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/matx.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/opencv.hpp>
#include <stdexcept>

#include "edit/operators/tone_mapping/ACES_tone_mapping_op.hpp"
#include "image/image.hpp"
#include "image/image_buffer.hpp"
#include "type/type.hpp"

namespace puerhlab {
/**
 * @brief A callback used to decode a raw file
 *
 * @param file
 * @param file_path
 * @param id
 */
void RawDecoder::Decode(std::vector<char> buffer, std::filesystem::path file_path,
                        std::shared_ptr<BufferQueue> result, image_id_t id,
                        std::shared_ptr<std::promise<image_id_t>> promise) {
  // TODO: Add Implementation
}

void RawDecoder::Decode(std::vector<char>&& buffer, std::shared_ptr<Image> source_img) {
  // TODO: Add Implementation
  LibRaw raw_processor;
  int    ret = raw_processor.open_buffer((void*)buffer.data(), buffer.size());
  if (ret != LIBRAW_SUCCESS) {
    throw std::runtime_error("RawDecoder: Unable to read raw file using LibRAW");
  }
  // Default set output color space to ACES2065-1 (AP0)
  raw_processor.imgdata.params.output_color   = 6;
  raw_processor.imgdata.params.output_bps     = 16;
  raw_processor.imgdata.params.gamm[0]        = 1.0;  // Linear gamma
  raw_processor.imgdata.params.gamm[1]        = 1.0;
  // raw_processor.imgdata.params.user_sat      = 11300;
  // raw_processor.imgdata.params.adjust_maximum_thr = 0.0f;
  raw_processor.imgdata.params.no_auto_bright = 0;  // Disable auto brightness
  raw_processor.imgdata.params.use_camera_wb  = 1;
  raw_processor.imgdata.params.highlight      = 8;
  // raw_processor.imgdata.params.half_size      = 1;

  // raw_processor.imgdata.rawparams.use_rawspeed    = 1;
  raw_processor.imgdata.rawparams.use_dngsdk  = 1;
  raw_processor.unpack();

  auto    img_unpacked = raw_processor.imgdata.rawdata.raw_image;
  auto    img_sizes    = raw_processor.imgdata.sizes;
  auto    color_coeffs = raw_processor.imgdata.rawdata.color.cam_xyz;
  auto    rgb_cam      = raw_processor.imgdata.rawdata.color.rgb_cam;

  // R G1 B G2
  auto    wbct_coeff   = raw_processor.imgdata.rawdata.color.WBCT_Coeffs;
  // auto d65_white_balance = wb_coeff[21];
  auto    wb_coeff     = raw_processor.imgdata.rawdata.color.cam_mul;
  auto    black        = raw_processor.imgdata.rawdata.color.black;
  auto    maximum      = raw_processor.imgdata.rawdata.color.maximum;
  auto    white        = raw_processor.imgdata.rawdata.color.white;

  float   max_gain     = std::max({wb_coeff[1], wb_coeff[2], wb_coeff[3], wb_coeff[4]});

  cv::Mat pre_debayer_mat(img_sizes.raw_height, img_sizes.raw_width, CV_16UC1, img_unpacked);
  pre_debayer_mat -= black;

  // white balance
  pre_debayer_mat.forEach<uint16_t>([&](uint16_t& pixel, const int* pos /* position */) {
    // Black level subtraction
    if (pixel > maximum) pixel = maximum;
    pixel = pixel * static_cast<uint16_t>(65535.0f / maximum);

    // Convert to 16-bit unsigned integer
    // if (pos[0] % 2 == 0 && pos[1] % 2 == 0) {
    //   pixel = static_cast<uint16_t>((float)pixel * wb_coeff[1] / max_gain);
    // } else if (pos[0] % 2 == 0 && pos[1] % 2 == 1) {
    //   pixel = static_cast<uint16_t>((float)pixel * wb_coeff[2] / max_gain);
    // } else if (pos[0] % 2 == 1 && pos[1] % 2 == 0) {
    //   pixel = static_cast<uint16_t>((float)pixel * wb_coeff[4] / max_gain);
    // } else {
    //   pixel = static_cast<uint16_t>((float)pixel * wb_coeff[3] / max_gain);
    // }

    // pixel = static_cast<uint16_t>(pixel * 16383.0f / maximum);
  });

  // Apply color space transform and debayer

  ImageBuffer pre_debayer_buffer{pre_debayer_mat};
  pre_debayer_buffer.SyncToGPU();

  cv::cuda::GpuMat debayer_mat;
  // pre_debayer_mat.convertTo(pre_debayer_mat, CV_16UC1);
  cv::cuda::cvtColor(pre_debayer_buffer.GetGPUData(), debayer_mat, cv::COLOR_BayerBG2RGB);

  static cv::Matx33f M_D65_to_D60(1.013033f, 0.006107f, -0.014948f, 0.007704f, 0.998150f,
                                  -0.005025f, -0.002836f, 0.004675f, 0.924644f);

  static cv::Matx33f xyz_ap0(1.0498110175f, 0.0000000000f, -0.0000974845f, -0.4959030231f,
                             1.3733130458f, 0.0982400361f, 0.0000000000f, 0.0000000000f,
                             0.9912520182f);

  static cv::Matx33f xyz_rgb(0.4124564f, 0.3575761f, 0.1804375f, 0.2126729f, 0.7151522f, 0.0721750f,
                             0.0193339f, 0.1191920f, 0.9503041f);

  cv::Matx33f rgb_cam_mat(rgb_cam[0][0], rgb_cam[0][1], rgb_cam[0][2], rgb_cam[1][0], rgb_cam[1][1],
                          rgb_cam[1][2], rgb_cam[2][0], rgb_cam[2][1], rgb_cam[2][2]);

  cv::Matx33f cam_xyz({color_coeffs[0][0], color_coeffs[0][1], color_coeffs[0][2],
                       color_coeffs[1][0], color_coeffs[1][1], color_coeffs[1][2],
                       color_coeffs[2][0], color_coeffs[2][1], color_coeffs[2][2]});
  cam_xyz             = cam_xyz.inv();

  cv::Matx33f to_aces = xyz_ap0 * M_D65_to_D60 * cam_xyz;
  cv::Mat     processed;
  debayer_mat.convertTo(debayer_mat, CV_32FC3, 1.0f / 65535.0f);
  debayer_mat.download(processed);

  cv::transform(processed, processed, to_aces);

  // cv::cuda::transform(debayer_mat, debayer_mat, cv::Mat(to_aces));

  // cv::Mat resized;
  // cv::resize(debayer_mat, resized, cv::Size(512, 512));
  // cv::cvtColor(resized, resized, cv::COLOR_RGB2BGR);
  // cv::imshow("Debayered Image", resized);
  // cv::waitKey(0);

  // raw_processor.dcraw_process();

  // auto img = raw_processor.dcraw_make_mem_image();
  // if (!img || img->type != LIBRAW_IMAGE_BITMAP) {
  //   throw std::runtime_error("RawDecoder: Unable to get processed image using LibRAW");
  // }

  // int width  = img->width;
  // int height = img->height;
  // if (img->colors != 3) {
  //   throw std::runtime_error("RawDecoder: Unsupported image format (cwwloww channel != 3)");
  // }
  // cv::Mat image_16u(height, width, CV_16UC3, img->data);
  // // cv::cvtColor(image_16u, image_16u, cv::COLOR_RGB2BGR);

  // cv::Mat image_32f;
  // image_16u.convertTo(image_32f, CV_32FC3, 1.0f / 65535.0f);

  // LibRaw::dcraw_clear_mem(img);
  raw_processor.recycle();
  source_img->LoadData({std::move(processed)});
}

void RawDecoder::Decode(std::vector<char> buffer, std::shared_ptr<Image> source_img,
                        std::shared_ptr<BufferQueue>              result,
                        std::shared_ptr<std::promise<image_id_t>> promise) {
  throw std::runtime_error("RawDecoder: Not implemented");
}

};  // namespace puerhlab
