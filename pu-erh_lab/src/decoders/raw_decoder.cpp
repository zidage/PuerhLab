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

#include <easy/profiler.h>
#include <libraw/libraw_const.h>
#include <opencv2/core/hal/interface.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/matx.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/opencv.hpp>
#include <stdexcept>

#include "decoders/processor/raw_processor.hpp"
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
  EASY_BLOCK("LibRaw Unpacking");
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
  raw_processor.imgdata.params.no_auto_bright = 0;  // Disable auto brightness
  raw_processor.imgdata.params.use_camera_wb  = 1;

  raw_processor.imgdata.rawparams.use_dngsdk  = 1;
  raw_processor.unpack();
  EASY_END_BLOCK;

  OpenCVRawProcessor processor{
      {false, false, {}, true, true, 0}, raw_processor.imgdata.rawdata, raw_processor};

  auto processed = processor.Process();

  raw_processor.recycle();
  source_img->LoadData(std::move(processed));
}

void RawDecoder::Decode(std::vector<char> buffer, std::shared_ptr<Image> source_img,
                        std::shared_ptr<BufferQueue>              result,
                        std::shared_ptr<std::promise<image_id_t>> promise) {
  throw std::runtime_error("RawDecoder: Not implemented");
}

};  // namespace puerhlab
