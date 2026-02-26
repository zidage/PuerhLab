//  Copyright 2025 Yurun Zi
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.

#include "decoders/raw_decoder.hpp"

#include <libraw/libraw_const.h>
#include <opencv2/core/hal/interface.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/matx.hpp>
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

  const ExifDisplayMetaData* exif_hint = source_img ? &source_img->exif_display_ : nullptr;
  RawProcessor processor{{true, false, true, 0}, raw_processor.imgdata.rawdata, raw_processor,
                         exif_hint};

  auto         processed = processor.Process();

  raw_processor.recycle();
  source_img->LoadOriginalData(std::move(processed));
}

void RawDecoder::Decode(std::vector<char> buffer, std::shared_ptr<Image> source_img,
                        std::shared_ptr<BufferQueue>              result,
                        std::shared_ptr<std::promise<image_id_t>> promise) {
  throw std::runtime_error("RawDecoder: Not implemented");
}

};  // namespace puerhlab
