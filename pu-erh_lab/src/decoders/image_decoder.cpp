/*
 * @file        pu-erh_lab/src/decoders/image_decoder.cpp
 * @brief       A decoder responsible for decoding image files
 * @author      Yurun Zi
 * @date        2025-03-28
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

#include "decoders/image_decoder.hpp"
#include "concurrency/thread_pool.hpp"
#include "image/image.hpp"

#include <algorithm>
#include <cstddef>
#include <exception>
#include <exiv2/exif.hpp>
#include <exiv2/image.hpp>
#include <fstream>
#include <opencv2/imgcodecs.hpp>
#include <vector>


namespace puerhlab {
/**
 * @brief Construct a new Image Decoder:: Image Decoder object
 *
 * @param thread_count
 * @param total_request
 */
ImageDecoder::ImageDecoder(size_t thread_count, uint32_t total_request)
    : _thread_pool(thread_count) {
  _total_request = std::min(MAX_REQUEST_SIZE, total_request);
  _decoded.reserve(_total_request);
}

/**
 * @brief
 *
 * @param image_path
 */
void ImageDecoder::ScheduleDecode(image_path_t image_path) {
  std::ifstream file(image_path);
  _thread_pool.SubmitFile(std::move(file), image_path, _decoded,
                          _next_request_id.fetch_add(1), DecodeImage);
}

/**
 * @brief
 *
 * @param file
 * @param file_path
 * @param id
 */
static void DecodeImage(std::ifstream &&file, file_path_t file_path,
                 std::vector<Image> result, uint32_t id) {
  // Load filestream to memory
  std::vector<char> buffer((std::istreambuf_iterator<char>(file)),
                           std::istreambuf_iterator<char>());
  file.close();

  cv::Mat image_datal(buffer.size(), 1, CV_8UC1, buffer.data());
  cv::Mat thumbnail = cv::imdecode(image_datal, cv::IMREAD_REDUCED_COLOR_8);

  try {
    auto exiv2_img = Exiv2::ImageFactory::open(buffer.data());
    Exiv2::ExifData &exifData = exiv2_img->exifData();

    // FIXME: Potential memory management bug
    result[id] =
        Image(std::move(file_path), ImageType::DEFAULT, std::move(exifData));
    result[id].LoadThumbnail(std::move(thumbnail));
  } catch (std::exception &e) {
    // TODO: Append error message to log
  }
}

}; // namespace puerhlab