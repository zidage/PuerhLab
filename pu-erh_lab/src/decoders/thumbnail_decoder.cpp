/*
 * @file        pu-erh_lab/src/decoders/thumbnail_decoder.hpp
 * @brief       A decoder used to decode the thumbnails of regular files, e.g.
 * .jpg
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

#include "decoders/thumbnail_decoder.hpp"

#include <future>
#include <memory>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>

#include "image/image.hpp"

namespace puerhlab {
/**
 * @brief A callback used to decode the thumbnail of a regular file
 *
 * @param buffer
 * @param file_path
 * @param result
 * @param id
 * @param promise
 */
void ThumbnailDecoder::Decode(std::vector<char> buffer, std::filesystem::path file_path,
                              std::shared_ptr<BufferQueue> result, image_id_t id,
                              std::shared_ptr<std::promise<image_id_t>> promise) {
  // Open the datastream as a cv::Mat image
  cv::Mat image_data((int)buffer.size(), 1, CV_8UC1, buffer.data());
  // Using IMREAD_REDUCED_COLOR_8 flag to get the low-res thumbnail image
  cv::Mat thumbnail = cv::imdecode(image_data, cv::IMREAD_REDUCED_COLOR_8);
  try {
    // Push the decoded image into the buffer queue
    std::shared_ptr<Image> img = std::make_shared<Image>(id, file_path, ImageType::DEFAULT);
    img->_exif_data            = Exiv2::ImageFactory::open((Exiv2::byte *)buffer.data(), buffer.size());
    img->_exif_data->readMetadata();
    img->_has_exif = !img->_exif_data->exifData().empty();
    img->LoadThumbnail(std::move(thumbnail));
    result->push(img);
    promise->set_value(id);
  } catch (std::exception &e) {
    // TODO: Append error message to log
  }
}

void ThumbnailDecoder::Decode(std::vector<char> buffer, std::shared_ptr<Image> source_img,
                              std::shared_ptr<BufferQueue> result, std::shared_ptr<std::promise<image_id_t>> promise) {
  // Open the datastream as a cv::Mat image
  cv::Mat image_data((int)buffer.size(), 1, CV_8UC1, buffer.data());
  // Using IMREAD_REDUCED_COLOR_8 flag to get the low-res thumbnail image
  cv::Mat thumbnail = cv::imdecode(image_data, cv::IMREAD_REDUCED_COLOR_8);
  source_img->LoadThumbnail(std::move(thumbnail));
  result->push(source_img);
  promise->set_value(source_img->_image_id);
}
};  // namespace puerhlab