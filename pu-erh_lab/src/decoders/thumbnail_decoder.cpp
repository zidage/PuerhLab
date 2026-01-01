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

#include "decoders/thumbnail_decoder.hpp"

#include <opencv2/core/hal/interface.h>

#include <future>
#include <memory>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#include <utility>

#include "image/image.hpp"
#include "image/image_buffer.hpp"

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
    img->exif_data_ = Exiv2::ImageFactory::open((Exiv2::byte*)buffer.data(), buffer.size());
    img->exif_data_->readMetadata();
    img->has_exif_ = !img->exif_data_->exifData().empty();

    img->LoadThumbnail({std::move(thumbnail)});
    result->push(img);
    promise->set_value(id);
  } catch (std::exception& e) {
    // TODO: Append error message to log
  }
}

void ThumbnailDecoder::Decode(std::vector<char> buffer, std::shared_ptr<Image> source_img,
                              std::shared_ptr<BufferQueue>              result,
                              std::shared_ptr<std::promise<image_id_t>> promise) {
  // Open the datastream as a cv::Mat image
  cv::Mat image_data((int)buffer.size(), 1, CV_32FC1, buffer.data());
  // Using IMREAD_REDUCED_COLOR_8 flag to get the low-res thumbnail image
  cv::Mat thumbnail = cv::imdecode(image_data, cv::IMREAD_COLOR);
  thumbnail.convertTo(thumbnail, CV_32FC3, 1.0 / 255.0);
  ImageBuffer thumbnail_data{std::move(thumbnail)};
  source_img->LoadThumbnail(std::move(thumbnail_data));
  result->push(source_img);
  promise->set_value(source_img->image_id_);
}
};  // namespace puerhlab