#include "decoders/thumbnail_decoder.hpp"
#include "image/image.hpp"
#include <future>
#include <memory>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>

namespace puerhlab {
void ThumbnailDecoder::Decode(
    std::vector<char> buffer, file_path_t file_path,
    std::shared_ptr<NonBlockingQueue<std::shared_ptr<Image>>> &result,
    image_id_t id, std::shared_ptr<std::promise<image_id_t>> promise) {
  // Open the datastream as a cv::Mat image
  cv::Mat image_data((int)buffer.size(), 1, CV_8UC1, buffer.data());
  // Using IMREAD_REDUCED_COLOR_8 flag to get the low-res thumbnail image
  cv::Mat thumbnail = cv::imdecode(image_data, cv::IMREAD_COLOR_RGB);
  cv::Mat resizedImage;
  cv::resize(thumbnail, resizedImage, cv::Size(), 0.1, 0.1, cv::INTER_NEAREST);

  try {
    auto exiv2_img = Exiv2::ImageFactory::open(
        (const Exiv2::byte *)buffer.data(), buffer.size());
    Exiv2::ExifData &exifData = exiv2_img->exifData();

    // Push the decoded image into the buffer queue
    std::shared_ptr<Image> img = std::make_shared<Image>(
        id, file_path, ImageType::DEFAULT, Exiv2::ExifData(exifData));
    img->LoadThumbnail(std::move(resizedImage));
    result->push(img);
    promise->set_value(id);
  } catch (std::exception &e) {
    // TODO: Append error message to log
  }
}
}; // namespace puerhlab