#include "decoders/thumbnail_decoder.hpp"
#include <future>

namespace puerhlab {
void ThumbnailDecoder::Decode(std::vector<char> buffer, file_path_t file_path,
                              std::vector<std::optional<Image>> &result,
                              uint32_t id, std::promise<uint32_t> promise) {
  // Open the datastream as a cv::Mat image
  cv::Mat image_data(buffer.size(), 1, CV_8UC1, buffer.data());
  // Using IMREAD_REDUCED_COLOR_8 flag to get the low-res thumbnail image
  cv::Mat thumbnail = cv::imdecode(image_data, cv::IMREAD_REDUCED_COLOR_8);
  try {
    auto exiv2_img = Exiv2::ImageFactory::open(
        (const Exiv2::byte *)buffer.data(), buffer.size());
    Exiv2::ExifData &exifData = exiv2_img->exifData();
    result.at(id) =
        Image(file_path, ImageType::DEFAULT, Exiv2::ExifData(exifData));
    result.at(id)->LoadThumbnail(std::move(thumbnail));
		promise.set_value(id);
  } catch (std::exception &e) {
    // TODO: Append error message to log
  }
}
}; // namespace puerhlab