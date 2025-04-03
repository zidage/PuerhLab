#include "decoders/decoder_manager.hpp"
#include "decoders/image_decoder.hpp"
#include "decoders/thumbnail_decoder.hpp"
#include <cstdint>
#include <future>
#include <memory>
#include <utility>
#include <vector>

namespace puerhlab {
/**
 * @brief Construct a new Image Decoder:: Image Decoder object
 *
 * @param thread_count
 * @param total_request
 */
DecoderManager::DecoderManager(size_t thread_count, uint32_t total_request)
    : _thread_pool(thread_count), _next_request_id(0),
      _decoded_buffer(total_request) {
  _total_request = std::min(MAX_REQUEST_SIZE, total_request);
}

/**
 * @brief
 *
 * @param image_path
 */
void DecoderManager::ScheduleDecode(
    image_path_t image_path,
    std::shared_ptr<std::promise<uint32_t>> decode_promise) {
  // Open file as an ifstream
  std::ifstream file(image_path, std::ios::binary | std::ios::ate);
  if (_next_request_id >= _total_request || !file.is_open()) {
    // FIXME: sanity check
    decode_promise->set_value(_next_request_id);
  }

  // Assign a decoder for the task
  // TODO: Dynamic objects creation
  std::shared_ptr<ImageDecoder> decoder = std::make_shared<ThumbnailDecoder>();

  // Read file into memory
  std::streamsize fileSize = file.tellg();
  file.seekg(0, std::ios::beg);
  std::vector<char> buffer(fileSize);
  if (!file.read(buffer.data(), fileSize)) {
    throw std::runtime_error("Could not read file");
  }
  file.close();

  // Submit a new decode request
  auto &decoded = _decoded_buffer;
  auto request_id = _next_request_id++;

  auto task = std::make_shared<std::packaged_task<void()>>(
      [decoder, buffer = std::move(buffer), image_path, &decoded, request_id,
       decode_promise]() mutable {
        decoder->Decode(std::move(buffer), image_path, decoded, request_id,
                        decode_promise);
      });

  _thread_pool.Submit([task]() { (*task)(); });
}
}; // namespace puerhlab