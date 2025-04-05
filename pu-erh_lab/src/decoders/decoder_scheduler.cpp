#include "decoders/decoder_scheduler.hpp"
#include "decoders/image_decoder.hpp"
#include "decoders/thumbnail_decoder.hpp"
#include "utils/queue/queue.hpp"
#include <cstdint>
#include <exception>
#include <fstream>
#include <future>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

namespace puerhlab {
/**
 * @brief Construct a new Image Decoder:: Image Decoder object
 *
 * @param thread_count
 * @param total_request
 */
DecoderScheduler::DecoderScheduler(size_t thread_count, uint32_t total_request)
    : _thread_pool(thread_count), _next_request_id(0) {
  _total_request = std::min(MAX_REQUEST_SIZE, total_request);
}

/**
 * @brief Send a decoding task to the scheduler
 *
 * @param image_path the path of the file to be decoded
 * @param decode_promise the corresponding promise to be collected
 */
void DecoderScheduler::ScheduleDecode(
    image_path_t image_path, DecodeType decode_type,
    std::shared_ptr<std::promise<uint32_t>> decode_promise) {
  if (_next_request_id >= _total_request) {
    // sanity check for buffer size
    decode_promise->set_exception(std::make_exception_ptr(
        std::runtime_error("Buffer capacity exceeded.")));
    return;
  }
  // Open file as an ifstream
  std::ifstream file(image_path, std::ios::binary | std::ios::ate);

  if (!file.is_open()) {
    // Check file status
    decode_promise->set_exception(std::make_exception_ptr(
        std::runtime_error("File not exists or no read permission.")));
    return;
  }

  // Assign a decoder for the task
  // TODO: Dynamic objects creation
  std::shared_ptr<ImageDecoder> decoder = std::make_shared<ThumbnailDecoder>();

  // Read file into memory
  std::streamsize fileSize = file.tellg();
  file.seekg(0, std::ios::beg);
  std::vector<char> buffer(fileSize);
  if (!file.read(buffer.data(), fileSize)) {
    decode_promise->set_exception(std::make_exception_ptr(
        std::runtime_error("File not exists or no read permission.")));
    return;
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