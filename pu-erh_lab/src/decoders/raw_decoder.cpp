#include "decoders/raw_decoder.hpp"
#include "type/type.hpp"
#include <memory>

namespace puerhlab {
/**
 * @brief
 *
 * @param file
 * @param file_path
 * @param id
 */
void RawDecoder::Decode(
    std::vector<char> buffer, file_path_t file_path,
    std::shared_ptr<NonBlockingQueue<std::shared_ptr<Image>>> &result,
    image_id_t id, std::shared_ptr<std::promise<image_id_t>> promise) {
  // TODO: Add Implementation
}
}; // namespace puerhlab
