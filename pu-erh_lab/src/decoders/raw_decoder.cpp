#include "decoders/raw_decoder.hpp"

namespace puerhlab {
/**
 * @brief
 *
 * @param file
 * @param file_path
 * @param id
 */
void RawDecoder::Decode(std::vector<char> buffer, file_path_t file_path,
                        NonBlockingQueue<std::optional<Image>> &result,
                        uint32_t id,
                        std::shared_ptr<std::promise<uint32_t>> promise) {
  // TODO: Add Implementation
}
}; // namespace puerhlab
