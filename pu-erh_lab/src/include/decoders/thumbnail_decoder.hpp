#include "decoders/image_decoder.hpp"

namespace puerhlab {
class ThumbnailDecoder : public ImageDecoder {
public:
  ThumbnailDecoder() = default;

  void Decode(std::vector<char> buffer, file_path_t file_path,
              std::vector<std::optional<Image>> &result, uint32_t id,
              std::promise<uint32_t> promise);
};
}; // namespace puerhlab