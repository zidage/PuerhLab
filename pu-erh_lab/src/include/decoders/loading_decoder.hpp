#pragma once

#include "decoders/image_decoder.hpp"

namespace puerhlab {

class LoadingDecoder : public ImageDecoder {
 public:
  virtual void Decode(std::vector<char> buffer, std::filesystem::path file_path,
                      std::shared_ptr<BufferQueue> result, image_id_t id,
                      std::shared_ptr<std::promise<image_id_t>> promise) = 0;
};
};  // namespace puerhlab