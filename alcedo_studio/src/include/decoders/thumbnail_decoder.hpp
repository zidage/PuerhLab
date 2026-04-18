//  Copyright 2025 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#pragma once

#include "data_decoder.hpp"
#include "type/type.hpp"

namespace alcedo {
class ThumbnailDecoder : public DataDecoder {
 public:
  ThumbnailDecoder() = default;

  void Decode(std::vector<char> buffer, std::filesystem::path file_path,
              std::shared_ptr<BufferQueue> result, image_id_t id,
              std::shared_ptr<std::promise<image_id_t>> promise);

  void Decode(std::vector<char> buffer, std::shared_ptr<Image> source_img,
              std::shared_ptr<BufferQueue>              result,
              std::shared_ptr<std::promise<image_id_t>> promise);
};
};  // namespace alcedo