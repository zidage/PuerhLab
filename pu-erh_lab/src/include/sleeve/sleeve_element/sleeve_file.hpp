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

#pragma once

#include <cstdint>
#include <memory>

#include "edit/history/edit_history.hpp"
#include "edit/history/version.hpp"
#include "image/image.hpp"
#include "sleeve_element.hpp"
#include "type/type.hpp"

namespace puerhlab {

/**
 * @brief A type of element, it contains an image file, its edit history, and other metadata used in
 * this software
 *
 */
class SleeveFile : public SleeveElement {
 private:
  std::shared_ptr<Image>       image_;

  std::shared_ptr<EditHistory> edit_history_;
  std::shared_ptr<Version>     current_version_;

 public:
  image_id_t image_id_;
  explicit SleeveFile(sl_element_id_t id, file_name_t element_name);
  explicit SleeveFile(sl_element_id_t id, file_name_t element_name, std::shared_ptr<Image> image);

  auto Clear() -> bool;

  auto Copy(sl_element_id_t new_id) const -> std::shared_ptr<SleeveElement>;
  auto GetImage() -> std::shared_ptr<Image>;
  void SetImage(const std::shared_ptr<Image> img);

  auto GetEditHistory() -> std::shared_ptr<EditHistory>;
  auto SetEditHistory(const std::shared_ptr<EditHistory> history) -> void;
  ~SleeveFile();
};
};  // namespace puerhlab