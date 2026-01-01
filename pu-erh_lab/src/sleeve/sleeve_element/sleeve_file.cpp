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

#include "sleeve/sleeve_element/sleeve_file.hpp"

#include <cstdint>
#include <memory>

#include "sleeve/sleeve_element/sleeve_element.hpp"

namespace puerhlab {
SleeveFile::~SleeveFile() {}
SleeveFile::SleeveFile(sl_element_id_t id, file_name_t element_name)
    : SleeveElement(id, element_name) {
  type_ = ElementType::FILE;
}
SleeveFile::SleeveFile(sl_element_id_t id, file_name_t element_name, std::shared_ptr<Image> image)
    : SleeveElement(id, element_name) {
  image_ = image;
  type_  = ElementType::FILE;
}

auto SleeveFile::Clear() -> bool {
  // FIXME: Add implementation
  return true;
}

auto SleeveFile::Copy(uint32_t new_id) const -> std::shared_ptr<SleeveElement> {
  std::shared_ptr<SleeveFile> new_file = std::make_shared<SleeveFile>(new_id, element_name_);
  new_file->edit_history_              = edit_history_;
  // TODO: Update the current_version pointer once finish implementing edit history module
  new_file->current_version_           = nullptr;
  // The image object is still reused
  new_file->image_                     = image_;
  return new_file;
}

auto SleeveFile::GetImage() -> std::shared_ptr<Image> { return image_; }
void SleeveFile::SetImage(const std::shared_ptr<Image> img) {
  image_        = img;
  image_id_     = img->image_id_;
  // Once a new image is set, the edit history will be replaced with a new one
  edit_history_ = std::make_shared<EditHistory>(this->element_id_);
}

auto SleeveFile::GetEditHistory() -> std::shared_ptr<EditHistory> { return edit_history_; }

auto SleeveFile::SetEditHistory(const std::shared_ptr<EditHistory> history) -> void {
  edit_history_ = history;
}
};  // namespace puerhlab
