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

#include "sleeve/sleeve_element/sleeve_element.hpp"

#include <chrono>

#include "utils/clock/time_provider.hpp"

namespace puerhlab {

SleeveElement::SleeveElement(sl_element_id_t id, file_name_t element_name)
    : element_id_(id), element_name_(element_name), ref_count_(0), pinned_(false) {
  this->SetAddTime();
}

SleeveElement::~SleeveElement() {}

auto SleeveElement::Copy(sl_element_id_t new_id) const -> std::shared_ptr<SleeveElement> {
  // TODO: Remove placeholder
  return nullptr;
}

auto SleeveElement::Clear() -> bool {
  // Placeholder
  return true;
}

void SleeveElement::SetAddTime() {
  added_time_         = std::chrono::system_clock::to_time_t(TimeProvider::Now());
  last_modified_time_ = added_time_;
}

void SleeveElement::SetLastModifiedTime() {
  last_modified_time_ = std::chrono::system_clock::to_time_t(TimeProvider::Now());
}

void SleeveElement::IncrementRefCount() { ++ref_count_; }

void SleeveElement::DecrementRefCount() {
  --ref_count_;
  if (ref_count_ <= 0) {
    sync_flag_ = SyncFlag::DELETED;
  }
}

auto SleeveElement::IsShared() -> bool { return ref_count_ > 1; }

void SleeveElement::SetSyncFlag(SyncFlag flag) { sync_flag_ = flag; }
};  // namespace puerhlab