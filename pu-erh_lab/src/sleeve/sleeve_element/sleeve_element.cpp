#include "sleeve/sleeve_element/sleeve_element.hpp"

#include <chrono>

#include "utils/clock/time_provider.hpp"

namespace puerhlab {

SleeveElement::SleeveElement(sl_element_id_t id, file_name_t element_name)
    : _element_id(id), _element_name(element_name), _ref_count(0), _pinned(false) {
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
  _added_time         = std::chrono::system_clock::to_time_t(TimeProvider::Now());
  _last_modified_time = _added_time;
}

void SleeveElement::SetLastModifiedTime() {
  _last_modified_time = std::chrono::system_clock::to_time_t(TimeProvider::Now());
}

void SleeveElement::IncrementRefCount() { ++_ref_count; }

void SleeveElement::DecrementRefCount() { --_ref_count; }

auto SleeveElement::IsShared() -> bool { return _ref_count > 1; }
};  // namespace puerhlab