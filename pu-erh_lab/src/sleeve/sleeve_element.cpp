#include "sleeve/sleeve_element.hpp"

#include <chrono>

namespace puerhlab {

SleeveElement::SleeveElement(sl_element_id_t id) : _element_id(id) { this->SetAddTime(); }

void SleeveElement::SetAddTime() {
  auto now            = std::chrono::system_clock::now();
  _added_time         = std::chrono::system_clock::to_time_t(now);
  _last_modified_time = _added_time;
}

void SleeveElement::SetLastModifiedTime() {
  auto now            = std::chrono::system_clock::now();
  _last_modified_time = std::chrono::system_clock::to_time_t(now);
}
};  // namespace puerhlab