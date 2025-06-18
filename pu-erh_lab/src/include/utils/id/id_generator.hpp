#pragma once

#include <cstdint>
namespace puerhlab {
namespace IncrID {
static uint32_t GenerateID() {
  static uint32_t counter = 1;
  return counter++;
}
};  // namespace IncrID
};  // namespace puerhlab