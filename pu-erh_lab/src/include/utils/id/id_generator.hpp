#pragma once

#include <concepts>
#include <cstdint>
namespace puerhlab {
namespace IncrID {
static uint32_t GenerateID() {
  static uint32_t counter = 1;
  return counter++;
}

template <typename IDType>
concept Incrementable = requires(IDType t) {
  { ++t } -> std::same_as<IDType&>;
};
template <Incrementable T>
class IDGenerator {
 private:
  T _counter;

 public:
  IDGenerator(T start_id) : _counter(start_id) {}
  auto GenerateID() -> T { return ++_counter; }
};
};  // namespace IncrID
};  // namespace puerhlab