#include "sleeve/sleeve_filter/filters/datetime_filter.hpp"

#include <json.hpp>

namespace puerhlab {
using json = nlohmann::json;
auto DatetimeFilter::ToJSON() -> std::wstring {
  json o{{"start_time", _start_time}, {"end_time", _end_time}};
  return o;
}
};  // namespace puerhlab