#pragma once

#include <utf8.h>

namespace conv {
auto ToBytes(std::wstring wstr) -> std::string;

auto FromBytes(std::string str) -> std::wstring;
};  // namespace conv
