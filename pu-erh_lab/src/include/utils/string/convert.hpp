#pragma once

#include <utf8.h>

namespace conv {
auto ToBytes(const std::wstring& wstr) -> std::string;
auto ToBytes(std::wstring&& wstr) -> std::string;

auto FromBytes(const std::string& str) -> std::wstring;
auto FromBytes(std::string&& str) -> std::wstring;
};  // namespace conv
