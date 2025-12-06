#include <iterator>
#include <string>

#include "utf8/checked.h"
#include "utils/string/convert.hpp"

namespace conv {
auto ToBytes(const std::wstring& wstr) -> std::string {
  std::string str;
  utf8::utf16to8(wstr.begin(), wstr.end(), std::back_inserter(str));
  return str;
}

auto ToBytes(std::wstring&& wstr) -> std::string {
  std::string str;
  utf8::utf16to8(wstr.begin(), wstr.end(), std::back_inserter(str));
  return str;
}

auto FromBytes(const std::string& str) -> std::wstring {
  std::wstring wstr;
  utf8::utf8to16(str.begin(), str.end(), std::back_inserter(wstr));
  return wstr;
}

auto FromBytes(std::string&& str) -> std::wstring {
  std::wstring wstr;
  utf8::utf8to16(str.begin(), str.end(), std::back_inserter(wstr));
  return wstr;
}
};  // namespace conv