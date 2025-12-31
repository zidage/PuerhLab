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