//  Copyright 2025 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#pragma once

#include <utf8.h>

namespace conv {
auto ToBytes(const std::wstring& wstr) -> std::string;
auto ToBytes(std::wstring&& wstr) -> std::string;

auto FromBytes(const std::string& str) -> std::wstring;
auto FromBytes(std::string&& str) -> std::wstring;
};  // namespace conv
