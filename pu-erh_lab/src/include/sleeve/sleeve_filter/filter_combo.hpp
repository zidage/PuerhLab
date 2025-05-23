/*
 * @file        pu-erh_lab/src/include/mapper/sleeve/sleeve_filter/filter_combo.hpp
 * @brief       A combination of a set of filters
 * @author      Yurun Zi
 * @date        2025-03-26
 * @license     MIT
 *
 * @copyright   Copyright (c) 2025 Yurun Zi
 */

// Copyright (c) 2025 Yurun Zi
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#pragma once

#include <cstdint>
#include <memory>
#include <set>
#include <unordered_map>
#include <vector>

#include "filters/sleeve_filter.hpp"
#include "type/type.hpp"

namespace puerhlab {
class FilterCombo {
 private:
  std::vector<SleeveFilter> _filters;

 public:
  auto GetFilters() -> std::vector<SleeveFilter> &;
  auto CreateIndexOn(std::shared_ptr<std::set<sl_element_id_t>> _lists) -> std::shared_ptr<std::set<sl_element_id_t>>;
};
};  // namespace puerhlab