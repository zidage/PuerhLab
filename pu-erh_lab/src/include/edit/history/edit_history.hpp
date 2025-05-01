/*
 * @file        pu-erh_lab/src/include/edit/history/edit_history.hpp
 * @brief       Data structure used to track all the edit history
 * @author      Yurun Zi
 * @date        2025-03-23
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
#include <list>
#include <unordered_map>

#include "type/type.hpp"
#include "version.hpp"

namespace puerhlab {
class VersionNode {
 private:
  Version               &_ver_ref;
  hash_t                 _commit_id;
  std::list<VersionNode> _branch;

  uint32_t               _ref_count;

 public:
  VersionNode(Version &ver_ref);
};

class EditHistory {
 private:
  hash_t                              _history_id;
  image_id_t                          _bound_image;

  std::time_t                         _added_time;
  std::time_t                         _last_modified_time;

  std::list<VersionNode>              _commit_tree;
  std::unordered_map<hash_t, Version> _version_storage;

 public:
  EditHistory(image_id_t _bound_image);
};
};  // namespace puerhlab