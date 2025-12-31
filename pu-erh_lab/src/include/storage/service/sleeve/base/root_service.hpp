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

#pragma once
#include <memory>

#include "sleeve/sleeve_base.hpp"
#include "sleeve/sleeve_element/sleeve_folder.hpp"
#include "storage/mapper/sleeve/base/base_mapper.hpp"
#include "storage/service/service_interface.hpp"
#include "type/type.hpp"

namespace puerhlab {
class RootService : ServiceInterface<RootService, sl_element_id_t, RootMapperParams, RootMapper,
                                     sl_element_id_t> {
 public:
  using ServiceInterface::ServiceInterface;
  static auto ToParams(const sl_element_id_t source) -> RootMapperParams;
  static auto FromParams(const RootMapperParams&& param) -> sl_element_id_t;
};
};  // namespace puerhlab