//  Copyright 2025 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#include "storage/service/sleeve/base/base_service.hpp"

#include "storage/mapper/sleeve/base/base_mapper.hpp"
#include "type/type.hpp"

namespace alcedo {
auto BaseService::ToParams(const sleeve_id_t source) -> BaseMapperParams { return {source}; }

auto BaseService::FromParams(BaseMapperParams&& param) -> sleeve_id_t { return param.id; }
};  // namespace alcedo