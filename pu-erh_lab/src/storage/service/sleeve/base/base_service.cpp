#include "storage/service/sleeve/base/base_service.hpp"

#include "storage/mapper/sleeve/base/base_mapper.hpp"
#include "type/type.hpp"

namespace puerhlab {
auto BaseService::ToParams(const sleeve_id_t source) -> BaseMapperParams { return {source}; }

auto BaseService::FromParams(const BaseMapperParams&& param) -> sleeve_id_t { return param.id; }
};  // namespace puerhlab