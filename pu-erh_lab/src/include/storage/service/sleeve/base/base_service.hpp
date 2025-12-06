#include "sleeve/sleeve_base.hpp"
#include "storage/mapper/sleeve/base/base_mapper.hpp"
#include "storage/service/service_interface.hpp"
#include "type/type.hpp"

namespace puerhlab {
class BaseService
    : ServiceInterface<BaseService, sleeve_id_t, BaseMapperParams, BaseMapper, sleeve_id_t> {
 public:
  using ServiceInterface::ServiceInterface;
  static auto ToParams(const sleeve_id_t source) -> BaseMapperParams;
  static auto FromParams(BaseMapperParams&& param) -> sleeve_id_t;
};
};  // namespace puerhlab