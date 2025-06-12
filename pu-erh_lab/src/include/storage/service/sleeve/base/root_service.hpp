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