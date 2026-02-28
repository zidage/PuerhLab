#include "ui/puerhlab_main/editor_dialog/modules/versioning.hpp"

namespace puerhlab::ui::versioning {

auto MakeTxCountLabel(size_t tx_count) -> QString {
  return QString("Uncommitted: %1 tx").arg(static_cast<qulonglong>(tx_count));
}

}  // namespace puerhlab::ui::versioning
