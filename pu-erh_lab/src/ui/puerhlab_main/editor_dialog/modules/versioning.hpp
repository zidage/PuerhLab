#pragma once

#include <cstddef>

#include <QString>

namespace puerhlab::ui::versioning {

auto MakeTxCountLabel(size_t tx_count) -> QString;

}  // namespace puerhlab::ui::versioning
