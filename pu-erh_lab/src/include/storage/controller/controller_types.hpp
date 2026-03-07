//  Copyright 2025 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#pragma once
#include <duckdb.h>

namespace puerhlab {
class ConnectionGuard {
 public:
  duckdb_connection conn_;

  ConnectionGuard(duckdb_connection conn);
  ConnectionGuard(ConnectionGuard&& other) noexcept;
  ~ConnectionGuard();
};
}  // namespace puerhlab