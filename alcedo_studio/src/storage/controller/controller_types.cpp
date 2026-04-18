//  Copyright 2025 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#include "storage/controller/controller_types.hpp"

namespace alcedo {
ConnectionGuard::ConnectionGuard(duckdb_connection conn) : conn_(conn) {}

ConnectionGuard::ConnectionGuard(ConnectionGuard&& other) noexcept : conn_(other.conn_) {
  other.conn_ = nullptr;
}

ConnectionGuard::~ConnectionGuard() { duckdb_disconnect(&conn_); }
};  // namespace alcedo