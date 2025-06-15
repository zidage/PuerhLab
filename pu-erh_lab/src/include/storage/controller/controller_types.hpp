#pragma once
#include <duckdb.h>

namespace puerhlab {
class ConnectionGuard {
 public:
  duckdb_connection _conn;

  ConnectionGuard(duckdb_connection conn);
  ConnectionGuard(ConnectionGuard&& other) noexcept;
  ~ConnectionGuard();
};
}  // namespace puerhlab