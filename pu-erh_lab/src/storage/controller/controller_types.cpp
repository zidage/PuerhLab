#include "storage/controller/controller_types.hpp"

namespace puerhlab {
ConnectionGuard::ConnectionGuard(duckdb_connection conn) : _conn(conn) {}

ConnectionGuard::~ConnectionGuard() { duckdb_disconnect(&_conn); }
};  // namespace puerhlab