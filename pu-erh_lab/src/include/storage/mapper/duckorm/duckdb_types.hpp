#pragma once

#include <cstddef>
#include <cstdint>
#include <variant>

namespace duckorm {
enum class DuckDBType : uint8_t {
  INT32,
  INT64,
  UINT32,
  UINT64,
  DOUBLE,
  VARCHAR,
  JSON,
  BOOLEAN,
  TIMESTAMP,
};

struct DuckFieldDesc {
  const char* name;
  DuckDBType  type;
  size_t      offset;
};

#define FIELD(type, field, field_type) \
  duckorm::DuckFieldDesc { #field, duckorm::DuckDBType::field_type, offsetof(type, field) }

using VarTypes = std::variant<int32_t, int64_t, uint32_t, uint64_t, double, const char*>;
};  // namespace duckorm