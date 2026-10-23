//  Copyright 2026 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#include "storage/store/mask_asset/mask_asset_store.hpp"

#include <zlib.h>

#include <cstdint>
#include <exception>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

#include "storage/mapper/duckorm/duckdb_orm.hpp"

namespace alcedo {
namespace {

/// Owns one prepared statement and the result of its last execution.
class PreparedStatement {
 public:
  PreparedStatement(duckdb_connection conn, const char* sql) {
    if (duckdb_prepare(conn, sql, &statement_) != DuckDBSuccess) {
      const char*       error   = duckdb_prepare_error(statement_);
      const std::string message = error != nullptr ? error : "MaskAssetStore prepare failed";
      duckdb_destroy_prepare(&statement_);
      throw std::runtime_error("MaskAssetStore: " + message);
    }
  }
  ~PreparedStatement() {
    if (has_result_) {
      duckdb_destroy_result(&result_);
    }
    duckdb_destroy_prepare(&statement_);
  }
  PreparedStatement(const PreparedStatement&)                                  = delete;
  auto               operator=(const PreparedStatement&) -> PreparedStatement& = delete;

  [[nodiscard]] auto Get() const -> duckdb_prepared_statement { return statement_; }

  void               BindText(idx_t index, std::string_view value) {
    Check(duckdb_bind_varchar_length(statement_, index, value.data(),
                                                   static_cast<idx_t>(value.size())));
  }

  /// Run the statement; the result stays valid until the next Execute or the destructor.
  auto Execute() -> duckdb_result& {
    if (has_result_) {
      duckdb_destroy_result(&result_);
      has_result_ = false;
    }
    const auto state = duckdb_execute_prepared(statement_, &result_);
    has_result_      = true;
    if (state != DuckDBSuccess) {
      const char* error = duckdb_result_error(&result_);
      throw std::runtime_error(std::string{"MaskAssetStore: "} +
                               (error != nullptr ? error : "statement failed"));
    }
    return result_;
  }

  static void Check(duckdb_state state) {
    if (state != DuckDBSuccess) {
      throw std::runtime_error("MaskAssetStore: parameter bind failed");
    }
  }

 private:
  duckdb_prepared_statement statement_  = nullptr;
  duckdb_result             result_     = {};
  bool                      has_result_ = false;
};

[[noreturn]] void FailCorrupt(const MaskAssetKey& key, std::string_view reason) {
  throw std::runtime_error("Mask asset " + std::string{key.Value()} +
                           " is corrupt: " + std::string{reason});
}

auto Compress(std::span<const std::uint8_t> pixels) -> std::vector<std::uint8_t> {
  if (pixels.size() > std::numeric_limits<uLong>::max()) {
    throw std::runtime_error("MaskAssetStore: the raster is too large for zlib");
  }
  const auto                source_size = static_cast<uLong>(pixels.size());
  uLongf                    packed_size = compressBound(source_size);
  std::vector<std::uint8_t> packed(packed_size);
  const int                 status =
      compress2(packed.data(), &packed_size, pixels.data(), source_size, Z_DEFAULT_COMPRESSION);
  if (status != Z_OK) {
    throw std::runtime_error("MaskAssetStore: zlib compression failed with status " +
                             std::to_string(status));
  }
  packed.resize(packed_size);
  return packed;
}

auto ReadText(duckdb_result& result, idx_t column) -> std::string {
  char*       raw  = duckdb_value_varchar(&result, column, 0);
  std::string text = raw != nullptr ? raw : "";
  if (raw != nullptr) {
    duckdb_free(raw);
  }
  return text;
}

}  // namespace

auto MaskAssetStore::Put(const MaskAsset& asset) -> bool {
  try {
    ValidateMaskAsset(asset);
  } catch (const std::exception& error) {
    throw std::runtime_error(std::string{"MaskAssetStore: invalid Mask asset: "} + error.what());
  }
  if (MakeMaskAssetKey(asset.descriptor, asset.pixels) != asset.key) {
    throw std::runtime_error("MaskAssetStore: Mask asset key " + std::string{asset.key.Value()} +
                             " does not match its pixels");
  }
  const auto        packed = Compress(asset.pixels);
  PreparedStatement statement(
      conn_,
      "INSERT INTO MaskAsset (asset_key, format_version, width, height, bounds_x, bounds_y, "
      "bounds_w, bounds_h, pixel_byte_count, codec, data) "
      "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) ON CONFLICT (asset_key) DO NOTHING;");
  const auto& bounds = asset.descriptor.reference_bounds;
  statement.BindText(1, asset.key.Value());
  PreparedStatement::Check(
      duckdb_bind_int32(statement.Get(), 2, static_cast<std::int32_t>(kMaskAssetFormatVersion)));
  PreparedStatement::Check(duckdb_bind_int32(
      statement.Get(), 3, static_cast<std::int32_t>(asset.descriptor.extent.width)));
  PreparedStatement::Check(duckdb_bind_int32(
      statement.Get(), 4, static_cast<std::int32_t>(asset.descriptor.extent.height)));
  PreparedStatement::Check(duckdb_bind_float(statement.Get(), 5, bounds.x));
  PreparedStatement::Check(duckdb_bind_float(statement.Get(), 6, bounds.y));
  PreparedStatement::Check(duckdb_bind_float(statement.Get(), 7, bounds.w));
  PreparedStatement::Check(duckdb_bind_float(statement.Get(), 8, bounds.h));
  PreparedStatement::Check(
      duckdb_bind_int64(statement.Get(), 9, static_cast<std::int64_t>(asset.pixels.size())));
  statement.BindText(10, kMaskAssetZlibCodec);
  PreparedStatement::Check(
      duckdb_bind_blob(statement.Get(), 11, packed.data(), static_cast<idx_t>(packed.size())));
  auto& result = statement.Execute();
  return duckdb_rows_changed(&result) > 0;
}

auto MaskAssetStore::Read(const MaskAssetKey& key) -> MaskAsset {
  PreparedStatement statement(
      conn_,
      "SELECT format_version, width, height, bounds_x, bounds_y, bounds_w, bounds_h, "
      "pixel_byte_count, codec, data FROM MaskAsset WHERE asset_key = ?;");
  statement.BindText(1, key.Value());
  auto& result = statement.Execute();
  if (duckdb_row_count(&result) == 0) {
    throw std::runtime_error("Mask asset " + std::string{key.Value()} +
                             " is missing from the project");
  }
  for (idx_t column = 0; column < 10; ++column) {
    if (duckdb_value_is_null(&result, column, 0)) {
      FailCorrupt(key, "a column is null");
    }
  }
  if (duckdb_value_int32(&result, 0, 0) != static_cast<std::int32_t>(kMaskAssetFormatVersion)) {
    FailCorrupt(key, "unsupported format version");
  }
  if (ReadText(result, 8) != kMaskAssetZlibCodec) {
    FailCorrupt(key, "unsupported codec");
  }
  const auto width      = duckdb_value_int32(&result, 1, 0);
  const auto height     = duckdb_value_int32(&result, 2, 0);
  const auto byte_count = duckdb_value_int64(&result, 7, 0);
  if (width < 1 || height < 1 || width > static_cast<std::int32_t>(kMaximumRasterMaskAxis) ||
      height > static_cast<std::int32_t>(kMaximumRasterMaskAxis) ||
      byte_count != static_cast<std::int64_t>(width) * height) {
    FailCorrupt(key, "dimensions do not match the byte count");
  }
  MaskAsset asset;
  asset.key = key;
  asset.descriptor.extent =
      Extent2D{static_cast<std::uint32_t>(width), static_cast<std::uint32_t>(height)};
  asset.descriptor.reference_bounds =
      NormalizedRect{duckdb_value_float(&result, 3, 0), duckdb_value_float(&result, 4, 0),
                     duckdb_value_float(&result, 5, 0), duckdb_value_float(&result, 6, 0)};
  asset.pixels.resize(static_cast<std::size_t>(byte_count));
  duckdb_blob packed     = duckdb_value_blob(&result, 9, 0);
  uLongf      pixel_size = static_cast<uLongf>(asset.pixels.size());
  const int   status =
      uncompress(asset.pixels.data(), &pixel_size, static_cast<const Bytef*>(packed.data),
                 static_cast<uLong>(packed.size));
  duckdb_free(packed.data);
  if (status != Z_OK) {
    FailCorrupt(key, "zlib data cannot be inflated (status " + std::to_string(status) + ")");
  }
  if (pixel_size != asset.pixels.size()) {
    FailCorrupt(key, "byte count mismatch");
  }
  if (MakeMaskAssetKey(asset.descriptor, asset.pixels) != key) {
    FailCorrupt(key, "key mismatch");
  }
  return asset;
}

auto MaskAssetStore::ListKeys() -> std::vector<MaskAssetKey> {
  PreparedStatement         statement(conn_, "SELECT asset_key FROM MaskAsset ORDER BY asset_key;");
  auto&                     result = statement.Execute();
  const auto                rows   = duckdb_row_count(&result);
  std::vector<MaskAssetKey> keys;
  keys.reserve(static_cast<std::size_t>(rows));
  for (idx_t row = 0; row < rows; ++row) {
    char* raw = duckdb_value_varchar(&result, 0, row);
    keys.emplace_back(std::string{raw != nullptr ? raw : ""});
    if (raw != nullptr) {
      duckdb_free(raw);
    }
  }
  return keys;
}

auto MaskAssetStore::DeleteKeysNotIn(const std::set<MaskAssetKey>& keep) -> std::size_t {
  std::vector<MaskAssetKey> remove;
  for (auto& key : ListKeys()) {
    if (!keep.contains(key)) {
      remove.push_back(std::move(key));
    }
  }
  if (remove.empty()) {
    return 0;
  }
  if (duckorm::begin_transaction(conn_) != DuckDBSuccess) {
    throw std::runtime_error("MaskAssetStore: cannot begin the delete transaction");
  }
  try {
    PreparedStatement statement(conn_, "DELETE FROM MaskAsset WHERE asset_key = ?;");
    for (const auto& key : remove) {
      statement.BindText(1, key.Value());
      (void)statement.Execute();
    }
    if (duckorm::commit_transaction(conn_) != DuckDBSuccess) {
      throw std::runtime_error("MaskAssetStore: cannot commit the delete transaction");
    }
  } catch (...) {
    duckorm::rollback_transaction(conn_);
    throw;
  }
  return remove.size();
}

}  // namespace alcedo
