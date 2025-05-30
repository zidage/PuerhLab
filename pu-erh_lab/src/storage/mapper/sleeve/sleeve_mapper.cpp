#include "storage/mapper/sleeve/sleeve_mapper.hpp"

#include <duckdb.h>
#include <easy/profiler.h>

#include <codecvt>
#include <cstdint>
#include <ctime>
#include <exception>
#include <filesystem>
#include <memory>
#include <optional>
#include <queue>
#include <unordered_map>

#include "concurrency/thread_local_resource.hpp"
#include "concurrency/thread_pool.hpp"
#include "sleeve/sleeve_element/sleeve_element.hpp"
#include "sleeve/sleeve_element/sleeve_file.hpp"
#include "sleeve/sleeve_element/sleeve_folder.hpp"
#include "storage/mapper/sleeve/query_prepare.hpp"
#include "storage/mapper/sleeve/statement_prepare.hpp"
#include "type/type.hpp"
#include "utils/queue/queue.hpp"

namespace puerhlab {
struct PreparedImageData {
  image_id_t             id;
  std::string            json;
  std::shared_ptr<Image> img;
};

SleeveMapper::SleeveMapper() {};

SleeveMapper::SleeveMapper(file_path_t db_path) {
  if (std::filesystem::exists(db_path)) {
    _initialized = true;
  }
  ConnectDB(db_path);
}

SleeveMapper::~SleeveMapper() {
  duckdb_disconnect(&_con);
  duckdb_close(&_db);
}

void SleeveMapper::ConnectDB(file_path_t db_path) {
  if (_db_connected) return;
  if (duckdb_open(db_path.string().c_str(), &_db) != DuckDBSuccess) {
    throw std::exception("DB cannot be created");
  }
  if (duckdb_connect(_db, &_con) != DuckDBSuccess) {
    throw std::exception("DB cannot be connected");
  }

  _db_connected = true;
  _prepare_storage.resize(32, Prepare(_con));
}

void SleeveMapper::InitDB() {
  if (!_db_connected) {
    throw std::exception("Must connect to a DB before initialization");
  }
  if (_initialized) {
    return;
  }

  duckdb_result result;
  if (duckdb_query(_con, Queries::init_table_query.c_str(), &result) != DuckDBSuccess) {
    auto error_message = duckdb_result_error(&result);
    duckdb_destroy_result(&result);
    throw std::exception(error_message);
  }
  _initialized        = true;

  duckdb_database &db = _db;
  ThreadLocalResource<duckdb_connection>::SetInitializer([&db] {
    auto ptr = std::make_unique<duckdb_connection>();
    if (duckdb_connect(db, ptr.get()) != DuckDBSuccess) {
      throw std::runtime_error("Failed to connect duckdb in thread");
    }
    return ptr;
  });
}

auto SleeveMapper::GetPrepare(uint8_t op, const std::string &query) -> Prepare & {
  Prepare &pre = _prepare_storage[op];
  if (!pre._prepared) {
    pre.GetStmtGuard(query);
  }
  return pre;
}

void SleeveMapper::CaptureElement(std::unordered_map<uint32_t, std::shared_ptr<SleeveElement>> &storage, Prepare &pre) {
  std::wstring_convert<std::codecvt_utf8<wchar_t>> conv;

  Prepare &folder_pre = GetPrepare(GET_OP(Operate::ADD, Table::Folder), Queries::folder_insert_query);
  Prepare &file_pre   = GetPrepare(GET_OP(Operate::ADD, Table::File), Queries::file_insert_query);

  for (auto &val : storage) {
    auto element = val.second;
    char added_time[32];
    char modified_time[32];
    std::strftime(added_time, sizeof(added_time), "%Y-%m-%d %H:%M:%S", std::gmtime(&element->_added_time));
    std::strftime(modified_time, sizeof(modified_time), "%Y-%m-%d %H:%M:%S",
                  std::gmtime(&element->_last_modified_time));

    duckdb_bind_uint32(pre._stmt, 1, element->_element_id);
    duckdb_bind_uint32(pre._stmt, 2, static_cast<uint32_t>(element->_type));
    duckdb_bind_varchar(pre._stmt, 3, conv.to_bytes(element->_element_name).c_str());
    duckdb_bind_varchar(pre._stmt, 4, added_time);
    duckdb_bind_varchar(pre._stmt, 5, modified_time);
    duckdb_bind_uint32(pre._stmt, 6, element->_ref_count);

    // Execute insertion of an element
    if (duckdb_execute_prepared(pre._stmt, &pre._result) != DuckDBSuccess) {
      auto error_message = duckdb_result_error(&pre._result);
      throw std::exception(error_message);
    }

    duckdb_destroy_result(&pre._result);

    if (element->_type == ElementType::FOLDER) {
      CaptureFolder(std::static_pointer_cast<SleeveFolder>(element), folder_pre);
    } else if (element->_type == ElementType::FILE) {
      CaptureFile(std::static_pointer_cast<SleeveFile>(element), file_pre);
    }
  }
}

void SleeveMapper::CaptureFolder(std::shared_ptr<SleeveFolder> folder, Prepare &pre) {
  auto folder_content = folder->ListElements();
  for (auto &content_id : *folder_content) {
    // FolderContent (folder_id BIGINT, element_id BIGINT);
    duckdb_bind_uint32(pre._stmt, 1, folder->_element_id);
    duckdb_bind_uint32(pre._stmt, 2, content_id);
    // Execute insertion of an folder content
    if (duckdb_execute_prepared(pre._stmt, &pre._result) != DuckDBSuccess) {
      auto error_message = duckdb_result_error(&pre._result);
      throw std::exception(error_message);
    }
    duckdb_destroy_result(&pre._result);
  }
  // Insert FilterCombo <-> Folder mapping
  Prepare &combo_folder_pre = GetPrepare(GET_OP(Operate::ADD, Table::ComboFolder), Queries::combo_insert_query);
  auto     combos           = folder->ListFilters();
  auto     folder_id        = folder->_element_id;
  for (filter_id_t filter_id : combos) {
    // "INSERT INTO ComboFolder (combo_id, folder_id) VALUES  (?,?);"
    duckdb_bind_uint32(combo_folder_pre._stmt, 1, filter_id);
    duckdb_bind_uint32(combo_folder_pre._stmt, 2, folder_id);
    if (duckdb_execute_prepared(combo_folder_pre._stmt, &combo_folder_pre._result) != DuckDBSuccess) {
      auto error_message = duckdb_result_error(&combo_folder_pre._result);
      throw std::exception(error_message);
    }
    duckdb_destroy_result(&combo_folder_pre._result);
  }
}

void SleeveMapper::CaptureFile(std::shared_ptr<SleeveFile> file, Prepare &pre) {
  duckdb_bind_uint32(pre._stmt, 1, file->_element_id);
  duckdb_bind_uint32(pre._stmt, 2, file->GetImage()->_image_id);
  if (duckdb_execute_prepared(pre._stmt, &pre._result) != DuckDBSuccess) {
    auto error_message = duckdb_result_error(&pre._result);
    throw std::exception(error_message);
  }
  // TODO: handle with edit histories
  // For now edit history will not be stored in DB, since I haven't started working on the editor.
  duckdb_destroy_result(&pre._result);
}

void SleeveMapper::CaptureFilters(std::unordered_map<uint32_t, std::shared_ptr<FilterCombo>> &filter_storage,
                                  Prepare                                                    &pre) {
  std::wstring_convert<std::codecvt_utf8<wchar_t>> conv;

  for (auto &combo_it : filter_storage) {
    auto  combo    = combo_it.second;
    auto  combo_id = combo_it.first;
    auto &filters  = combo->GetFilters();
    for (auto &filter : filters) {
      // INSERT INTO Filter (combo_id,types,data)
      duckdb_bind_uint32(pre._stmt, 1, combo_id);
      duckdb_bind_uint32(pre._stmt, 2, static_cast<uint32_t>(filter._type));
      duckdb_bind_varchar(pre._stmt, 3, conv.to_bytes(filter.ToJSON()).c_str());
      if (duckdb_execute_prepared(pre._stmt, &pre._result) != DuckDBSuccess) {
        auto error_message = duckdb_result_error(&pre._result);
        throw std::exception(error_message);
      }
    }

    duckdb_destroy_result(&pre._result);
  }
}

void SleeveMapper::CaptureSleeve(const std::shared_ptr<SleeveBase>       sleeve_base,
                                 const std::shared_ptr<ImagePoolManager> image_pool) {
  if (!_initialized || !_db_connected) {
    throw std::exception("Cannot connect to a valid sleeve db");
  }
  std::unordered_map<uint32_t, std::shared_ptr<SleeveElement>> &storage = sleeve_base->GetStorage();
  if (_has_sleeve && _captured_sleeve_id != sleeve_base->_sleeve_id) {
    throw std::exception("A sleeve has already been captured and cannot be overwritten");
  }

  // Since base will not be alternated again, base_pre will not be in the _prepare_storage
  if (!_has_sleeve) {
    Prepare base_pre{_con};
    base_pre.GetStmtGuard(Queries::base_insert_query);

    // Try to insert the sleeve base if there is no sleeve captured
    duckdb_bind_uint32(base_pre._stmt, 1, sleeve_base->_sleeve_id);
    if (duckdb_execute_prepared(base_pre._stmt, &base_pre._result) != DuckDBSuccess) {
      auto error_message = duckdb_result_error(&base_pre._result);
      throw std::exception(error_message);
    }
    duckdb_destroy_result(&base_pre._result);
    // The same goes to root
    Prepare root_pre{_con};
    root_pre.GetStmtGuard(Queries::root_insert_query);
    // Insert the root
    duckdb_bind_uint64(root_pre._stmt, 1, static_cast<uint64_t>(sleeve_base->_root->_element_id));
    if (duckdb_execute_prepared(root_pre._stmt, &root_pre._result) != DuckDBSuccess) {
      auto error_message = duckdb_result_error(&root_pre._result);
      throw std::exception(error_message);
    }
    duckdb_destroy_result(&root_pre._result);
  }

  Prepare &element_pre = GetPrepare(GET_OP(Operate::ADD, Table::Element), Queries::element_insert_query);
  // Capture elements
  CaptureElement(storage, element_pre);

  // Capture filters
  Prepare &filter_pre = GetPrepare(GET_OP(Operate::ADD, Table::Filter), Queries::filter_insert_query);
  CaptureFilters(sleeve_base->GetFilterStorage(), filter_pre);

  Prepare &img_pre = GetPrepare(GET_OP(Operate::ADD, Table::Image), Queries::image_insert_query);
  CaptureImagePool(image_pool, img_pre);
}

void SleeveMapper::CaptureImagePool(const std::shared_ptr<ImagePoolManager> image_pool, Prepare &pre) {
  EASY_FUNCTION(profiler::colors::Cyan);
  ThreadPool thread_pool{16};
  if (!_initialized || !_db_connected) {
    throw std::exception("Cannot connect to a valid sleeve db");
  }

  auto                                &pool = image_pool->GetPool();
  BlockingMPMCQueue<PreparedImageData> exif_jsons{348};
  EASY_BLOCK("Submitting tasks");
  for (auto &pool_val : pool) {
    auto img = pool_val.second;
    thread_pool.Submit([img, &exif_jsons]() {
      std::string result = img->ExifToJson();
      EASY_BLOCK("Write finished to Queue");
      exif_jsons.push({img->_image_id, std::move(result), img});
      EASY_END_BLOCK;
    });
  }
  std::wstring_convert<std::codecvt_utf8<wchar_t>> conv;
  duckdb_appender                                  appender;
  duckdb_appender_create(_con, nullptr, "Image", &appender);
  EASY_END_BLOCK;

  EASY_BLOCK("Reclaim converted EXIF and write to DB");
  for (size_t i = 0; i < pool.size(); ++i) {
    EASY_BLOCK("Write data to DB");
    auto result = exif_jsons.pop();
    auto json   = result.json;
    // INSERT INTO Image (id,image_path,file_name,type,metadata)
    auto img    = result.img;
    duckdb_append_uint32(appender, img->_image_id);
    duckdb_append_varchar(appender, img->_image_path.string().c_str());
    duckdb_append_varchar(appender, conv.to_bytes(img->_image_name).c_str());
    duckdb_append_uint32(appender, static_cast<uint32_t>(img->_image_type));
    duckdb_append_varchar(appender, json.c_str());
    duckdb_appender_end_row(appender);
    EASY_END_BLOCK;
  }
  duckdb_appender_flush(appender);
  duckdb_appender_destroy(&appender);
}

void SleeveMapper::AddFilter(const std::shared_ptr<SleeveFolder> sleeve_folder,
                             const std::shared_ptr<FilterCombo> filter, Prepare &pre) {
  if (!_initialized || !_db_connected) {
    throw std::exception("Cannot connect to a valid sleeve db");
  }

  Prepare &filter_pre = GetPrepare(GET_OP(Operate::ADD, Table::Filter), Queries::filter_insert_query);
}
};  // namespace puerhlab