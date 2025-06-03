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
#include <string>
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
  // Insert Folder
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
  // Remove existing file image mapping
  RemoveFile(file->_element_id);

  // Insert File
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
      duckdb_destroy_result(&pre._result);
    }
  }
}

void SleeveMapper::RecaptureFolder(std::shared_ptr<SleeveFolder> folder) {
  if (!_initialized || !_db_connected) {
    throw std::exception("Cannot connect to a valid sleeve db");
  }
  // Remove existing folder content mapping
  RemoveFolder(folder->_element_id);

  Prepare &folder_pre = GetPrepare(GET_OP(Operate::ADD, Table::Folder), Queries::folder_insert_query);
  CaptureFolder(folder, folder_pre);
}

void SleeveMapper::RecaptureFile(std::shared_ptr<SleeveFile> file) {
  if (!_initialized || !_db_connected) {
    throw std::exception("Cannot connect to a valid sleeve db");
  }
  // Remove existing file image mapping
  RemoveFile(file->_element_id);

  Prepare &file_pre = GetPrepare(GET_OP(Operate::ADD, Table::File), Queries::file_insert_query);
  CaptureFile(file, file_pre);
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

  CaptureImagePool(image_pool);
}

void SleeveMapper::CaptureImagePool(const std::shared_ptr<ImagePoolManager> image_pool) {
  ThreadPool thread_pool{8};
  if (!_initialized || !_db_connected) {
    throw std::exception("Cannot connect to a valid sleeve db");
  }

  auto                                &pool = image_pool->GetPool();
  BlockingMPMCQueue<PreparedImageData> exif_jsons{348};
  for (auto &pool_val : pool) {
    auto img = pool_val.second;
    thread_pool.Submit([img, &exif_jsons]() {
      std::string result = img->ExifToJson();
      exif_jsons.push({img->_image_id, std::move(result), img});
    });
  }
  duckdb_appender appender;
  duckdb_appender_create(_con, nullptr, "Image", &appender);

  for (size_t i = 0; i < pool.size(); ++i) {
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
  }
  duckdb_appender_flush(appender);
  duckdb_appender_destroy(&appender);
}

void SleeveMapper::AddFilterCombo(const sl_element_id_t folder_id, const std::shared_ptr<FilterCombo> combo) {
  if (!_initialized || !_db_connected) {
    throw std::exception("Cannot connect to a valid sleeve db");
  }

  Prepare &filter_pre       = GetPrepare(GET_OP(Operate::ADD, Table::Filter), Queries::filter_insert_query);
  Prepare &combo_folder_pre = GetPrepare(GET_OP(Operate::ADD, Table::ComboFolder), Queries::filter_insert_query);
  auto     combo_id         = combo->filter_id;

  duckdb_bind_uint32(combo_folder_pre._stmt, 1, combo_id);
  duckdb_bind_uint32(combo_folder_pre._stmt, 2, folder_id);
  if (duckdb_execute_prepared(combo_folder_pre._stmt, &combo_folder_pre._result) != DuckDBSuccess) {
    auto error_message = duckdb_result_error(&combo_folder_pre._result);
    throw std::exception(error_message);
  }
  duckdb_destroy_result(&combo_folder_pre._result);

  auto &filters = combo->GetFilters();

  for (auto &filter : filters) {
    // INSERT INTO Filter (combo_id,types,data)
    duckdb_bind_uint32(filter_pre._stmt, 1, combo_id);
    duckdb_bind_uint32(filter_pre._stmt, 2, static_cast<uint32_t>(filter._type));
    duckdb_bind_varchar(filter_pre._stmt, 3, conv.to_bytes(filter.ToJSON()).c_str());
    if (duckdb_execute_prepared(filter_pre._stmt, &filter_pre._result) != DuckDBSuccess) {
      auto error_message = duckdb_result_error(&filter_pre._result);
      throw std::exception(error_message);
    }
    duckdb_destroy_result(&filter_pre._result);
  }
}

void SleeveMapper::AddImage(const Image &image) {
  if (!_initialized || !_db_connected) {
    throw std::exception("Cannot connect to a valid sleeve db");
  }

  Prepare &img_pre = GetPrepare(GET_OP(Operate::ADD, Table::Filter), Queries::image_insert_query);
  // INSERT INTO Image (id,image_path,file_name,type,metadata)
  duckdb_bind_uint32(img_pre._stmt, 1, image._image_id);
  duckdb_bind_varchar(img_pre._stmt, 2, conv.to_bytes(image._image_path.wstring()).c_str());
  duckdb_bind_varchar(img_pre._stmt, 3, conv.to_bytes(image._image_name).c_str());
  duckdb_bind_uint32(img_pre._stmt, 4, static_cast<uint32_t>(image._image_type));
  duckdb_bind_varchar(img_pre._stmt, 5, image.ExifToJson().c_str());
  if (duckdb_execute_prepared(img_pre._stmt, &img_pre._result) != DuckDBSuccess) {
    auto error_message = duckdb_result_error(&img_pre._result);
    throw std::exception(error_message);
  }
  duckdb_destroy_result(&img_pre._result);
}

auto SleeveMapper::GetImage(const image_id_t id) -> std::shared_ptr<Image> {
  if (!_initialized || !_db_connected) {
    throw std::exception("Cannot connect to a valid sleeve db");
  }

  Prepare &img_pre = GetPrepare(GET_OP(Operate::LOOKUP, Table::Image), Queries::image_lookup_query);
  duckdb_bind_uint32(img_pre._stmt, 1, id);
  if (duckdb_execute_prepared(img_pre._stmt, &img_pre._result) != DuckDBSuccess) {
    auto error_message = duckdb_result_error(&img_pre._result);
    throw std::exception(error_message);
  }
  idx_t row_count = duckdb_row_count(&img_pre._result);
  if (row_count != 1) {
    duckdb_destroy_result(&img_pre._result);
    throw std::exception("No image found");  // No image found
  }

  // Image (id,image_path,file_name,type,metadata)
  auto image_id   = duckdb_value_uint32(&img_pre._result, 0, 0);
  auto img_path   = std::filesystem::path(duckdb_value_varchar(&img_pre._result, 1, 0));
  auto file_name  = conv.from_bytes(duckdb_value_varchar(&img_pre._result, 2, 0));
  auto type       = static_cast<ImageType>(duckdb_value_uint32(&img_pre._result, 3, 0));
  auto metadata   = duckdb_value_varchar(&img_pre._result, 4, 0);

  auto img        = std::make_shared<Image>(image_id, img_path, file_name, type);

  img->_exif_json = nlohmann::json::parse(metadata);
  img->_exif_display.ExtractFromJson(img->_exif_json);

  duckdb_destroy_result(&img_pre._result);

  return img;
}

void SleeveMapper::UpdateImage(const Image &image, const image_id_t id) {
  if (!_initialized || !_db_connected) {
    throw std::exception("Cannot connect to a valid sleeve db");
  }

  Prepare &img_pre = GetPrepare(GET_OP(Operate::EDIT, Table::Image), Queries::image_update_query);
  // INSERT INTO Image (id,image_path,file_name,type,metadata)
  duckdb_bind_uint32(img_pre._stmt, 1, id);
  duckdb_bind_varchar(img_pre._stmt, 2, conv.to_bytes(image._image_path.wstring()).c_str());
  duckdb_bind_varchar(img_pre._stmt, 3, conv.to_bytes(image._image_name).c_str());
  duckdb_bind_uint32(img_pre._stmt, 4, static_cast<uint32_t>(image._image_type));
  duckdb_bind_varchar(img_pre._stmt, 5, image.ExifToJson().c_str());
  if (duckdb_execute_prepared(img_pre._stmt, &img_pre._result) != DuckDBSuccess) {
    auto error_message = duckdb_result_error(&img_pre._result);
    throw std::exception(error_message);
  }
  duckdb_destroy_result(&img_pre._result);
}

void SleeveMapper::RemoveImage(const image_id_t id) {
  if (!_initialized || !_db_connected) {
    throw std::exception("Cannot connect to a valid sleeve db");
  }

  Prepare &img_pre = GetPrepare(GET_OP(Operate::DELETE, Table::Image), Queries::image_delete_query);
  duckdb_bind_uint32(img_pre._stmt, 1, id);
  if (duckdb_execute_prepared(img_pre._stmt, &img_pre._result) != DuckDBSuccess) {
    auto error_message = duckdb_result_error(&img_pre._result);
    throw std::exception(error_message);
  }
}

void SleeveMapper::AddElement(const SleeveElement &element) {
  if (!_initialized || !_db_connected) {
    throw std::exception("Cannot connect to a valid sleeve db");
  }

  Prepare &element_pre = GetPrepare(GET_OP(Operate::ADD, Table::Element), Queries::element_insert_query);
  char     added_time[32];
  char     modified_time[32];
  std::strftime(added_time, sizeof(added_time), "%Y-%m-%d %H:%M:%S", std::gmtime(&element._added_time));
  std::strftime(modified_time, sizeof(modified_time), "%Y-%m-%d %H:%M:%S", std::gmtime(&element._last_modified_time));

  duckdb_bind_uint32(element_pre._stmt, 1, element._element_id);
  duckdb_bind_uint32(element_pre._stmt, 2, static_cast<uint32_t>(element._type));
  duckdb_bind_varchar(element_pre._stmt, 3, conv.to_bytes(element._element_name).c_str());
  duckdb_bind_varchar(element_pre._stmt, 4, added_time);
  duckdb_bind_varchar(element_pre._stmt, 5, modified_time);
  duckdb_bind_uint32(element_pre._stmt, 6, element._ref_count);

  if (duckdb_execute_prepared(element_pre._stmt, &element_pre._result) != DuckDBSuccess) {
    auto error_message = duckdb_result_error(&element_pre._result);
    throw std::exception(error_message);
  }

  if (element._type == ElementType::FOLDER) {
    Prepare &folder_pre = GetPrepare(GET_OP(Operate::ADD, Table::Folder), Queries::folder_insert_query);
    CaptureFolder(std::static_pointer_cast<SleeveFolder>(std::make_shared<SleeveElement>(element)), folder_pre);
  } else if (element._type == ElementType::FILE) {
    Prepare &file_pre = GetPrepare(GET_OP(Operate::ADD, Table::File), Queries::file_insert_query);
    CaptureFile(std::static_pointer_cast<SleeveFile>(std::make_shared<SleeveElement>(element)), file_pre);
  }
}

void SleeveMapper::RemoveElement(const SleeveElement &element) {
  if (!_initialized || !_db_connected) {
    throw std::exception("Cannot connect to a valid sleeve db");
  }

  Prepare &element_pre = GetPrepare(GET_OP(Operate::DELETE, Table::Element), Queries::element_delete_query);
  duckdb_bind_uint32(element_pre._stmt, 1, element._element_id);
  if (duckdb_execute_prepared(element_pre._stmt, &element_pre._result) != DuckDBSuccess) {
    auto error_message = duckdb_result_error(&element_pre._result);
    throw std::exception(error_message);
  }

  if (element._type == ElementType::FOLDER) {
    RemoveFolder(element._element_id);
  } else if (element._type == ElementType::FILE) {
    RemoveFile(element._element_id);
  }
}

void SleeveMapper::EditElement(const sl_element_id_t element_id, const std::shared_ptr<SleeveElement> element) {
  if (!_initialized || !_db_connected) {
    throw std::exception("Cannot connect to a valid sleeve db");
  }

  Prepare &element_pre = GetPrepare(GET_OP(Operate::EDIT, Table::Element), Queries::element_update_query);
  char     added_time[32];
  char     modified_time[32];
  std::strftime(added_time, sizeof(added_time), "%Y-%m-%d %H:%M:%S", std::gmtime(&element->_added_time));
  std::strftime(modified_time, sizeof(modified_time), "%Y-%m-%d %H:%M:%S", std::gmtime(&element->_last_modified_time));

  duckdb_bind_uint32(element_pre._stmt, 1, element_id);
  duckdb_bind_uint32(element_pre._stmt, 2, static_cast<uint32_t>(element->_type));
  duckdb_bind_varchar(element_pre._stmt, 3, conv.to_bytes(element->_element_name).c_str());
  duckdb_bind_varchar(element_pre._stmt, 4, added_time);
  duckdb_bind_varchar(element_pre._stmt, 5, modified_time);
  duckdb_bind_uint32(element_pre._stmt, 6, element->_ref_count);

  if (duckdb_execute_prepared(element_pre._stmt, &element_pre._result) != DuckDBSuccess) {
    auto error_message = duckdb_result_error(&element_pre._result);
    throw std::exception(error_message);
  }

  duckdb_destroy_result(&element_pre._result);
  if (element->_type == ElementType::FOLDER) {
    Prepare &folder_pre = GetPrepare(GET_OP(Operate::ADD, Table::Folder), Queries::folder_insert_query);
    CaptureFolder(std::static_pointer_cast<SleeveFolder>(element), folder_pre);
  } else if (element->_type == ElementType::FILE) {
    Prepare &file_pre = GetPrepare(GET_OP(Operate::ADD, Table::File), Queries::file_insert_query);
    CaptureFile(std::static_pointer_cast<SleeveFile>(element), file_pre);
  }
}

auto SleeveMapper::GetElement(const sl_element_id_t element_id) -> std::shared_ptr<SleeveElement> {
  if (!_initialized || !_db_connected) {
    throw std::exception("Cannot connect to a valid sleeve db");
  }

  Prepare &element_pre = GetPrepare(GET_OP(Operate::LOOKUP, Table::Element), Queries::element_lookup_query);
  duckdb_bind_uint32(element_pre._stmt, 1, element_id);
  if (duckdb_execute_prepared(element_pre._stmt, &element_pre._result) != DuckDBSuccess) {
    auto error_message = duckdb_result_error(&element_pre._result);
    throw std::exception(error_message);
  }
  idx_t row_count = duckdb_row_count(&element_pre._result);
  if (row_count != 1) {
    duckdb_destroy_result(&element_pre._result);
    return nullptr;  // No element found
  }

  // Element (id,type,element_name,added_time,modified_time,ref_count)
  auto               element_id_val = duckdb_value_uint32(&element_pre._result, 0, 0);
  auto               type           = static_cast<ElementType>(duckdb_value_uint32(&element_pre._result, 1, 0));
  auto               element_name   = conv.from_bytes(duckdb_value_varchar(&element_pre._result, 2, 0));
  auto               added_time     = duckdb_value_varchar(&element_pre._result, 3, 0);
  auto               modified_time  = duckdb_value_varchar(&element_pre._result, 4, 0);
  auto               ref_count      = duckdb_value_uint32(&element_pre._result, 5, 0);

  std::tm            tm_added{};
  std::tm            tm_modified{};

  std::istringstream a_ss(added_time);
  std::istringstream m_ss(modified_time);
  a_ss >> std::get_time(&tm_added, "%Y-%m-%d %H:%M:%S");
  m_ss >> std::get_time(&tm_modified, "%Y-%m-%d %H:%M:%S");

  std::shared_ptr<SleeveElement> element;
  if (type == ElementType::FILE) {
    element = std::make_shared<SleeveFile>(element_id_val, element_name);
  } else if (type == ElementType::FOLDER) {
    element = std::make_shared<SleeveFolder>(element_id_val, element_name);
  }
  element->_added_time         = std::mktime(&tm_added);
  element->_last_modified_time = std::mktime(&tm_modified);
  element->_ref_count          = ref_count;

  return element;
}

void SleeveMapper::RemoveFolder(const sl_element_id_t folder_id) {
  if (!_initialized || !_db_connected) {
    throw std::exception("Cannot connect to a valid sleeve db");
  }

  // Remove all folder content
  Prepare &folder_pre = GetPrepare(GET_OP(Operate::DELETE, Table::Folder), Queries::folder_delete_query);
  duckdb_bind_uint32(folder_pre._stmt, 1, folder_id);
  if (duckdb_execute_prepared(folder_pre._stmt, &folder_pre._result) != DuckDBSuccess) {
    auto error_message = duckdb_result_error(&folder_pre._result);
    throw std::exception(error_message);
  }
  duckdb_destroy_result(&folder_pre._result);

  // Remove all bound filter combos
  RemoveFilterComboByFolderId(folder_id);
}

void SleeveMapper::RemoveFile(const sl_element_id_t file_id) {
  if (!_initialized || !_db_connected) {
    throw std::exception("Cannot connect to a valid sleeve db");
  }

  Prepare &file_pre = GetPrepare(GET_OP(Operate::DELETE, Table::File), Queries::file_delete_query);
  duckdb_bind_uint32(file_pre._stmt, 1, file_id);
  if (duckdb_execute_prepared(file_pre._stmt, &file_pre._result) != DuckDBSuccess) {
    auto error_message = duckdb_result_error(&file_pre._result);
    throw std::exception(error_message);
  }
  duckdb_destroy_result(&file_pre._result);
}

void SleeveMapper::RemoveFilterComboByComboId(const sl_element_id_t combo_id) {
  if (!_initialized || !_db_connected) {
    throw std::exception("Cannot connect to a valid sleeve db");
  }

  Prepare &combo_pre = GetPrepare(GET_OP(Operate::DELETE, Table::Filter), Queries::combo_delete_query_combo_id);
  duckdb_bind_uint32(combo_pre._stmt, 1, combo_id);
  if (duckdb_execute_prepared(combo_pre._stmt, &combo_pre._result) != DuckDBSuccess) {
    auto error_message = duckdb_result_error(&combo_pre._result);
    throw std::exception(error_message);
  }
  duckdb_destroy_result(&combo_pre._result);
}

void SleeveMapper::RemoveFilterComboByFolderId(const sl_element_id_t folder_id) {
  if (!_initialized || !_db_connected) {
    throw std::exception("Cannot connect to a valid sleeve db");
  }

  Prepare &combo_pre = GetPrepare(GET_OP(Operate::DELETE, Table::ComboFolder), Queries::combo_delete_query_folder_id);
  duckdb_bind_uint32(combo_pre._stmt, 1, folder_id);
  if (duckdb_execute_prepared(combo_pre._stmt, &combo_pre._result) != DuckDBSuccess) {
    auto error_message = duckdb_result_error(&combo_pre._result);
    throw std::exception(error_message);
  }
  duckdb_destroy_result(&combo_pre._result);
}

void SleeveMapper::EditFilterCombo(const sl_element_id_t folder_id, const std::shared_ptr<FilterCombo> combo) {
  if (!_initialized || !_db_connected) {
    throw std::exception("Cannot connect to a valid sleeve db");
  }

  RemoveFilterComboByFolderId(folder_id);
  AddFilterCombo(folder_id, combo);
}

auto SleeveMapper::GetSleeveBase() -> std::shared_ptr<SleeveBase> {
  if (!_db_connected) {
    throw std::exception("Cannot connect to a valid sleeve db");
  }

  if (_has_sleeve) {
    throw std::exception("Sleeve base has already been restored");
  }

  // Get sleeve base
  Prepare base_pre{_con};
  base_pre.GetStmtGuard(Queries::base_lookup_query);
  if (duckdb_execute_prepared(base_pre._stmt, &base_pre._result) != DuckDBSuccess) {
    auto error_message = duckdb_result_error(&base_pre._result);
    throw std::exception(error_message);
  }
  idx_t row_count = duckdb_row_count(&base_pre._result);
  if (row_count != 1) {
    duckdb_destroy_result(&base_pre._result);
    throw std::exception("No sleeve base found");
  }
  duckdb_destroy_result(&base_pre._result);
  sleeve_id_t sleeve_id = duckdb_value_uint32(&base_pre._result, 0, 0);
  _captured_sleeve_id   = sleeve_id;
  _has_sleeve           = true;
  _initialized          = true;
  auto    sleeve_base   = std::make_shared<SleeveBase>(sleeve_id);

  // Get root element
  Prepare root_pre{_con};
  root_pre.GetStmtGuard(Queries::root_lookup_query);
  duckdb_bind_uint32(root_pre._stmt, 1, sleeve_id);
  if (duckdb_execute_prepared(root_pre._stmt, &root_pre._result) != DuckDBSuccess) {
    auto error_message = duckdb_result_error(&root_pre._result);
    throw std::exception(error_message);
  }
  idx_t root_row_count = duckdb_row_count(&root_pre._result);
  if (root_row_count != 1) {
    duckdb_destroy_result(&root_pre._result);
    throw std::exception("Bad database file, no root found");
  }
  duckdb_destroy_result(&root_pre._result);
  sl_element_id_t root_id = duckdb_value_uint32(&root_pre._result, 0, 0);

  auto            root    = GetElement(root_id);
  sleeve_base->_root      = std::static_pointer_cast<SleeveFolder>(root);

  // Get all elements under the root
  Prepare &folder_pre     = GetPrepare(GET_OP(Operate::LOOKUP, Table::Folder), Queries::folder_content_lookup_query);
  duckdb_bind_uint32(folder_pre._stmt, 1, root_id);
  if (duckdb_execute_prepared(folder_pre._stmt, &folder_pre._result) != DuckDBSuccess) {
    auto error_message = duckdb_result_error(&folder_pre._result);
    throw std::exception(error_message);
  }
  idx_t content_row_count = duckdb_row_count(&folder_pre._result);
  for (idx_t i = 0; i < content_row_count; ++i) {
    sl_element_id_t element_id = duckdb_value_uint32(&folder_pre._result, 2, i);
    auto            element    = GetElement(element_id);
    if (element->_type == ElementType::FOLDER) {
      auto folder = std::static_pointer_cast<SleeveFolder>(element);
    } else if (element->_type == ElementType::FILE) {
      auto file = std::static_pointer_cast<SleeveFile>(element);
    }
  }
}
};  // namespace puerhlab