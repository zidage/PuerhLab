#include <gtest/gtest.h>

#include <exiv2/exiv2.hpp>

#include <filesystem>
#include <future>
#include <memory>
#include <string>
#include <vector>

#include "app/import_service.hpp"
#include "app/project_service.hpp"
#include "app/sleeve_filter_service.hpp"
#include "edit/operators/operator_registeration.hpp"
#include "sleeve/sleeve_element/sleeve_element.hpp"
#include "sleeve/sleeve_filter/filter_combo.hpp"
#include "type/supported_file_type.hpp"
#include "utils/clock/time_provider.hpp"

namespace puerhlab {
class FilterServiceTests : public ::testing::Test {
 protected:
  std::filesystem::path db_path_;
  std::filesystem::path meta_path_;

  void SetUp() override {
    TimeProvider::Refresh();
    Exiv2::LogMsg::setLevel(Exiv2::LogMsg::Level::mute);
    RegisterAllOperators();

    db_path_   = std::filesystem::temp_directory_path() / "filter_service_test.db";
    meta_path_ = std::filesystem::temp_directory_path() / "filter_service_test.json";

    if (std::filesystem::exists(db_path_)) {
      std::filesystem::remove(db_path_);
    }
    if (std::filesystem::exists(meta_path_)) {
      std::filesystem::remove(meta_path_);
    }
  }

  void TearDown() override {
    if (std::filesystem::exists(db_path_)) {
      std::filesystem::remove(db_path_);
    }
    if (std::filesystem::exists(meta_path_)) {
      std::filesystem::remove(meta_path_);
    }
  }

  static auto LoadBatchToRoot(ProjectService& project) -> uint32_t {
    auto fs_service       = project.GetSleeveService();
    auto img_pool_service = project.GetImagePoolService();

    std::unique_ptr<ImportService> import_service =
        std::make_unique<ImportServiceImpl>(fs_service, img_pool_service);

    const image_path_t batch_dir = std::string(TEST_IMG_PATH) + "/raw/batch";

    std::vector<image_path_t> paths;
    for (const auto& img : std::filesystem::directory_iterator(batch_dir)) {
      if (!img.is_directory() && is_supported_file(img.path())) {
        paths.push_back(img.path().string());
      }
    }

    std::shared_ptr<ImportJob> import_job = std::make_shared<ImportJob>();

    std::promise<ImportResult> final_result;
    auto                       final_result_future = final_result.get_future();

    import_job->on_finished_ = [&final_result](const ImportResult& result) {
      final_result.set_value(result);
    };

    import_job = import_service->ImportToFolder(paths, L"", {}, import_job);
    EXPECT_NE(import_job, nullptr);

    final_result_future.wait();
    ImportResult result = final_result_future.get();

    EXPECT_EQ(result.requested_, static_cast<uint32_t>(paths.size()));
    EXPECT_EQ(result.failed_, 0u);

    EXPECT_NE(import_job->import_log_, nullptr);
    auto snapshot = import_job->import_log_->Snapshot();
    import_service->SyncImports(snapshot, L"");

    return result.imported_;
  }
};

TEST_F(FilterServiceTests, ASTCreationTest) {
  try {
    FieldCondition cond{
        .field_ = FilterField::ExifCameraModel,
        .op_    = CompareOp::EQUALS,
        .value_ = std::wstring(L"Canon EOS 5D Mark IV"),
    };
    FilterNode root{FilterNode::Type::Condition, {}, {}, std::move(cond), std::nullopt};
    (void)root;
  } catch (const std::exception& e) {
    FAIL() << "Exception during AST creation: " << e.what();
  }
}

TEST_F(FilterServiceTests, SQLCompilationTest) {
  FieldCondition cond{
      .field_ = FilterField::ExifCameraModel,
      .op_    = CompareOp::EQUALS,
      .value_ = std::wstring(L"Canon EOS 5D Mark IV"),
  };
  FilterNode root{FilterNode::Type::Condition, {}, {}, std::move(cond), std::nullopt};

  std::wstring sql          = FilterSQLCompiler::Compile(root);
  std::wstring expected_sql = L"(json_extract(metadata, '$.Model') = 'Canon EOS 5D Mark IV')";
  EXPECT_EQ(sql, expected_sql);
}

TEST_F(FilterServiceTests, ComplexFilterSQLTest) {
  FieldCondition cond1{
      .field_ = FilterField::ExifCameraModel,
      .op_    = CompareOp::EQUALS,
      .value_ = std::wstring(L"Nikon D850"),
  };
  FilterNode node1{FilterNode::Type::Condition, {}, {}, std::move(cond1), std::nullopt};

  FieldCondition cond2{
      .field_ = FilterField::FileExtension,
      .op_    = CompareOp::ENDS_WITH,
      .value_ = std::wstring(L".NEF"),
  };
  FilterNode node2{FilterNode::Type::Condition, {}, {}, std::move(cond2), std::nullopt};

  FilterNode root{FilterNode::Type::Logical, FilterOp::AND, {node1, node2}, {}, std::nullopt};

  std::wstring sql = FilterSQLCompiler::Compile(root);
  std::wstring expected_sql =
      L"((json_extract(metadata, '$.Model') = 'Nikon D850') AND (UPPER(file_name) LIKE '%.NEF'))";
  EXPECT_EQ(sql, expected_sql);
}

TEST_F(FilterServiceTests, BetweenConditionSQLTest) {
  FieldCondition cond{
      .field_        = FilterField::ExifISO,
      .op_           = CompareOp::BETWEEN,
      .value_        = int64_t(100),
      .second_value_ = int64_t(800),
  };
  FilterNode root{FilterNode::Type::Condition, {}, {}, std::move(cond), std::nullopt};

  std::wstring sql          = FilterSQLCompiler::Compile(root);
  std::wstring expected_sql = L"(json_extract(metadata, '$.ISO')::INT BETWEEN 100 AND 800)";
  EXPECT_EQ(sql, expected_sql);
}

TEST_F(FilterServiceTests, FolderIndexTest_Model) {
  ProjectService       project(db_path_, meta_path_);
  const uint32_t       imported = LoadBatchToRoot(project);
  ASSERT_GT(imported, 0u);

  auto sleeve_service = project.GetSleeveService();
  auto root_folder = sleeve_service->Read<std::shared_ptr<SleeveElement>>(
      [](FileSystem& fs) { return fs.Get(L"/", false); });
  ASSERT_NE(root_folder, nullptr);

  SleeveFilterService filter_service(project.GetStorageService());

  FieldCondition cond{
      .field_ = FilterField::ExifCameraModel,
      .op_    = CompareOp::CONTAINS,
      .value_ = std::wstring(L"D850"),
  };
  FilterNode root{FilterNode::Type::Condition, {}, {}, std::move(cond), std::nullopt};

  const auto filter_id   = filter_service.CreateFilterCombo(root);
  auto       result_opt  = filter_service.ApplyFilterOn(filter_id, root_folder->element_id_);
  ASSERT_TRUE(result_opt.has_value());

  EXPECT_EQ(result_opt->size(), 5u);
}

TEST_F(FilterServiceTests, FolderIndexTest_FileExtension) {
  ProjectService       project(db_path_, meta_path_);
  const uint32_t       imported = LoadBatchToRoot(project);
  ASSERT_GT(imported, 0u);

  auto sleeve_service = project.GetSleeveService();
  auto root_folder = sleeve_service->Read<std::shared_ptr<SleeveElement>>(
      [](FileSystem& fs) { return fs.Get(L"/", false); });
  ASSERT_NE(root_folder, nullptr);

  SleeveFilterService filter_service(project.GetStorageService());

  FieldCondition cond{
      .field_ = FilterField::FileExtension,
      .op_    = CompareOp::ENDS_WITH,
      .value_ = std::wstring(L".NEF"),
  };
  FilterNode root{FilterNode::Type::Condition, {}, {}, std::move(cond), std::nullopt};

  const auto filter_id  = filter_service.CreateFilterCombo(root);
  auto       result_opt = filter_service.ApplyFilterOn(filter_id, root_folder->element_id_);
  ASSERT_TRUE(result_opt.has_value());

  EXPECT_EQ(result_opt->size(), 5u);
}

TEST_F(FilterServiceTests, FolderIndexTest_Aperature) {
  ProjectService       project(db_path_, meta_path_);
  const uint32_t       imported = LoadBatchToRoot(project);
  ASSERT_GT(imported, 0u);

  auto sleeve_service = project.GetSleeveService();
  auto root_folder = sleeve_service->Read<std::shared_ptr<SleeveElement>>(
      [](FileSystem& fs) { return fs.Get(L"/", false); });
  ASSERT_NE(root_folder, nullptr);

  SleeveFilterService filter_service(project.GetStorageService());

  FieldCondition cond{
      .field_ = FilterField::ExifAperture,
      .op_    = CompareOp::GREATER_THAN,
      .value_ = double(5.6),
  };
  FilterNode root{FilterNode::Type::Condition, {}, {}, std::move(cond), std::nullopt};

  const auto filter_id  = filter_service.CreateFilterCombo(root);
  auto       result_opt = filter_service.ApplyFilterOn(filter_id, root_folder->element_id_);
  ASSERT_TRUE(result_opt.has_value());

  EXPECT_EQ(result_opt->size(), 12u);
}

TEST_F(FilterServiceTests, FolderIndexTest_ISO) {
  ProjectService       project(db_path_, meta_path_);
  const uint32_t       imported = LoadBatchToRoot(project);
  ASSERT_GT(imported, 0u);

  auto sleeve_service = project.GetSleeveService();
  auto root_folder = sleeve_service->Read<std::shared_ptr<SleeveElement>>(
      [](FileSystem& fs) { return fs.Get(L"/", false); });
  ASSERT_NE(root_folder, nullptr);

  SleeveFilterService filter_service(project.GetStorageService());

  FieldCondition cond{
      .field_        = FilterField::ExifISO,
      .op_           = CompareOp::BETWEEN,
      .value_        = int64_t(100),
      .second_value_ = int64_t(400),
  };
  FilterNode root{FilterNode::Type::Condition, {}, {}, std::move(cond), std::nullopt};

  const auto filter_id  = filter_service.CreateFilterCombo(root);
  auto       result_opt = filter_service.ApplyFilterOn(filter_id, root_folder->element_id_);
  ASSERT_TRUE(result_opt.has_value());

  EXPECT_EQ(result_opt->size(), 12u);
}

TEST_F(FilterServiceTests, FolderIndexTest_FocalLength) {
  ProjectService       project(db_path_, meta_path_);
  const uint32_t       imported = LoadBatchToRoot(project);
  ASSERT_GT(imported, 0u);

  auto sleeve_service = project.GetSleeveService();
  auto root_folder = sleeve_service->Read<std::shared_ptr<SleeveElement>>(
      [](FileSystem& fs) { return fs.Get(L"/", false); });
  ASSERT_NE(root_folder, nullptr);

  SleeveFilterService filter_service(project.GetStorageService());

  FieldCondition cond{
      .field_ = FilterField::ExifFocalLength,
      .op_    = CompareOp::LESS_THAN,
      .value_ = double(150.0),
  };
  FilterNode root{FilterNode::Type::Condition, {}, {}, std::move(cond), std::nullopt};

  const auto filter_id  = filter_service.CreateFilterCombo(root);
  auto       result_opt = filter_service.ApplyFilterOn(filter_id, root_folder->element_id_);
  ASSERT_TRUE(result_opt.has_value());

  EXPECT_EQ(result_opt->size(), 13u);
}

TEST_F(FilterServiceTests, FolderIndexTest_Combined) {
  ProjectService       project(db_path_, meta_path_);
  const uint32_t       imported = LoadBatchToRoot(project);
  ASSERT_GT(imported, 0u);

  auto sleeve_service = project.GetSleeveService();
  auto root_folder = sleeve_service->Read<std::shared_ptr<SleeveElement>>(
      [](FileSystem& fs) { return fs.Get(L"/", false); });
  ASSERT_NE(root_folder, nullptr);

  SleeveFilterService filter_service(project.GetStorageService());

  FieldCondition cond1{
      .field_ = FilterField::ExifCameraModel,
      .op_    = CompareOp::CONTAINS,
      .value_ = std::wstring(L"D850"),
  };
  FilterNode node1{FilterNode::Type::Condition, {}, {}, std::move(cond1), std::nullopt};

  FieldCondition cond2{
      .field_ = FilterField::ExifFocalLength,
      .op_    = CompareOp::LESS_THAN,
      .value_ = double(150.0),
  };
  FilterNode node2{FilterNode::Type::Condition, {}, {}, std::move(cond2), std::nullopt};

  FilterNode root{FilterNode::Type::Logical, FilterOp::AND, {node1, node2}, {}, std::nullopt};

  const auto filter_id  = filter_service.CreateFilterCombo(root);
  auto       result_opt = filter_service.ApplyFilterOn(filter_id, root_folder->element_id_);
  ASSERT_TRUE(result_opt.has_value());

  EXPECT_EQ(result_opt->size(), 1u);
}

TEST_F(FilterServiceTests, FolderIndexTest_NoMatch) {
  ProjectService       project(db_path_, meta_path_);
  const uint32_t       imported = LoadBatchToRoot(project);
  ASSERT_GT(imported, 0u);

  auto sleeve_service = project.GetSleeveService();
  auto root_folder = sleeve_service->Read<std::shared_ptr<SleeveElement>>(
      [](FileSystem& fs) { return fs.Get(L"/", false); });
  ASSERT_NE(root_folder, nullptr);

  SleeveFilterService filter_service(project.GetStorageService());

  FieldCondition cond{
      .field_ = FilterField::ExifCameraModel,
      .op_    = CompareOp::CONTAINS,
      .value_ = std::wstring(L"A7"),
  };
  FilterNode root{FilterNode::Type::Condition, {}, {}, std::move(cond), std::nullopt};

  const auto filter_id  = filter_service.CreateFilterCombo(root);
  auto       result_opt = filter_service.ApplyFilterOn(filter_id, root_folder->element_id_);
  ASSERT_TRUE(result_opt.has_value());

  EXPECT_EQ(result_opt->size(), 0u);
}

TEST_F(FilterServiceTests, FolderIndexTest_DateRange) {
  ProjectService       project(db_path_, meta_path_);
  const uint32_t       imported = LoadBatchToRoot(project);
  ASSERT_GT(imported, 0u);

  auto sleeve_service = project.GetSleeveService();
  auto root_folder = sleeve_service->Read<std::shared_ptr<SleeveElement>>(
      [](FileSystem& fs) { return fs.Get(L"/", false); });
  ASSERT_NE(root_folder, nullptr);

  SleeveFilterService filter_service(project.GetStorageService());

  FieldCondition cond{
      .field_        = FilterField::CaptureDate,
      .op_           = CompareOp::BETWEEN,
      .value_        = std::tm{0, 0, 0, 1, 0, 125, 0, 0, -1},    // Jan 1, 2025
      .second_value_ = std::tm{0, 0, 0, 31, 11, 125, 0, 0, -1},  // Dec 31, 2025
  };
  FilterNode root{FilterNode::Type::Condition, {}, {}, std::move(cond), std::nullopt};

  const auto filter_id  = filter_service.CreateFilterCombo(root);
  auto       result_opt = filter_service.ApplyFilterOn(filter_id, root_folder->element_id_);
  ASSERT_TRUE(result_opt.has_value());

  EXPECT_EQ(result_opt->size(), 13u);
}

}  // namespace puerhlab
