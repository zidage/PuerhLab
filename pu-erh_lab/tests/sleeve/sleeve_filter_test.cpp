#include <gtest/gtest.h>
#include <memory>

#include "sleeve/sleeve_filter/filter_combo.hpp"
#include "sleeve/sleeve_manager.hpp"
#include "sleeve_test_fixture.hpp"
#include "type/supported_file_type.hpp"
#include "utils/string/convert.hpp"

using namespace puerhlab;

TEST_F(SleeveFilterTests, ASTCreationTest) {
  try {
    FieldCondition cond{
        .field = FilterField::ExifCameraModel,
        .op    = CompareOp::EQUALS,
        .value = std::wstring(L"Canon EOS 5D Mark IV"),
    };
    FilterNode root{FilterNode::Type::Condition, {}, {}, std::move(cond), std::nullopt};
  } catch (const std::exception& e) {
    FAIL() << "Exception during AST creation: " << e.what();
  }
}

TEST_F(SleeveFilterTests, SQLCompilationTest) {
  FieldCondition cond{
      .field = FilterField::ExifCameraModel,
      .op    = CompareOp::EQUALS,
      .value = std::wstring(L"Canon EOS 5D Mark IV"),
  };
  FilterNode root{FilterNode::Type::Condition, {}, {}, std::move(cond), std::nullopt};

  std::wstring sql = FilterSQLCompiler::Compile(root);
  std::wcout << L"Generated SQL: " << sql << std::endl;
  std::wstring expected_sql = L"(json_extract(metadata, '$.Model') = 'Canon EOS 5D Mark IV')";
  EXPECT_EQ(sql, expected_sql);
}

TEST_F(SleeveFilterTests, ComplexFilterSQLTest) {
  FieldCondition cond1{
      .field = FilterField::ExifCameraModel,
      .op    = CompareOp::EQUALS,
      .value = std::wstring(L"Nikon D850"),
  };
  FilterNode node1{FilterNode::Type::Condition, {}, {}, std::move(cond1), std::nullopt};

  FieldCondition cond2{
      .field = FilterField::FileExtension,
      .op    = CompareOp::ENDS_WITH,
      .value = std::wstring(L".NEF"),
  };
  FilterNode node2{FilterNode::Type::Condition, {}, {}, std::move(cond2), std::nullopt};

  FilterNode root{FilterNode::Type::Logical, FilterOp::AND, {node1, node2}, {}, std::nullopt};

  std::wstring sql = FilterSQLCompiler::Compile(root);
  std::wcout << L"Generated SQL for complex filter: " << sql << std::endl;
  std::wstring expected_sql =
      L"((json_extract(metadata, '$.Model') = 'Nikon D850') AND (UPPER(file_name) LIKE '%.NEF'))";
  EXPECT_EQ(sql, expected_sql);
}

TEST_F(SleeveFilterTests, BetweenConditionSQLTest) {
  FieldCondition cond{
      .field        = FilterField::ExifISO,
      .op           = CompareOp::BETWEEN,
      .value        = int64_t(100),
      .second_value = int64_t(800),
  };
  FilterNode root{FilterNode::Type::Condition, {}, {}, std::move(cond), std::nullopt};

  std::wstring sql = FilterSQLCompiler::Compile(root);
  std::wcout << L"Generated SQL for BETWEEN condition: " << sql << std::endl;
  std::wstring expected_sql = L"(json_extract(metadata, '$.ISO')::INT BETWEEN 100 AND 800)";
  EXPECT_EQ(sql, expected_sql);
}

TEST_F(SleeveFilterTests, FolderIndexTest_Model) {
  {
    SleeveManager manager(GetDBPath());
    image_path_t              path = std::string(TEST_IMG_PATH) + "/raw/batch";
    std::vector<image_path_t> imgs;
    for (const auto& img : std::filesystem::directory_iterator(path)) {
      if (!img.is_directory() && is_supported_file(img.path())) imgs.push_back(img.path());
    }
    manager.LoadToPath(imgs, L"");

    // Create a filter node
    FieldCondition cond{
        .field = FilterField::ExifCameraModel,
        .op    = CompareOp::CONTAINS,
        .value = std::wstring(L"D850"),
    };
    FilterNode   root{FilterNode::Type::Condition, {}, {}, std::move(cond), std::nullopt};

    std::wstring sql = FilterSQLCompiler::Compile(root);

    auto filter = std::make_shared<FilterCombo>(1, root);
    auto fs = manager.GetFilesystem();
    auto result = fs->ApplyFilterToFolder("", filter);
    EXPECT_EQ(result.size(), 5);  // Expecting 5 images matching the filter
  }
}

TEST_F(SleeveFilterTests, FolderIndexTest_FileExtension) {
  {
    SleeveManager manager(GetDBPath());
    image_path_t              path = std::string(TEST_IMG_PATH) + "/raw/batch";
    std::vector<image_path_t> imgs;
    for (const auto& img : std::filesystem::directory_iterator(path)) {
      if (!img.is_directory() && is_supported_file(img.path())) imgs.push_back(img.path());
    }
    manager.LoadToPath(imgs, L"");

    // Create a filter node
    FieldCondition cond{
        .field = FilterField::FileExtension,
        .op    = CompareOp::ENDS_WITH,
        .value = std::wstring(L".NEF"),
    };
    FilterNode   root{FilterNode::Type::Condition, {}, {}, std::move(cond), std::nullopt};

    std::wstring sql = FilterSQLCompiler::Compile(root);

    auto filter = std::make_shared<FilterCombo>(1, root);
    auto fs = manager.GetFilesystem();
    auto result = fs->ApplyFilterToFolder("", filter);
    EXPECT_EQ(result.size(), 5);  // Expecting 4 images matching the filter
  }
}

TEST_F(SleeveFilterTests, FolderIndexTest_Aperature) {
  {
    SleeveManager manager(GetDBPath());
    image_path_t              path = std::string(TEST_IMG_PATH) + "/raw/batch";
    std::vector<image_path_t> imgs;
    for (const auto& img : std::filesystem::directory_iterator(path)) {
      if (!img.is_directory() && is_supported_file(img.path())) imgs.push_back(img.path());
    }
    manager.LoadToPath(imgs, L"");

    // Create a filter node
    FieldCondition cond{
        .field = FilterField::ExifAperture,
        .op    = CompareOp::GREATER_THAN,
        .value = double(5.6),
    };
    FilterNode   root{FilterNode::Type::Condition, {}, {}, std::move(cond), std::nullopt};

    std::wstring sql = FilterSQLCompiler::Compile(root);

    auto filter = std::make_shared<FilterCombo>(1, root);
    auto fs = manager.GetFilesystem();
    auto result = fs->ApplyFilterToFolder("", filter);
    EXPECT_EQ(result.size(), 12);  // Expecting 12 images matching the filter
  }
}

TEST_F(SleeveFilterTests, FolderIndexTest_ISO) {
  {
    SleeveManager manager(GetDBPath());
    image_path_t              path = std::string(TEST_IMG_PATH) + "/raw/batch";
    std::vector<image_path_t> imgs;
    for (const auto& img : std::filesystem::directory_iterator(path)) {
      if (!img.is_directory() && is_supported_file(img.path())) imgs.push_back(img.path());
    }
    manager.LoadToPath(imgs, L"");

    // Create a filter node
    FieldCondition cond{
        .field = FilterField::ExifISO,
        .op    = CompareOp::BETWEEN,
        .value = int64_t(100),
        .second_value = int64_t(400),
    };
    FilterNode   root{FilterNode::Type::Condition, {}, {}, std::move(cond), std::nullopt};

    std::wstring sql = FilterSQLCompiler::Compile(root);

    auto filter = std::make_shared<FilterCombo>(1, root);
    auto fs = manager.GetFilesystem();
    auto result = fs->ApplyFilterToFolder("", filter);
    EXPECT_EQ(result.size(), 12);  // Expecting 12 images matching the filter
  }
}

TEST_F(SleeveFilterTests, FolderIndexTest_FocalLength) {
  {
    SleeveManager manager(GetDBPath());
    image_path_t              path = std::string(TEST_IMG_PATH) + "/raw/batch";
    std::vector<image_path_t> imgs;
    for (const auto& img : std::filesystem::directory_iterator(path)) {
      if (!img.is_directory() && is_supported_file(img.path())) imgs.push_back(img.path());
    }
    manager.LoadToPath(imgs, L"");

    // Create a filter node
    FieldCondition cond{
        .field = FilterField::ExifFocalLength,
        .op    = CompareOp::LESS_THAN,
        .value = double(150.0),
    };
    FilterNode   root{FilterNode::Type::Condition, {}, {}, std::move(cond), std::nullopt};

    std::wstring sql = FilterSQLCompiler::Compile(root);

    auto filter = std::make_shared<FilterCombo>(1, root);
    auto fs = manager.GetFilesystem();
    auto result = fs->ApplyFilterToFolder("", filter);
    EXPECT_EQ(result.size(), 13);  // Expecting 13 images matching the filter
  }
}

TEST_F(SleeveFilterTests, FolderIndexTest_Combined) {
  {
    SleeveManager manager(GetDBPath());
    image_path_t              path = std::string(TEST_IMG_PATH) + "/raw/batch";
    std::vector<image_path_t> imgs;
    for (const auto& img : std::filesystem::directory_iterator(path)) {
      if (!img.is_directory() && is_supported_file(img.path())) imgs.push_back(img.path());
    }
    manager.LoadToPath(imgs, L"");

    // Create a filter node
    FieldCondition cond1{
        .field = FilterField::ExifCameraModel,
        .op    = CompareOp::CONTAINS,
        .value = std::wstring(L"D850"),
    };
    FilterNode node1{FilterNode::Type::Condition, {}, {}, std::move(cond1), std::nullopt};

    FieldCondition cond2{
        .field = FilterField::ExifFocalLength,
        .op    = CompareOp::LESS_THAN,
        .value = double(150.0),
    };
    FilterNode node2{FilterNode::Type::Condition, {}, {}, std::move(cond2), std::nullopt};

    FilterNode root{FilterNode::Type::Logical, FilterOp::AND, {node1, node2}, {}, std::nullopt};

    std::wstring sql = FilterSQLCompiler::Compile(root);

    auto filter = std::make_shared<FilterCombo>(1, root);
    auto fs = manager.GetFilesystem();
    auto result = fs->ApplyFilterToFolder("", filter);
    EXPECT_EQ(result.size(), 1);  // Expecting 1 image matching the filter
  }
}

TEST_F(SleeveFilterTests, FolderIndexTest_NoMatch) {
  {
    SleeveManager manager(GetDBPath());
    image_path_t              path = std::string(TEST_IMG_PATH) + "/raw/batch";
    std::vector<image_path_t> imgs;
    for (const auto& img : std::filesystem::directory_iterator(path)) {
      if (!img.is_directory() && is_supported_file(img.path())) imgs.push_back(img.path());
    }
    manager.LoadToPath(imgs, L"");

    // Create a filter node
    FieldCondition cond{
        .field = FilterField::ExifCameraModel,
        .op    = CompareOp::CONTAINS,
        .value = std::wstring(L"A7"),
    };
    FilterNode   root{FilterNode::Type::Condition, {}, {}, std::move(cond), std::nullopt};

    std::wstring sql = FilterSQLCompiler::Compile(root);

    auto filter = std::make_shared<FilterCombo>(1, root);
    auto fs = manager.GetFilesystem();
    auto result = fs->ApplyFilterToFolder("", filter);
    EXPECT_EQ(result.size(), 0);  // Expecting no images matching the filter
  }
}

TEST_F(SleeveFilterTests, FolderIndexTest_DateRange) {
  {
    SleeveManager manager(GetDBPath());
    image_path_t              path = std::string(TEST_IMG_PATH) + "/raw/batch";
    std::vector<image_path_t> imgs;
    for (const auto& img : std::filesystem::directory_iterator(path)) {
      if (!img.is_directory() && is_supported_file(img.path())) imgs.push_back(img.path());
    }
    manager.LoadToPath(imgs, L"");

    // Create a filter node
    FieldCondition cond{
        .field = FilterField::CaptureDate,
        .op    = CompareOp::BETWEEN,
        .value = std::tm{0,0,0,1,0,125},  // Jan 1, 2025
        .second_value = std::tm{0,0,0,31,11,125}, // Dec 31, 2025
    };
    FilterNode   root{FilterNode::Type::Condition, {}, {}, std::move(cond), std::nullopt};

    std::wstring sql = FilterSQLCompiler::Compile(root);

    auto filter = std::make_shared<FilterCombo>(1, root);
    auto fs = manager.GetFilesystem();
    auto result = fs->ApplyFilterToFolder("", filter);
    EXPECT_EQ(result.size(), 13);  // Expecting 13 images matching the filter
  }
}