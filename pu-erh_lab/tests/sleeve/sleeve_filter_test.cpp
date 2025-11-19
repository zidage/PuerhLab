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
      .value = std::wstring(L".nef"),
  };
  FilterNode node2{FilterNode::Type::Condition, {}, {}, std::move(cond2), std::nullopt};

  FilterNode root{FilterNode::Type::Logical, FilterOp::AND, {node1, node2}, {}, std::nullopt};

  std::wstring sql = FilterSQLCompiler::Compile(root);
  std::wcout << L"Generated SQL for complex filter: " << sql << std::endl;
  std::wstring expected_sql =
      L"((json_extract(metadata, '$.Model') = 'Nikon D850') AND (LOWER(file_name) LIKE '%.nef'))";
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
  std::wstring expected_sql = L"(json_extract(metadata, '$.ISO') BETWEEN 100 AND 800)";
  EXPECT_EQ(sql, expected_sql);
}

TEST_F(SleeveFilterTests, FolderIndexTest) {
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
    EXPECT_EQ(result.size(), 4);  // Expecting 4 images matching the filter
  }
} 