#include <gtest/gtest.h>

#include "sleeve/sleeve_filter/filter_combo.hpp"
#include "sleeve_test_fixture.hpp"

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
  std::wstring expected_sql = L"(json_extract_scalar(metadata, '$.Model') = 'Canon EOS 5D Mark IV')";
  EXPECT_EQ(sql, expected_sql);
}