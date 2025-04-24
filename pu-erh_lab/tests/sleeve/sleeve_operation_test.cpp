#include <gtest/gtest.h>

#include <codecvt>
#include <optional>
#include <string>

#include "gtest/gtest.h"
#include "sleeve/sleeve_base.hpp"
#include "sleeve/sleeve_element/sleeve_element.hpp"


TEST(SleeveOperationTest, NormalTest1) {
  using namespace puerhlab;
  SleeveBase sl{0};

  sl.CreateElementToPath(L"root", L"test", ElementType::FOLDER);
  auto element = sl.AccessElementByPath(L"root/test");
  ASSERT_TRUE(element.has_value());
  ASSERT_EQ(element.value()->_element_id, 1);
  ASSERT_EQ(element.value()->_element_name, L"test");
  ASSERT_EQ(element.value()->_ref_count, 1);

  auto element_1 = sl.CreateElementToPath(L"root/test", L"test_file", ElementType::FILE);
  element_1      = sl.AccessElementByPath(L"root/test/test_file");
  ASSERT_TRUE(element_1.has_value());
  ASSERT_EQ(element_1.value()->_element_id, 2);
  ASSERT_EQ(element_1.value()->_element_name, L"test_file");
  ASSERT_EQ(element_1.value()->_ref_count, 1);

  auto element_2 = sl.CreateElementToPath(L"root", L"test_file", ElementType::FILE);
  element_2      = sl.AccessElementByPath(L"root/test_file");
  ASSERT_TRUE(element_2.has_value());
  ASSERT_EQ(element_2.value()->_element_id, 3);
  ASSERT_EQ(element_2.value()->_element_name, L"test_file");
  ASSERT_EQ(element_2.value()->_ref_count, 1);

  std::wstring                                     tree = sl.Tree(L"root");
  std::wstring_convert<std::codecvt_utf8<wchar_t>> conv;
  std::cout << conv.to_bytes(tree) << std::endl;
}

TEST(SleeveOperationTest, NormalTest2) {
  using namespace puerhlab;
  SleeveBase sl{0};

  sl.CreateElementToPath(L"root", L"test", ElementType::FOLDER);

  sl.CreateElementToPath(L"root/test", L"test_file", ElementType::FILE);

  auto element_1 = sl.RemoveElementInPath(L"root/test", L"test_file");
  ASSERT_TRUE(element_1.has_value());
  ASSERT_EQ(element_1.value()->_element_id, 2);
  ASSERT_EQ(element_1.value()->_element_name, L"test_file");
  ASSERT_EQ(element_1.value()->_ref_count, 0);

  ASSERT_EQ(sl.AccessElementByPath(L"root/test/test_file"), std::nullopt);

  auto element_2 = sl.RemoveElementInPath(L"root", L"test");
  ASSERT_TRUE(element_2.has_value());
  ASSERT_EQ(element_2.value()->_element_id, 1);
  ASSERT_EQ(element_2.value()->_element_name, L"test");
  ASSERT_EQ(element_2.value()->_ref_count, 0);
  ASSERT_EQ(sl.AccessElementByPath(L"root/test"), std::nullopt);

  std::wstring                                     tree = sl.Tree(L"root");
  std::wstring_convert<std::codecvt_utf8<wchar_t>> conv;
  std::cout << conv.to_bytes(tree) << std::endl;
}

/**
 * @brief Edge case 1: create element under a file instead of a folder
 *
 */
TEST(SleeveOperationTest, EdgeTest1) {
  using namespace puerhlab;
  SleeveBase sl{0};

  sl.CreateElementToPath(L"root", L"test", ElementType::FOLDER);

  sl.CreateElementToPath(L"root/test", L"test", ElementType::FILE);

  auto element_1 = sl.CreateElementToPath(L"root/test", L"test", ElementType::FOLDER);
  ASSERT_FALSE(element_1.has_value());
}

/**
 * @brief Edge case 2: create a repeatedly named file to the same folder
 *
 */
TEST(SleeveOperationTest, EdgeTest2) {
  using namespace puerhlab;
  SleeveBase sl{0};

  sl.CreateElementToPath(L"root", L"test", ElementType::FOLDER);

  sl.CreateElementToPath(L"root/test", L"test", ElementType::FOLDER);

  auto element_1 = sl.CreateElementToPath(L"root/test", L"test", ElementType::FOLDER);
  ASSERT_FALSE(element_1.has_value());
}

/**
 * @brief Create duplicated named elements to different locations
 *
 */
TEST(SleeveOperationTest, EdgeTest3) {
  using namespace puerhlab;
  SleeveBase sl{0};

  sl.CreateElementToPath(L"root", L"test", ElementType::FOLDER);

  auto element_1 = sl.CreateElementToPath(L"root/test", L"test", ElementType::FOLDER);
  ASSERT_TRUE(element_1.has_value());
  ASSERT_EQ(element_1.value()->_element_name, L"test");
  ASSERT_EQ(element_1.value()->_ref_count, 1);

  auto element_2 = sl.CreateElementToPath(L"root/test/test", L"test", ElementType::FILE);
  ASSERT_TRUE(element_2.has_value());
  ASSERT_EQ(element_2.value()->_element_name, L"test");
  ASSERT_EQ(element_2.value()->_ref_count, 1);

  std::wstring                                     tree = sl.Tree(L"root");
  std::wstring_convert<std::codecvt_utf8<wchar_t>> conv;
  std::cout << conv.to_bytes(tree) << std::endl;
}

/**
 * @brief Test the correctness of the file copy functionality
 *
 */
TEST(SleeveOperationTest, CopyTest1) {
  using namespace puerhlab;
  SleeveBase sl{0};

  sl.CreateElementToPath(L"root", L"monday", ElementType::FOLDER);
  sl.CreateElementToPath(L"root", L"tuesday", ElementType::FOLDER);

  auto element_1 = sl.CreateElementToPath(L"root/monday", L"broken", ElementType::FILE);
  ASSERT_TRUE(element_1.has_value());
  ASSERT_EQ(element_1.value()->_element_name, L"broken");
  ASSERT_EQ(element_1.value()->_ref_count, 1);

  auto element_2 = sl.CreateElementToPath(L"root/tuesday", L"hope", ElementType::FILE);
  ASSERT_TRUE(element_2.has_value());
  ASSERT_EQ(element_2.value()->_element_name, L"hope");
  ASSERT_EQ(element_2.value()->_ref_count, 1);

  auto element_3 = sl.CopyElement(L"root/monday/broken", L"root/tuesday");
  ASSERT_TRUE(element_3.has_value());
  ASSERT_EQ(element_3.value()->_element_name, L"broken");
  ASSERT_EQ(element_3.value()->_ref_count, 2);

  std::wstring                                     tree = sl.Tree(L"root");
  std::wstring_convert<std::codecvt_utf8<wchar_t>> conv;
  std::cout << conv.to_bytes(tree) << std::endl;
}

/**
 * @brief Test the correctness of the file copy functionality
 *
 */
TEST(SleeveOperationTest, CopyTest2) {
  using namespace puerhlab;
  SleeveBase sl{0};

  sl.CreateElementToPath(L"root", L"monday", ElementType::FOLDER);
  sl.CreateElementToPath(L"root", L"tuesday", ElementType::FOLDER);

  auto element_1 = sl.CreateElementToPath(L"root/monday", L"broken", ElementType::FILE);
  ASSERT_TRUE(element_1.has_value());
  ASSERT_EQ(element_1.value()->_element_name, L"broken");
  ASSERT_EQ(element_1.value()->_ref_count, 1);

  auto element_2 = sl.CreateElementToPath(L"root/tuesday", L"hope", ElementType::FILE);
  ASSERT_TRUE(element_2.has_value());
  ASSERT_EQ(element_2.value()->_element_name, L"hope");
  ASSERT_EQ(element_2.value()->_ref_count, 1);

  auto element_3 = sl.CopyElement(L"root/monday/broken", L"root/tuesday");
  ASSERT_TRUE(element_3.has_value());
  ASSERT_EQ(element_3.value()->_element_name, L"broken");
  ASSERT_EQ(element_3.value()->_ref_count, 2);

  auto element_4 = sl.CopyElement(L"root/tuesday", L"root/monday");
  ASSERT_TRUE(element_4.has_value());
  ASSERT_EQ(element_4.value()->_element_name, L"broken");
  ASSERT_EQ(element_4.value()->_ref_count, 1);

  auto element_5 = sl.AccessElementByPath(L"root/monday/tuesday/broken");
  ASSERT_TRUE(element_5.has_value());
  ASSERT_EQ(element_5.value()->_element_name, L"broken");
  ASSERT_EQ(element_5.value()->_ref_count, 3);

  std::wstring                                     tree = sl.Tree(L"root");
  std::wstring_convert<std::codecvt_utf8<wchar_t>> conv;
  std::cout << conv.to_bytes(tree) << std::endl;
}