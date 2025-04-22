#include "sleeve/sleeve_base.hpp"
#include "sleeve/sleeve_element/sleeve_element.hpp"

#include <gtest/gtest.h>
#include <optional>

TEST(SleeveOperationTest, NormalTest1) {
  using namespace puerhlab;
  SleeveBase sl{0};

  sl.CreateElementToPath(L"root", L"test", ElementType::FOLDER);
  auto new_element = sl.AccessElementByPath(L"root/test");
  ASSERT_TRUE(new_element.has_value());
  ASSERT_EQ(new_element.value()->_element_id, 1);
  ASSERT_EQ(new_element.value()->_element_name, L"test");
  ASSERT_EQ(new_element.value()->_ref_count, 1);

  auto new_element_1 = sl.CreateElementToPath(L"root/test", L"test_file", ElementType::FILE);
  new_element_1 = sl.AccessElementByPath(L"root/test/test_file");
  ASSERT_TRUE(new_element_1.has_value());
  ASSERT_EQ(new_element_1.value()->_element_id, 2);
  ASSERT_EQ(new_element_1.value()->_element_name, L"test_file");
  ASSERT_EQ(new_element_1.value()->_ref_count, 1);

  auto new_element_2 = sl.CreateElementToPath(L"root", L"test_file", ElementType::FILE);
  new_element_2 = sl.AccessElementByPath(L"root/test_file");
  ASSERT_TRUE(new_element_2.has_value());
  ASSERT_EQ(new_element_2.value()->_element_id, 3);
  ASSERT_EQ(new_element_2.value()->_element_name, L"test_file");
  ASSERT_EQ(new_element_2.value()->_ref_count, 1);
}

TEST(SleeveOperationTest, NormalTest2) {
  using namespace puerhlab;
  SleeveBase sl{0};

  sl.CreateElementToPath(L"root", L"test", ElementType::FOLDER);

  sl.CreateElementToPath(L"root/test", L"test_file", ElementType::FILE);

  auto new_element_1 = sl.RemoveElementInPath(L"root/test", L"test_file");
  ASSERT_TRUE(new_element_1.has_value());
  ASSERT_EQ(new_element_1.value()->_element_id, 2);
  ASSERT_EQ(new_element_1.value()->_element_name, L"test_file");
  ASSERT_EQ(new_element_1.value()->_ref_count, 0);

  ASSERT_EQ(sl.AccessElementByPath(L"root/test/test_file"), std::nullopt);

  auto new_element_2 = sl.RemoveElementInPath(L"root", L"test");
  ASSERT_TRUE(new_element_2.has_value());
  ASSERT_EQ(new_element_2.value()->_element_id, 1);
  ASSERT_EQ(new_element_2.value()->_element_name, L"test");
  ASSERT_EQ(new_element_2.value()->_ref_count, 0);
  ASSERT_EQ(sl.AccessElementByPath(L"root/test"), std::nullopt);
}

TEST(SleeveOperationTest, EdgeTest1) {
  using namespace puerhlab;
}