#include <gtest/gtest.h>
#include <exception>
#include <filesystem>

#include "sleeve/sleeve_element/sleeve_element.hpp"
#include "sleeve/sleeve_filesystem.hpp"

std::filesystem::path db_path(
    "D:\\Projects\\pu-erh_lab\\pu-erh_lab\\tests\\resources\\temp_folder\\test.db");

namespace puerhlab {
TEST(SleeveFSTest, SimpleTest1) {
{
  try {
  FileSystem fs{db_path, 0};
  fs.InitRoot();
  } catch (std::exception& e) {
    FAIL();
  }
}
  if (std::filesystem::exists(db_path)) {
    std::filesystem::remove(db_path);
  }
}

TEST(SleeveFSTest, SimpleTest2) {
{
  try {
  FileSystem fs{db_path, 0};
  fs.InitRoot();

  fs.Create(L"", L"FILE", ElementType::FILE);
  auto new_file = fs.Get("/FILE", false);
  EXPECT_FALSE(new_file == nullptr);
  EXPECT_EQ(new_file->_element_name, L"FILE");
  EXPECT_EQ(new_file->_element_id, 1);
  EXPECT_EQ(new_file->_ref_count, 1);
  } catch (std::exception& e) {
    FAIL();
  }
}
  if (std::filesystem::exists(db_path)) {
    std::filesystem::remove(db_path);
  }
}
};