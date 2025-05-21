#include "storage/mapper/sleeve/sleeve_mapper.hpp"

#include <gtest/gtest.h>
#include <filesystem>


using namespace puerhlab;

std::filesystem::path db_path("D:\\Projects\\pu-erh_lab\\pu-erh_lab\\tests\\resources\\temp_folder\\test.db");
TEST(SleeveMapperTest, InitTest1) {
  {
    SleeveMapper mapper{db_path};
    mapper.InitDB();
    EXPECT_TRUE(std::filesystem::exists(db_path));
  }
  std::filesystem::remove(db_path.string());
}