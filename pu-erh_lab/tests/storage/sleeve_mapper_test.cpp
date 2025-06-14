#include "storage/mapper/sleeve/sleeve_mapper.hpp"

#include <gtest/gtest.h>
#include <easy/profiler.h>
#include <exception>
#include <exiv2/error.hpp>
#include <filesystem>
#include <stdexcept>

#include "storage/controller/db_controller.hpp"

using namespace puerhlab;

std::filesystem::path db_path("D:\\Projects\\pu-erh_lab\\pu-erh_lab\\tests\\resources\\temp_folder\\test.db");
TEST(SleeveMapperTest, InitTest1) {
  if (std::filesystem::exists(db_path)) {
    std::filesystem::remove(db_path);
  }
  {
    try {
      DBController db_ctr{db_path};
      db_ctr.InitializeDB();
    } catch (std::exception &e) {
      std::cout << e.what() << std::endl;
    }
  }
  std::filesystem::remove(db_path);
}

TEST(SleeveMapperTest, DISABLED_SimpleCaptureTest1) {
}

TEST(SleeveMapperTest, SimpleCaptureTest2) {

}