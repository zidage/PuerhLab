#include "storage/mapper/sleeve/sleeve_mapper.hpp"

#include <easy/profiler.h>
#include <gtest/gtest.h>

#include <exception>
#include <exiv2/error.hpp>
#include <filesystem>
#include <stdexcept>

#include "sleeve/sleeve_manager.hpp"
#include "storage/controller/db_controller.hpp"
#include "storage/controller/image/image_controller.hpp"
#include "storage/mapper/duckorm/duckdb_types.hpp"
#include "type/type.hpp"

using namespace puerhlab;

std::filesystem::path db_path(
    "D:\\Projects\\pu-erh_lab\\pu-erh_lab\\tests\\resources\\temp_folder\\test.db");
TEST(SleeveMapperTest, InitTest1) {
  if (std::filesystem::exists(db_path)) {
    std::filesystem::remove(db_path);
  }
  {
    try {
      DBController db_ctr{db_path};
      db_ctr.InitializeDB();
    } catch (std::exception& e) {
      std::cout << e.what() << std::endl;
    }
  }
  std::filesystem::remove(db_path);
}

TEST(SleeveMapperTest, SimpleCaptureTest1) {
  if (std::filesystem::exists(db_path)) {
    std::filesystem::remove(db_path);
  }
  {
    try {
      Exiv2::LogMsg::setLevel(Exiv2::LogMsg::Level::mute);
      DBController db_ctr{db_path};
      db_ctr.InitializeDB();
      ImageController img_ctr{db_ctr.GetConnectionGuard()};

      SleeveManager   manager{};
      image_path_t    path =
          L"D:\\Projects\\pu-erh_lab\\pu-erh_"
          L"lab\\tests\\resources\\sample_images\\jpg";
      std::vector<image_path_t> imgs;
      for (const auto& img : std::filesystem::directory_iterator(path)) {
        imgs.push_back(img.path());
      }
      manager.LoadToPath(imgs, L"root");
      img_ctr.CaptureImagePool(manager.GetPool());
    } catch (std::exception& e) {
      std::cout << e.what() << std::endl;
      FAIL();
    }
  }
}

TEST(SleeveMapperTest, SimpleCaptureTest2) {}