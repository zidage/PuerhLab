
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
TEST_DB_PATH);

std::filesystem::path raw_path(
    std::string(TEST_IMG_PATH) + std::string("/raw/batch"));
TEST(SleeveMapperTest, DISABLED_InitTest1) {
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
  // std::filesystem::remove(db_path);
}

TEST(SleeveMapperTest, SimpleCaptureTest1) {
  if (std::filesystem::exists(db_path)) {
    std::filesystem::remove(db_path);
  }
  {
    try {
      Exiv2::LogMsg::setLevel(Exiv2::LogMsg::Level::mute);

      // DBController db_ctr{db_path};
      // db_ctr.InitializeDB();
      // ImageController img_ctr{db_ctr.GetConnectionGuard()};

      SleeveManager   manager{db_path};
      std::vector<image_path_t> imgs;
      for (const auto& img : std::filesystem::directory_iterator(raw_path)) {
        imgs.push_back(img.path());
      }
      manager.LoadToPath(imgs, L"");

      // img_ctr.CaptureImagePool(manager.GetPool());
    } catch (std::exception& e) {
      std::cout << e.what() << std::endl;
      FAIL();
    }
  }
}

TEST(SleeveMapperTest, DISABLED_SimpleCaptureTest2) {}