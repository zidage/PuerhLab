#include "storage/mapper/sleeve/sleeve_mapper.hpp"

#include <gtest/gtest.h>
#include <easy/profiler.h>
#include <exception>
#include <exiv2/error.hpp>
#include <filesystem>
#include <stdexcept>

#include "sleeve/sleeve_manager.hpp"


using namespace puerhlab;

std::filesystem::path db_path("D:\\Projects\\pu-erh_lab\\pu-erh_lab\\tests\\resources\\temp_folder\\test.db");
TEST(SleeveMapperTest, DISABLED_InitTest1) {
  if (std::filesystem::exists(db_path)) {
    std::filesystem::remove(db_path);
  }
  {
    try {
      SleeveMapper mapper{db_path};
      mapper.InitDB();
      EXPECT_TRUE(std::filesystem::exists(db_path));
    } catch (std::exception &e) {
      std::cout << e.what() << std::endl;
    }
  }
  std::filesystem::remove(db_path);
}

TEST(SleeveMapperTest, DISABLED_SimpleCaptureTest1) {
  if (std::filesystem::exists(db_path)) {
    std::filesystem::remove(db_path);
  }
  {
    SleeveMapper  mapper{db_path};
    SleeveManager manager{};
    image_path_t  path =
        L"D:\\Projects\\pu-erh_lab\\pu-erh_"
        L"lab\\tests\\resources\\sample_images\\jpg";
    std::vector<image_path_t> imgs;
    for (const auto &img : std::filesystem::directory_iterator(path)) {
      imgs.push_back(img.path());
    }
    manager.LoadToPath(imgs, L"root");
    try {
      mapper.InitDB();
      mapper.CaptureSleeve(manager.GetBase(), manager.GetPool());
    } catch (std::exception &e) {
      std::cout << e.what() << std::endl;
    }
  }
  std::filesystem::remove(db_path);
}

TEST(SleeveMapperTest, SimpleCaptureTest2) {
  EASY_PROFILER_ENABLE;
  if (std::filesystem::exists(db_path)) {
    std::filesystem::remove(db_path);
  }
  {
    Exiv2::LogMsg::setLevel(Exiv2::LogMsg::Level::mute);
    SleeveMapper  mapper{db_path};
    SleeveManager manager{};
    image_path_t  path =
        L"D:\\Projects\\pu-erh_lab\\pu-erh_"
        L"lab\\tests\\resources\\sample_images\\dng_100";
    std::vector<image_path_t> imgs;
    for (const auto &img : std::filesystem::directory_iterator(path)) {
      imgs.push_back(img.path());
    }
    manager.LoadToPath(imgs, L"root");
    try {
      mapper.InitDB();
      mapper.CaptureSleeve(manager.GetBase(), manager.GetPool());
    } catch (std::exception &e) {
      std::cout << e.what() << std::endl;
    }
  }
  profiler::dumpBlocksToFile("D:\\Projects\\pu-erh_lab\\pu-erh_lab\\tests\\resources\\temp_folder\\test_profile.prof");
  // std::filesystem::remove(db_path.string());
}