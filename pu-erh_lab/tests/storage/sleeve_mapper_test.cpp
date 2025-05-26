#include "storage/mapper/sleeve/sleeve_mapper.hpp"
#include "sleeve/sleeve_manager.hpp"

#include <gtest/gtest.h>

#include <exception>
#include <filesystem>
#include <stdexcept>

using namespace puerhlab;

std::filesystem::path db_path("D:\\Projects\\pu-erh_lab\\pu-erh_lab\\tests\\resources\\temp_folder\\test.db");
TEST(SleeveMapperTest, DISABLED_InitTest1) {
  {
    try {
    SleeveMapper mapper{db_path};
    mapper.InitDB();
    EXPECT_TRUE(std::filesystem::exists(db_path));
    } catch (std::exception &e) {
      std::cout << e.what() << std::endl;    
    }
  }
  std::filesystem::remove(db_path.string());
}

TEST(SleeveMapperTest, DISABLED_SimpleCaptureTest1) {
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
      mapper.CaptureSleeve(manager.GetBase());
    } catch (std::exception &e) {
      std::cout << e.what() << std::endl;
    }
  }
  std::filesystem::remove(db_path.string());
}

TEST(SleeveMapperTest, SimpleCaptureTest2) {
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
      mapper.CaptureImagePool(manager.GetPool());
    } catch (std::exception &e) {
      std::cout << e.what() << std::endl;
    }
  }
  // std::filesystem::remove(db_path.string());
}