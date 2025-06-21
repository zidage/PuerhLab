#include <gtest/gtest.h>

#include <exception>
#include <filesystem>
#include <stdexcept>

#include "sleeve/sleeve_element/sleeve_element.hpp"
#include "sleeve/sleeve_filesystem.hpp"
#include "utils/clock/time_provider.hpp"
#include "utils/string/convert.hpp"

std::filesystem::path db_path(
    "D:\\Projects\\pu-erh_lab\\pu-erh_lab\\tests\\resources\\temp_folder\\test.db");

namespace puerhlab {
TEST(SleeveFSTest, InitTest1) {
  TimeProvider::Refresh();
  {
    try {
      FileSystem fs{db_path, 0};
      fs.InitRoot();
    } catch (std::exception& e) {
      std::cout << e.what() << std::endl;
      FAIL();
    }
  }
  if (std::filesystem::exists(db_path)) {
    std::filesystem::remove(db_path);
  }
}

TEST(SleeveFSTest, AddGetTest1) {
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
      std::cout << e.what() << std::endl;
      FAIL();
    }
  }
  if (std::filesystem::exists(db_path)) {
    std::filesystem::remove(db_path);
  }
}

TEST(SleeveFSTest, AddGetTest2) {
  try {
    FileSystem fs{db_path, 0};
    fs.InitRoot();

    fs.Create(L"", L"Folder", ElementType::FOLDER);
    fs.Create(L"/Folder", L"Subfolder", ElementType::FOLDER);
    fs.Create(L"/Folder", L"File", ElementType::FILE);

    auto folder = fs.Get(L"/Folder", false);
    EXPECT_FALSE(folder == nullptr);
    auto subfolder = fs.Get(L"/Folder/Subfolder", false);
    EXPECT_FALSE(subfolder == nullptr);
    auto file = fs.Get(L"/Folder/File", false);
    EXPECT_FALSE(file == nullptr);
    
    std::cout << conv::ToBytes(fs.Tree(L"/"));
  } catch (std::exception& e) {
    std::cout << e.what() << std::endl;
    FAIL();
  }

  if (std::filesystem::exists(db_path)) {
    std::filesystem::remove(db_path);
  }
}

TEST(SleeveFSTest, DISABLED_ReInitTest1) {
  {
    try {
      FileSystem fs{db_path, 0};
      fs.InitRoot();

      fs.Create(L"", L"Folder", ElementType::FOLDER);
      fs.Create(L"/Folder", L"Subfolder", ElementType::FOLDER);
      fs.Create(L"/Folder", L"File", ElementType::FILE);

      auto folder = fs.Get(L"/Folder", false);
      EXPECT_FALSE(folder == nullptr);
      auto subfolder = fs.Get(L"/Folder/Subfolder", false);
      EXPECT_FALSE(subfolder == nullptr);
      auto file = fs.Get(L"/Folder/File", false);
      EXPECT_FALSE(file == nullptr);
    } catch (std::exception& e) {
      std::cout << e.what() << std::endl;
      FAIL();
    }
  }

  // Sync function has not been implemented
  {
    try {
      // Auto recovered start id has not been implemented
      FileSystem fs{db_path, 3};
      fs.InitRoot();

      auto folder = fs.Get(L"/Folder", false);
      EXPECT_FALSE(folder == nullptr);
      auto subfolder = fs.Get(L"/Folder/Subfolder", false);
      EXPECT_FALSE(subfolder == nullptr);
      auto file = fs.Get(L"/Folder/File", false);
      EXPECT_FALSE(file == nullptr);
    } catch (std::exception& e) {
      std::cout << e.what() << std::endl;
      FAIL();
    }
  }
}

TEST(SleeveFSTest, DeleteTest1) {
  try {
    FileSystem fs{db_path, 0};
    fs.InitRoot();

    fs.Create(L"", L"FILE", ElementType::FILE);
    auto new_file = fs.Get("/FILE", false);
    EXPECT_FALSE(new_file == nullptr);
    EXPECT_EQ(new_file->_element_name, L"FILE");
    EXPECT_EQ(new_file->_element_id, 1);
    EXPECT_EQ(new_file->_ref_count, 1);

    fs.Delete(L"/FILE");
    fs.Get("/FILE", false);
  } catch (std::runtime_error& e) {
    std::cout << "Expected exception: " << e.what() << std::endl;
  } catch (std::exception& e) {
    std::cout << e.what() << std::endl;
    FAIL();
  }

  if (std::filesystem::exists(db_path)) {
    std::filesystem::remove(db_path);
  }
}

TEST(SleeveFSTest, DeleteTest2) {
  try {
    FileSystem fs{db_path, 0};
    fs.InitRoot();

    fs.Create(L"", L"Folder", ElementType::FOLDER);
    fs.Create(L"/Folder", L"Subfolder", ElementType::FOLDER);
    fs.Create(L"/Folder", L"File", ElementType::FILE);

    fs.Delete(L"/Folder");
    fs.Get("/Folder/Subfolder", false);
  } catch (std::runtime_error& e) {
    std::cout << "Expected exception: " << e.what() << std::endl;
  } catch (std::exception& e) {
    std::cout << e.what() << std::endl;
    FAIL();
  }

  if (std::filesystem::exists(db_path)) {
    std::filesystem::remove(db_path);
  }
}

TEST(SleeveFSTest, CopyTest1) {
  try {
    FileSystem fs{db_path, 0};
    fs.InitRoot();

    fs.Create(L"", L"Folder", ElementType::FOLDER);
    fs.Create(L"/Folder", L"Subfolder", ElementType::FOLDER);
    fs.Create(L"/Folder/Subfolder", L"Linux", ElementType::FILE);
    fs.Create(L"/Folder", L"File", ElementType::FILE);
    fs.Copy(L"/Folder/Subfolder", L"/");

    std::cout << conv::ToBytes(fs.Tree(L"/"));
  } catch (std::runtime_error& e) {
    std::cout << "Expected exception: " << e.what() << std::endl;
  } catch (std::exception& e) {
    std::cout << e.what() << std::endl;
    FAIL();
  }

  if (std::filesystem::exists(db_path)) {
    std::filesystem::remove(db_path);
  }
}

TEST(SleeveFSTest, CoWTest1) {
  try {
    FileSystem fs{db_path, 0};
    fs.InitRoot();

    fs.Create(L"", L"Folder", ElementType::FOLDER);
    fs.Create(L"/Folder", L"Subfolder", ElementType::FOLDER);
    fs.Create(L"/Folder/Subfolder", L"Linux", ElementType::FILE);
    fs.Create(L"/Folder", L"File", ElementType::FILE);
    fs.Copy(L"/Folder/Subfolder", L"/");

    fs.Create(L"/Folder/Subfolder", L"Windows", ElementType::FILE);

    std::cout << conv::ToBytes(fs.Tree(L"/"));
  } catch (std::runtime_error& e) {
    std::cout << "Expected exception: " << e.what() << std::endl;
  } catch (std::exception& e) {
    std::cout << e.what() << std::endl;
    FAIL();
  }

  if (std::filesystem::exists(db_path)) {
    std::filesystem::remove(db_path);
  }
}
};  // namespace puerhlab