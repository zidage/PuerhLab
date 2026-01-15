#include "app/sleeve_service.hpp"

#include <gtest/gtest.h>

#include <filesystem>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "sleeve/sleeve_element/sleeve_element.hpp"
#include "utils/clock/time_provider.hpp"
#include "utils/string/convert.hpp"


namespace puerhlab {
class SleeveServiceTests : public ::testing::Test {
 protected:
  std::filesystem::path db_path_;
  std::filesystem::path meta_path_;

  void                  SetUp() override {
    TimeProvider::Refresh();
    db_path_ = std::filesystem::temp_directory_path() / "sleeve_service_test.db";
    meta_path_ = std::filesystem::temp_directory_path() / "sleeve_service_test.json";
    if (std::filesystem::exists(db_path_)) {
      std::filesystem::remove(db_path_);
    }
    if (std::filesystem::exists(meta_path_)) {
      std::filesystem::remove(meta_path_);
    }
  }

  void TearDown() override {
    if (std::filesystem::exists(db_path_)) {
      std::filesystem::remove(db_path_);
    }
    if (std::filesystem::exists(meta_path_)) {
      std::filesystem::remove(meta_path_);
    }
  }
};

TEST_F(SleeveServiceTests, InitAndCreateTest) {
  SleeveServiceImpl service(db_path_, meta_path_, 0);

  auto              write_result = service.Write<std::shared_ptr<SleeveElement>>(
      [](FileSystem& fs) { return fs.Create(L"", L"Folder", ElementType::FOLDER); });
  EXPECT_NE(write_result.first, nullptr);
  EXPECT_TRUE(write_result.second.success_);

  service.Write<std::shared_ptr<SleeveElement>>(
      [](FileSystem& fs) { return fs.Create(L"/Folder", L"File", ElementType::FILE); });

  auto file = service.Read<std::shared_ptr<SleeveElement>>(
      [](FileSystem& fs) { return fs.Get(L"/Folder/File", false); });
  ASSERT_NE(file, nullptr);
  EXPECT_EQ(file->element_name_, L"File");
}

TEST_F(SleeveServiceTests, DeleteTest) {
  SleeveServiceImpl service(db_path_, meta_path_, 0);

  service.Write<std::shared_ptr<SleeveElement>>(
      [](FileSystem& fs) { return fs.Create(L"", L"File", ElementType::FILE); });
  service.Write<bool>([](FileSystem& fs) {
    fs.Delete(L"/File");
    return true;
  });

  EXPECT_THROW(service.Read<std::shared_ptr<SleeveElement>>(
                   [](FileSystem& fs) { return fs.Get(L"/File", false); }),
               std::runtime_error);
}

TEST_F(SleeveServiceTests, CopyTest) {
  SleeveServiceImpl service(db_path_, meta_path_, 0);

  service.Write<std::shared_ptr<SleeveElement>>(
      [](FileSystem& fs) { return fs.Create(L"", L"Folder", ElementType::FOLDER); });
  service.Write<std::shared_ptr<SleeveElement>>(
      [](FileSystem& fs) { return fs.Create(L"/Folder", L"Subfolder", ElementType::FOLDER); });
  service.Write<std::shared_ptr<SleeveElement>>(
      [](FileSystem& fs) { return fs.Create(L"/Folder/Subfolder", L"Linux", ElementType::FILE); });

  service.Write<bool>([](FileSystem& fs) {
    fs.Copy(L"/Folder/Subfolder", L"/");
    return true;
  });

  auto tree     = service.Read<std::wstring>([](FileSystem& fs) { return fs.Tree(L"/"); });
  auto tree_str = conv::ToBytes(tree);
  EXPECT_NE(tree_str.find("Subfolder"), std::string::npos);
  EXPECT_NE(tree_str.find("Linux"), std::string::npos);
}

TEST_F(SleeveServiceTests, SaveLoadTest) {
  {
    SleeveServiceImpl service(db_path_, meta_path_, 0);
    service.Write<std::shared_ptr<SleeveElement>>(
        [](FileSystem& fs) { return fs.Create(L"", L"Folder", ElementType::FOLDER); });
    service.Write<std::shared_ptr<SleeveElement>>(
        [](FileSystem& fs) { return fs.Create(L"/Folder", L"File", ElementType::FILE); });

    service.SaveSleeve(meta_path_);
  }

  SleeveServiceImpl reloaded_service(meta_path_);
  auto              file = reloaded_service.Read<std::shared_ptr<SleeveElement>>(
      [](FileSystem& fs) { return fs.Get(L"/Folder/File", false); });
  ASSERT_NE(file, nullptr);
  EXPECT_EQ(file->element_name_, L"File");
}

TEST_F(SleeveServiceTests, FuzzyCreateCopyTest) {
  std::wstring first_tree;
  {
    SleeveServiceImpl         service(db_path_, meta_path_, 0);

    std::mt19937              gen(42);
    std::vector<std::wstring> known_paths;
    std::vector<std::wstring> known_folders;
    known_paths.push_back(L"");
    known_folders.push_back(L"");

    auto generate_name = [&gen](int length = 8) {
      static const std::wstring chars =
          L"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
      std::uniform_int_distribution<> dist(0, static_cast<int>(chars.size() - 1));
      std::wstring                    result;
      for (int i = 0; i < length; ++i) {
        result += chars[dist(gen)];
      }
      return result;
    };

    constexpr int kOperations = 200;
    for (int i = 0; i < kOperations; ++i) {
      std::uniform_int_distribution<> op_dist(0, 1);
      int                             op = op_dist(gen);

      if (known_paths.size() < 2 && op == 1) {
        op = 0;
      }

      if (op == 0) {
        std::uniform_int_distribution<size_t> parent_dist(0, known_folders.size() - 1);
        std::wstring                          parent = known_folders[parent_dist(gen)];
        std::wstring                          name   = generate_name();
        ElementType type = (gen() % 2 == 0) ? ElementType::FOLDER : ElementType::FILE;

        try {
          service.Write<std::shared_ptr<SleeveElement>>(
              [&](FileSystem& fs) { return fs.Create(parent, name, type); });
          std::wstring new_path = parent + L"/" + name;
          known_paths.push_back(new_path);
          if (type == ElementType::FOLDER) {
            known_folders.push_back(new_path);
          }
        } catch (const std::exception&) {
          // Ignore invalid ops to keep fuzz running
        }
      } else {
        std::uniform_int_distribution<size_t> from_dist(0, known_paths.size() - 1);
        std::uniform_int_distribution<size_t> dest_dist(0, known_folders.size() - 1);
        std::wstring                          from_path = known_paths[from_dist(gen)];
        std::wstring                          to_parent = known_folders[dest_dist(gen)];

        try {
          service.Write<bool>([&](FileSystem& fs) {
            fs.Copy(from_path, to_parent);
            return true;
          });
        } catch (const std::exception&) {
          // Ignore invalid ops to keep fuzz running
        }
      }
      std::cout << "\r\033[2KCompleted operation " << (i + 1) << " / " << kOperations << std::flush;
    }

    first_tree = service.Read<std::wstring>([](FileSystem& fs) { return fs.Tree(L"/"); });
    service.SaveSleeve(meta_path_);
  }
  std::cout << std::endl;

  SleeveServiceImpl reloaded_service(meta_path_);
  auto              second_tree =
      reloaded_service.Read<std::wstring>([](FileSystem& fs) { return fs.Tree(L"/"); });

  EXPECT_EQ(conv::ToBytes(first_tree), conv::ToBytes(second_tree));
}

}  // namespace puerhlab
