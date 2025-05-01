#include <gtest/gtest.h>

#include <codecvt>
#include <memory>
#include <optional>
#include <random>
#include <string>

#include "gtest/gtest.h"
#include "sleeve/sleeve_base.hpp"
#include "sleeve/sleeve_element/sleeve_element.hpp"
#include "sleeve/sleeve_element/sleeve_folder.hpp"

TEST(SleeveOperationTest, NormalTest1) {
  using namespace puerhlab;
  SleeveBase sl{0};

  sl.CreateElementToPath(L"root", L"test", ElementType::FOLDER);
  auto element = sl.AccessElementByPath(L"root/test");
  ASSERT_TRUE(element.has_value());
  ASSERT_EQ(element.value()->_element_id, 1);
  ASSERT_EQ(element.value()->_element_name, L"test");
  ASSERT_EQ(element.value()->_ref_count, 1);

  auto element_1 = sl.CreateElementToPath(L"root/test", L"test_file", ElementType::FILE);
  element_1      = sl.AccessElementByPath(L"root/test/test_file");
  ASSERT_TRUE(element_1.has_value());
  ASSERT_EQ(element_1.value()->_element_id, 2);
  ASSERT_EQ(element_1.value()->_element_name, L"test_file");
  ASSERT_EQ(element_1.value()->_ref_count, 1);

  auto element_2 = sl.CreateElementToPath(L"root", L"test_file", ElementType::FILE);
  element_2      = sl.AccessElementByPath(L"root/test_file");
  ASSERT_TRUE(element_2.has_value());
  ASSERT_EQ(element_2.value()->_element_id, 3);
  ASSERT_EQ(element_2.value()->_element_name, L"test_file");
  ASSERT_EQ(element_2.value()->_ref_count, 1);

  std::wstring                                     tree = sl.Tree(L"root");
  std::wstring_convert<std::codecvt_utf8<wchar_t>> conv;
  std::cout << conv.to_bytes(tree) << std::endl;
}

TEST(SleeveOperationTest, NormalTest2) {
  using namespace puerhlab;
  SleeveBase sl{0};

  sl.CreateElementToPath(L"root", L"test", ElementType::FOLDER);

  sl.CreateElementToPath(L"root/test", L"test_file", ElementType::FILE);

  auto element_1 = sl.RemoveElementInPath(L"root/test", L"test_file");
  ASSERT_TRUE(element_1.has_value());
  ASSERT_EQ(element_1.value()->_element_id, 2);
  ASSERT_EQ(element_1.value()->_element_name, L"test_file");
  ASSERT_EQ(element_1.value()->_ref_count, 0);

  ASSERT_EQ(sl.AccessElementByPath(L"root/test/test_file"), std::nullopt);

  auto element_2 = sl.RemoveElementInPath(L"root", L"test");
  ASSERT_TRUE(element_2.has_value());
  ASSERT_EQ(element_2.value()->_element_id, 1);
  ASSERT_EQ(element_2.value()->_element_name, L"test");
  ASSERT_EQ(element_2.value()->_ref_count, 0);
  ASSERT_EQ(sl.AccessElementByPath(L"root/test"), std::nullopt);

  std::wstring                                     tree = sl.Tree(L"root");
  std::wstring_convert<std::codecvt_utf8<wchar_t>> conv;
  std::cout << conv.to_bytes(tree) << std::endl;
}

/**
 * @brief Edge case 1: create element under a file instead of a folder
 *
 */
TEST(SleeveOperationTest, EdgeTest1) {
  using namespace puerhlab;
  SleeveBase sl{0};

  sl.CreateElementToPath(L"root", L"test", ElementType::FOLDER);

  sl.CreateElementToPath(L"root/test", L"test", ElementType::FILE);

  auto element_1 = sl.CreateElementToPath(L"root/test", L"test", ElementType::FOLDER);
  ASSERT_FALSE(element_1.has_value());
}

/**
 * @brief Edge case 2: create a repeatedly named file to the same folder
 *
 */
TEST(SleeveOperationTest, EdgeTest2) {
  using namespace puerhlab;
  SleeveBase sl{0};

  sl.CreateElementToPath(L"root", L"test", ElementType::FOLDER);

  sl.CreateElementToPath(L"root/test", L"test", ElementType::FOLDER);

  auto element_1 = sl.CreateElementToPath(L"root/test", L"test", ElementType::FOLDER);
  ASSERT_FALSE(element_1.has_value());
}

/**
 * @brief Create duplicated named elements to different locations
 *
 */
TEST(SleeveOperationTest, EdgeTest3) {
  using namespace puerhlab;
  SleeveBase sl{0};

  sl.CreateElementToPath(L"root", L"test", ElementType::FOLDER);

  auto element_1 = sl.CreateElementToPath(L"root/test", L"test", ElementType::FOLDER);
  ASSERT_TRUE(element_1.has_value());
  ASSERT_EQ(element_1.value()->_element_name, L"test");
  ASSERT_EQ(element_1.value()->_ref_count, 1);

  auto element_2 = sl.CreateElementToPath(L"root/test/test", L"test", ElementType::FILE);
  ASSERT_TRUE(element_2.has_value());
  ASSERT_EQ(element_2.value()->_element_name, L"test");
  ASSERT_EQ(element_2.value()->_ref_count, 1);

  std::wstring                                     tree = sl.Tree(L"root");
  std::wstring_convert<std::codecvt_utf8<wchar_t>> conv;
  std::cout << conv.to_bytes(tree) << std::endl;
}

/**
 * @brief Test the correctness of the file copy functionality
 *
 */
TEST(SleeveOperationTest, CopyTest1) {
  using namespace puerhlab;
  SleeveBase sl{0};

  sl.CreateElementToPath(L"root", L"monday", ElementType::FOLDER);
  sl.CreateElementToPath(L"root", L"tuesday", ElementType::FOLDER);

  auto element_1 = sl.CreateElementToPath(L"root/monday", L"broken", ElementType::FILE);
  ASSERT_TRUE(element_1.has_value());
  ASSERT_EQ(element_1.value()->_element_name, L"broken");
  ASSERT_EQ(element_1.value()->_ref_count, 1);

  auto element_2 = sl.CreateElementToPath(L"root/tuesday", L"hope", ElementType::FILE);
  ASSERT_TRUE(element_2.has_value());
  ASSERT_EQ(element_2.value()->_element_name, L"hope");
  ASSERT_EQ(element_2.value()->_ref_count, 1);

  auto element_3 = sl.CopyElement(L"root/monday/broken", L"root/tuesday");
  ASSERT_TRUE(element_3.has_value());
  ASSERT_EQ(element_3.value()->_element_name, L"broken");
  ASSERT_EQ(element_3.value()->_ref_count, 2);

  std::wstring                                     tree = sl.Tree(L"root");
  std::wstring_convert<std::codecvt_utf8<wchar_t>> conv;
  std::cout << conv.to_bytes(tree) << std::endl;
}

/**
 * @brief Test the correctness of the file copy functionality
 *
 */
TEST(SleeveOperationTest, CopyTest2) {
  using namespace puerhlab;
  SleeveBase sl{0};

  sl.CreateElementToPath(L"root", L"monday", ElementType::FOLDER);
  sl.CreateElementToPath(L"root", L"tuesday", ElementType::FOLDER);

  auto element_1 = sl.CreateElementToPath(L"root/monday", L"broken", ElementType::FILE);
  ASSERT_TRUE(element_1.has_value());
  ASSERT_EQ(element_1.value()->_element_name, L"broken");
  ASSERT_EQ(element_1.value()->_ref_count, 1);

  auto element_2 = sl.CreateElementToPath(L"root/tuesday", L"hope", ElementType::FILE);
  ASSERT_TRUE(element_2.has_value());
  ASSERT_EQ(element_2.value()->_element_name, L"hope");
  ASSERT_EQ(element_2.value()->_ref_count, 1);

  auto element_3 = sl.CopyElement(L"root/monday/broken", L"root/tuesday");
  ASSERT_TRUE(element_3.has_value());
  ASSERT_EQ(element_3.value()->_element_name, L"broken");
  ASSERT_EQ(element_3.value()->_ref_count, 2);

  auto element_4 = sl.CopyElement(L"root/tuesday", L"root/monday");
  ASSERT_TRUE(element_4.has_value());
  ASSERT_EQ(element_4.value()->_element_name, L"tuesday");
  ASSERT_EQ(element_4.value()->_ref_count, 2);

  auto element_5 = sl.AccessElementByPath(L"root/monday/tuesday/broken");
  ASSERT_TRUE(element_5.has_value());
  ASSERT_EQ(element_5.value()->_element_name, L"broken");
  ASSERT_EQ(element_5.value()->_ref_count, 3);

  std::wstring                                     tree = sl.Tree(L"root");
  std::wstring_convert<std::codecvt_utf8<wchar_t>> conv;
  std::cout << conv.to_bytes(tree) << std::endl;
}

/**
 * @brief Test the correctness of the file copy functionality
 *
 */
TEST(SleeveOperationTest, CopyEdgeTest1) {
  using namespace puerhlab;
  SleeveBase sl{0};

  sl.CreateElementToPath(L"root", L"monday", ElementType::FOLDER);
  sl.CreateElementToPath(L"root", L"tuesday", ElementType::FOLDER);

  auto element_1 = sl.CreateElementToPath(L"root/monday", L"broken", ElementType::FILE);
  ASSERT_TRUE(element_1.has_value());
  ASSERT_EQ(element_1.value()->_element_name, L"broken");
  ASSERT_EQ(element_1.value()->_ref_count, 1);

  auto element_2 = sl.CreateElementToPath(L"root/tuesday", L"hope", ElementType::FILE);
  ASSERT_TRUE(element_2.has_value());
  ASSERT_EQ(element_2.value()->_element_name, L"hope");
  ASSERT_EQ(element_2.value()->_ref_count, 1);

  auto element_3 = sl.CopyElement(L"root/monday/broken", L"root/tuesday");
  ASSERT_TRUE(element_3.has_value());
  ASSERT_EQ(element_3.value()->_element_name, L"broken");
  ASSERT_EQ(element_3.value()->_ref_count, 2);

  auto element_4 = sl.CopyElement(L"root/tuesday", L"root/monday");
  ASSERT_TRUE(element_4.has_value());
  ASSERT_EQ(element_4.value()->_element_name, L"tuesday");
  ASSERT_EQ(element_4.value()->_ref_count, 2);

  sl.CopyElement(L"root/monday", L"root/tuesday");

  std::wstring                                     tree = sl.Tree(L"root");
  std::wstring_convert<std::codecvt_utf8<wchar_t>> conv;
  std::cout << conv.to_bytes(tree) << std::endl;
}

/**
 * @brief Test the correctness of the file copy functionality
 *
 */
TEST(SleeveOperationTest, CopyEdgeTest2) {
  using namespace puerhlab;
  SleeveBase sl{0};

  sl.CreateElementToPath(L"root", L"B", ElementType::FOLDER);
  sl.CreateElementToPath(L"root", L"C", ElementType::FOLDER);

  sl.CreateElementToPath(L"root/B", L"D", ElementType::FILE);
  sl.CopyElement(L"root/B/D", L"root/C");
  sl.CopyElement(L"root/C", L"root/B");
  sl.CopyElement(L"root/B", L"root/C");
  sl.GetWriteGuard(L"root/B/C/D");
  std::wstring                                     tree = sl.Tree(L"root");
  std::wstring_convert<std::codecvt_utf8<wchar_t>> conv;
  std::cout << conv.to_bytes(tree) << std::endl;
}

/**
 * @brief Test the correctness of the file copy functionality
 *
 */
TEST(SleeveOperationTest, CopyEdgeTest3) {
  using namespace puerhlab;
  SleeveBase sl{0};

  sl.CreateElementToPath(L"root", L"B", ElementType::FOLDER);
  sl.CreateElementToPath(L"root", L"C", ElementType::FOLDER);

  sl.CreateElementToPath(L"root/B", L"D", ElementType::FOLDER);
  sl.CreateElementToPath(L"root/C", L"E", ElementType::FOLDER);
  sl.CreateElementToPath(L"root/C/E", L"F", ElementType::FOLDER);
  auto result1 = sl.CopyElement(L"root/B", L"root/C/E/F");
  sl.CopyElement(L"root/C", L"root/B/D");
  std::wstring                                     tree = sl.Tree(L"root");
  std::wstring_convert<std::codecvt_utf8<wchar_t>> conv;
  std::cout << conv.to_bytes(tree) << std::endl;
}

/**
 * @brief Test the correctness of the file copy functionality
 *
 */
 TEST(SleeveOperationTest, CopyEdgeTest4) {
  using namespace puerhlab;
  SleeveBase sl{0};

  sl.CreateElementToPath(L"root", L"B", ElementType::FOLDER);
  sl.CreateElementToPath(L"root", L"C", ElementType::FOLDER);

  sl.CreateElementToPath(L"root/B", L"D", ElementType::FOLDER);
  sl.CreateElementToPath(L"root/C", L"E", ElementType::FOLDER);
  auto result1 = sl.CopyElement(L"root/C", L"root/B/D");
  EXPECT_TRUE(result1.has_value());
  EXPECT_EQ(result1.value()->_ref_count, 2);
  EXPECT_EQ(result1.value()->_element_name, L"C");

  sl.CopyElement(L"root/B/D", L"root/C/E");

  std::wstring                                     tree = sl.Tree(L"root");
  std::wstring_convert<std::codecvt_utf8<wchar_t>> conv;
  std::cout << conv.to_bytes(tree) << std::endl;
}

/**
 * @brief Test the correctness of the file copy functionality
 *
 */
TEST(SleeveOperationTest, CopyRemoveTest2) {
  using namespace puerhlab;
  SleeveBase sl{0};

  sl.CreateElementToPath(L"root", L"monday", ElementType::FOLDER);
  sl.CreateElementToPath(L"root", L"tuesday", ElementType::FOLDER);

  auto element_1 = sl.CreateElementToPath(L"root/monday", L"broken", ElementType::FILE);
  ASSERT_TRUE(element_1.has_value());
  ASSERT_EQ(element_1.value()->_element_name, L"broken");
  ASSERT_EQ(element_1.value()->_ref_count, 1);

  auto element_2 = sl.CreateElementToPath(L"root/tuesday", L"hope", ElementType::FILE);
  ASSERT_TRUE(element_2.has_value());
  ASSERT_EQ(element_2.value()->_element_name, L"hope");
  ASSERT_EQ(element_2.value()->_ref_count, 1);

  auto element_3 = sl.CopyElement(L"root/monday/broken", L"root/tuesday");
  ASSERT_TRUE(element_3.has_value());
  ASSERT_EQ(element_3.value()->_element_name, L"broken");
  ASSERT_EQ(element_3.value()->_ref_count, 2);

  auto element_4 = sl.CopyElement(L"root/tuesday", L"root/monday");
  ASSERT_TRUE(element_4.has_value());
  ASSERT_EQ(element_4.value()->_element_name, L"tuesday");
  ASSERT_EQ(element_4.value()->_ref_count, 2);

  auto element_5 = sl.RemoveElementInPath(L"root/monday/tuesday/broken");
  ASSERT_TRUE(element_5.has_value());
  ASSERT_EQ(element_5.value()->_element_name, L"broken");
  ASSERT_EQ(element_5.value()->_ref_count, 2);

  auto element_6 = sl.AccessElementByPath(L"root/monday/tuesday");
  ASSERT_TRUE(element_6.has_value());
  ASSERT_EQ(element_6.value()->_element_name, L"tuesday");
  ASSERT_EQ(element_6.value()->_ref_count, 1);

  std::wstring                                     tree = sl.Tree(L"root");
  std::wstring_convert<std::codecvt_utf8<wchar_t>> conv;
  std::cout << conv.to_bytes(tree) << std::endl;
}

/**
 * @brief Test the correctness of the file movement functionality
 *
 */
TEST(SleeveOperationTest, MoveTest1) {
  using namespace puerhlab;
  SleeveBase sl{0};

  sl.CreateElementToPath(L"root", L"B", ElementType::FOLDER);
  sl.CreateElementToPath(L"root", L"C", ElementType::FOLDER);

  sl.CreateElementToPath(L"root/B", L"D", ElementType::FILE);
  sl.MoveElement(L"root/B/D", L"root/C");
  std::wstring                                     tree = sl.Tree(L"root");
  std::wstring_convert<std::codecvt_utf8<wchar_t>> conv;
  std::cout << conv.to_bytes(tree) << std::endl;
}

/**
 * @brief Test the correctness of the file movement functionality
 *
 */
 TEST(SleeveOperationTest, MoveTest2) {
  using namespace puerhlab;
  SleeveBase sl{0};

  sl.CreateElementToPath(L"root", L"B", ElementType::FOLDER);
  sl.CreateElementToPath(L"root", L"C", ElementType::FOLDER);

  sl.CreateElementToPath(L"root/B", L"D", ElementType::FOLDER);
  sl.CreateElementToPath(L"root/C", L"E", ElementType::FOLDER);
  auto result1 = sl.CopyElement(L"root/C", L"root/B/D");
  EXPECT_TRUE(result1.has_value());
  EXPECT_EQ(result1.value()->_ref_count, 2);
  EXPECT_EQ(result1.value()->_element_name, L"C");

  sl.CopyElement(L"root/B/D", L"root/C/E");

  sl.MoveElement(L"root/C/E/D", L"root");
  sl.RemoveElementInPath(L"root/D");
  std::wstring                                     tree = sl.Tree(L"root");
  std::wstring_convert<std::codecvt_utf8<wchar_t>> conv;
  std::cout << conv.to_bytes(tree) << std::endl;
}

/**
 * @brief Test the correctness of the file movement functionality
 *
 */
TEST(SleeveOperationTest, SubFolderTest1) {
  using namespace puerhlab;

  SleeveBase sl{0};

  sl.CreateElementToPath(L"root", L"B", ElementType::FOLDER);
  sl.CreateElementToPath(L"root", L"C", ElementType::FOLDER);

  sl.CreateElementToPath(L"root/B", L"D", ElementType::FOLDER);
  sl.CreateElementToPath(L"root/C", L"E", ElementType::FOLDER);

  EXPECT_TRUE(
      sl.IsSubFolder(std::dynamic_pointer_cast<SleeveFolder>(sl.AccessElementByPath(L"root/C").value()), L"root/C/E"));
}

/**
 * @brief Test the correctness of the file copy functionality
 *
 */
TEST(SleeveOperationTest, FuzzingTest1) {
  using namespace puerhlab;

  class SleeveBaseFuzzTest {
   public:
    SleeveBase                       sl{0};
    std::unordered_set<std::wstring> existing_paths;
    std::unordered_set<std::wstring> existing_folders;

    static std::wstring              RandomWString(size_t length) {
      static std::mt19937                       rng{std::random_device{}()};
      static std::uniform_int_distribution<int> dist(97, 122);  // ASCII 'a'-'z'
      std::wstring                              result;
      for (size_t i = 0; i < length; ++i) {
        result += static_cast<wchar_t>(dist(rng));
      }
      return result;
    }

    std::wstring RandomExistingPath() {
      static std::mt19937                   rng{std::random_device{}()};
      std::vector<std::wstring>             path_pool(existing_paths.begin(), existing_paths.end());
      std::uniform_int_distribution<size_t> dist(0, path_pool.size() - 1);
      return path_pool[dist(rng)];
    }

    std::wstring RandomExistingFolder() {
      static std::mt19937                   rng{std::random_device{}()};
      std::vector<std::wstring>             path_pool(existing_folders.begin(), existing_folders.end());
      std::uniform_int_distribution<size_t> dist(0, path_pool.size() - 1);
      return path_pool[dist(rng)];
    }

    ElementType RandomElementType() { return (rand() % 2 == 0) ? ElementType::FILE : ElementType::FOLDER; }

    void        Test() {
      constexpr int kIterations = 1000;
      existing_folders.insert(L"root");
      existing_paths.insert(L"root");

      for (int i = 0; i < kIterations; ++i) {
        // Create
        std::wstring parent_path = RandomExistingFolder();
        std::wstring name        = RandomWString(5);
        ElementType  type        = RandomElementType();

        auto         created     = sl.CreateElementToPath(parent_path, name, type);
        if (created.has_value()) {
          std::wstring new_path = parent_path + L"/" + name;
          if (type == ElementType::FOLDER) {
            existing_folders.insert(new_path);
          }
          existing_paths.insert(new_path);
          ASSERT_EQ(created.value()->_element_name, name);
        }

        // Copy
        if (rand() % 4 == 0) {
          if (existing_paths.size() > 1) {
            std::wstring src_path = RandomExistingPath();
            std::wstring dst_path = RandomExistingFolder();
            if (src_path == dst_path) {
              continue;
            }
            auto copied = sl.CopyElement(src_path, dst_path);
            if (copied.has_value()) {
              std::wstring copied_path = dst_path + L"/" + copied.value()->_element_name;
              if (copied.value()->_type == ElementType::FOLDER) existing_folders.insert(copied_path);
              existing_paths.insert(copied_path);
            }
          }
        }

        // Access
        // std::wstring access_path = RandomExistingPath();
        // auto accessed = sl.AccessElementByPath(access_path);
        // if (accessed.has_value()) {
        //   ASSERT_GE(accessed.value()->_ref_count, 1);
        // }
      }
      auto print = false;
      if (print) {
        std::wstring                                     tree = sl.Tree(L"root");
        std::wstring_convert<std::codecvt_utf8<wchar_t>> conv;
        std::cout << conv.to_bytes(tree) << std::endl;
      }
    }
  };

  SleeveBaseFuzzTest test;
  test.Test();
}