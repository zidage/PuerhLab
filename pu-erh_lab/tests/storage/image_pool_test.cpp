#include <gtest/gtest.h>

#include <cstddef>
#include <exception>
#include <exiv2/exif.hpp>
#include <memory>
#include <random>
#include <vector>

#include "image/image.hpp"
#include "storage/image_pool/image_pool_manager.hpp"
#include "type/type.hpp"

namespace puerhlab {
TEST(ImagePoolTest, SimpleTest1) {
  // Using default setting, thumbnail cache 64, full img cache 3
  ImagePoolManager                    img_pool{};

  std::vector<std::shared_ptr<Image>> images_thumb;
  image_id_t                          start_id = 0;
  for (int i = 0; i < 128; i++) {
    auto img = std::make_shared<Image>(start_id++, L"PATH", ImageType::DEFAULT);
    images_thumb.push_back(img);
    img_pool.Insert(img);
  }

  std::vector<std::shared_ptr<Image>> images_full;
  for (int i = 0; i < 8; i++) {
    auto img = std::make_shared<Image>(start_id++, L"PATH_FOR_FULL", ImageType::DEFAULT);
    images_full.push_back(img);
    img_pool.Insert(img);
  }

  // Cache init test, first cache miss
  img_pool.RecordAccess(images_thumb[0]->image_id_, AccessType::THUMB);
  EXPECT_TRUE(img_pool.PoolContains(images_thumb[0]->image_id_));
  EXPECT_TRUE(img_pool.CacheContains(images_thumb[0]->image_id_, AccessType::THUMB));

  img_pool.RecordAccess(images_full[0]->image_id_, AccessType::FULL_IMG);
  img_pool.RecordAccess(images_full[1]->image_id_, AccessType::FULL_IMG);
  img_pool.RecordAccess(images_full[2]->image_id_, AccessType::FULL_IMG);
  EXPECT_TRUE(img_pool.AccessElement(images_full[0]->image_id_, AccessType::FULL_IMG).has_value());
  img_pool.RecordAccess(images_full[3]->image_id_, AccessType::FULL_IMG);
  EXPECT_FALSE(img_pool.CacheContains(images_full[1]->image_id_, AccessType::FULL_IMG));
  EXPECT_TRUE(img_pool.CacheContains(images_full[0]->image_id_, AccessType::FULL_IMG));
  EXPECT_TRUE(img_pool.CacheContains(images_full[2]->image_id_, AccessType::FULL_IMG));
  EXPECT_TRUE(img_pool.CacheContains(images_full[3]->image_id_, AccessType::FULL_IMG));
}

TEST(ImagePoolTest, SimpleTest2) {
  // Using default setting, thumbnail cache 64, full img cache 3
  ImagePoolManager                    img_pool{32, 3};

  std::vector<std::shared_ptr<Image>> images_thumb;
  image_id_t                          start_id = 0;
  for (int i = 0; i < 128; i++) {
    auto img = std::make_shared<Image>(start_id++, L"PATH", ImageType::DEFAULT);
    images_thumb.push_back(img);
    img_pool.Insert(img);
  }

  std::vector<std::shared_ptr<Image>> images_full;
  for (int i = 0; i < 8; i++) {
    auto img = std::make_shared<Image>(start_id++, L"PATH_FOR_FULL", ImageType::DEFAULT);
    images_full.push_back(img);
    img_pool.Insert(img);
  }

  // Cache flood test
  for (int i = 0; i < 128; i++) {
    img_pool.RecordAccess(images_thumb[i]->image_id_, AccessType::THUMB);
  }

  int cache_hit    = 0;
  int total_access = 0;
  for (int i = 0; i < 128; i++) {
    if (img_pool.AccessElement(images_thumb[i]->image_id_, AccessType::THUMB).has_value()) {
      cache_hit++;
    }
    total_access++;
  }

  std::cout << "Hit rate: " << (double)(cache_hit) / (double)(total_access) * 100 << "%\n";
}

TEST(ImagePoolTest, RandomTest1) {
  try {
    // Using default setting, thumbnail cache 64, full img cache 3
    ImagePoolManager                    img_pool{32, 3};

    std::vector<std::shared_ptr<Image>> images_thumb;
    image_id_t                          start_id = 0;
    for (int i = 0; i < 128; i++) {
      auto img = std::make_shared<Image>(start_id++, L"PATH", ImageType::DEFAULT);
      images_thumb.push_back(img);
      img_pool.Insert(img);
    }

    std::vector<std::shared_ptr<Image>> images_full;
    for (int i = 0; i < 8; i++) {
      auto img = std::make_shared<Image>(start_id++, L"PATH_FOR_FULL", ImageType::DEFAULT);
      images_full.push_back(img);
      img_pool.Insert(img);
    }

    // Cache flood test
    for (int i = 0; i < 128; i++) {
      img_pool.RecordAccess(images_thumb[i]->image_id_, AccessType::THUMB);
    }

    // Random number generator
    static std::mt19937                   rng{std::random_device{}()};
    std::uniform_int_distribution<size_t> dist(0, images_thumb.size() - 1);

    constexpr int                         total_iter   = 1000;
    int                                   cache_hit    = 0;
    int                                   total_access = 0;
    for (int i = 0; i < total_iter; i++) {
      if (img_pool.AccessElement(images_thumb[dist(rng)]->image_id_, AccessType::THUMB).has_value()) {
        cache_hit++;
      }
      total_access++;
    }
    std::cout << "Hit rate: " << (double)(cache_hit) / (double)(total_access) * 100 << "%\n";
  } catch (std::exception &e) {
    std::cout << "Unexpected exception thrown in RandomTest1" << std::endl;
    std::cout << e.what() << std::endl;
  }
}
};  // namespace puerhlab
