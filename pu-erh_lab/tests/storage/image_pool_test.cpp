#include <gtest/gtest.h>

#include <cstddef>
#include <exiv2/exif.hpp>
#include <memory>
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
    auto img = std::make_shared<Image>(start_id++, L"PATH", ImageType::DEFAULT, Exiv2::ExifData());
    images_thumb.push_back(img);
    img_pool.Insert(img);
  }

  std::vector<std::shared_ptr<Image>> images_full;
  for (int i = 0; i < 8; i++) {
    auto img = std::make_shared<Image>(start_id++, L"PATH_FOR_FULL", ImageType::DEFAULT, Exiv2::ExifData());
    images_full.push_back(img);
    img_pool.Insert(img);
  }

	// Cache init test, first cache miss
  img_pool.RecordAccess(images_thumb[0]->_image_id, AccessType::THUMB);
  EXPECT_TRUE(img_pool.PoolContains(images_thumb[0]->_image_id));
	EXPECT_TRUE(img_pool.CacheContains(images_thumb[0]->_image_id, AccessType::THUMB));

  img_pool.RecordAccess(images_full[0]->_image_id, AccessType::FULL_IMG);
  img_pool.RecordAccess(images_full[1]->_image_id, AccessType::FULL_IMG);
  img_pool.RecordAccess(images_full[2]->_image_id, AccessType::FULL_IMG);
  EXPECT_TRUE(img_pool.AccessElement(images_full[0]->_image_id, AccessType::FULL_IMG).has_value()); 
  img_pool.RecordAccess(images_full[3]->_image_id, AccessType::FULL_IMG);
	EXPECT_FALSE(img_pool.CacheContains(images_full[1]->_image_id, AccessType::FULL_IMG));
}
};  // namespace puerhlab
