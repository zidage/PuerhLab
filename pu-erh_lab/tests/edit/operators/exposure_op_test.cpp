#include "edit/operators/exposure_op.hpp"

#include <gtest/gtest.h>

#include <utility>

#include "sleeve/sleeve_manager.hpp"


namespace puerhlab {
class ExposureTests : public ::testing::Test {
 protected:
  std::filesystem::path db_path_;

  // Run before any unit test runs
  void                  SetUp() override {
    // Create a unique db file location
    db_path_ = std::filesystem::temp_directory_path() / "test_db.db";
    // Make sure there is not existing db
    if (std::filesystem::exists(db_path_)) {
      std::filesystem::remove(db_path_);
    }
  }

  // Run before any unit test runs
  void TearDown() override {
    // Clean up the DB file
    if (std::filesystem::exists(db_path_)) {
      std::filesystem::remove(db_path_);
    }
  }
};

TEST_F(ExposureTests, AdjustmentTest1) {
  {
    SleeveManager manager{db_path_};
    ImageLoader   image_loader(128, 8, 0);
    image_path_t  path =
        L"D:\\Projects\\pu-erh_lab\\pu-erh_"
        L"lab\\tests\\resources\\sample_images\\jpg";
    std::vector<image_path_t> imgs;
    for (const auto& img : std::filesystem::directory_iterator(path)) {
      imgs.push_back(img.path());
    }

    auto display_callback = [](size_t idx, std::weak_ptr<Image> img) {
      auto  access_img = img.lock();
      auto& thumbnail  = access_img->GetThumbnailBuffer();
      cv::imshow(std::format("img_before: {}", idx), access_img->GetThumbnailData());
      cv::waitKey(1);
      ExposureOp op{1.0f};
      auto       adjusted = op.Apply(thumbnail);

      access_img->LoadThumbnail(std::move(adjusted));

      cv::imshow(std::format("img_after: {}", idx), access_img->GetThumbnailData());
      cv::waitKey(1);
    };

    manager.LoadToPath(imgs, L"");
    auto view = manager.GetView();
    view->UpdateView(L"");
    view->LoadPreview(0, 20, display_callback);
    cv::waitKey();
  }
}
}  // namespace puerhlab