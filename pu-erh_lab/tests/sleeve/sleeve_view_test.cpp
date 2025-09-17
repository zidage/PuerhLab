#include "sleeve/sleeve_view.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <exception>
#include <format>
#include <memory>
#include <opencv2/highgui.hpp>
#include <random>

#include "sleeve/sleeve_manager.hpp"

std::filesystem::path db_path(TEST_DB_PATH);

image_path_t          path = std::string(TEST_IMG_PATH) + std::string("/jpg");
namespace puerhlab {
auto loaded_callback = [](size_t idx, std::weak_ptr<Image> img) {
  // std::cout << "Get image " << img.lock()->_image_path << " at index " << idx << "\n";
};

auto display_callback = [](size_t idx, std::weak_ptr<Image> img) {
  auto access_img = img.lock();
  cv::imshow(std::format("img: {}", idx), access_img->GetThumbnailData());
  cv::waitKey(1);
};

TEST(SleeveViewTest, SimpleTest1) {
  Exiv2::LogMsg::setLevel(Exiv2::LogMsg::Level::mute);
  {
    SleeveManager             manager{db_path};
    ImageLoader               image_loader(128, 8, 0);

    std::vector<image_path_t> imgs;
    for (const auto& img : std::filesystem::directory_iterator(path)) {
      imgs.push_back(img.path());
    }
    manager.LoadToPath(imgs, L"");
    auto view = manager.GetView();
    view->UpdateView(L"");
    view->LoadPreview(0, 20, display_callback);
    cv::waitKey();
  }

  if (std::filesystem::exists(db_path)) {
    std::filesystem::remove(db_path);
  }
}

TEST(SleeveViewTest, DISABLED_SimpleTest2) {
  {
    SleeveManager             manager{db_path};
    image_path_t              path = std::string(TEST_IMG_PATH) + std::string("/jpg");
    std::vector<image_path_t> imgs;
    for (const auto& img : std::filesystem::directory_iterator(path)) {
      imgs.push_back(img.path());
    }
    manager.LoadToPath(imgs, L"");
    auto view = manager.GetView();
    view->UpdateView(L"");

    // Simulate scrolling down
    auto start = std::chrono::high_resolution_clock::now();
    view->LoadPreview(0, 20, loaded_callback);
    auto                                      end      = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;
    std::cout << "From 0 to 20, time consumed: " << duration.count() << "ms" << std::endl;

    start = std::chrono::high_resolution_clock::now();
    view->LoadPreview(10, 30, loaded_callback);
    end      = std::chrono::high_resolution_clock::now();
    duration = end - start;
    std::cout << "From 10 to 30, time consumed: " << duration.count() << "ms" << std::endl;

    start = std::chrono::high_resolution_clock::now();
    view->LoadPreview(20, 40, loaded_callback);
    end      = std::chrono::high_resolution_clock::now();
    duration = end - start;
    std::cout << "From 20 to 40, time consumed: " << duration.count() << "ms" << std::endl;

    // Simulate scrolling up
    start = std::chrono::high_resolution_clock::now();
    view->LoadPreview(10, 30, loaded_callback);
    end      = std::chrono::high_resolution_clock::now();
    duration = end - start;
    std::cout << "From 10 to 30, time consumed: " << duration.count() << "ms" << std::endl;
  }

  if (std::filesystem::exists(db_path)) {
    std::filesystem::remove(db_path);
  }
}

TEST(SleeveViewTest, DISABLED_FuzzyTest1) {
  {
    SleeveManager manager{db_path};
    std::vector<image_path_t> imgs;
    for (const auto& img : std::filesystem::directory_iterator(path)) {
      imgs.push_back(img.path());
    }
    manager.LoadToPath(imgs, L"");
    auto view = manager.GetView();
    view->UpdateView(L"");

    // Simulate random scrolling:
    size_t                                    total_size = manager.GetImgCount();
    uint32_t                                  view_size  = 50;
    static std::mt19937                       rng{std::random_device{}()};
    std::uniform_int_distribution<size_t>     dist(0, total_size - view_size - 1);
    std::chrono::duration<double, std::milli> total_duration(0);
    constexpr int                             iter_count = 1000;
    for (int i = 0; i < iter_count; i++) {
      try {
        auto   start   = std::chrono::high_resolution_clock::now();
        size_t low_idx = dist(rng);
        view->LoadPreview(low_idx, low_idx + view_size, loaded_callback);
        auto                                      end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> duration = end - start;
        total_duration += duration;
        if (i % 50 == 0)
          std::cout << std::format("Iteration: {}, Avg. View Loading Time: {}ms", i,
                                   total_duration.count() / i)
                    << std::endl;
      } catch (std::exception& e) {
        std::cout << std::format("Test aborted, successful iterations: {}. Cause of abortion: {}",
                                 i, e.what())
                  << std::endl;
        break;
      }
    }
  }
  if (std::filesystem::exists(db_path)) {
    std::filesystem::remove(db_path);
  }
}
}  // namespace puerhlab