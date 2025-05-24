#include "sleeve/sleeve_view.hpp"

#include <gtest/gtest.h>
#include <cstdint>
#include <exception>
#include <memory>
#include <random>

#include "sleeve/sleeve_manager.hpp"


namespace puerhlab {
auto loaded_callback = [](size_t idx, std::weak_ptr<Image> img) {
  // std::cout << "Get image " << img.lock()->_image_path << " at index " << idx << "\n";
};

TEST(SleeveViewTest, DISABLED_SimpleTest1) {
  SleeveManager manager{};
  ImageLoader   image_loader(128, 8, 0);
  image_path_t  path =
      L"D:\\Projects\\pu-erh_lab\\pu-erh_"
      L"lab\\tests\\resources\\sample_images\\jpg";
  std::vector<image_path_t> imgs;
  for (const auto &img : std::filesystem::directory_iterator(path)) {
    imgs.push_back(img.path());
  }
  manager.LoadToPath(imgs, L"root");
  auto view = manager.GetView();
  view->UpdateView(L"root");
  view->LoadPreview(0, 20, loaded_callback);
}

TEST(SleeveViewTest, DISABLED_SimpleTest2) {
  SleeveManager manager{};
  image_path_t  path =
      L"D:\\Projects\\pu-erh_lab\\pu-erh_"
      L"lab\\tests\\resources\\sample_images\\jpg";
  std::vector<image_path_t> imgs;
  for (const auto &img : std::filesystem::directory_iterator(path)) {
    imgs.push_back(img.path());
  }
  manager.LoadToPath(imgs, L"root");
  auto view = manager.GetView();
  view->UpdateView(L"root");

  // Simulate scrolling down
  auto start = std::chrono::high_resolution_clock::now();
  view->LoadPreview(0, 20, loaded_callback);
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> duration = end - start;
  std::cout << "From 0 to 20, time consumed: " << duration.count() << "ms" << std::endl;

  start = std::chrono::high_resolution_clock::now();
  view->LoadPreview(10, 30, loaded_callback);
  end = std::chrono::high_resolution_clock::now();
  duration = end - start;
  std::cout << "From 10 to 30, time consumed: " << duration.count() << "ms" << std::endl;

  start = std::chrono::high_resolution_clock::now();
  view->LoadPreview(20, 40, loaded_callback);
  end = std::chrono::high_resolution_clock::now();
  duration = end - start;
  std::cout << "From 20 to 40, time consumed: " << duration.count() << "ms" << std::endl;

  // Simulate scrolling up
  start = std::chrono::high_resolution_clock::now();
  view->LoadPreview(10, 30, loaded_callback);
  end = std::chrono::high_resolution_clock::now();
  duration = end - start;
  std::cout << "From 10 to 30, time consumed: " << duration.count() << "ms" << std::endl;
}

TEST(SleeveViewTest, FuzzyTest1) {
  SleeveManager manager{};
  image_path_t  path =
      L"D:\\Projects\\pu-erh_lab\\pu-erh_"
      L"lab\\tests\\resources\\sample_images\\jpg";
  std::vector<image_path_t> imgs;
  for (const auto &img : std::filesystem::directory_iterator(path)) {
    imgs.push_back(img.path());
  }
  manager.LoadToPath(imgs, L"root");
  auto view = manager.GetView();
  view->UpdateView(L"root");

  // Simulate random scrolling:
  size_t total_size = manager.GetImgCount();
  uint32_t view_size = 50;
  static std::mt19937                   rng{std::random_device{}()};
  std::uniform_int_distribution<size_t> dist(0, total_size - view_size - 1);
  std::chrono::duration<double, std::milli> total_duration(0);
  constexpr int iter_count = 1000;
  for (int i = 0; i < iter_count; i++) {
    try {
    auto start = std::chrono::high_resolution_clock::now();
    uint32_t low_idx = dist(rng);
    view->LoadPreview(low_idx, low_idx + view_size, loaded_callback);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;
    total_duration += duration;
    if (i % 50 == 0)
      std::cout << std::format("Iteration: {}, Avg. View Loading Time: {}ms", i, total_duration.count() / i) << std::endl;
    } catch (std::exception &e) {
      std::cout << std::format("Test aborted, successful iterations: {}. Cause of abortion: {}", i, e.what()) << std::endl;
      return;
    }
  }
}
}  // namespace puerhlab