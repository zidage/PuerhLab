#include "sleeve/sleeve_view.hpp"

#include <gtest/gtest.h>
#include <memory>

#include "sleeve/sleeve_manager.hpp"


namespace puerhlab {
TEST(SleeveViewTest, SimpleTest1) {
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
  view->LoadPreview(0, 20, [](size_t idx, std::weak_ptr<Image> img) {
    std::cout << "Get image " << img.lock()->_image_path << " at index " << idx << "\n";
  });
}
}  // namespace puerhlab