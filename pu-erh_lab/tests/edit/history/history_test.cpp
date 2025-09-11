#include "edit/history/edit_history.hpp"
#include "history_test_fixation.hpp"
#include "type/type.hpp"

using namespace puerhlab;

TEST_F(EditHistoryTests, CreateEditHistory) {
  sl_element_id_t image_id = 12345;
  EditHistory    history(image_id);
  EXPECT_EQ(history.GetBoundImage(), image_id);
  EXPECT_NE(history.GetHistoryId(), 0);
  EXPECT_NE(history.GetAddTime(), 0);
  EXPECT_EQ(history.GetAddTime(), history.GetLastModifiedTime());
}

TEST_F(EditHistoryTests, CommitVersion) {
  sl_element_id_t image_id = 12345;
  EditHistory    history(image_id);
  Version        ver1(image_id);
  Version        ver2(image_id);
  auto           ver1_id = history.CommitVersion(std::move(ver1));
  auto           ver2_id = history.CommitVersion(std::move(ver2));
  EXPECT_NE(ver1_id, ver2_id);
  EXPECT_NO_THROW(history.GetVersion(ver1_id));
  EXPECT_NO_THROW(history.GetVersion(ver2_id));
}
