//  Copyright 2026 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

// LibrarySelection rules that SelectionState.qml had: no-op changes, replace, prune, and the
// target maps of the selected images.

#include "ui/alcedo_main/album_backend/library_selection.hpp"

#include <gtest/gtest.h>

#include <QSignalSpy>
#include <QVariantList>
#include <QVariantMap>

namespace alcedo::ui::test {
namespace {

auto Item(uint element_id, uint image_id, const QString& file_name) -> QVariantMap {
  return QVariantMap{{QStringLiteral("elementId"), element_id},
                     {QStringLiteral("imageId"), image_id},
                     {QStringLiteral("fileName"), file_name},
                     {QStringLiteral("isHdr"), false}};
}

TEST(LibrarySelectionTest, SetSelectedIsNoOpWhenStateMatches) {
  LibrarySelection selection(nullptr);
  QSignalSpy       changed(&selection, &LibrarySelection::SelectionChanged);

  selection.SetImageSelected(5, 50, QStringLiteral("a.arw"), false, false);
  EXPECT_EQ(changed.count(), 0);
  selection.SetImageSelected(5, 50, QStringLiteral("a.arw"), false, true);
  EXPECT_EQ(changed.count(), 1);
  selection.SetImageSelected(5, 50, QStringLiteral("a.arw"), false, true);
  EXPECT_EQ(changed.count(), 1);
  selection.Replace(QVariantList{Item(5, 50, QStringLiteral("a.arw"))});
  EXPECT_EQ(changed.count(), 1);
  selection.PruneElements(QVariantList{7});
  EXPECT_EQ(changed.count(), 1);
  selection.Clear();
  EXPECT_EQ(changed.count(), 2);
  selection.Clear();
  EXPECT_EQ(changed.count(), 2);
}

TEST(LibrarySelectionTest, ReplaceAndPruneKeepTheSelectedImages) {
  LibrarySelection selection(nullptr);
  selection.Replace(QVariantList{Item(3, 30, QStringLiteral("c.arw")), Item(1, 10, QString()),
                                 Item(0, 99, QStringLiteral("ignored"))});
  ASSERT_EQ(selection.SelectedCount(), 2);
  EXPECT_TRUE(selection.IsSelected(3));
  EXPECT_TRUE(selection.IsSelected(1));
  EXPECT_FALSE(selection.IsSelected(0));
  EXPECT_EQ(selection.SelectedElementIds(), (std::vector<sl_element_id_t>{1, 3}));
  const QVariantList before_prune = selection.SelectedItems();
  ASSERT_EQ(before_prune.size(), 2);
  EXPECT_EQ(before_prune.at(1).toMap().value(QStringLiteral("imageId")).toUInt(), 30u);
  // An empty file name gets the placeholder name, as in QML.
  EXPECT_FALSE(before_prune.at(0).toMap().value(QStringLiteral("fileName")).toString().isEmpty());

  selection.Prune({3});
  EXPECT_FALSE(selection.IsSelected(3));
  EXPECT_EQ(selection.SelectedElementIds(), (std::vector<sl_element_id_t>{1}));
  const QVariantList items = selection.SelectedItems();
  ASSERT_EQ(items.size(), 1);
  EXPECT_EQ(items.front().toMap().value(QStringLiteral("fileId")).toUInt(), 1u);
}

TEST(LibrarySelectionTest, RevisionIncrementsOnEachChangeOnly) {
  LibrarySelection selection(nullptr);
  const int        start = selection.Revision();
  selection.SetImageSelected(5, 50, QStringLiteral("a.arw"), false, true);
  EXPECT_EQ(selection.Revision(), start + 1);
  selection.SetImageSelected(5, 50, QStringLiteral("a.arw"), false, true);
  EXPECT_EQ(selection.Revision(), start + 1);
  selection.AddItems(QVariantList{Item(5, 50, QStringLiteral("a.arw"))});
  EXPECT_EQ(selection.Revision(), start + 1);
  selection.AddItems(QVariantList{Item(6, 60, QStringLiteral("b.arw"))});
  EXPECT_EQ(selection.Revision(), start + 2);
}

TEST(LibrarySelectionTest, AddItemsKeepsTheSelection) {
  LibrarySelection selection(nullptr);
  selection.Replace(QVariantList{Item(1, 10, QStringLiteral("a.arw"))});
  selection.AddItems(
      QVariantList{Item(2, 20, QStringLiteral("b.arw")), Item(1, 10, QStringLiteral("a.arw"))});
  EXPECT_EQ(selection.SelectedElementIds(), (std::vector<sl_element_id_t>{1, 2}));
}

TEST(LibrarySelectionTest, RangeItemsReplaceOnlyWhatTheOpenRangeAdded) {
  LibrarySelection selection(nullptr);
  selection.Replace(
      QVariantList{Item(1, 10, QStringLiteral("a.arw")), Item(3, 30, QStringLiteral("c.arw"))});

  // The band grows over 2, 3 and 4: 3 was selected before the range.
  selection.SetRangeItems(QVariantList{Item(2, 20, QStringLiteral("b.arw")),
                                       Item(3, 30, QStringLiteral("c.arw")),
                                       Item(4, 40, QStringLiteral("d.arw"))});
  EXPECT_EQ(selection.SelectedElementIds(), (std::vector<sl_element_id_t>{1, 2, 3, 4}));

  // The band shrinks to 4 only: 2 leaves, 3 stays because the range did not add it.
  selection.SetRangeItems(QVariantList{Item(4, 40, QStringLiteral("d.arw"))});
  EXPECT_EQ(selection.SelectedElementIds(), (std::vector<sl_element_id_t>{1, 3, 4}));

  // After EndRange, a new range keeps 4.
  selection.EndRange();
  selection.SetRangeItems(QVariantList{Item(5, 50, QStringLiteral("e.arw"))});
  selection.SetRangeItems(QVariantList{});
  EXPECT_EQ(selection.SelectedElementIds(), (std::vector<sl_element_id_t>{1, 3, 4}));
}

}  // namespace
}  // namespace alcedo::ui::test
