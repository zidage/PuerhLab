//  Copyright 2026 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#include <gtest/gtest.h>

#include <QCoreApplication>
#include <QDir>
#include <QLocale>
#include <QSettings>
#include <QSignalSpy>
#include <QTemporaryDir>

#include "ui/alcedo_main/language_manager.hpp"

namespace alcedo::ui::test {
namespace {

class LanguageManagerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    ASSERT_TRUE(settings_dir_.isValid());
    QCoreApplication::setOrganizationName("PuerhLabTest");
    QCoreApplication::setApplicationName("LanguageManagerTest");
    QSettings::setDefaultFormat(QSettings::IniFormat);
    QSettings::setPath(QSettings::IniFormat, QSettings::UserScope, settings_dir_.path());
    QSettings settings;
    settings.clear();
    settings.sync();
  }

  QTemporaryDir settings_dir_;
};

TEST_F(LanguageManagerTest, ResolveSystemLanguageCode_MapsChineseToZhCn) {
  EXPECT_EQ(LanguageManager::ResolveSystemLanguageCode(QLocale("zh-CN")), "zh-CN");
  EXPECT_EQ(LanguageManager::ResolveSystemLanguageCode(QLocale("zh-Hans")), "zh-CN");
  EXPECT_EQ(LanguageManager::ResolveSystemLanguageCode(QLocale("en-US")), "en");
  EXPECT_EQ(LanguageManager::ResolveSystemLanguageCode(QLocale("fr-FR")), "en");
}

TEST_F(LanguageManagerTest, SetLanguage_PersistsAcrossInstances) {
  auto* app = QCoreApplication::instance();
  ASSERT_NE(app, nullptr);

  LanguageManager first(app);
  QSignalSpy       changed_spy(&first, &LanguageManager::LanguageChanged);

  first.setLanguage("zh-CN");

  EXPECT_EQ(first.CurrentLanguageCode(), "zh-CN");
  EXPECT_EQ(first.EffectiveLanguageCode(), "zh-CN");
  EXPECT_EQ(changed_spy.count(), 1);

  LanguageManager second(app);
  EXPECT_EQ(second.CurrentLanguageCode(), "zh-CN");
  EXPECT_EQ(second.EffectiveLanguageCode(), "zh-CN");
}

TEST_F(LanguageManagerTest, SetLanguage_SystemSelectionPersistsLiteralSystemCode) {
  auto* app = QCoreApplication::instance();
  ASSERT_NE(app, nullptr);

  LanguageManager manager(app);
  manager.setLanguage("zh-CN");
  manager.setLanguage("system");

  QSettings settings;
  EXPECT_EQ(settings.value("ui/language").toString(), "system");
}

}  // namespace
}  // namespace alcedo::ui::test
