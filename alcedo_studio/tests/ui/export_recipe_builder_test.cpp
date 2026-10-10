//  Copyright 2026 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

// ExportRecipeBuilder rules that ExportInspectorPanel.qml had: the defaults, the limits of the
// size values, the bit depths of each format, and the output folder rule.

#include "ui/alcedo_main/album_backend/export_recipe_builder.hpp"

#include <gtest/gtest.h>

#include <QList>
#include <QLocale>
#include <QVariantMap>

namespace alcedo::ui::test {
namespace {

using Builder = ExportRecipeBuilder;

TEST(ExportRecipeBuilderTest, DefaultsMatchPreviousPanelDefaults) {
  const ExportRecipeOptions defaults = Builder::Defaults();
  EXPECT_EQ(defaults.format, QStringLiteral("JPEG"));
  EXPECT_EQ(defaults.bit_depth, 8);
  EXPECT_EQ(defaults.quality, 95);
  EXPECT_EQ(defaults.png_level, 5);
  EXPECT_EQ(defaults.tiff_compression, QStringLiteral("NONE"));
  EXPECT_TRUE(defaults.include_metadata);
  EXPECT_TRUE(defaults.embed_icc_profile);
  EXPECT_EQ(defaults.ultra_hdr_quality, 95);
  EXPECT_TRUE(defaults.ultra_hdr_dither);
  EXPECT_FALSE(defaults.subfolder_enabled);
  EXPECT_EQ(defaults.subfolder, QStringLiteral("Processed"));
  EXPECT_EQ(defaults.resize_mode, QStringLiteral("original"));
  EXPECT_EQ(defaults.long_edge, 2048);
  EXPECT_EQ(defaults.width, 2048);
  EXPECT_EQ(defaults.height, 2048);
  EXPECT_EQ(defaults.print_width, 210);
  EXPECT_EQ(defaults.print_height, 297);
  EXPECT_EQ(defaults.print_unit, QStringLiteral("mm"));
  EXPECT_EQ(defaults.dpi, 300);
  EXPECT_EQ(defaults.file_name_pattern, QStringLiteral("{source}"));
  EXPECT_FALSE(Builder::Validate(defaults).has_value());

  // The map form that the panel reads has the same values.
  Builder           builder(nullptr);
  const QVariantMap map = builder.DefaultOptions();
  EXPECT_EQ(map.value(QStringLiteral("format")).toString(), QStringLiteral("JPEG"));
  EXPECT_EQ(map.value(QStringLiteral("long_edge")).toInt(), 2048);
  EXPECT_EQ(map.value(QStringLiteral("dpi")).toInt(), 300);
}

TEST(ExportRecipeBuilderTest, BitDepthNotAllowedForFormatIsRejected) {
  EXPECT_EQ(Builder::AllowedBitDepths(QStringLiteral("JPEG")), (QList<int>{8}));
  EXPECT_EQ(Builder::AllowedBitDepths(QStringLiteral("PNG")), (QList<int>{8, 16}));
  EXPECT_EQ(Builder::AllowedBitDepths(QStringLiteral("TIFF")), (QList<int>{8, 16, 32}));
  EXPECT_EQ(Builder::AllowedBitDepths(QStringLiteral("EXR")), (QList<int>{16, 32}));
  EXPECT_EQ(Builder::PreferredBitDepth(QStringLiteral("JPEG")), 8);
  EXPECT_EQ(Builder::PreferredBitDepth(QStringLiteral("EXR")), 16);

  ExportRecipeOptions options = Builder::Defaults();
  options.bit_depth           = 16;
  const auto error            = Builder::Validate(options);
  ASSERT_TRUE(error.has_value());
  EXPECT_EQ(error->field, QStringLiteral("bit_depth"));
  EXPECT_EQ(error->message, QStringLiteral("bit_depth for JPEG must be one of: 8"));

  options.format    = QStringLiteral("EXR");
  options.bit_depth = 8;
  EXPECT_EQ(Builder::Validate(options)->field, QStringLiteral("bit_depth"));
  options.bit_depth = 32;
  EXPECT_FALSE(Builder::Validate(options).has_value());

  options.format          = QStringLiteral("WEBP");
  const auto format_error = Builder::Validate(options);
  ASSERT_TRUE(format_error.has_value());
  EXPECT_EQ(format_error->message, QStringLiteral("format must be one of: JPEG, PNG, TIFF, EXR"));
}

TEST(ExportRecipeBuilderTest, SizeValuesOutsideLimitsAreRejectedWithTheRange) {
  ExportRecipeOptions options = Builder::Defaults();
  options.resize_mode         = QStringLiteral("long_edge");
  options.long_edge           = 100;
  const auto long_edge_error  = Builder::Validate(options);
  ASSERT_TRUE(long_edge_error.has_value());
  EXPECT_EQ(long_edge_error->field, QStringLiteral("long_edge"));
  EXPECT_EQ(long_edge_error->message, QStringLiteral("long_edge must be in [256, 16384]"));
  options.long_edge = 256;
  EXPECT_FALSE(Builder::Validate(options).has_value());
  options.long_edge = 16384.5;
  EXPECT_TRUE(Builder::Validate(options).has_value());

  // Only the values of the selected mode are checked.
  options.long_edge   = 100;
  options.resize_mode = QStringLiteral("bounds");
  options.width       = 100000;
  options.height      = 0;
  EXPECT_EQ(Builder::Validate(options)->field, QStringLiteral("height"));
  options.height = 1;
  EXPECT_FALSE(Builder::Validate(options).has_value());

  options.resize_mode = QStringLiteral("print");
  options.print_width = 0;
  EXPECT_EQ(Builder::Validate(options)->field, QStringLiteral("print_width"));
  options.print_width = 100000;
  options.dpi         = 10001;
  EXPECT_EQ(Builder::Validate(options)->message, QStringLiteral("dpi must be in [1, 10000]"));
  options.dpi = 1;
  EXPECT_FALSE(Builder::Validate(options).has_value());

  options.quality = 0;
  EXPECT_EQ(Builder::Validate(options)->field, QStringLiteral("quality"));
  options.quality   = 100;
  options.png_level = 10;
  EXPECT_EQ(Builder::Validate(options)->message, QStringLiteral("png_level must be in [0, 9]"));
}

TEST(ExportRecipeBuilderTest, PanelTextValuesAreReadLikeThePanelFields) {
  Builder builder(nullptr);
  // Trimmed integer text is valid; text that is not a number or not an integer is not.
  EXPECT_TRUE(
      builder.ValidateOptions(QVariantMap{{"resize_mode", "long_edge"}, {"long_edge", " 512 "}})
          .value(QStringLiteral("valid"))
          .toBool());
  EXPECT_FALSE(
      builder.ValidateOptions(QVariantMap{{"resize_mode", "long_edge"}, {"long_edge", "5x"}})
          .value(QStringLiteral("valid"))
          .toBool());
  EXPECT_FALSE(builder.ValidateOptions(QVariantMap{{"resize_mode", "bounds"}, {"width", "12.5"}})
                   .value(QStringLiteral("valid"))
                   .toBool());
  // Print sizes are read in the default locale, as Number.fromLocaleString did.
  const QString print_width = QLocale().toString(210.5);
  EXPECT_TRUE(builder
                  .ValidateOptions(QVariantMap{
                      {"resize_mode", "print"}, {"print_width", print_width}, {"dpi", "300"}})
                  .value(QStringLiteral("valid"))
                  .toBool());
  const QVariantMap rejected =
      builder.ValidateOptions(QVariantMap{{"resize_mode", "print"}, {"print_height", ""}});
  EXPECT_FALSE(rejected.value(QStringLiteral("valid")).toBool());
  EXPECT_EQ(rejected.value(QStringLiteral("field")).toString(), QStringLiteral("print_height"));
}

TEST(ExportRecipeBuilderTest, OutputFolderAppendsEnabledSubfolder) {
  ExportRecipeOptions options = Builder::Defaults();
  options.out_dir             = QStringLiteral("D:/Exports/");
  EXPECT_EQ(Builder::OutputFolder(options), QStringLiteral("D:/Exports/"));
  options.subfolder_enabled = true;
  options.subfolder         = QStringLiteral("  Web  ");
  EXPECT_EQ(Builder::OutputFolder(options), QStringLiteral("D:/Exports/Web"));
  options.out_dir = QStringLiteral("D:\\Exports\\\\");
  EXPECT_EQ(Builder::OutputFolder(options), QStringLiteral("D:\\Exports/Web"));
  options.subfolder = QStringLiteral("   ");
  EXPECT_EQ(Builder::OutputFolder(options), QStringLiteral("D:\\Exports\\\\"));
}

TEST(ExportRecipeBuilderTest, StartWithRejectedOptionDoesNotReachTheHandler) {
  Builder             builder(nullptr);
  ExportRecipeOptions options = Builder::Defaults();
  options.resize_mode         = QStringLiteral("long_edge");
  options.long_edge           = 100;
  const auto error            = builder.Start(options, {});
  ASSERT_TRUE(error.has_value());
  EXPECT_EQ(error->field, QStringLiteral("long_edge"));
}

}  // namespace
}  // namespace alcedo::ui::test
