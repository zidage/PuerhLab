#pragma once

#include <easy/profiler.h>
#include <gtest/gtest.h>

#include <exiv2/exiv2.hpp>
#include <filesystem>

#include "edit/operators/operator_registeration.hpp"


namespace puerhlab {
class PipelineTests : public ::testing::Test {
 protected:
  std::filesystem::path db_path_;

  // Run before any unit test runs
  void                  SetUp() override {
    EASY_PROFILER_ENABLE;
    Exiv2::LogMsg::setLevel(Exiv2::LogMsg::Level::mute);
    // Create a unique db file location
    db_path_ = std::filesystem::temp_directory_path() / "test_db.db";
    // Make sure there is not existing db
    if (std::filesystem::exists(db_path_)) {
      std::filesystem::remove(db_path_);
    }
    RegisterAllOperators();
  }

  // Run before any unit test runs
  void TearDown() override {
    // Clean up the DB file
    if (std::filesystem::exists(db_path_)) {
      std::filesystem::remove(db_path_);
    }
    profiler::dumpBlocksToFile(
      "D:\\Projects\\pu-erh_lab\\pu-erh_lab\\tests\\resources\\temp_folder\\test_profile.prof");
  }
};
}  // namespace puerhlab