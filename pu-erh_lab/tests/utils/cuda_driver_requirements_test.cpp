//  Copyright 2026 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#include <gtest/gtest.h>

#include "utils/cuda/cuda_driver_requirements.hpp"

TEST(CudaDriverRequirementsTest, FormatsCudaDriverVersionNumbers) {
  EXPECT_EQ(puerhlab::cuda::FormatCudaVersion(12080), "12.8");
  EXPECT_EQ(puerhlab::cuda::FormatCudaVersion(13010), "13.1");
  EXPECT_EQ(puerhlab::cuda::FormatCudaVersion(0), "unknown");
}

TEST(CudaDriverRequirementsTest, ComparesDriverVersionsAgainstMinimumRequirement) {
  EXPECT_TRUE(puerhlab::cuda::IsCudaDriverVersionSupported(12080));
  EXPECT_TRUE(puerhlab::cuda::IsCudaDriverVersionSupported(12100));
  EXPECT_FALSE(puerhlab::cuda::IsCudaDriverVersionSupported(12070));
  EXPECT_TRUE(puerhlab::cuda::IsCudaDriverVersionSupported(12070, 12070));
}
