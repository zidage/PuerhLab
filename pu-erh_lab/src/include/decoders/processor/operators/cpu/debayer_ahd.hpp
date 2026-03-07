//  Copyright 2025 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#pragma once

#include <opencv2/core.hpp>

namespace puerhlab {
namespace CPU {
void BayerRGGB2RGB_AHD(cv::Mat& bayer);
};
};  // namespace puerhlab