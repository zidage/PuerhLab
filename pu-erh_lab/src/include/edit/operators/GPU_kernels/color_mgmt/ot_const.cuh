//  Copyright 2025 Yurun Zi
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.

#pragma once

#include <cuda_runtime.h>

// Gamut Compression Constants
__constant__ float smooth_cusps               = 0.12f;
__constant__ float smooth_J                   = 0.0f;
__constant__ float smooth_M                   = 0.27f;
__constant__ float cusp_mid_blend             = 1.3f;

__constant__ float focus_gain_blend           = 0.3f;
__constant__ float focus_adjust_gain          = 0.55f;
__constant__ float focus_distance             = 1.35f;
__constant__ float focus_distant_scaling      = 1.75f;

__constant__ float compression_func_params[4] = {0.75f, 1.1f, 1.3f, 1.0f};

__constant__ int   gamut_table_size           = 360;


