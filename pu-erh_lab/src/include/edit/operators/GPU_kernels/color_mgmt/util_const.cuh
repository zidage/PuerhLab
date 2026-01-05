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

#include <__clang_cuda_runtime_wrapper.h>
#include <cuda_runtime.h>

// JMh Constants

__constant__ float ra                         = 2.f;
__constant__ float ba                         = 0.05f;

__constant__ float L_A_CONST                  = 100.f;
__constant__ float Y_b_CONST                  = 20.f;
__constant__ float k_CONST                    = 0.00199600798f;
__constant__ float k4_CONST                   = 1.58726383e-11f;
__constant__ float F_L_CONST                  = 0.793700516f;
__constant__ float n_CONST = 0.200000003f;
__constant__ float z_CONST = 1.92721355f;
__constant__ float F_L_W_CONST = 0.907519162f;
__constant__ float A_W_CONST = 12.9472103f;

__constant__ float surround[3]                = {0.9f, 0.59f, 0.9f};

__constant__ float panlrcm[9]                 = {460.f,  460.f, 460.f,  451.f,  -891.f,
                                                 -220.f, 288.f, -261.f, -6300.f};

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

__constant__ float AP0_to_XYZ[9]     = {0.986519,  0.023971,  -0.010490, 0.359689, 0.714586,
                                        -0.074275, -0.000386, 0.000029,  1.000356};

__constant__ float AP0_XYZ_TO_RGB[9] = {
    1.04981101f, -0.495903015f,    0.f,           0.f,         1.37331307f,
    0.f,         -9.74845389e-05f, 0.0982400328f, 0.991252005f};

__constant__ float AP1_to_XYZ[9]       = {0.687209f, 0.159347f,  0.153444f, 0.282567f, 0.664611f,
                                          0.052823f, -0.005795f, 0.003998f, 1.001797f};

__constant__ float CAM16_XYZ_TO_RGB[9] = {0.364074498f,  -0.222245097f, -0.002067619f,
                                          0.594700813f,  1.07385552f,   0.0488260463f,
                                          0.0411012731f, 0.14794533f,   0.950387537f};