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

// Lightweight wrapper for easy_profiler's EASY_BLOCK
// - On MSVC (_MSC_VER), use easy_profiler if available
// - On other compilers, EASY_BLOCK becomes a no-op
//
// Usage:
//   #include "utils/profiler/profiler.hpp"
//   EASY_BLOCK("My block");
//   EASY_BLOCK("Colored", profiler::colors::Red);
//
// Note: Only EASY_BLOCK is provided by this wrapper as requested.

#ifndef PUERH_UTILS_PROFILER_WRAPPER_HPP
#define PUERH_UTILS_PROFILER_WRAPPER_HPP

// Prefer using existing definition if project already included easy/profiler.h
#if !defined(EASY_BLOCK)
#if defined(_MSC_VER)
// MSVC: include easy_profiler only if header is available; otherwise, fall back to no-op.
#if defined(__has_include)
#if __has_include(<easy/profiler.h>)
#include <easy/profiler.h>
#endif
#else
// Fallback path: rely on BUILD_WITH_EASY_PROFILER if provided by build system
#ifdef BUILD_WITH_EASY_PROFILER
#include <easy/profiler.h>
#endif
#endif
#endif  // _MSC_VER

// If EASY_BLOCK is still not defined, provide a no-op
#if !defined(EASY_BLOCK)
#define EASY_BLOCK(...)
#endif

#endif  // !defined(EASY_BLOCK)

#endif  // PUERH_UTILS_PROFILER_WRAPPER_HPP
