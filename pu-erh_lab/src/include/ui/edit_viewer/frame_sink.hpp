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

#include <optional>

namespace puerhlab {
struct ViewportRenderRegion {
  int   x_            = 0;
  int   y_            = 0;
  float scale_x_      = 1.0f;
  float scale_y_      = 1.0f;
};

enum class FramePresentationMode {
  ViewportTransformed,
  RoiFrame,
};

class IFrameSink {
 public:
  virtual ~IFrameSink() {}

  virtual void    EnsureSize(int width, int height) = 0;

  virtual float4* MapResourceForWrite()             = 0;

  virtual void    UnmapResource()                   = 0;

  virtual void    NotifyFrameReady()                = 0;

  // Get the size of the frame
  virtual int     GetWidth() const                  = 0;
  virtual int     GetHeight() const                 = 0;

  // Returns ROI parameters derived from the current viewer transform (if any).
  virtual auto    GetViewportRenderRegion() const -> std::optional<ViewportRenderRegion> {
    return std::nullopt;
  }

  // Sets how the next presented frame should be displayed.
  virtual void    SetNextFramePresentationMode(FramePresentationMode) {}
};
}  // namespace puerhlab
