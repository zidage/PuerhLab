#pragma once

#include <cstdint>
#include <opencv2/core/cuda.hpp>
#include <opencv2/opencv.hpp>
#include <vector>

namespace puerhlab {
class ImageBuffer {
 private:
  cv::Mat          _cpu_data;
  // TODO: NOT USED FOR NOW: 2025-6-30
  cv::cuda::GpuMat _gpu_data;

  bool             _data_valid;

  // TODO: NOT USED FOR NOW: 2025-6-30
  void             SyncToGPU();
  void             SyncToCPU();

 public:
  ImageBuffer() = default;
  ImageBuffer(std::vector<uint8_t> buffer);

  void                    ReadFromVectorBuffer(std::vector<uint8_t> buffer);

  const cv::Mat&          GetCPUData();
  const cv::cuda::GpuMat& GetGPUData();
};
};  // namespace puerhlab