#include <opencv2/core/cuda.hpp>
#include <opencv2/core/cuda_stream_accessor.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>

#include "decoders/processor/operators/gpu/cuda_color_space_conv.hpp"


namespace puerhlab {
namespace CUDA {
__constant__ float M_const[9];

__global__ void    ApplyColorMatrixKernel(const uchar* srcptr, uchar* dstptr, int rows, int cols,
                                          size_t src_step, size_t dst_step) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= cols || y >= rows) return;

  const float* src_p = (const float*)(srcptr + y * src_step) + x * 3;
  float*       dst_p = (float*)(dstptr + y * dst_step) + x * 3;

  float        r     = src_p[0];
  float        g     = src_p[1];
  float        b     = src_p[2];

  dst_p[0]           = M_const[0] * r + M_const[1] * g + M_const[2] * b;
  dst_p[1]           = M_const[3] * r + M_const[4] * g + M_const[5] * b;
  dst_p[2]           = M_const[6] * r + M_const[7] * g + M_const[8] * b;
}

void ApplyColorMatrix_Helper(const cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst,
                             const cv::Mat& matrix, cv::cuda::Stream& stream) {
  CV_Assert(src.type() == CV_32FC3);
  CV_Assert(matrix.isContinuous() && matrix.rows == 3 && matrix.cols == 3 &&
            matrix.type() == CV_32F);

  if (dst.empty() || dst.size() != src.size() || dst.type() != src.type())
    dst.create(src.size(), src.type());

  cudaStream_t cudaStream = cv::cuda::StreamAccessor::getStream(stream);

  cudaMemcpyToSymbolAsync(M_const, matrix.data, 9 * sizeof(float), 0, cudaMemcpyHostToDevice,
                          cudaStream);

  dim3 block(32, 32);
  dim3 grid((src.cols + block.x - 1) / block.x, (src.rows + block.y - 1) / block.y);

  ApplyColorMatrixKernel<<<grid, block, 0, cudaStream>>>(src.data, dst.data, src.rows, src.cols,
                                                         src.step, dst.step);
}

void ApplyColorMatrix(cv::cuda::GpuMat& img, const float rgb_cam[][4]) {
  cv::Matx33f      rgb_cam_matrix({rgb_cam[0][0], rgb_cam[0][1], rgb_cam[0][2], rgb_cam[1][0],
                                   rgb_cam[1][1], rgb_cam[1][2], rgb_cam[2][0], rgb_cam[2][1],
                                   rgb_cam[2][2]});
  cv::cuda::Stream stream;
  ApplyColorMatrix_Helper(img, img, cv::Mat(rgb_cam_matrix), stream);
  stream.waitForCompletion();
}
};  // namespace CUDA
};  // namespace puerhlab