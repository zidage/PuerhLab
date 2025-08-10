#include <opencv2/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/opencv.hpp>

namespace puerhlab {
inline static void GPUCvtColor(cv::Mat& src, cv::Mat& dst, int code) {
  cv::UMat uSrc, uDst;
  src.copyTo(uSrc);
  cv::cvtColor(uSrc, uDst, code);
  uDst.copyTo(dst);
}

inline static void GPUCvtColor(cv::UMat& src, cv::UMat& dst, int code) {
  cv::cvtColor(src, dst, code);
}
};  // namespace puerhlab