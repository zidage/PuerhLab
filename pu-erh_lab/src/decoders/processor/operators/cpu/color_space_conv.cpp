#include "decoders/processor/operators/cpu/color_space_conv.hpp"

namespace puerhlab {
namespace CPU {
void ApplyColorMatrix(cv::Mat& img, const float rgb_cam[][4]) {
  cv::Matx33f rgb_cam_matrix({rgb_cam[0][0], rgb_cam[0][1], rgb_cam[0][2], rgb_cam[1][0],
                              rgb_cam[1][1], rgb_cam[1][2], rgb_cam[2][0], rgb_cam[2][1],
                              rgb_cam[2][2]});

  cv::transform(img, img, rgb_cam_matrix);
}
};  // namespace CPU
};  // namespace puerhlab