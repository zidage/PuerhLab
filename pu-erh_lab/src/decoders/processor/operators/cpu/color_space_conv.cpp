#include "decoders/processor/operators/cpu/color_space_conv.hpp"

#include <cmath>
#include <opencv2/core.hpp>
#include <utility>

#include "decoders/processor/operators/cpu/raw_proc_utils.hpp"

namespace puerhlab {
namespace CPU {
static const cv::Matx33f Bradford_CAT = {0.8951f,  -0.7502f, 0.0389f, 0.2664f, 1.7135f,
                                         -0.0685f, -0.1614f, 0.0367f, 1.0296f};

void                     ApplyColorMatrix(cv::Mat& img, const float rgb_cam[][4]) {
  cv::Matx33f rgb_cam_matrix({rgb_cam[0][0], rgb_cam[0][1], rgb_cam[0][2], rgb_cam[1][0],
                                                  rgb_cam[1][1], rgb_cam[1][2], rgb_cam[2][0], rgb_cam[2][1],
                                                  rgb_cam[2][2]});

  cv::transform(img, img, rgb_cam_matrix);
}

void ApplyColorMatrix(cv::Mat& img, const float rgb_cam[][4], const float* pre_mul,
                      const float* cam_mul) {
  cv::Matx33f normalized_pre_mul({pre_mul[0] / pre_mul[1], 0.f, 0.f, 0.f, pre_mul[1] / pre_mul[1],
                                  0.f, 0.f, 0.f, pre_mul[2] / pre_mul[1]});
  cv::Matx33f normalized_cam_mul({cam_mul[0] / cam_mul[1], 0.f, 0.f, 0.f, cam_mul[1] / cam_mul[1],
                                  0.f, 0.f, 0.f, cam_mul[2] / cam_mul[1]});
  cv::Matx33f pre_to_cam_matrix = {
      normalized_cam_mul(0, 0) / normalized_pre_mul(0, 0), 0.f, 0.f, 0.f,
      normalized_cam_mul(1, 1) / normalized_pre_mul(1, 1), 0.f, 0.f, 0.f,
      normalized_cam_mul(2, 2) / normalized_pre_mul(2, 2)};

  cv::Matx33f rgb_cam_matrix({rgb_cam[0][0], rgb_cam[0][1], rgb_cam[0][2], rgb_cam[1][0],
                              rgb_cam[1][1], rgb_cam[1][2], rgb_cam[2][0], rgb_cam[2][1],
                              rgb_cam[2][2]});

  cv::Matx33f total_matrix = rgb_cam_matrix * pre_to_cam_matrix;
  cv::transform(img, img, total_matrix);
}

static inline std::pair<float, float> PlanckianLocusApprox(int target_K) {
  double T = static_cast<double>(target_K);
  if (T < 1667.0 || T > 25000.0)
    throw std::runtime_error("CCT out of supported range (1667K - 25000K)");

  double x;

  if (T <= 4000.0) {
    x = -0.2661239e9 / (T * T * T) - 0.2343580e6 / (T * T) + 0.8776956e3 / T + 0.179910;
  } else {
    x = -3.0258469e9 / (T * T * T) + 2.1070379e6 / (T * T) + 0.2226347e3 / T + 0.240390;
  }

  double y = -3.0 * x * x + 2.87 * x - 0.275;

  return {static_cast<float>(x), static_cast<float>(y)};
}

static inline cv::Matx33f GetGainMatrixForWb(int target_K, cv::Matx33f cam_xyz) {
  // Step 1: Get target white point xy
  auto        xy = PlanckianLocusApprox(target_K);
  // Step 2: Convert to XYZ
  float       X  = xy.first / xy.second;
  float       Y  = 1.0f;
  float       Z  = (1.0f - xy.first - xy.second) / xy.second;
  cv::Matx31f target_wp(X, Y, Z);
  // Step 3: Convert to camera space
  // target_wp = Bradford_CAT * target_wp;
  cv::Matx33f cam_xyz_inv       = cam_xyz.inv();
  cv::Matx31f cam_response      = cam_xyz * target_wp;

  // Step 4: Get gain matrix (Raw gains to make white point 1,1,1)
  float       g_r               = 1.f / (cam_response(0, 0) + 1e-6f);
  float       g_g               = 1.f / (cam_response(1, 0) + 1e-6f);
  float       g_b               = 1.f / (cam_response(2, 0) + 1e-6f);

  // Normalize to Green = 1.0
  cv::Matx33f final_gain_matrix = {g_r / g_g, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f, g_b / g_g};
  return final_gain_matrix;
}

void ApplyColorMatrix(cv::Mat& img, const float rgb_cam[][4], const float* pre_mul,
                      const float* cam_mul, const int wb_coeffs[][4],
                      std::pair<int, int> user_temp_indices, int user_wb,
                      const float cam_xyz[][3]) {
  cv::Matx33f rgb_cam_matrix({rgb_cam[0][0], rgb_cam[0][1], rgb_cam[0][2], rgb_cam[1][0],
                              rgb_cam[1][1], rgb_cam[1][2], rgb_cam[2][0], rgb_cam[2][1],
                              rgb_cam[2][2]});

  cv::Matx33f normalized_pre_mul({pre_mul[0] / pre_mul[1], 0.f, 0.f, 0.f, pre_mul[1] / pre_mul[1],
                                  0.f, 0.f, 0.f, pre_mul[2] / pre_mul[1]});

  cv::Matx33f cam_xyz_matrix({cam_xyz[0][0], cam_xyz[0][1], cam_xyz[0][2], cam_xyz[1][0],
                              cam_xyz[1][1], cam_xyz[1][2], cam_xyz[2][0], cam_xyz[2][1],
                              cam_xyz[2][2]});
  if (user_temp_indices.first == user_temp_indices.second) {
    // Exact match
    int  idx    = user_temp_indices.first;
    auto wb_mul = wb_coeffs[idx];
    if (wb_mul[1] < 0.1f) {
      // Invalid wb coeffs, fall back to planckian
      auto        gain_matrix        = CPU::GetGainMatrixForWb(user_wb, cam_xyz_matrix);
      // cv::Matx33f total_matrix =
      //     rgb_cam_matrix *
      //     gain_matrix;  // Note: gain_matrix is already relative to G=1, but here we might need
      //     to
      // account for pre_mul if we want to replace it.
      // Actually, if we fall back to planckian, we usually want to replace the pre_mul entirely.
      // But to be consistent with the rest of the function which calculates pre_to_user:
      cv::Matx33f pre_to_user_matrix = {gain_matrix(0, 0) / normalized_pre_mul(0, 0), 0.f, 0.f, 0.f,
                                        gain_matrix(1, 1) / normalized_pre_mul(1, 1), 0.f, 0.f, 0.f,
                                        gain_matrix(2, 2) / normalized_pre_mul(2, 2)};
      cv::Matx33f total_matrix_final = rgb_cam_matrix * pre_to_user_matrix;
      cv::transform(img, img, total_matrix_final);
      return;
    }
    cv::Matx33f wb_matrix({(float)wb_mul[0] / (float)wb_mul[1], 0.f, 0.f, 0.f, 1.0f, 0.f, 0.f, 0.f,
                           (float)wb_mul[2] / (float)wb_mul[1]});
    cv::Matx33f pre_to_user_matrix = {wb_matrix(0, 0) / normalized_pre_mul(0, 0), 0.f, 0.f, 0.f,
                                      wb_matrix(1, 1) / normalized_pre_mul(1, 1), 0.f, 0.f, 0.f,
                                      wb_matrix(2, 2) / normalized_pre_mul(2, 2)};
    cv::Matx33f total_matrix       = rgb_cam_matrix * pre_to_user_matrix;
    cv::transform(img, img, total_matrix);
    return;
  }

  // Interpolate between two wb coeffs
  int  idx1    = user_temp_indices.first;
  int  idx2    = user_temp_indices.second;
  auto wb_mul1 = wb_coeffs[idx1];
  auto wb_mul2 = wb_coeffs[idx2];

  if (wb_mul1[1] < 0.1f && wb_mul2[1] < 0.1f) {
    // Invalid wb coeffs, fall back to planckian
    auto        gain_matrix        = CPU::GetGainMatrixForWb(user_wb, cam_xyz_matrix);
    cv::Matx33f pre_to_user_matrix = {gain_matrix(0, 0) / normalized_pre_mul(0, 0), 0.f, 0.f, 0.f,
                                      gain_matrix(1, 1) / normalized_pre_mul(1, 1), 0.f, 0.f, 0.f,
                                      gain_matrix(2, 2) / normalized_pre_mul(2, 2)};
    cv::Matx33f total_matrix       = rgb_cam_matrix * pre_to_user_matrix;
    cv::transform(img, img, total_matrix);
    return;
  }

  // Extrapolate if one of the wb coeffs is invalid
  if (wb_mul1[1] < 0.1f || wb_mul2[1] < 0.1f) {
    int         valid_idx = (wb_mul1[1] < 0.1f) ? idx2 : idx1;
    auto        wb_mul    = wb_coeffs[valid_idx];

    // 1. Get the valid anchor gain matrix
    cv::Matx33f anchor_gain({(float)wb_mul[0] / (float)wb_mul[1], 0.f, 0.f, 0.f, 1.0f, 0.f, 0.f,
                             0.f, (float)wb_mul[2] / (float)wb_mul[1]});

    // 2. Get the theoretical gain at the anchor temperature
    int         anchor_temp             = CPU::GetTempForWBIndices(valid_idx);
    cv::Matx33f theoretical_anchor_gain = CPU::GetGainMatrixForWb(anchor_temp, cam_xyz_matrix);

    // 3. Compute calibration factor (Calibration Factor = Actual / Theoretical)
    // This is a diagonal matrix representing the deviation of the camera's actual response from the
    // Planckian theoretical value
    cv::Matx33f calibration_factor      = anchor_gain.div(theoretical_anchor_gain);

    // 4. Get the theoretical gain at the target temperature
    auto        target_theoretical_gain = CPU::GetGainMatrixForWb(user_wb, cam_xyz_matrix);

    // 5. Apply calibration factor to get the final gain
    auto        gain_matrix             = target_theoretical_gain.mul(calibration_factor);

    cv::Matx33f pre_to_user_matrix = {gain_matrix(0, 0) / normalized_pre_mul(0, 0), 0.f, 0.f, 0.f,
                                      gain_matrix(1, 1) / normalized_pre_mul(1, 1), 0.f, 0.f, 0.f,
                                      gain_matrix(2, 2) / normalized_pre_mul(2, 2)};
    cv::Matx33f total_matrix       = rgb_cam_matrix * pre_to_user_matrix;
    cv::transform(img, img, total_matrix);
    return;
  }

  // Both are valid, linear interpolate
  float ratio = (user_wb - CPU::GetTempForWBIndices(idx1)) /
                (CPU::GetTempForWBIndices(idx2) - CPU::GetTempForWBIndices(idx1));
  cv::Matx33f wb_matrix1({(float)wb_mul1[0] / (float)wb_mul1[1], 0.f, 0.f, 0.f, 1.0f, 0.f, 0.f, 0.f,
                          (float)wb_mul1[2] / (float)wb_mul1[1]});
  cv::Matx33f wb_matrix2({(float)wb_mul2[0] / (float)wb_mul2[1], 0.f, 0.f, 0.f, 1.0f, 0.f, 0.f, 0.f,
                          (float)wb_mul2[2] / (float)wb_mul2[1]});
  // Interpolated wb matrix
  cv::Matx33f wb_matrix          = wb_matrix1 * (1.0f - ratio) + wb_matrix2 * ratio;
  cv::Matx33f pre_to_user_matrix = {wb_matrix(0, 0) / normalized_pre_mul(0, 0), 0.f, 0.f, 0.f,
                                    wb_matrix(1, 1) / normalized_pre_mul(1, 1), 0.f, 0.f, 0.f,
                                    wb_matrix(2, 2) / normalized_pre_mul(2, 2)};
  cv::Matx33f total_matrix       = rgb_cam_matrix * pre_to_user_matrix;
  cv::transform(img, img, total_matrix);
}
};  // namespace CPU
};  // namespace puerhlab