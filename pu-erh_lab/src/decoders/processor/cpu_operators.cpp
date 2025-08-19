#include "decoders/processor/cpu_operators.hpp"

#include <easy/profiler.h>

#include <cmath>
#include <cstdint>
#include <opencv2/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>

#include "omp.h"

namespace puerhlab {
namespace CPU {
/**
 * @brief
 *
 * @param raw_clipped
 * @param guide
 * @param mask_clipped
 * @param out
 * @param channel 0: R, 3: B
 * @param window_size
 * @param sigma_spatial
 * @param sigma_guide
 * @param min_valid_neighbors
 */
void        recoverHighlights_RatioGuided(const cv::Mat& raw_clipped, const cv::Mat& guide,
                                          const cv::Mat& mask_clipped, cv::Mat& out, int channel,
                                          int window_size = 25, float sigma_spatial = 8.0f,
                                          float sigma_guide = 0.1f, int min_valid_neighbors = 3,
                                          float feather_radius = 15.0f, float edge_sigma = 0.02f) {
  // cv::Mat resized;
  // cv::resize(raw_clipped, resized, cv::Size(1024, 1024));
  // cv::imshow("Mask", resized);
  // cv::waitKey(0);

  //   CV_Assert(raw_clipped.type() == CV_32FC1 && guide.type() == CV_32FC1 &&
  //                    mask_clipped.type() == CV_8UC1);
  //   CV_Assert(raw_clipped.size() == guide.size() && raw_clipped.size() == mask_clipped.size());
  //   CV_Assert(window_size % 2 == 1);

  //   out                       = raw_clipped.clone();
  //   const int   half_win      = window_size / 2;
  //   const float spatial_denom = -1.0f / (2 * sigma_spatial * sigma_spatial);
  //   const float guide_denom   = -1.0f / (2 * sigma_guide * sigma_guide);

  // #pragma omp parallel for schedule(dynamic)
  //   for (int y = channel; y < raw_clipped.rows - 1; y += 2) {
  //     for (int x = channel; x < raw_clipped.cols - 1; x += 2) {
  //       if (mask_clipped.at<uchar>(y, x) != 0 && y - 1 >= 0 && x - 1 >= 0) {
  //         float       total_ratio  = 0.0f;
  //         float       total_weight = 0.0f;

  //         const float guide_val_center = (guide.at<float>(y, x + 1) + guide.at<float>(y, x - 1) +
  //                                         guide.at<float>(y - 1, x) + guide.at<float>(y + 1, x))
  //                                         /
  //                                        4.0f;

  //         int valid_neighbor = 0;
  //         for (int j = -half_win; j <= half_win; j += 2) {
  //           for (int i = -half_win; i <= half_win; i += 2) {
  //             const int ny = y + j;
  //             const int nx = x + i;

  //             if (ny >= 1 && ny < raw_clipped.rows - 1 && nx >= 1 && nx < raw_clipped.cols - 1) {
  //               // int color_idx = (((y % 2) * 2) + (x % 2)) % 4;
  //               if (mask_clipped.at<uchar>(ny, nx) == 0) {
  //                 ++valid_neighbor;
  //                 const float guide_val_neighbor = guide.at<float>(ny, nx + 1);
  //                 if (guide_val_neighbor > 1e-6f) {
  //                   const float dist_sq    = static_cast<float>(i * i + j * j);
  //                   const float spatial_w  = std::exp(dist_sq * spatial_denom);
  //                   const float guide_diff = guide_val_center - guide_val_neighbor;
  //                   const float guide_w    = std::exp((guide_diff * guide_diff) * guide_denom);

  //                   const float weight     = spatial_w * guide_w;
  //                   const float ratio = raw_clipped.at<float>(ny, nx) / guide_val_neighbor;

  //                   total_ratio += weight * ratio;
  //                   total_weight += weight;
  //                 }
  //               }
  //             }
  //           }
  //         }

  //         if (total_weight > 1e-6f && valid_neighbor >= min_valid_neighbors) {
  //           out.at<float>(y, x) = (total_ratio / total_weight) * guide_val_center;
  //         }
  //         // else: Fallback strategy can be implemented here if needed.
  //       }
  //     }
  //   }
  CV_Assert(raw_clipped.type() == CV_32FC1 && guide.type() == CV_32FC1 &&
                   mask_clipped.type() == CV_8UC1);
  CV_Assert(raw_clipped.size() == guide.size() && raw_clipped.size() == mask_clipped.size());
  CV_Assert(window_size % 2 == 1);

  cv::Mat non_clipped = (mask_clipped == 0);  // uchar, 255 for non-clipped
  cv::Mat dist;
  // distanceTransform expects binary image with non-zero = foreground; use CV_32F result
  non_clipped.convertTo(non_clipped, CV_8U);
  cv::distanceTransform(non_clipped, dist, cv::DIST_L2, 3);
  dist.convertTo(dist, CV_32F);

  cv::Mat gx, gy;
  cv::Sobel(guide, gx, CV_32F, 1, 0, 5);
  cv::Sobel(guide, gy, CV_32F, 0, 1, 5);
  cv::Mat grad_mag;
  cv::magnitude(gx, gy, grad_mag);

  double gmin, gmax;
  cv::minMaxLoc(grad_mag, &gmin, &gmax);
  if (gmax > 1e-12) grad_mag /= static_cast<float>(gmax);

  out                       = raw_clipped.clone();
  const int   half_win      = window_size / 2;
  const float spatial_denom = -1.0f / (1.8f * sigma_spatial * sigma_spatial);
  const float guide_denom   = -1.0f / (1.8f * sigma_guide * sigma_guide);
  const float eps           = 1e-6f;

#pragma omp parallel for schedule(dynamic)
  for (int y = channel; y < raw_clipped.rows - 1; y += 2) {
    for (int x = channel; x < raw_clipped.cols - 1; x += 2) {
      if (mask_clipped.at<uchar>(y, x) != 0) {
        float       total_ratio       = 0.0f;
        float       total_raw         = 0.0f;
        float       total_weight      = 0.0f;

        const float guide_val_center  = guide.at<float>(y, x + 1);
        int         valid_neighbor    = 0;

        // small safety for guide center
        const float guide_center_safe = std::max(guide_val_center, eps);

        for (int j = -half_win; j <= half_win; j += 2) {
          for (int i = -half_win; i <= half_win; i += 2) {
            const int ny = y + j;
            const int nx = x + i;
            if (ny >= 1 && ny < raw_clipped.rows - 1 && nx >= 1 && nx < raw_clipped.cols - 1) {
              if (mask_clipped.at<uchar>(ny, nx) == 0) {
                float guide_val_neighbor = guide.at<float>(ny, nx + 1);
                if (guide_val_neighbor <= eps) continue;

                ++valid_neighbor;
                const float dist_sq    = static_cast<float>(i * i + j * j);
                const float spatial_w  = std::exp(dist_sq * spatial_denom);
                const float guide_diff = guide_center_safe - guide_val_neighbor;
                const float guide_w    = std::exp((guide_diff * guide_diff) * guide_denom);

                const float edge_factor_neighbor =
                    std::exp(-(grad_mag.at<float>(ny, nx) * grad_mag.at<float>(ny, nx)) /
                                    (edge_sigma * edge_sigma));
                const float edge_factor_center =
                    std::exp(-(grad_mag.at<float>(y, x) * grad_mag.at<float>(y, x)) /
                                    (edge_sigma * edge_sigma));
                const float edge_mul = std::min(edge_factor_neighbor, edge_factor_center);

                const float weight   = spatial_w * guide_w * edge_mul;

                const float raw_neighbor = raw_clipped.at<float>(ny, nx);
                const float ratio        = raw_neighbor / std::max(guide_val_neighbor, eps);

                total_ratio += weight * ratio;
                total_raw += weight * raw_neighbor;
                total_weight += weight;
              }
            }
          }
        }  // end neighborhood loops

        if (total_weight > 1e-8f && valid_neighbor >= min_valid_neighbors) {
          const float recovered =
              (total_ratio / total_weight) * guide_center_safe;  // 你的原始恢复值
          const float local_avg = (total_raw / total_weight);    // 邻域真实值的加权平均

          // based on distance to non-clipped pixels, do soft blend:
          float       d         = dist.at<float>(y, x);
          // normalize to [0,1] over feather_radius
          float       t         = d / feather_radius;
          if (t < 0.f) t = 0.f;
          if (t > 1.f) t = 1.f;
          // use smoothstep for smoother transition
          float tt            = t * t * (3.f - 2.f * t);  // smoothstep(t)

          // blend: near boundary (d small) prefer local_avg, deeper inside prefer recovered
          float final_val     = (1.f - tt) * local_avg + tt * recovered;
          out.at<float>(y, x) = final_val;
        } else {
          if (valid_neighbor > 0 && total_weight > 1e-8f) {
            out.at<float>(y, x) = (total_raw / total_weight);
          } else {
            out.at<float>(y, x) = std::min(raw_clipped.at<float>(y, x), guide_center_safe);
          }
        }
      }
    }
  }  // end for y
       }

void WhiteBalanceCorrectionAndHighlightRestore(cv::Mat& img, LibRaw& raw_processor,
                                               std::array<float, 4>& black_level, const float* wb) {
  img.convertTo(img, CV_32FC1, 1.0f / 65535.0f);
  int       w = img.cols;
  int       h = img.rows;

  cv::Mat1f R_plane(h, w, 0.0f);
  cv::Mat1f B_plane(h, w, 0.0f);
  cv::Mat1f G_plane(h, w, 0.0f);
  cv::Mat1f G_guide(h, w, 0.0f);

  cv::Mat1b R_clipped(h, w, (uchar)0.0);
  cv::Mat1b B_clipped(h, w, (uchar)0.0);
  cv::Mat1b G_clipped(h, w, (uchar)0.0);

  // cv::Mat1b  G2_clipped(h, w, (uchar)0.0);

  if (raw_processor.imgdata.color.as_shot_wb_applied != 1) {
    float min       = black_level[0];
    float maximum   = static_cast<float>(raw_processor.imgdata.color.maximum) / 65535.0f - min;

    float threshold = maximum * 0.8f;
    img.forEach<float>([&](float& pixel, const int* pos) {
      int y             = pos[0];
      int x             = pos[1];
      int color_idx     = raw_processor.COLOR(pos[0], pos[1]);

      pixel             = std::max(0.0f, pixel - black_level[color_idx]);

      float muled_pixel = pixel;
      float mask        = (color_idx == 0 || color_idx == 2) ? 1.0f : 0.0f;
      //
      float wb_mul      = (wb[color_idx] / wb[1]) * mask + (1.0f - mask);

      muled_pixel       = muled_pixel * wb_mul;

      //
      // muled_pixel *= (1.0f / maximum);
      // muled_pixel       =
      pixel             = muled_pixel;

      switch (color_idx) {
        case 0:
          if (pixel >= threshold) R_clipped(y, x) = 255;
          R_plane(y, x) = pixel;
          break;
        case 2:
          if (pixel >= threshold) B_clipped(y, x) = 255;
          B_plane(y, x) = pixel;
          break;
        case 1:
        case 3:
          G_plane(y, x) = pixel;
          break;
        default:
          return;
      }
    });

    bool recover = true;
    if (recover) {
      recoverHighlights_RatioGuided(B_plane, G_plane, B_clipped, B_plane, 0, 29, 2.0f, 0.001f);
      recoverHighlights_RatioGuided(R_plane, G_plane, R_clipped, R_plane, 0, 29, 2.0f, 0.001f);
    }

    img = B_plane + R_plane + G_plane;
  }
}

void BayerRGGB2RGB_AHD(cv::Mat& bayer, bool use_AHD, float maximum) {
  if (!use_AHD) {
    bayer.convertTo(bayer, CV_16UC1, 65535.0f);
    cv::cvtColor(bayer, bayer, cv::COLOR_BayerBG2RGB);
    bayer.convertTo(bayer, CV_32FC1, 1.0f / 65535.0f);
    return;
  }

  static int remap[4] = {0, 1, 3, 2};
  const int  h        = bayer.rows;
  const int  w        = bayer.cols;

  cv::Mat1f  mask_R(h, w, 0.0f);
  cv::Mat1f  mask_G_r(h, w, 0.0f);
  cv::Mat1f  mask_G_b(h, w, 0.0f);
  cv::Mat1f  mask_B(h, w, 0.0f);

  EASY_BLOCK("RG1G2B Masks")
  // build masks using forEacll
  bayer.forEach<float>([&](float&, const int* pos) {
    int y           = pos[0];
    int x           = pos[1];
    int color_index = remap[(((y % 2) * 2) + (x % 2)) % 4];  // assume 0:R,1:G_r,2:B,3:G_b
    switch (color_index) {
      case 0:
        mask_R(y, x) = 1.0f;
        break;
      case 1:
        mask_G_r(y, x) = 1.0f;
        break;
      case 2:
        mask_B(y, x) = 1.0f;
        break;
      case 3:
        mask_G_b(y, x) = 1.0f;
        break;
      default:
        break;
    }
  });
  EASY_END_BLOCK;

  cv::Mat1f raw(bayer);

  cv::Mat1f G_final = bayer.clone();
  EASY_BLOCK("Directional Green Estimates and G_final Generation")
#pragma omp parallel for schedule(dynamic)
  for (int y = 2; y < h - 2; ++y) {
    for (int x = 2; x < w - 2; ++x) {
      if (mask_R(y, x) > 0.5f || mask_B(y, x) > 0.5f) {
        float center  = raw(y, x);
        float h_avg   = 0.5f * (raw(y, x - 1) + raw(y, x + 1));
        float h_diff  = 0.25f * (2.0f * center - raw(y, x - 2) - raw(y, x + 2));
        // G_h(y, x)     = h_avg + h_diff;

        float v_avg   = 0.5f * (raw(y - 1, x) + raw(y + 1, x));
        float v_diff  = 0.25f * (2.0f * center - raw(y - 2, x) - raw(y + 2, x));
        // G_v(y, x)     = v_avg + v_diff;

        // Calculate G final now
        float Dh      = std::abs(raw(y, x - 1) - raw(y, x + 1));
        float Dv      = std::abs(raw(y - 1, x) - raw(y + 1, x));
        G_final(y, x) = (Dh < Dv) ? (h_avg + h_diff) : (v_avg + v_diff);
      }
    }
  }
  EASY_END_BLOCK;

  // initialize R_final and B_final (only at their native positions)
  cv::Mat1f R_final = bayer.mul(mask_R);
  cv::Mat1f B_final = bayer.mul(mask_B);

  EASY_BLOCK("Get R/B_final")
// Fill missing R and B using pattern-aware interpolation with forEach
#pragma omp parallel for schedule(dynamic)
  for (int y = 1; y < h - 1; ++y) {
    for (int x = 1; x < w - 1; ++x) {
      // --- fill missing R ---
      if (mask_R(y, x) < 0.5f) {
        float estimateR = 0.0f;
        if (mask_G_r(y, x) > 0.5f) {
          // G on R row: R neighbors are left/right
          float left  = R_final(y, x - 1) - G_final(y, x - 1);
          float right = R_final(y, x + 1) - G_final(y, x + 1);
          estimateR   = G_final(y, x) + 0.5f * (left + right);
        } else if (mask_G_b(y, x) > 0.5f) {
          // G on B row: R neighbors are up/down
          float up   = R_final(y - 1, x) - G_final(y - 1, x);
          float down = R_final(y + 1, x) - G_final(y + 1, x);
          estimateR  = G_final(y, x) + 0.5f * (up + down);
        } else if (mask_B(y, x) > 0.5f) {
          // At a B pixel: R is on diagonals
          float d1  = R_final(y - 1, x - 1) - G_final(y - 1, x - 1);
          float d2  = R_final(y - 1, x + 1) - G_final(y - 1, x + 1);
          float d3  = R_final(y + 1, x - 1) - G_final(y + 1, x - 1);
          float d4  = R_final(y + 1, x + 1) - G_final(y + 1, x + 1);
          estimateR = G_final(y, x) + 0.25f * (d1 + d2 + d3 + d4);
        } else {
          // fallback: simple average of immediate neighbors
          float left  = R_final(y, x - 1) - G_final(y, x - 1);
          float right = R_final(y, x + 1) - G_final(y, x + 1);
          float up    = R_final(y - 1, x) - G_final(y - 1, x);
          float down  = R_final(y + 1, x) - G_final(y + 1, x);
          estimateR   = G_final(y, x) + 0.25f * (left + right + up + down);
        }
        R_final(y, x) = estimateR;
      }

      // --- fill missing B ---
      if (mask_B(y, x) < 0.5f) {
        float estimateB = 0.0f;
        if (mask_G_b(y, x) > 0.5f) {
          // G on B row: B neighbors are left/right
          float left  = B_final(y, x - 1) - G_final(y, x - 1);
          float right = B_final(y, x + 1) - G_final(y, x + 1);
          estimateB   = G_final(y, x) + 0.5f * (left + right);
        } else if (mask_G_r(y, x) > 0.5f) {
          // G on R row: B neighbors are up/down
          float up   = B_final(y - 1, x) - G_final(y - 1, x);
          float down = B_final(y + 1, x) - G_final(y + 1, x);
          estimateB  = G_final(y, x) + 0.5f * (up + down);
        } else if (mask_R(y, x) > 0.5f) {
          // At an R pixel: B is on diagonals
          float d1  = B_final(y - 1, x - 1) - G_final(y - 1, x - 1);
          float d2  = B_final(y - 1, x + 1) - G_final(y - 1, x + 1);
          float d3  = B_final(y + 1, x - 1) - G_final(y + 1, x - 1);
          float d4  = B_final(y + 1, x + 1) - G_final(y + 1, x + 1);
          estimateB = G_final(y, x) + 0.25f * (d1 + d2 + d3 + d4);
        } else {
          // fallback
          float left  = B_final(y, x - 1) - G_final(y, x - 1);
          float right = B_final(y, x + 1) - G_final(y, x + 1);
          float up    = B_final(y - 1, x) - G_final(y - 1, x);
          float down  = B_final(y + 1, x) - G_final(y + 1, x);
          estimateB   = G_final(y, x) + 0.25f * (left + right + up + down);
        }
        B_final(y, x) = estimateB;
      }
    }
  }
  EASY_END_BLOCK;

  EASY_BLOCK("Channel Merging")
  // merge channels into a 3-channel RGB image
  std::vector<cv::Mat> channels = {R_final, G_final, B_final};
  cv::merge(channels, bayer);

  maximum *= 0.98f;
  bayer.forEach<cv::Vec3f>([&](cv::Vec3f& pixel, const int*) {
    pixel[0] = fminf(1.0f, fmaxf(0.0f, pixel[0] / maximum));
    pixel[1] = fminf(1.0f, fmaxf(0.0f, pixel[1] / maximum));
    pixel[2] = fminf(1.0f, fmaxf(0.0f, pixel[2] / maximum));
  });

  EASY_END_BLOCK;
}

void ApplyColorMatrix(cv::Mat& img, const float rgb_cam[][4]) {
  cv::Matx33f rgb_cam_matrix({rgb_cam[0][0], rgb_cam[0][1], rgb_cam[0][2], rgb_cam[1][0],
                              rgb_cam[1][1], rgb_cam[1][2], rgb_cam[2][0], rgb_cam[2][1],
                              rgb_cam[2][2]});

  cv::transform(img, img, rgb_cam_matrix);
}

auto CalculateBlackLevel(const libraw_rawdata_t& raw_data) -> std::array<float, 4> {
  const auto           base_black_level = static_cast<float>(raw_data.color.black);
  std::array<float, 4> black_level      = {
      base_black_level + static_cast<float>(raw_data.color.cblack[0]),
      base_black_level + static_cast<float>(raw_data.color.cblack[1]),
      base_black_level + static_cast<float>(raw_data.color.cblack[2]),
      base_black_level + static_cast<float>(raw_data.color.cblack[3])};

  if (raw_data.color.cblack[4] == 2 && raw_data.color.cblack[5] == 2) {
    for (unsigned int x = 0; x < raw_data.color.cblack[4]; ++x) {
      for (unsigned int y = 0; y < raw_data.color.cblack[5]; ++y) {
        const auto index   = y * 2 + x;
        black_level[index] = raw_data.color.cblack[6 + index];
      }
    }
  }

  for (float& level : black_level) {
    level /= 65535.0f;
  }
  return black_level;
}

auto GetWBCoeff(const libraw_rawdata_t& raw_data) -> const float* { return raw_data.color.cam_mul; }

};  // namespace CPU
};  // namespace puerhlab