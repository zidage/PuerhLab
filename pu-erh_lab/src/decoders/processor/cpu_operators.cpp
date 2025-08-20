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
void recoverHighlights_RatioGuided(const cv::Mat& raw_clipped, const cv::Mat& guide,
                                   const cv::Mat& mask_clipped, cv::Mat& out, int channel_y,
                                   int channel_x, int step_y, int step_x, int window_size = 25,
                                   float sigma_spatial = 25.0f, float sigma_guide = 2.f,
                                   int min_valid_neighbors = 0, float feather_radius = 10000.0f,
                                   float edge_sigma = 0.0f, float residual_beta = 0.5f) {
  CV_Assert(raw_clipped.type() == CV_32FC1 && guide.type() == CV_32FC1 &&
            mask_clipped.type() == CV_8UC1);
  CV_Assert(raw_clipped.size() == guide.size() && raw_clipped.size() == mask_clipped.size());
  CV_Assert(window_size % 2 == 1);

  cv::Mat non_clipped = (mask_clipped == 0);  // uchar, 255 for non-clipped
  cv::Mat dist;
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
  const float spatial_denom = -1.0f / (2.f * sigma_spatial * sigma_spatial);
  const float guide_denom   = -1.0f / (2.f * sigma_guide * sigma_guide);
  const float eps           = 1e-6f;

#pragma omp parallel for schedule(dynamic)
  for (int y = channel_y; y < raw_clipped.rows - 2; y += step_y) {
    for (int x = channel_x; x < raw_clipped.cols - 2; x += step_x) {
      if (mask_clipped.at<uchar>(y, x) != 0 && x > 1 && y > 1) {
        // gather weighted sums for weighted linear regression r = a*g + b
        float S_w = 0.f, S_g = 0.f, S_r = 0.f, S_g2 = 0.f, S_gr = 0.f;
        int   valid_neighbor = 0;
        float min_r = 1e30f, max_r = -1e30f;
        float guide_center_safe;
        if (channel_x - channel_y != 0) {
          guide_center_safe = std::max((guide.at<float>(y, x + 2) + guide.at<float>(y, x - 2) +
                                        guide.at<float>(y - 2, x) + guide.at<float>(y + 2, x)) /
                                           4.0f,
                                       eps);
        } else {
          // safe guide center (average 4-neighbors to reduce noise / avoid clipped guide)
          guide_center_safe =
              std::max((guide.at<float>(y + 1, x + 1) + guide.at<float>(y - 1, x - 1) +
                        guide.at<float>(y - 1, x + 1) + guide.at<float>(y + 1, x - 1)) /
                           4.0f,
                       eps);
        }

        // Collect neighbor residuals array for median (optional) - small vector
        std::vector<std::pair<float, float>> residuals;  // pair(weight, residual)

        for (int j = -half_win; j <= half_win; j += 2) {
          for (int i = -half_win; i <= half_win; i += 2) {
            const int ny = y + j;
            const int nx = x + i;
            if (ny >= 2 && ny < raw_clipped.rows - 2 && nx >= 2 && nx < raw_clipped.cols - 2) {
              if (mask_clipped.at<uchar>(ny, nx) == 0) {
                // safe guide center (average 4-neighbors to reduce noise / avoid clipped guide)
                float guide_val_neighbor;

                if (channel_x - channel_y != 0) {
                  guide_val_neighbor = (guide.at<float>(ny, nx + 1) + guide.at<float>(ny, nx - 1) +
                                        guide.at<float>(ny - 1, nx) + guide.at<float>(ny + 1, nx)) /
                                       4.0f;
                } else {
                  guide_val_neighbor =
                      (guide.at<float>(ny + 1, nx + 1) + guide.at<float>(ny - 1, nx - 1) +
                       guide.at<float>(ny - 1, nx + 1) + guide.at<float>(ny + 1, nx - 1)) /
                      4.0f;
                }

                if (guide_val_neighbor <= eps) continue;

                // spatial weight
                const float dist_sq    = static_cast<float>(i * i + j * j);
                const float spatial_w  = std::exp(dist_sq * spatial_denom);

                // guide similarity weight (gaussian on guide intensity difference)
                const float guide_diff = guide_center_safe - guide_val_neighbor;
                const float guide_w    = std::exp((guide_diff * guide_diff) * guide_denom);

                // edge-aware factor (min of center/neighbor edge factors)
                const float edge_factor_neighbor =
                    std::exp(-(grad_mag.at<float>(ny, nx) * grad_mag.at<float>(ny, nx)) /
                             (edge_sigma * edge_sigma));
                const float edge_factor_center =
                    std::exp(-(grad_mag.at<float>(y, x) * grad_mag.at<float>(y, x)) /
                             (edge_sigma * edge_sigma));
                const float edge_mul = std::min(edge_factor_neighbor, edge_factor_center);

                const float w        = spatial_w * guide_w * edge_mul;
                const float r        = raw_clipped.at<float>(ny, nx);
                const float g        = guide_val_neighbor;

                // accumulate
                S_w += w;
                S_g += w * g;
                S_r += w * r;
                S_g2 += w * g * g;
                S_gr += w * g * r;

                min_r = std::min(min_r, r);
                max_r = std::max(max_r, r);
                ++valid_neighbor;
              }
            }
          }
        }  // neighborhood

        if (S_w > 1e-8f && valid_neighbor >= min_valid_neighbors) {
          // weighted linear regression solve
          const float denom = (S_w * S_g2 - S_g * S_g);
          float       a = 0.f, b = 0.f;
          bool        model_ok = true;
          if (std::abs(denom) > 1e-8f) {
            a = (S_w * S_gr - S_g * S_r) / denom;
            b = (S_g2 * S_r - S_g * S_gr) / denom;
          } else {
            model_ok = false;
          }

          // fallback: if regression unstable, use weighted average
          // local weighted average:
          float local_avg = S_r / S_w;

          float predicted = local_avg;
          if (model_ok) {
            predicted = a * guide_center_safe + b;
          }

          // compute a small weighted residual correction:
          // Estimate predicted at local avg guide and take difference to local_avg raw
          float local_avg_g         = S_g / S_w;
          float predicted_at_localg = a * local_avg_g + b;
          float detail_offset       = (local_avg - predicted_at_localg);  // captures local HF bias
          float corrected           = predicted + residual_beta * detail_offset;

          // clamp to neighbor min/max to avoid wild extrapolation
          corrected                 = std::min(std::max(corrected, min_r), max_r);

          // feather with local_avg based on distance transform (near boundary prefer local_avg)
          float d                   = dist.at<float>(y, x);
          float t                   = d / feather_radius;
          if (t < 0.f) t = 0.f;
          if (t > 1.f) t = 1.f;
          float tt            = t * t * (3.f - 2.f * t);

          float final_val     = (1.f - tt) * local_avg + tt * corrected;
          out.at<float>(y, x) = final_val;
        } else if (valid_neighbor > 0 && S_w > 1e-8f) {
          // weak case: fallback to local weighted average
          // out.at<float>(y, x) = (S_r / S_w);
        }
      }  // if clipped
    }  // for x
  }  // for y
}

void WhiteBalanceCorrectionAndHighlightRestore(cv::Mat& img, LibRaw& raw_processor,
                                               std::array<float, 4>& black_level, const float* wb) {
  img.convertTo(img, CV_32FC1, 1.0f / 65535.0f);
  int       w = img.cols;
  int       h = img.rows;

  cv::Mat1f G_guide(h, w, 0.0f);

  cv::Mat1b R_clipped(h, w, (uchar)0.0);
  cv::Mat1b B_clipped(h, w, (uchar)0.0);
  cv::Mat1b G1_clipped(h, w, (uchar)0.0);
  cv::Mat1b G2_clipped(h, w, (uchar)0.0);

  // cv::Mat1b  G2_clipped(h, w, (uchar)0.0);

  if (raw_processor.imgdata.color.as_shot_wb_applied != 1) {
    float min       = black_level[0];
    float maximum   = static_cast<float>(raw_processor.imgdata.color.maximum) / 65535.0f - min;

    float threshold = maximum * 0.7f;
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

      pixel             = muled_pixel;
      // switch (color_idx) {
      //   case 0:
      //     if (pixel >= threshold) R_clipped(y, x) = 255;
      //     break;
      //   case 2:
      //     if (pixel >= threshold) B_clipped(y, x) = 255;
      //     break;
      //   case 1:
      //     if (pixel < threshold)
      //       G_guide(y, x) = pixel;
      //     else
      //       G1_clipped(y, x) = 255;
      //     break;
      //   case 3:
      //     if (pixel < threshold)
      //       G_guide(y, x) = pixel;
      //     else
      //       G2_clipped(y, x) = 255;
      //     break;
      //   default:
      //     return;
      // }
    });

    // cv::Mat resized;
    // cv::resize(G_guide, resized, cv::Size(512, 512));

    // cv::imshow("G_guide", resized);
    // cv::waitKey(0);

    // bool recover = false;
    // if (recover) {
    //   recoverHighlights_RatioGuided(img, G_guide, B_clipped, img, 1, 1, 2, 2);
    //   recoverHighlights_RatioGuided(img, G_guide, R_clipped, img, 0, 0, 2, 2);
    //   // recoverHighlights_RatioGuided(img, G_guide, G1_clipped, img, 0, 1, 2, 2);
    //   // recoverHighlights_RatioGuided(img, G_guide, G2_clipped, img, 1, 0, 2, 2);
    // }

    // img.forEach<float>([&](float& pixel, const int* pos) {
    //   int color_idx = raw_processor.COLOR(pos[0], pos[1]);
    // });
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

  // maximum *= 1.5f;
  // bayer.forEach<cv::Vec3f>([&](cv::Vec3f& pixel, const int*) {
  //   pixel[0] = fminf(1.0f, fmaxf(0.0f, pixel[0] / maximum));
  //   pixel[1] = fminf(1.0f, fmaxf(0.0f, pixel[1] / maximum));
  //   pixel[2] = fminf(1.0f, fmaxf(0.0f, pixel[2] / maximum));
  // });

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