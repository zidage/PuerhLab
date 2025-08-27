#include "decoders/processor/operators/cpu/debayer_amaze.hpp"

#include <algorithm>  // for std::sort
#include <array>      // for std::array
#include <opencv2/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgproc.hpp>  // For cv::merge
#include <vector>               // for std::vector

#include "decoders/processor/operators/cpu/raw_proc_utils.hpp"

#define InRangeX(x, raw) ((x) >= 0 && (x) < (raw.cols))
#define InRangeY(y, raw) ((y) >= 0 && (y) < (raw.rows))

namespace puerhlab {
namespace CPU {

enum class CH { R, G, B };
static float eps      = 1e-8f;
// RGGB Bayer Pattern Mapping:
// 0 -> R at (even, even)
// 1 -> G at (even, odd)
// 2 -> B at (odd, odd)
// 3 -> G at (odd, even)
static int   remap[4] = {0, 1, 3, 2};
#define COLOR(y, x) remap[(((y % 2) * 2) + (x % 2))]

inline static float DirectionVariation(const cv::Mat1f& raw, int y, int x) {
  float v_h = 0.f, v_v = 0.f, v_d1 = 0.f, v_d2 = 0.f;

  // Horizontal Variation
  if (InRangeX(x - 2, raw) && InRangeX(x + 2, raw)) {
    v_h += std::abs(2 * raw(y, x) - raw(y, x - 2) - raw(y, x + 2));
  } else if (InRangeX(x - 1, raw) && InRangeX(x + 1, raw)) {
    v_h += std::abs(raw(y, x - 1) - raw(y, x + 1));
  }

  // Vertical Variation
  if (InRangeY(y - 2, raw) && InRangeY(y + 2, raw)) {
    v_v += std::abs(2 * raw(y, x) - raw(y - 2, x) - raw(y + 2, x));
  } else if (InRangeY(y - 1, raw) && InRangeY(y + 1, raw)) {
    v_v += std::abs(raw(y - 1, x) - raw(y + 1, x));
  }

  // Sum of Horizontal and Vertical gives the first part of the decision
  return v_h + v_v;
}

inline static void InterpolateGreen(const cv::Mat1f& raw, cv::Mat1f& G, cv::Mat1f& v_buffer) {
  int h = raw.rows;
  int w = raw.cols;

#pragma omp parallel for
  for (int y = 2; y < h - 2; ++y) {
    for (int x = 2; x < w - 2; ++x) {
      // Only interpolate at R and B locations (where G is initially 0)
      if (COLOR(y, x) == 0 || COLOR(y, x) == 2) {
        // Step 1: Calculate Horizontal and Vertical Gradients at this R/B location
        // The gradient is estimated using the difference of the green neighbors
        float v_h = std::abs(raw(y, x - 1) - raw(y, x + 1)) +
                    std::abs(2 * raw(y, x) - raw(y, x - 2) - raw(y, x + 2));
        float v_v = std::abs(raw(y - 1, x) - raw(y + 1, x)) +
                    std::abs(2 * raw(y, x) - raw(y - 2, x) - raw(y + 2, x));

        v_buffer(y, x) = v_h + v_v;  // Store for chroma interpolation later

        // Step 2: Calculate Horizontal and Vertical Green estimates
        float g_h      = (raw(y, x - 1) + raw(y, x + 1)) / 2.0f +
                    (2 * raw(y, x) - raw(y, x - 2) - raw(y, x + 2)) / 4.0f;
        float g_v = (raw(y - 1, x) + raw(y + 1, x)) / 2.0f +
                    (2 * raw(y, x) - raw(y - 2, x) - raw(y + 2, x)) / 4.0f;

        // Step 3: Weighted average based on gradients
        if (v_h < v_v) {
          G(y, x) = g_h;
        } else if (v_v < v_h) {
          G(y, x) = g_v;
        } else {
          G(y, x) = (g_h + g_v) / 2.0f;
        }
      }
    }
  }
}

inline static void InterpolateChroma(const cv::Mat1f& raw, cv::Mat1f& R, cv::Mat1f& G, cv::Mat1f& B,
                                     const cv::Mat1f& v_buffer) {
  int h = R.rows;
  int w = R.cols;

  // --- Step 1: Interpolate R and B at G locations ---
#pragma omp parallel for
  for (int y = 1; y < h - 1; ++y) {
    for (int x = 1; x < w - 1; ++x) {
      if (COLOR(y, x) == 1 || COLOR(y, x) == 3) {  // If it's a G pixel
        // Interpolate R
        float r_h = 0, r_v = 0;
        int   c_h = 0, c_v = 0;
        if (COLOR(y, x - 1) == 0) {
          r_h += raw(y, x - 1) - G(y, x - 1);
          c_h++;
        }
        if (COLOR(y, x + 1) == 0) {
          r_h += raw(y, x + 1) - G(y, x + 1);
          c_h++;
        }
        if (COLOR(y - 1, x) == 0) {
          r_v += raw(y - 1, x) - G(y - 1, x);
          c_v++;
        }
        if (COLOR(y + 1, x) == 0) {
          r_v += raw(y + 1, x) - G(y + 1, x);
          c_v++;
        }

        float diff_r = 0;
        if (c_h > 0 && c_v > 0) {
          diff_r = (r_h / c_h + r_v / c_v) / 2.0f;
        } else if (c_h > 0) {
          diff_r = r_h / c_h;
        } else if (c_v > 0) {
          diff_r = r_v / c_v;
        }
        R(y, x)   = G(y, x) + diff_r;

        // Interpolate B
        float b_h = 0, b_v = 0;
        c_h = 0, c_v = 0;
        if (COLOR(y, x - 1) == 2) {
          b_h += raw(y, x - 1) - G(y, x - 1);
          c_h++;
        }
        if (COLOR(y, x + 1) == 2) {
          b_h += raw(y, x + 1) - G(y, x + 1);
          c_h++;
        }
        if (COLOR(y - 1, x) == 2) {
          b_v += raw(y - 1, x) - G(y - 1, x);
          c_v++;
        }
        if (COLOR(y + 1, x) == 2) {
          b_v += raw(y + 1, x) - G(y + 1, x);
          c_v++;
        }

        float diff_b = 0;
        if (c_h > 0 && c_v > 0) {
          diff_b = (b_h / c_h + b_v / c_v) / 2.0f;
        } else if (c_h > 0) {
          diff_b = b_h / c_h;
        } else if (c_v > 0) {
          diff_b = b_v / c_v;
        }
        B(y, x) = G(y, x) + diff_b;
      }
    }
  }

  // --- Step 2: Interpolate missing chroma at R/B locations ---
#pragma omp parallel for
  for (int y = 1; y < h - 1; ++y) {
    for (int x = 1; x < w - 1; ++x) {
      if (COLOR(y, x) == 0) {  // At an R location, interpolate B
        float diff_b_sum = 0;
        int   count      = 0;
        if (InRangeX(x - 1, raw) && InRangeY(y - 1, raw)) {
          diff_b_sum += B(y - 1, x - 1) - G(y - 1, x - 1);
          count++;
        }
        if (InRangeX(x + 1, raw) && InRangeY(y - 1, raw)) {
          diff_b_sum += B(y - 1, x + 1) - G(y - 1, x + 1);
          count++;
        }
        if (InRangeX(x - 1, raw) && InRangeY(y + 1, raw)) {
          diff_b_sum += B(y + 1, x - 1) - G(y + 1, x - 1);
          count++;
        }
        if (InRangeX(x + 1, raw) && InRangeY(y + 1, raw)) {
          diff_b_sum += B(y + 1, x + 1) - G(y + 1, x + 1);
          count++;
        }
        if (count > 0) B(y, x) = G(y, x) + diff_b_sum / count;

      } else if (COLOR(y, x) == 2) {  // At a B location, interpolate R
        float diff_r_sum = 0;
        int   count      = 0;
        if (InRangeX(x - 1, raw) && InRangeY(y - 1, raw)) {
          diff_r_sum += R(y - 1, x - 1) - G(y - 1, x - 1);
          count++;
        }
        if (InRangeX(x + 1, raw) && InRangeY(y - 1, raw)) {
          diff_r_sum += R(y - 1, x + 1) - G(y - 1, x + 1);
          count++;
        }
        if (InRangeX(x - 1, raw) && InRangeY(y + 1, raw)) {
          diff_r_sum += R(y + 1, x - 1) - G(y + 1, x - 1);
          count++;
        }
        if (InRangeX(x + 1, raw) && InRangeY(y + 1, raw)) {
          diff_r_sum += R(y + 1, x + 1) - G(y + 1, x + 1);
          count++;
        }
        if (count > 0) R(y, x) = G(y, x) + diff_r_sum / count;
      }
    }
  }
}

inline static float MedianOf3x3(const cv::Mat1f& G, const cv::Mat1f& R_B, int y, int x) {
  std::array<float, 9> neigh;
  int                  k = 0;
  for (int dy = -1; dy <= 1; ++dy) {
    for (int dx = -1; dx <= 1; ++dx) {
      neigh[k++] = R_B(y + dy, x + dx) - G(y + dy, x + dx);
    }
  }
  std::sort(neigh.begin(), neigh.end());
  return neigh[4];
}

inline static void FalseColorSuppression(cv::Mat1f& R, cv::Mat1f& G, cv::Mat1f& B) {
  int h = R.rows;
  int w = R.cols;
#pragma omp parallel for
  for (int y = 1; y < h - 1; ++y) {
    for (int x = 1; x < w - 1; ++x) {
      float cd_R       = R(y, x) - G(y, x);
      float cd_B       = B(y, x) - G(y, x);

      float med_cd_R   = MedianOf3x3(G, R, y, x);
      float med_cd_B   = MedianOf3x3(G, B, y, x);

      float diff_R     = std::abs(cd_R - med_cd_R);
      float diff_B     = std::abs(cd_B - med_cd_B);

      float max_diff_R = (std::abs(med_cd_R) + 0.02f) * 2.0f;
      float max_diff_B = (std::abs(med_cd_B) + 0.02f) * 2.0f;

      if (diff_R > max_diff_R) {
        R(y, x) = G(y, x) + med_cd_R;
      }
      if (diff_B > max_diff_B) {
        B(y, x) = G(y, x) + med_cd_B;
      }
    }
  }
}

void BayerRGGB2RGB_AMaZe(cv::Mat& bayer_io) {
  const int h   = bayer_io.rows;
  const int w   = bayer_io.cols;

  cv::Mat1f raw = bayer_io;
  cv::Mat1f R(h, w, 0.0f);
  cv::Mat1f G(h, w, 0.0f);
  cv::Mat1f B(h, w, 0.0f);

#pragma omp parallel for
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      switch (COLOR(y, x)) {
        case 0:
          R(y, x) = raw(y, x);
          break;  // R
        case 1:
          G(y, x) = raw(y, x);
          break;  // G
        case 2:
          B(y, x) = raw(y, x);
          break;  // B
        case 3:
          G(y, x) = raw(y, x);
          break;  // G
      }
    }
  }

  cv::Mat1f v_buffer(cv::Size(w, h), 0.0f);

  InterpolateGreen(raw, G, v_buffer);

  InterpolateChroma(raw, R, G, B, v_buffer);

  FalseColorSuppression(R, G, B);

  std::vector<cv::Mat> channels = {B, G, R};  // OpenCV's merge expects BGR order
  cv::merge(channels, bayer_io);
}

};  // namespace CPU
};  // namespace puerhlab