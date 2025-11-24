#include "decoders/processor/operators/cpu/highlight_reconstruct.hpp"

#include <libraw/libraw_types.h>
#include <opencv2/core/hal/interface.h>

#include <cmath>
#include <opencv2/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#define HL_POWERF 3.0f

static int fc[2][2] = {{0, 1}, {2, 1}};  // R=0, G1=1, B=2, G2=1
#ifndef FC
#define FC(y, x) fc[(y) & 1][(x) & 1]
#endif

namespace puerhlab {
namespace CPU {
// Helper: Fast cubic root and power to avoid expensive powf
static inline float fast_pow_inv_3(float x) { return cbrtf(x); }
static inline float fast_pow_3(float x) { return x * x * x; }

inline static float sqr(float x) { return x * x; }

inline static auto  GetBlackSom(const libraw_rawdata_t& raw_data) -> std::array<float, 3> {
  const auto           base_black_level = static_cast<float>(raw_data.color.black);
  std::array<float, 3> black_level      = {
      (base_black_level + static_cast<float>(raw_data.color.cblack[0])) / 65535.0f,
      (base_black_level + static_cast<float>(raw_data.color.cblack[1])) / 65535.0f,
      (base_black_level + static_cast<float>(raw_data.color.cblack[2])) / 65535.0f};

  return black_level;
}

inline static auto GetScaleMul(const libraw_rawdata_t& raw_data) -> std::array<float, 3> {
  // auto                 pre_mul     = raw_data.color.pre_mul;
  auto                 cam_mul = raw_data.color.cam_mul;

  // float                max_cam_mul = std::max({cam_mul[0], cam_mul[1], cam_mul[2]});

  auto                 c_white = raw_data.color.maximum;
  auto                 c_black = raw_data.color.black;

  std::array<float, 3> scale_mul;
  for (int c = 0; c < 3; ++c) {
    scale_mul[c] = (cam_mul[c] / cam_mul[1]) / (((int)c_white - (int)c_black) / 65535.0f);
  }

  return scale_mul;
}

inline static auto GetClMax(const libraw_rawdata_t& raw_data) -> std::array<float, 3> {
  std::array<float, 3> cl_max;

  int                  c_white    = (int)(raw_data.color.maximum);

  std::array<float, 3> cblack_som = GetBlackSom(raw_data);
  std::array<float, 3> scale_mul  = GetScaleMul(raw_data);

  for (int c = 0; c < 3; ++c) {
    cl_max[c] = ((c_white - cblack_som[c]) * scale_mul[c]) / 65535.0f;
  }

  return cl_max;
}

inline static auto GetChMax(cv::Mat& img) -> std::array<float, 3> {
  cv::Mat resized;
  cv::resize(img, resized, cv::Size(img.cols / 8, img.rows / 8));

  std::array<float, 3> ch_max = {0.f, 0.f, 0.f};

  resized.forEach<cv::Vec3f>([&](cv::Vec3f& pixel, const int*) {
    if (pixel[0] > ch_max[0]) {
      ch_max[0] = pixel[0];
    }
    if (pixel[1] > ch_max[1]) {
      ch_max[1] = pixel[1];
    }
    if (pixel[2] > ch_max[2]) {
      ch_max[2] = pixel[2];
    }
  });

  return ch_max;
}

size_t round_size(const size_t size, const size_t alignment) {
  // Round the size of a buffer to the closest higher multiple
  return ((size % alignment) == 0) ? size : ((size - 1) / alignment + 1) * alignment;
}

static inline char mask_dilate(const unsigned char* in, const int w1) {
  if (in[0]) return 1;

  if (in[-w1 - 1] | in[-w1] | in[-w1 + 1] | in[-1] | in[1] | in[w1 - 1] | in[w1] | in[w1 + 1])
    return 1;

  const int w2 = 2 * w1;
  const int w3 = 3 * w1;
  return (in[-w3 - 2] | in[-w3 - 1] | in[-w3] | in[-w3 + 1] | in[-w3 + 2] | in[-w2 - 3] |
          in[-w2 - 2] | in[-w2 - 1] | in[-w2] | in[-w2 + 1] | in[-w2 + 2] | in[-w2 + 3] |
          in[-w1 - 3] | in[-w1 - 2] | in[-w1 + 2] | in[-w1 + 3] | in[-3] | in[-2] | in[2] | in[3] |
          in[w1 - 3] | in[w1 - 2] | in[w1 + 2] | in[w1 + 3] | in[w2 - 3] | in[w2 - 2] | in[w2 - 1] |
          in[w2] | in[w2 + 1] | in[w2 + 2] | in[w2 + 3] | in[w3 - 2] | in[w3 - 1] | in[w3] |
          in[w3 + 1] | in[w3 + 2])
             ? 1
             : 0;
}

static inline size_t _raw_to_cmap(const size_t width, const size_t row, const size_t col) {
  return (row / 3) * width + (col / 3);
}

static inline float _calc_linear_refavg(const float* in, const int color) {
  const float ins[4] = {powf(fmaxf(0.0f, in[0]), 1.0f / HL_POWERF),
                        powf(fmaxf(0.0f, in[1]), 1.0f / HL_POWERF),
                        powf(fmaxf(0.0f, in[2]), 1.0f / HL_POWERF), 0.0f};
  const float opp[4] = {0.5f * (ins[1] + ins[2]), 0.5f * (ins[0] + ins[2]),
                        0.5f * (ins[0] + ins[1]), 0.0f};

  return powf(opp[color], HL_POWERF);
}

static inline float _calc_refavg(const float* in, const int row, const int col, const int height,
                                 const int width, const float* correction) {
  const int color   = FC(row, col);
  float     mean[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  float     cnt[4]  = {0.0f, 0.0f, 0.0f, 0.0f};

  // Loop limits logic remains the same
  const int dymin   = std::max(0, row - 1);
  const int dxmin   = std::max(0, col - 1);
  const int dymax   = std::min(height - 1, row + 2);
  const int dxmax   = std::min(width - 1, col + 2);

  for (int dy = dymin; dy < dymax; ++dy) {
    for (int dx = dxmin; dx < dxmax; ++dx) {
      const float val = fmaxf(0.0f, in[dy * width + dx]);
      const int   c   = FC(dy, dx);
      mean[c] += val;
      cnt[c] += 1.0f;
    }
  }

  // Optimization: correction is now based on D65 (pre_mul)
  for (int c = 0; c < 3; ++c) {
    // Use cbrtf for speed optimization over powf(x, 1.0/3.0)
    mean[c] = (cnt[c] > 0.f) ? fast_pow_inv_3(correction[c] * mean[c] / cnt[c]) : 0.f;
  }

  const float croot_refavg[4] = {0.5f * (mean[1] + mean[2]),  // Reconstruct Red from G/B
                                 0.5f * (mean[0] + mean[2]),  // Reconstruct Green from R/B
                                 0.5f * (mean[0] + mean[1]),  // Reconstruct Blue from R/G
                                 0.0f};

  // Use simple multiplication for power of 3
  return fast_pow_3(croot_refavg[color]);
}

/**
 * @brief Adapted from
 * https://github.com/darktable-org/darktable/blob/master/src/iop/hlreconstruct/opposed.c
 *
 * @param img
 * @param raw_processor
 */
void HighlightReconstruct(cv::Mat& img, LibRaw& raw_processor) {
  const int          width         = img.cols;
  const int          height        = img.rows;

  static const float hilight_magic = 0.987f;
  const float        clip_val      = hilight_magic;
  const float        clips[3]      = {clip_val, clip_val, clip_val};

  auto               cam_mul       = raw_processor.imgdata.color.cam_mul;
  auto               pre_mul       = raw_processor.imgdata.color.pre_mul;

  // --- FIX START: Use D65 (pre_mul) for reconstruction correlation ---
  // If we use cam_mul (Scene WB) on a Sun highlight (D65), we skew the ratios.
  // We fallback to cam_mul only if pre_mul looks invalid (all zeros).
  const float*       wb_basis      = (pre_mul[0] > 0.1f && pre_mul[1] > 0.1f) ? pre_mul : cam_mul;

  float              correction[4] = {
      (wb_basis[1] > 0.f) ? (wb_basis[0] / wb_basis[1]) : 1.f,  // R normalized to G
      1.f,                                                                   // G is anchor
      (wb_basis[1] > 0.f) ? (wb_basis[2] / wb_basis[1]) : 1.f,               // B normalized to G
      0.f};
  // --- FIX END ---

  const size_t               m_width    = width / 3;
  const size_t               m_height   = height / 3;
  const size_t               m_size     = round_size((size_t)(m_width + 1) * (m_height + 1), 16);

  bool                       anyclipped = false;
  cv::Mat1f                  input(img);
  auto                       input_data = input.ptr<float>(0);

  // Using vector is fine, but ensure zero initialization
  std::vector<unsigned char> mask_buf(6 * m_size, 0);

// 1. Build Clipping Mask
#pragma omp parallel for
  for (int row = 1; row < m_height - 1; ++row) {
    for (int col = 1; col < m_width - 1; ++col) {
      // ... [Keep existing mask generation logic] ...
      char         mbuff[3] = {0, 0, 0};
      const size_t grp      = 3 * (row * width + col);
      for (int y = -1; y <= 1; ++y) {
        for (int x = -1; x <= 1; ++x) {
          const size_t idx   = grp + y * width + x;
          const int    color = FC(row + y, col + x);
          if (input_data[idx] >= clips[color]) mbuff[color]++;
        }
      }
      for (int c = 0; c < 3; ++c) {
        if (mbuff[c]) {
          mask_buf[c * m_size + row * m_width + col] = 1;
          anyclipped = true;  // This race condition is benign (bool becomes true repeatedly)
        }
      }
    }
  }

  if (!anyclipped) return;

  float sums[4] = {0.f, 0.f, 0.f, 0.f};
  float cnts[4] = {0.f, 0.f, 0.f, 0.f};

// 2. Dilate Mask (Expand area of interest)
#pragma omp parallel for
  for (int row = 3; row < static_cast<int>(m_height) - 3; ++row) {
    for (int col = 3; col < static_cast<int>(m_width) - 3; ++col) {
      const int mx              = row * m_width + col;
      // Calculate dilated masks for the "Expanded" buffer area (indices 3, 4, 5)
      mask_buf[3 * m_size + mx] = mask_dilate(mask_buf.data() + 0 * m_size + mx, m_width);
      mask_buf[4 * m_size + mx] = mask_dilate(mask_buf.data() + 1 * m_size + mx, m_width);
      mask_buf[5 * m_size + mx] = mask_dilate(mask_buf.data() + 2 * m_size + mx, m_width);
    }
  }

  const float lo_clips[4] = {0.95f * clips[0], 0.95f * clips[1], 0.95f * clips[2], 1.0f};

// 3. Calculate Chromaticity Offset (Global Reference)
// Note: Global offset can be dangerous if image has mixed highlight colors (e.g. Sun vs Blue Sky).
// Ideally, this should be local, but optimizing the existing global logic first:
#pragma omp parallel for
  for (int row = 3; row < height - 3; ++row) {
    for (int col = 3; col < width - 3; ++col) {
      const int   color = FC(row, col);
      const float inval = input(row, col);

      // Check if pixel is Valid (unclipped) but High (near clip) AND inside the dilated mask area
      if ((inval < clips[color]) && (inval > lo_clips[color]) &&
          (mask_buf[(color + 3) * m_size + _raw_to_cmap(m_width, row, col)])) {
        // Calculate what the value "should" be based on neighbors + D65 correction
        float ref = _calc_refavg(input_data, row, col, height, width, correction);

#pragma omp atomic
        sums[color] += (inval - ref);
#pragma omp atomic
        cnts[color] += 1.0f;
      }
    }
  }

  float chrominance[4] = {0.f, 0.f, 0.f, 0.f};
  for (int c = 0; c < 3; ++c) {
    chrominance[c] = (cnts[c] > 100.f) ? (sums[c] / cnts[c]) * (cam_mul[c] / cam_mul[1]) : 0.f;
  }

  // 4. Apply Reconstruction
  cv::Mat1f result = input.clone();
#pragma omp parallel for
  for (int row = 0; row < height; ++row) {
    for (int col = 0; col < width; ++col) {
      const int   color = FC(row, col);
      const float inval = std::max(0.0f, input(row, col));  // Safe access

      if (inval >= clips[color]) {
        const float ref           = _calc_refavg(input_data, row, col, height, width, correction);

        // Apply global offset
        float       reconstructed = ref + chrominance[color];

        // Safety: Ensure we don't reconstruct LOWER than the clipped value
        result(row, col)          = std::max(inval, reconstructed);
      }
    }
  }
  img = result;
}

};  // namespace CPU
};  // namespace puerhlab