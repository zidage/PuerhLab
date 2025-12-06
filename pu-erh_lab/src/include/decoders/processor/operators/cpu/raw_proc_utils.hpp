#pragma once

#include <cstdint>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "hwy/highway.h"

namespace puerhlab {
namespace CPU {
enum class LightSourceType : uint8_t {
  UNKNOWN      = 0,
  DAYLIGHT     = 1,
  FLUORESCENT  = 2,
  TUNGSTEN     = 3,
  FLASH        = 4,
  FINE_WEATHER = 9,
  CLOUDY       = 10,
  SHADE        = 11,
  STD_A        = 17,
  D55          = 20,
  D65          = 21,
  D75          = 22,
  D50          = 23
};

const int   COLOR_TEMP_TABLE_SIZE              = 24;
const float EXIF_WB_MAP[COLOR_TEMP_TABLE_SIZE] = {
    0.0f, 5500.0f, 4200.0f, 2850.0f, 5500.0f, 5200.0f, 0.0f,    0.0f,
    0.0f, 5200.0f, 6500.0f, 7500.0f, 0.0f,    0.0f,    0.0f,    0.0f,
    0.0f, 2850.0f, 0.0f,    0.0f,    5500.0f, 6500.0f, 7500.0f, 5000.0f};

inline static float GetTempForWBIndices(int idx) {
  if (idx < 0 || idx >= COLOR_TEMP_TABLE_SIZE) {
    return -1.0f;
  }
  return EXIF_WB_MAP[idx];
}

/**
 * @brief Get the WB Indices For Temp object. If exact match is found, both indices will be the
 * same. Otherwise, the lower and upper indices surrounding the temperature are returned.
 *
 * @param temp
 * @return std::pair<int, int>
 */
inline static std::pair<int, int> GetWBIndicesForTemp(float temp) {
  std::pair<int, int> result{-1, -1};  // lower index, upper index
  float               lower_tmp = -1.0f;
  float               upper_tmp = -1.0f;
  for (int i = 1; i < COLOR_TEMP_TABLE_SIZE; ++i) {
    if (EXIF_WB_MAP[i] == 0.0f) continue;
    if (EXIF_WB_MAP[i] <= temp) {
      if (EXIF_WB_MAP[i] == temp) {
        result.first  = i;
        result.second = i;
        // Exact match
        return result;
      }
      if (EXIF_WB_MAP[i] > lower_tmp) {
        lower_tmp    = EXIF_WB_MAP[i];
        result.first = i;
      }
    } else {
      if (upper_tmp < 0.0f || EXIF_WB_MAP[i] < upper_tmp) {
        upper_tmp     = EXIF_WB_MAP[i];
        result.second = i;
      }
    }
  }
  return result;
}

void boxblur2(const cv::Mat1f& src, cv::Mat1f& dst, cv::Mat1f& temp, int startY, int startX, int H,
              int W, int box);

void boxblur_resamp(const cv::Mat1f& src, cv::Mat1f& dst, cv::Mat1f& temp, int H, int W, int box,
                    int samp);

inline static void DebuggingPreview(cv::Mat& src) {
  cv::Mat resized;
  cv::resize(src, resized, cv::Size(512, 512));
  cv::imshow("Debugging Preview", resized);
  cv::waitKey(0);
}
};  // namespace CPU
};  // namespace puerhlab