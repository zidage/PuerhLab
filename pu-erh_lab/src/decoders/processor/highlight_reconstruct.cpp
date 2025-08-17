#include "decoders/processor/highlight_reconstruct.hpp"

#include <opencv2/core/hal/interface.h>

#include <opencv2/opencv.hpp>

namespace puerhlab {
void OpenCVHighlightRecovery::ProcessHighlights(const std::vector<cv::Mat>& bayer_planes,
                                                const RecoveryParams&       params) {
  _params       = params;
  _color_planes = bayer_planes;
  _segment_masks.resize(4);
  _plane_segments.resize(4);

  for (int plane = 0; plane < 4; ++plane) {
    ProcessPlane(plane);
  }

  FinalizeRecovery();
}

auto OpenCVHighlightRecovery::GetProcessedPlanes() -> std::vector<cv::Mat>& {
  return _color_planes;
}

void OpenCVHighlightRecovery::ProcessPlane(int plane_idx) {
  cv::Mat& plane          = _color_planes[plane_idx];

  cv::Mat  highlight_mask = CreateHighlightMask(plane);

  cv::Mat  combined_mask  = MorphologicalCombine(highlight_mask);

  cv::Mat  labels, stats, centeroids;
  int      num_labels =
      cv::connectedComponentsWithStats(combined_mask, labels, stats, centeroids, 8, CV_32S);
  _segment_masks[plane_idx] = labels;

  AnalyzeSegments(plane_idx, labels, stats, centeroids, num_labels);

  InpaintPlane(plane_idx);
}

auto OpenCVHighlightRecovery::CreateHighlightMask(const cv::Mat& plane) -> cv::Mat {
  cv::Mat mask;
  cv::threshold(plane, mask, _params.clip_threshold, 255, cv::THRESH_BINARY);
  mask.convertTo(mask, CV_8UC1);

  cv::Mat extended_mask;
  cv::threshold(plane, extended_mask, _params.clip_threshold * 0.8f, 255, cv::THRESH_BINARY);
  extended_mask.convertTo(extended_mask, CV_8UC1);

  // 使用形态学操作连接相邻区域
  cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
  cv::dilate(mask, mask, kernel, cv::Point(-1, -1), 1);

  // 与扩展掩码求交集
  cv::bitwise_and(mask, extended_mask, mask);

  return mask;
}

auto OpenCVHighlightRecovery::MorphologicalCombine(const cv::Mat& mask) -> cv::Mat {
  if (_params.morphological_radius <= 0) {
    return mask.clone();
  }

  cv::Mat kernel = cv::getStructuringElement(
      cv::MORPH_ELLIPSE,
      cv::Size(2 * _params.morphological_radius + 1, 2 * _params.morphological_radius + 1));

  cv::Mat combined;
  cv::morphologyEx(mask, combined, cv::MORPH_CLOSE, kernel);

  return combined;
}

void OpenCVHighlightRecovery::AnalyzeSegments(int plane_idx, const cv::Mat& labels,
                                              const cv::Mat& stats, const cv::Mat& centroids,
                                              int num_labels) {
  const cv::Mat& plane = _color_planes[plane_idx];
  _plane_segments[plane_idx].clear();

  // 跳过背景标签 (label 0)
  for (int label = 1; label < num_labels; ++label) {
    SegmentInfo segment;
    segment.label = label;
    segment.bbox =
        cv::Rect(stats.at<int>(label, cv::CC_STAT_LEFT), stats.at<int>(label, cv::CC_STAT_TOP),
                 stats.at<int>(label, cv::CC_STAT_WIDTH), stats.at<int>(label, cv::CC_STAT_HEIGHT));

    // 查找最佳候选点
    FindBestCandidate(plane_idx, label, segment);

    if (segment.confidence > 0.1f) {
      _plane_segments[plane_idx].push_back(segment);
    }
  }
}

void OpenCVHighlightRecovery::FindBestCandidate(int plane_idx, int label, SegmentInfo& segment) {
  const cv::Mat& plane        = _color_planes[plane_idx];
  const cv::Mat& labels       = _segment_masks[plane_idx];

  // 创建当前分段的掩码
  cv::Mat        segment_mask = (labels == label);

  // 找到分段边界
  cv::Mat        border_mask;
  cv::Mat        kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
  cv::erode(segment_mask, border_mask, kernel);
  border_mask = segment_mask - border_mask;  // 边界像素

  // 在边界区域寻找候选点
  std::vector<cv::Point> border_points;
  cv::findNonZero(border_mask, border_points);

  float       best_score = -1.0f;
  cv::Point2f best_candidate(0, 0);

  for (const cv::Point& pt : border_points) {
    // 检查该点是否为未剪切点
    if (plane.at<float>(pt.x, pt.y) < _params.clip_threshold) {
      float score = EvaluateCandidatePoint(plane, pt);

      if (score > best_score) {
        best_score     = score;
        best_candidate = cv::Point2f(pt.x, pt.y);
      }
    }
  }

  segment.best_candidate = best_candidate;
  segment.confidence     = best_score;
  segment.border_points  = border_points;
}

float OpenCVHighlightRecovery::EvaluateCandidatePoint(const cv::Mat&   plane,
                                                      const cv::Point& point) {
  cv::Rect roi(point.x - 2, point.y - 2, 5, 5);
  roi &= cv::Rect(0, 0, plane.cols, plane.rows);

  if (roi.width < 3 || roi.height < 3) {
    return 0.0f;
  }

  cv::Mat    local_patch = plane(roi);

  // 计算局部统计量
  cv::Scalar mean, stddev;
  cv::meanStdDev(local_patch, mean, stddev);

  // 计算中值
  std::vector<float> values;
  // local_patch.convertTo(local_patch, CV_32FC1);

  local_patch.forEach<float>([&](float pixel, const int*) { values.push_back(pixel); });
  std::sort(values.begin(), values.end());
  float median           = values[values.size() / 2];

  // 评分函数
  float smoothness_score = 1.0f / (1.0f + stddev[0]);      // 平滑度
  float brightness_score = std::min(median / 0.5f, 1.0f);  // 亮度适中

  return smoothness_score * brightness_score * _params.candidate_weight;
}

void OpenCVHighlightRecovery::InpaintPlane(int plane_idx) {
  cv::Mat&       plane  = _color_planes[plane_idx];
  const cv::Mat& labels = _segment_masks[plane_idx];

  for (const SegmentInfo& segment : _plane_segments[plane_idx]) {
    if (segment.confidence < 0.1f) continue;

    // 创建当前分段的修复掩码
    cv::Mat segment_mask = (labels == segment.label);
    cv::Mat inpaint_mask;

    // 只修复实际剪切的像素
    cv::Mat clipped_mask;
    cv::threshold(plane, clipped_mask, _params.clip_threshold, 255, cv::THRESH_BINARY);
    clipped_mask.convertTo(clipped_mask, CV_8UC1);

    cv::bitwise_and(segment_mask, clipped_mask, inpaint_mask);

    if (cv::countNonZero(inpaint_mask) == 0) continue;

    // 使用改进的修复策略
    InpaintSegmentAdvanced(plane, inpaint_mask, segment);
  }
}

void OpenCVHighlightRecovery::InpaintSegmentAdvanced(cv::Mat& plane, const cv::Mat& inpaint_mask,
                                                     const SegmentInfo& segment) {
  // 获取候选点的值作为参考
  float   reference_value = plane.at<float>(static_cast<int>(segment.best_candidate.y),
                                            static_cast<int>(segment.best_candidate.x));

  // 方法1: 基于距离的渐变填充
  cv::Mat distance_transform;
  cv::distanceTransform(255 - inpaint_mask, distance_transform, cv::DIST_L2, cv::DIST_MASK_PRECISE);

  // 方法2: 使用OpenCV的快速行进修复
  cv::Mat inpainted_plane = plane.clone();
  cv::inpaint(plane, inpaint_mask, inpainted_plane, 3, cv::INPAINT_TELEA);

  // 方法3: 伪色度修复 (核心创新)
  PerformPseudoChromacityInpainting(plane, inpaint_mask, segment, reference_value);

  // 组合多种方法的结果
  cv::Mat final_result;
  cv::addWeighted(plane, 0.7f, inpainted_plane, 0.3f, 0, final_result);
  final_result.copyTo(plane, inpaint_mask);
}

void OpenCVHighlightRecovery::PerformPseudoChromacityInpainting(cv::Mat&           plane,
                                                                const cv::Mat&     inpaint_mask,
                                                                const SegmentInfo& segment,
                                                                float reference_value) {
  std::vector<float> neighbor_means(4, 0.0f);
  CalculateNeighborPlaneMeans(segment.bbox, neighbor_means);

  // 计算伪色度差值
  float pseudo_chromacity    = reference_value - neighbor_means[0];  // 简化版本

  // 在立方根空间进行计算以提高稳定性
  float cube_root_chromacity = std::cbrt(std::abs(pseudo_chromacity));
  if (pseudo_chromacity < 0) cube_root_chromacity = -cube_root_chromacity;

  // 应用修复值
  cv::Mat repair_value =
      cv::Mat::ones(plane.size(), CV_32F) *
      (neighbor_means[0] + cube_root_chromacity * cube_root_chromacity * cube_root_chromacity);

  repair_value.copyTo(plane, inpaint_mask);
}

void OpenCVHighlightRecovery::CalculateNeighborPlaneMeans(const cv::Rect&     bbox,
                                                          std::vector<float>& means) {
  // 扩展区域用于计算邻域均值
  cv::Rect expanded_bbox = bbox;
  expanded_bbox.x        = std::max(0, bbox.x - 10);
  expanded_bbox.y        = std::max(0, bbox.y - 10);
  expanded_bbox.width    = std::min(_color_planes[0].cols - expanded_bbox.x, bbox.width + 20);
  expanded_bbox.height   = std::min(_color_planes[0].rows - expanded_bbox.y, bbox.height + 20);

  for (int i = 0; i < 4; ++i) {
    cv::Mat    roi      = _color_planes[i](expanded_bbox);
    cv::Scalar mean_val = cv::mean(roi);
    means[i]            = mean_val[0];
  }
}

void OpenCVHighlightRecovery::FinalizeRecovery() {
  // 对所有颜色平面应用轻微的高斯模糊以减少色度噪声
  for (int i = 0; i < 4; ++i) {
    cv::Mat blurred;
    cv::GaussianBlur(_color_planes[i], blurred, cv::Size(3, 3), 0.5);

    // 只在修复区域应用模糊
    cv::Mat repair_mask;
    cv::threshold(_color_planes[i], repair_mask, _params.clip_threshold * 0.99f, 255,
                  cv::THRESH_BINARY);
    repair_mask.convertTo(repair_mask, CV_8UC1);

    blurred.copyTo(_color_planes[i], repair_mask);
  }
}
};  // namespace puerhlab