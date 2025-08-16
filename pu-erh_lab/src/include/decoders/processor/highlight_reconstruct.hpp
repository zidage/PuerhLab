#pragma once

#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/ximgproc.hpp>
#include <unordered_map>
#include <vector>

namespace puerhlab {
class OpenCVHighlightRecovery {
 public:
  struct RecoveryParams {
    float clip_threshold;
    int   morphological_radius;
    float candidate_weight;
    bool  enable_debug_viz;

    RecoveryParams()
        : clip_threshold(0.98f),
          morphological_radius(4),
          candidate_weight(0.5f),
          enable_debug_viz(false) {}
  };

  struct SegmentInfo {
    int                    label;
    cv::Rect               bbox;
    cv::Point2f            best_candidate;
    float                  confidence;
    std::vector<cv::Point> border_points;
  };

 private:
  RecoveryParams                        _params;
  std::vector<cv::Mat>                  _color_planes;    // RGGB planes
  std::vector<cv::Mat>                  _segment_masks;   // segment mask for each plane
  std::vector<std::vector<SegmentInfo>> _plane_segments;  // segement info for each plane

  void                                  ProcessPlane(int plane_idx);
  auto                                  CreateHighlightMask(const cv::Mat& plane) -> cv::Mat;
  auto                                  MorphologicalCombine(const cv::Mat& mask) -> cv::Mat;
  void  AnalyzeSegments(int plane_idx, const cv::Mat& labels, const cv::Mat& stats,
                        const cv::Mat& centroids, int num_labels);
  void  FindBestCandidate(int plane_idx, int label, SegmentInfo& segment);
  float EvaluateCandidatePoint(const cv::Mat& plane, const cv::Point& point);
  void  InpaintPlane(int plane_idx);
  void  InpaintSegmentAdvanced(cv::Mat& plane, const cv::Mat& inpaint_mask,
                               const SegmentInfo& segment);

 public:
  void ProcessHighlights(const std::vector<cv::Mat>& bayer_planes,
                         const RecoveryParams&       params = {});
  auto GetProcessedPlanes() -> std::vector<cv::Mat>&;
};
};  // namespace puerhlab