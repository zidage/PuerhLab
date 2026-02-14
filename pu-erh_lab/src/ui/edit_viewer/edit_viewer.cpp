//  Copyright 2025 Yurun Zi
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.

#include "ui/edit_viewer/edit_viewer.hpp"

#include <GL/gl.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <qopenglext.h>
#include <qoverload.h>
#include <QApplication>
#include <QEasingCurve>
#include <QSurfaceFormat>
#include <QByteArray>
#include <QPainter>
#include <QPainterPath>
#include <QPen>
#include <QPolygonF>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <mutex>
#include <vector>

namespace puerhlab {

static const char* vertexShaderSource = R"(
#version 330 core
layout(location = 0) in vec2 position;

uniform vec2 uScale;
uniform vec2 uPan;
uniform float uZoom;

out vec2 vTexCoord;

void main() {
  // Letterbox scale, then user zoom/pan for interactive view controls
  vec2 pos = position * uScale * uZoom + uPan;
  gl_Position = vec4(pos, 0.0, 1.0);

  vec2 uv = (position + 1.0) * 0.5;
  vTexCoord = vec2(uv.x, 1.0 - uv.y); // flip Y
}
)";

static const char* fragmentShaderSource = R"(
#version 330 core
uniform sampler2D textureSampler;
in vec2 vTexCoord;
out vec4 FragColor;
void main() {
    FragColor = texture(textureSampler, vTexCoord);
}
)";

static const char* histogramClearShaderSource = R"(
#version 430 core
layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

layout(std430, binding = 0) buffer HistogramCounts {
  uint counts[];
};

uniform int uCount;

void main() {
  uint idx = gl_GlobalInvocationID.x;
  if (idx < uint(uCount)) {
    counts[idx] = 0u;
  }
}
)";

static const char* histogramComputeShaderSource = R"(
#version 430 core
layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

layout(binding = 0) uniform sampler2D uSourceTex;

layout(std430, binding = 0) buffer HistogramCounts {
  uint counts[];
};

uniform int uBins;
uniform int uSampleSize;

void main() {
  uvec2 gid = gl_GlobalInvocationID.xy;
  if (gid.x >= uint(uSampleSize) || gid.y >= uint(uSampleSize)) {
    return;
  }

  vec2 uv = (vec2(gid) + vec2(0.5)) / float(uSampleSize);
  vec3 rgb = clamp(textureLod(uSourceTex, uv, 0.0).rgb, 0.0, 1.0);

  int r = int(rgb.r * float(uBins - 1) + 0.5);
  int g = int(rgb.g * float(uBins - 1) + 0.5);
  int b = int(rgb.b * float(uBins - 1) + 0.5);

  atomicAdd(counts[r], 1u);
  atomicAdd(counts[uBins + g], 1u);
  atomicAdd(counts[uBins * 2 + b], 1u);
}
)";

static const char* histogramNormalizeShaderSource = R"(
#version 430 core
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

layout(std430, binding = 0) readonly buffer HistogramCounts {
  uint counts[];
};

layout(std430, binding = 1) writeonly buffer HistogramNormalized {
  float normalized[];
};

uniform int uBins;

void main() {
  const int count = uBins * 3;
  uint max_count = 1u;
  for (int i = 0; i < count; ++i) {
    max_count = max(max_count, counts[i]);
  }

  const float inv = 1.0 / float(max_count);
  for (int i = 0; i < count; ++i) {
    normalized[i] = float(counts[i]) * inv;
  }
}
)";

constexpr float kCropMinSize               = 1e-4f;
constexpr float kCropCornerHitRadiusPx     = 12.0f;
constexpr float kCropEdgeHitRadiusPx       = 10.0f;
constexpr float kCropCornerDrawRadiusPx    = 4.0f;
constexpr float kCropRotateDegreesPerPixel = 0.25f;
constexpr float kPi                        = 3.14159265358979323846f;

auto Clamp01(float v) -> float { return std::clamp(v, 0.0f, 1.0f); }

auto NormalizeAngleDegrees(float angle_degrees) -> float {
  if (!std::isfinite(angle_degrees)) {
    return 0.0f;
  }
  angle_degrees = std::fmod(angle_degrees, 360.0f);
  if (angle_degrees > 180.0f) {
    angle_degrees -= 360.0f;
  } else if (angle_degrees < -180.0f) {
    angle_degrees += 360.0f;
  }
  return angle_degrees;
}

auto ClampAspect(float aspect) -> float { return std::max(aspect, 1e-4f); }

auto SafeAspect(int image_width, int image_height) -> float {
  if (image_width <= 0 || image_height <= 0) {
    return 1.0f;
  }
  return ClampAspect(static_cast<float>(image_width) / static_cast<float>(image_height));
}

auto UvToMetric(const QPointF& uv, float aspect) -> QPointF {
  const float a = ClampAspect(aspect);
  return QPointF(static_cast<float>(uv.x()) * a, static_cast<float>(uv.y()));
}

auto MetricToUv(const QPointF& metric, float aspect) -> QPointF {
  const float a = ClampAspect(aspect);
  return QPointF(static_cast<float>(metric.x()) / a, static_cast<float>(metric.y()));
}

auto Dot2(const QPointF& a, const QPointF& b) -> float {
  return (static_cast<float>(a.x()) * static_cast<float>(b.x())) +
         (static_cast<float>(a.y()) * static_cast<float>(b.y()));
}

auto RotateVector(const QPointF& v, float angle_degrees) -> QPointF {
  const float a = NormalizeAngleDegrees(angle_degrees) * (kPi / 180.0f);
  const float c = std::cos(a);
  const float s = std::sin(a);
  return QPointF((c * static_cast<float>(v.x())) - (s * static_cast<float>(v.y())),
                 (s * static_cast<float>(v.x())) + (c * static_cast<float>(v.y())));
}

auto InverseRotateVector(const QPointF& v, float angle_degrees) -> QPointF {
  const float a = NormalizeAngleDegrees(angle_degrees) * (kPi / 180.0f);
  const float c = std::cos(a);
  const float s = std::sin(a);
  return QPointF((c * static_cast<float>(v.x())) + (s * static_cast<float>(v.y())),
                 (-s * static_cast<float>(v.x())) + (c * static_cast<float>(v.y())));
}

auto MakeRectFromCenterSize(const QPointF& center, float width, float height) -> QRectF {
  return QRectF(center.x() - (static_cast<qreal>(width) * 0.5),
                center.y() - (static_cast<qreal>(height) * 0.5), width, height);
}

auto ClampCropRectForRotation(const QRectF& rect, float angle_degrees, float aspect) -> QRectF {
  const float a_metric = ClampAspect(aspect);
  QRectF r = rect.normalized();
  float  w = std::clamp(static_cast<float>(r.width()), kCropMinSize, 1.0f);
  float  h = std::clamp(static_cast<float>(r.height()), kCropMinSize, 1.0f);
  QPointF center_uv = r.center();
  center_uv.setX(Clamp01(static_cast<float>(center_uv.x())));
  center_uv.setY(Clamp01(static_cast<float>(center_uv.y())));
  QPointF center_m = UvToMetric(center_uv, a_metric);

  float hw = (w * a_metric) * 0.5f;
  float hh = h * 0.5f;

  const float a = NormalizeAngleDegrees(angle_degrees) * (kPi / 180.0f);
  const float c = std::abs(std::cos(a));
  const float s = std::abs(std::sin(a));

  float extent_x = (c * hw) + (s * hh);
  float extent_y = (s * hw) + (c * hh);
  if (extent_x > (a_metric * 0.5f) || extent_y > 0.5f) {
    const float sx = (extent_x > 0.0f) ? ((a_metric * 0.5f) / extent_x) : 1.0f;
    const float sy = (extent_y > 0.0f) ? (0.5f / extent_y) : 1.0f;
    const float scale = std::clamp(std::min(sx, sy), 0.0f, 1.0f);
    hw = std::max((kCropMinSize * a_metric) * 0.5f, hw * scale);
    hh = std::max(kCropMinSize * 0.5f, hh * scale);
    extent_x = (c * hw) + (s * hh);
    extent_y = (s * hw) + (c * hh);
  }

  center_m.setX(std::clamp(static_cast<float>(center_m.x()), extent_x, a_metric - extent_x));
  center_m.setY(std::clamp(static_cast<float>(center_m.y()), extent_y, 1.0f - extent_y));

  const QPointF final_center_uv = MetricToUv(center_m, a_metric);
  const float   final_w_uv = std::clamp((hw * 2.0f) / a_metric, kCropMinSize, 1.0f);
  const float   final_h_uv = std::clamp(hh * 2.0f, kCropMinSize, 1.0f);
  return MakeRectFromCenterSize(final_center_uv, final_w_uv, final_h_uv);
}

auto RotatedCropCornersUv(const QRectF& rect, float angle_degrees, float aspect) -> std::array<QPointF, 4> {
  const float   a_metric = ClampAspect(aspect);
  const QPointF center_m = UvToMetric(rect.center(), a_metric);
  const float   hw = std::max((kCropMinSize * a_metric) * 0.5f,
                              static_cast<float>(rect.width()) * a_metric * 0.5f);
  const float   hh = std::max(kCropMinSize * 0.5f, static_cast<float>(rect.height()) * 0.5f);
  const std::array<QPointF, 4> local = {QPointF(-hw, -hh), QPointF(hw, -hh),
                                         QPointF(hw, hh), QPointF(-hw, hh)};

  std::array<QPointF, 4> corners{};
  for (size_t i = 0; i < local.size(); ++i) {
    corners[i] = MetricToUv(center_m + RotateVector(local[i], angle_degrees), a_metric);
  }
  return corners;
}

auto IsPointInsideRotatedCrop(const QPointF& point_uv, const QRectF& rect, float angle_degrees,
                              float aspect) -> bool {
  const float   a_metric = ClampAspect(aspect);
  const QPointF local = InverseRotateVector(UvToMetric(point_uv, a_metric) - UvToMetric(rect.center(), a_metric),
                                            angle_degrees);
  const float   hw = std::max((kCropMinSize * a_metric) * 0.5f,
                              static_cast<float>(rect.width()) * a_metric * 0.5f);
  const float   hh = std::max(kCropMinSize * 0.5f, static_cast<float>(rect.height()) * 0.5f);
  return std::abs(static_cast<float>(local.x())) <= hw &&
         std::abs(static_cast<float>(local.y())) <= hh;
}

auto PointSegmentDistanceSquared(const QPointF& p, const QPointF& a, const QPointF& b) -> float {
  const QPointF ab = b - a;
  const float   ab_len2 = Dot2(ab, ab);
  if (ab_len2 <= 1e-8f) {
    const float dx = static_cast<float>(p.x() - a.x());
    const float dy = static_cast<float>(p.y() - a.y());
    return (dx * dx) + (dy * dy);
  }
  const float t = std::clamp(Dot2(p - a, ab) / ab_len2, 0.0f, 1.0f);
  const QPointF proj = a + (ab * t);
  const float dx = static_cast<float>(p.x() - proj.x());
  const float dy = static_cast<float>(p.y() - proj.y());
  return (dx * dx) + (dy * dy);
}

auto LerpPoint(const QPointF& a, const QPointF& b, float t) -> QPointF {
  return QPointF(a.x() + (b.x() - a.x()) * t, a.y() + (b.y() - a.y()) * t);
}

QtEditViewer::QtEditViewer(QWidget* parent) : QOpenGLWidget(parent) {
  // Connect the frame ready signal to the update slot.
  // Use QueuedConnection explicitly so that signals from worker threads are
  // processed on the next event-loop iteration of the GUI thread.
  connect(this, &QtEditViewer::RequestUpdate, this, QOverload<>::of(&QtEditViewer::update),
          Qt::QueuedConnection);

  // Blocking resize requests until the current resize is done
  connect(this, &QtEditViewer::RequestResize, this, &QtEditViewer::OnResizeGL,
          Qt::BlockingQueuedConnection);

  zoom_animation_ = new QVariantAnimation(this);
  zoom_animation_->setDuration(kZoomAnimationDurationMs);
  zoom_animation_->setEasingCurve(QEasingCurve::InOutCubic);
  zoom_animation_->setStartValue(0.0);
  zoom_animation_->setEndValue(1.0);
  connect(zoom_animation_, &QVariantAnimation::valueChanged, this,
          [this](const QVariant& value) {
            const float t = std::clamp(static_cast<float>(value.toDouble()), 0.0f, 1.0f);
            const float zoom = zoom_animation_start_ +
                               ((zoom_animation_target_ - zoom_animation_start_) * t);
            const QVector2D pan = pan_animation_start_ +
                                  ((pan_animation_target_ - pan_animation_start_) * t);
            ApplyViewTransform(zoom, pan, true);
          });
  connect(zoom_animation_, &QVariantAnimation::finished, this,
          [this]() { ApplyViewTransform(zoom_animation_target_, pan_animation_target_, true); });

  click_toggle_timer_ = new QTimer(this);
  click_toggle_timer_->setSingleShot(true);
  connect(click_toggle_timer_, &QTimer::timeout, this, [this]() {
    if (!pending_click_toggle_) {
      return;
    }
    pending_click_toggle_ = false;
    ToggleClickZoomAt(pending_click_toggle_pos_);
  });
}

QtEditViewer::~QtEditViewer() {
  makeCurrent();
  // Clean up OpenGL resources
  FreeAllBuffers();
  FreeHistogramResources();
  delete program_;
  program_ = nullptr;
  doneCurrent();

  if (staging_ptr_) {
    cudaFree(staging_ptr_);
    staging_ptr_   = nullptr;
    staging_bytes_ = 0;
  }
}

void QtEditViewer::EnsureSize(int width, int height) {
  bool need_resize = false;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto& target_buf = buffers_[render_target_idx_];
    if (target_buf.width != width || target_buf.height != height) {
      // Prepare the alternate buffer for the new size without dropping the currently shown one.
      render_target_idx_ = write_idx_;
      const auto& write_buf = buffers_[render_target_idx_];
      need_resize = (write_buf.width != width || write_buf.height != height);
    }
  }

  // Emit outside the mutex to avoid deadlock with the UI thread (slot locks the same mutex).
  if (need_resize) {
    emit RequestResize(width, height);
  }

  // Ensure staging buffer is available for worker thread writes.
  const size_t needed_bytes = static_cast<size_t>(width) * static_cast<size_t>(height) * sizeof(float4);
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (needed_bytes > 0 && needed_bytes > staging_bytes_) {
      float4* new_staging_ptr = nullptr;
      const cudaError_t alloc_err = cudaMalloc(reinterpret_cast<void**>(&new_staging_ptr), needed_bytes);
      if (alloc_err != cudaSuccess) {
        qWarning("Failed to allocate CUDA staging buffer (%zu bytes): %s", needed_bytes,
                 cudaGetErrorString(alloc_err));
      } else {
        if (staging_ptr_) {
          cudaFree(staging_ptr_);
        }
        staging_ptr_ = new_staging_ptr;
        staging_bytes_ = needed_bytes;
      }
    }
  }

  // Resize request is emitted in the size-change branch above.
}

auto QtEditViewer::GetViewZoom() const -> float {
  std::lock_guard<std::mutex> view_lock(view_state_mutex_);
  return view_zoom_;
}

void QtEditViewer::StopZoomAnimation() {
  if (zoom_animation_ && zoom_animation_->state() == QAbstractAnimation::Running) {
    zoom_animation_->stop();
  }
}

auto QtEditViewer::ClampPanForZoom(float zoom, const QVector2D& pan) const -> QVector2D {
  const int image_width = buffers_[active_idx_].width;
  const int image_height = buffers_[active_idx_].height;
  if (image_width <= 0 || image_height <= 0) {
    return QVector2D(0.0f, 0.0f);
  }

  const float dpr = devicePixelRatioF();
  const float vw  = std::max(1.0f, static_cast<float>(width()) * dpr);
  const float vh  = std::max(1.0f, static_cast<float>(height()) * dpr);
  if (vw <= 0.0f || vh <= 0.0f) {
    return QVector2D(0.0f, 0.0f);
  }

  const float imgW      = static_cast<float>(std::max(1, image_width));
  const float imgH      = static_cast<float>(std::max(1, image_height));
  const float winAspect = vw / vh;
  const float imgAspect = imgW / imgH;

  float sx = 1.0f;
  float sy = 1.0f;
  if (imgAspect > winAspect) {
    sy = winAspect / imgAspect;
  } else {
    sx = imgAspect / winAspect;
  }
  sx = std::max(sx, 1e-4f);
  sy = std::max(sy, 1e-4f);

  const float clamped_zoom = std::clamp(zoom, kMinInteractiveZoom, kMaxInteractiveZoom);
  const float max_pan_x = std::max(0.0f, (sx * clamped_zoom) - 1.0f);
  const float max_pan_y = std::max(0.0f, (sy * clamped_zoom) - 1.0f);
  return QVector2D(std::clamp(pan.x(), -max_pan_x, max_pan_x),
                   std::clamp(pan.y(), -max_pan_y, max_pan_y));
}

auto QtEditViewer::ComputeAnchoredPan(float target_zoom, const QPointF& anchor_widget_pos,
                                      const QVector2D& fallback_pan) const -> QVector2D {
  const int image_width = buffers_[active_idx_].width;
  const int image_height = buffers_[active_idx_].height;
  if (image_width <= 0 || image_height <= 0) {
    return fallback_pan;
  }

  const auto anchor_uv = WidgetPointToImageUv(anchor_widget_pos, image_width, image_height);
  if (!anchor_uv.has_value()) {
    return fallback_pan;
  }

  const float dpr = devicePixelRatioF();
  const float vw  = std::max(1.0f, static_cast<float>(width()) * dpr);
  const float vh  = std::max(1.0f, static_cast<float>(height()) * dpr);
  if (vw <= 0.0f || vh <= 0.0f) {
    return fallback_pan;
  }

  const float imgW      = static_cast<float>(std::max(1, image_width));
  const float imgH      = static_cast<float>(std::max(1, image_height));
  const float winAspect = vw / vh;
  const float imgAspect = imgW / imgH;

  float sx = 1.0f;
  float sy = 1.0f;
  if (imgAspect > winAspect) {
    sy = winAspect / imgAspect;
  } else {
    sx = imgAspect / winAspect;
  }
  sx = std::max(sx, 1e-4f);
  sy = std::max(sy, 1e-4f);

  const float px   = static_cast<float>(anchor_widget_pos.x()) * dpr;
  const float py   = static_cast<float>(anchor_widget_pos.y()) * dpr;
  const float ndcX = (2.0f * px / vw) - 1.0f;
  const float ndcY = 1.0f - (2.0f * py / vh);
  const float imgX = (2.0f * Clamp01(static_cast<float>(anchor_uv->x()))) - 1.0f;
  const float imgY = 1.0f - (2.0f * Clamp01(static_cast<float>(anchor_uv->y())));
  return QVector2D(ndcX - (imgX * sx * target_zoom), ndcY - (imgY * sy * target_zoom));
}

void QtEditViewer::ApplyViewTransform(float zoom, const QVector2D& pan, bool emit_zoom_signal) {
  const float clamped_zoom = std::clamp(zoom, kMinInteractiveZoom, kMaxInteractiveZoom);
  const QVector2D clamped_pan = ClampPanForZoom(clamped_zoom, pan);
  float       prev_zoom    = clamped_zoom;
  {
    std::lock_guard<std::mutex> view_lock(view_state_mutex_);
    prev_zoom = view_zoom_;
    view_zoom_ = clamped_zoom;
    view_pan_  = clamped_pan;
  }
  UpdateViewportRenderRegionCache();
  update();
  if (emit_zoom_signal && std::abs(prev_zoom - clamped_zoom) > 1e-5f) {
    emit ViewZoomChanged(clamped_zoom);
  }
}

void QtEditViewer::AnimateViewTo(float target_zoom, const std::optional<QPointF>& anchor_widget_pos,
                                 const std::optional<QVector2D>& explicit_target_pan) {
  target_zoom = std::clamp(target_zoom, kMinInteractiveZoom, kMaxInteractiveZoom);

  float     start_zoom = kMinInteractiveZoom;
  QVector2D start_pan(0.0f, 0.0f);
  {
    std::lock_guard<std::mutex> view_lock(view_state_mutex_);
    start_zoom = view_zoom_;
    start_pan  = view_pan_;
  }

  QVector2D target_pan = explicit_target_pan.value_or(start_pan);
  if (!explicit_target_pan.has_value() && anchor_widget_pos.has_value()) {
    target_pan = ComputeAnchoredPan(target_zoom, *anchor_widget_pos, start_pan);
  }
  target_pan = ClampPanForZoom(target_zoom, target_pan);

  if (std::abs(start_zoom - target_zoom) <= 1e-5f &&
      (start_pan - target_pan).lengthSquared() <= 1e-8f) {
    ApplyViewTransform(target_zoom, target_pan, true);
    return;
  }

  StopZoomAnimation();
  zoom_animation_start_  = start_zoom;
  zoom_animation_target_ = target_zoom;
  pan_animation_start_   = start_pan;
  pan_animation_target_  = target_pan;

  if (!zoom_animation_) {
    ApplyViewTransform(target_zoom, target_pan, true);
    return;
  }

  zoom_animation_->setDuration(kZoomAnimationDurationMs);
  zoom_animation_->setStartValue(0.0);
  zoom_animation_->setEndValue(1.0);
  zoom_animation_->start();
}

void QtEditViewer::ToggleClickZoomAt(const QPointF& anchor_widget_pos) {
  if (click_zoom_toggle_active_) {
    click_zoom_toggle_active_ = false;
    AnimateViewTo(click_zoom_restore_zoom_, std::nullopt, click_zoom_restore_pan_);
    return;
  }

  {
    std::lock_guard<std::mutex> view_lock(view_state_mutex_);
    click_zoom_restore_zoom_ = view_zoom_;
    click_zoom_restore_pan_  = view_pan_;
  }
  click_zoom_toggle_active_ = true;
  AnimateViewTo(kSingleClickZoomFactor, anchor_widget_pos);
}

void QtEditViewer::ToggleDoubleClickZoomAt(const QPointF& anchor_widget_pos) {
  float target_zoom = kSingleClickZoomFactor;
  bool  zoom_in     = true;
  {
    std::lock_guard<std::mutex> view_lock(view_state_mutex_);
    target_zoom = std::clamp(double_click_zoom_target_, kMinInteractiveZoom, kMaxInteractiveZoom);
    zoom_in     = double_click_zoom_in_next_;
    double_click_zoom_in_next_ = !double_click_zoom_in_next_;
  }

  click_zoom_toggle_active_ = false;
  if (zoom_in) {
    AnimateViewTo(target_zoom, anchor_widget_pos);
    return;
  }
  AnimateViewTo(kMinInteractiveZoom, std::nullopt, QVector2D(0.0f, 0.0f));
}

void QtEditViewer::ResetView() {
  click_zoom_toggle_active_ = false;
  pending_click_toggle_     = false;
  suppress_next_click_release_toggle_ = false;
  if (click_toggle_timer_ && click_toggle_timer_->isActive()) {
    click_toggle_timer_->stop();
  }
  {
    std::lock_guard<std::mutex> view_lock(view_state_mutex_);
    double_click_zoom_in_next_ = true;
  }
  StopZoomAnimation();
  ApplyViewTransform(kMinInteractiveZoom, QVector2D(0.0f, 0.0f), true);
}

void QtEditViewer::SetCropToolEnabled(bool enabled) {
  bool zoom_reset = false;
  if (enabled) {
    StopZoomAnimation();
    click_zoom_toggle_active_ = false;
    pending_click_toggle_     = false;
    suppress_next_click_release_toggle_ = false;
    if (click_toggle_timer_ && click_toggle_timer_->isActive()) {
      click_toggle_timer_->stop();
    }
  }
  {
    std::lock_guard<std::mutex> view_lock(view_state_mutex_);
    crop_tool_enabled_ = enabled;
    if (enabled) {
      zoom_reset = (std::abs(view_zoom_ - kMinInteractiveZoom) > 1e-5f) ||
                   (view_pan_.lengthSquared() > 1e-8f);
      view_zoom_ = kMinInteractiveZoom;
      view_pan_  = QVector2D(0.0f, 0.0f);
      double_click_zoom_in_next_ = true;
    }
    if (!enabled) {
      crop_drag_mode_             = CropDragMode::None;
      crop_drag_corner_           = CropCorner::None;
      crop_drag_edge_             = CropEdge::None;
      crop_drag_rotation_degrees_ = 0.0f;
      crop_drag_fixed_corner_uv_  = QPointF();
      crop_drag_anchor_widget_pos_ = QPointF();
    }
  }
  if (!enabled) {
    unsetCursor();
  }
  UpdateViewportRenderRegionCache();
  update();
  if (zoom_reset) {
    emit ViewZoomChanged(kMinInteractiveZoom);
  }
}

void QtEditViewer::SetCropOverlayVisible(bool visible) {
  {
    std::lock_guard<std::mutex> view_lock(view_state_mutex_);
    crop_overlay_visible_ = visible;
  }
  update();
}

auto QtEditViewer::ClampCropRect(const QRectF& rect) -> QRectF {
  QRectF r = rect.normalized();
  float  x = Clamp01(static_cast<float>(r.x()));
  float  y = Clamp01(static_cast<float>(r.y()));
  float  w = std::clamp(static_cast<float>(r.width()), kCropMinSize, 1.0f);
  float  h = std::clamp(static_cast<float>(r.height()), kCropMinSize, 1.0f);
  x        = std::clamp(x, 0.0f, 1.0f - w);
  y        = std::clamp(y, 0.0f, 1.0f - h);
  return QRectF(x, y, w, h);
}

void QtEditViewer::SetCropOverlayRectNormalized(float x, float y, float w, float h) {
  QRectF adjusted_rect;
  bool   rect_changed = false;
  {
    std::lock_guard<std::mutex> view_lock(view_state_mutex_);
    const QRectF clamped = ClampCropRect(QRectF(x, y, w, h));
    const QRectF adjusted =
        ClampCropRectForRotation(clamped, crop_overlay_rotation_degrees_, crop_overlay_metric_aspect_);
    rect_changed =
        std::abs(static_cast<float>(adjusted.x() - crop_overlay_rect_.x())) > 1e-6f ||
        std::abs(static_cast<float>(adjusted.y() - crop_overlay_rect_.y())) > 1e-6f ||
        std::abs(static_cast<float>(adjusted.width() - crop_overlay_rect_.width())) > 1e-6f ||
        std::abs(static_cast<float>(adjusted.height() - crop_overlay_rect_.height())) > 1e-6f;
    crop_overlay_rect_ = adjusted;
    adjusted_rect      = adjusted;
  }
  if (rect_changed) {
    emit CropOverlayRectChanged(static_cast<float>(adjusted_rect.x()),
                                static_cast<float>(adjusted_rect.y()),
                                static_cast<float>(adjusted_rect.width()),
                                static_cast<float>(adjusted_rect.height()), false);
  }
  update();
}

void QtEditViewer::SetCropOverlayRotationDegrees(float angle_degrees) {
  QRectF adjusted_rect;
  bool   rect_changed = false;
  {
    std::lock_guard<std::mutex> view_lock(view_state_mutex_);
    crop_overlay_rotation_degrees_ = NormalizeAngleDegrees(angle_degrees);
    const QRectF adjusted = ClampCropRectForRotation(crop_overlay_rect_, crop_overlay_rotation_degrees_,
                                                     crop_overlay_metric_aspect_);
    rect_changed = std::abs(static_cast<float>(adjusted.x() - crop_overlay_rect_.x())) > 1e-6f ||
                   std::abs(static_cast<float>(adjusted.y() - crop_overlay_rect_.y())) > 1e-6f ||
                   std::abs(static_cast<float>(adjusted.width() - crop_overlay_rect_.width())) > 1e-6f ||
                   std::abs(static_cast<float>(adjusted.height() - crop_overlay_rect_.height())) > 1e-6f;
    crop_overlay_rect_ = adjusted;
    adjusted_rect      = adjusted;
  }

  if (rect_changed) {
    emit CropOverlayRectChanged(static_cast<float>(adjusted_rect.x()),
                                static_cast<float>(adjusted_rect.y()),
                                static_cast<float>(adjusted_rect.width()),
                                static_cast<float>(adjusted_rect.height()), false);
  }
  update();
}

void QtEditViewer::ResetCropOverlayRectToFull() {
  SetCropOverlayRectNormalized(0.0f, 0.0f, 1.0f, 1.0f);
}

auto QtEditViewer::GetViewportRenderRegion() const -> std::optional<ViewportRenderRegion> {
  std::lock_guard<std::mutex> view_lock(view_state_mutex_);
  return viewport_render_region_cache_;
}

void QtEditViewer::SetNextFramePresentationMode(FramePresentationMode mode) {
  pending_frame_presentation_mode_.store(mode, std::memory_order_release);
  pending_presentation_mode_valid_.store(true, std::memory_order_release);
}

void QtEditViewer::SetHistogramFrameExpected(bool expected_fast_preview) {
  histogram_expect_fast_frame_.store(expected_fast_preview, std::memory_order_release);
}

void QtEditViewer::SetHistogramUpdateIntervalMs(int interval_ms) {
  histogram_update_interval_ms_ = std::max(0, interval_ms);
}

auto QtEditViewer::GetHistogramBufferId() const -> GLuint {
  if (!histogram_resources_ready_) {
    return 0;
  }
  return histogram_norm_ssbo_;
}

auto QtEditViewer::GetHistogramBinCount() const -> int { return kHistogramBins; }

auto QtEditViewer::HasHistogramData() const -> bool {
  return histogram_has_data_.load(std::memory_order_acquire);
}

float4* QtEditViewer::MapResourceForWrite() {
  mutex_.lock();

  // IMPORTANT: Do NOT map the OpenGL PBO from this thread. On Windows this often
  // fails with "invalid OpenGL or DirectX context" because the GL context is
  // owned by the GUI thread.
  if (!staging_ptr_ || staging_bytes_ == 0) {
    mutex_.unlock();
    return nullptr;
  }

  return staging_ptr_;
}

void QtEditViewer::UnmapResource() {
  // Mark which buffer should receive the pending frame.
  pending_frame_idx_.store(render_target_idx_, std::memory_order_release);
  mutex_.unlock();
}

void QtEditViewer::NotifyFrameReady() {
  histogram_pending_frame_.store(histogram_expect_fast_frame_.load(std::memory_order_acquire),
                                 std::memory_order_release);
  // Wake the UI thread to update the display
  emit RequestUpdate();
}

void QtEditViewer::initializeGL() {
  initializeOpenGLFunctions();

  program_ = new QOpenGLShaderProgram();
  if (!program_->addShaderFromSourceCode(QOpenGLShader::Vertex, vertexShaderSource)) {
    qWarning("Vertex shader compile failed: %s", program_->log().toUtf8().constData());
  }
  if (!program_->addShaderFromSourceCode(QOpenGLShader::Fragment, fragmentShaderSource)) {
    qWarning("Fragment shader compile failed: %s", program_->log().toUtf8().constData());
  }
  if (!program_->link()) {
    qWarning("Shader program link failed: %s", program_->log().toUtf8().constData());
  }

  // Static full-screen quad (never modified)
  float vertices[] = {
      -1.0f, -1.0f,
       1.0f, -1.0f,
      -1.0f,  1.0f,
       1.0f,  1.0f,
  };
  glGenBuffers(1, &vbo_);
  glBindBuffer(GL_ARRAY_BUFFER, vbo_);
  glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (buffers_[active_idx_].width <= 0 || buffers_[active_idx_].height <= 0) {
      buffers_[active_idx_].width  = std::max(1, this->width());
      buffers_[active_idx_].height = std::max(1, this->height());
    }
  }

  InitBuffer(buffers_[active_idx_], buffers_[active_idx_].width, buffers_[active_idx_].height);
  InitHistogramResources();
  UpdateViewportRenderRegionCache();
}

bool QtEditViewer::InitBuffer(GLBuffer& buffer, int width, int height) {
  if (width <= 0 || height <= 0) {
    qWarning("InitBuffer skipped: invalid size %dx%d", width, height);
    return false;
  }

  FreeBuffer(buffer);

  glGenTextures(1, &buffer.texture);
  glBindTexture(GL_TEXTURE_2D, buffer.texture);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, width, height, 0, GL_RGBA, GL_FLOAT, nullptr);

  cudaError_t err =
      cudaGraphicsGLRegisterImage(&buffer.cuda_resource, buffer.texture, GL_TEXTURE_2D,
                                  cudaGraphicsRegisterFlagsWriteDiscard);
  if (err != cudaSuccess) {
    qWarning("Failed to register texture with CUDA: %s", cudaGetErrorString(err));
    FreeBuffer(buffer);
    return false;
  }

  glBindTexture(GL_TEXTURE_2D, 0);

  buffer.width  = width;
  buffer.height = height;
  return true;
}

auto QtEditViewer::BuildComputeProgram(const char* source, const char* debug_name,
                                       GLuint& out_program) -> bool {
  if (!source) {
    return false;
  }

  if (out_program != 0) {
    glDeleteProgram(out_program);
    out_program = 0;
  }

  const GLuint shader = glCreateShader(GL_COMPUTE_SHADER);
  glShaderSource(shader, 1, &source, nullptr);
  glCompileShader(shader);

  GLint compile_ok = GL_FALSE;
  glGetShaderiv(shader, GL_COMPILE_STATUS, &compile_ok);
  if (compile_ok != GL_TRUE) {
    GLint log_len = 0;
    glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &log_len);
    std::vector<char> log(static_cast<size_t>(std::max(0, log_len)) + 1, '\0');
    if (log_len > 0) {
      glGetShaderInfoLog(shader, log_len, nullptr, log.data());
    }
    qWarning("%s compute shader compile failed: %s", debug_name, log.data());
    glDeleteShader(shader);
    return false;
  }

  const GLuint program = glCreateProgram();
  glAttachShader(program, shader);
  glLinkProgram(program);
  glDeleteShader(shader);

  GLint link_ok = GL_FALSE;
  glGetProgramiv(program, GL_LINK_STATUS, &link_ok);
  if (link_ok != GL_TRUE) {
    GLint log_len = 0;
    glGetProgramiv(program, GL_INFO_LOG_LENGTH, &log_len);
    std::vector<char> log(static_cast<size_t>(std::max(0, log_len)) + 1, '\0');
    if (log_len > 0) {
      glGetProgramInfoLog(program, log_len, nullptr, log.data());
    }
    qWarning("%s compute program link failed: %s", debug_name, log.data());
    glDeleteProgram(program);
    return false;
  }

  out_program = program;
  return true;
}

bool QtEditViewer::InitHistogramResources() {
  if (histogram_resources_ready_) {
    return true;
  }

  auto* gl_context = context();
  if (!gl_context) {
    return false;
  }

  const QSurfaceFormat format = gl_context->format();
  const bool has_compute_support =
      (format.majorVersion() > 4 || (format.majorVersion() == 4 && format.minorVersion() >= 3)) ||
      gl_context->hasExtension(QByteArrayLiteral("GL_ARB_compute_shader"));
  if (!has_compute_support) {
    qWarning("QtEditViewer histogram disabled: OpenGL compute shaders are not supported.");
    return false;
  }

  if (!BuildComputeProgram(histogramClearShaderSource, "HistogramClear",
                           histogram_clear_program_)) {
    FreeHistogramResources();
    return false;
  }
  if (!BuildComputeProgram(histogramComputeShaderSource, "HistogramCompute",
                           histogram_compute_program_)) {
    FreeHistogramResources();
    return false;
  }
  if (!BuildComputeProgram(histogramNormalizeShaderSource, "HistogramNormalize",
                           histogram_normalize_program_)) {
    FreeHistogramResources();
    return false;
  }

  const GLsizeiptr count_bytes = static_cast<GLsizeiptr>(sizeof(GLuint) * kHistogramBins * 3);
  const GLsizeiptr norm_bytes  = static_cast<GLsizeiptr>(sizeof(float) * kHistogramBins * 3);

  glGenBuffers(1, &histogram_count_ssbo_);
  glBindBuffer(GL_SHADER_STORAGE_BUFFER, histogram_count_ssbo_);
  glBufferData(GL_SHADER_STORAGE_BUFFER, count_bytes, nullptr, GL_DYNAMIC_DRAW);

  glGenBuffers(1, &histogram_norm_ssbo_);
  glBindBuffer(GL_SHADER_STORAGE_BUFFER, histogram_norm_ssbo_);
  glBufferData(GL_SHADER_STORAGE_BUFFER, norm_bytes, nullptr, GL_DYNAMIC_DRAW);
  glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

  histogram_clear_count_loc_    = glGetUniformLocation(histogram_clear_program_, "uCount");
  histogram_compute_tex_loc_    = glGetUniformLocation(histogram_compute_program_, "uSourceTex");
  histogram_compute_bins_loc_   = glGetUniformLocation(histogram_compute_program_, "uBins");
  histogram_compute_sample_loc_ = glGetUniformLocation(histogram_compute_program_, "uSampleSize");
  histogram_norm_bins_loc_      = glGetUniformLocation(histogram_normalize_program_, "uBins");

  histogram_resources_ready_ = histogram_count_ssbo_ != 0 && histogram_norm_ssbo_ != 0;
  histogram_has_data_.store(false, std::memory_order_release);
  last_histogram_update_time_ = {};
  return histogram_resources_ready_;
}

void QtEditViewer::FreeHistogramResources() {
  if (histogram_count_ssbo_) {
    glDeleteBuffers(1, &histogram_count_ssbo_);
    histogram_count_ssbo_ = 0;
  }
  if (histogram_norm_ssbo_) {
    glDeleteBuffers(1, &histogram_norm_ssbo_);
    histogram_norm_ssbo_ = 0;
  }
  if (histogram_clear_program_) {
    glDeleteProgram(histogram_clear_program_);
    histogram_clear_program_ = 0;
  }
  if (histogram_compute_program_) {
    glDeleteProgram(histogram_compute_program_);
    histogram_compute_program_ = 0;
  }
  if (histogram_normalize_program_) {
    glDeleteProgram(histogram_normalize_program_);
    histogram_normalize_program_ = 0;
  }

  histogram_clear_count_loc_    = -1;
  histogram_compute_tex_loc_     = -1;
  histogram_compute_bins_loc_    = -1;
  histogram_compute_sample_loc_  = -1;
  histogram_norm_bins_loc_       = -1;
  histogram_resources_ready_     = false;
  histogram_has_data_.store(false, std::memory_order_release);
  histogram_pending_frame_.store(false, std::memory_order_release);
}

auto QtEditViewer::ComputeViewportRenderRegion(int image_width, int image_height) const
    -> std::optional<ViewportRenderRegion> {
  const float dpr = devicePixelRatioF();
  const float vw  = std::max(1.0f, float(width()) * dpr);
  const float vh  = std::max(1.0f, float(height()) * dpr);
  if (vw <= 0.0f || vh <= 0.0f) {
    return std::nullopt;
  }

  float zoom = 1.0f;
  QVector2D pan(0.0f, 0.0f);
  int       base_width  = image_width;
  int       base_height = image_height;
  {
    std::lock_guard<std::mutex> view_lock(view_state_mutex_);
    zoom = std::max(view_zoom_, 1e-4f);
    pan  = view_pan_;
    if (render_reference_width_ > 0 && render_reference_height_ > 0) {
      base_width  = render_reference_width_;
      base_height = render_reference_height_;
    }
  }

  if (base_width <= 0 || base_height <= 0) {
    return std::nullopt;
  }

  const float imgW = static_cast<float>(base_width);
  const float imgH = static_cast<float>(base_height);
  const float winAspect = vw / vh;
  const float imgAspect = imgW / imgH;

  float sx = 1.0f;
  float sy = 1.0f;
  if (imgAspect > winAspect) {
    sy = winAspect / imgAspect;
  } else {
    sx = imgAspect / winAspect;
  }
  sx = std::max(sx, 1e-4f);
  sy = std::max(sy, 1e-4f);

  const float inv_x = 1.0f / (sx * zoom);
  const float inv_y = 1.0f / (sy * zoom);

  float px_min = (-1.0f - pan.x()) * inv_x;
  float px_max = (1.0f - pan.x()) * inv_x;
  float py_min = (-1.0f - pan.y()) * inv_y;
  float py_max = (1.0f - pan.y()) * inv_y;

  if (px_min > px_max) std::swap(px_min, px_max);
  if (py_min > py_max) std::swap(py_min, py_max);

  px_min = std::clamp(px_min, -1.0f, 1.0f);
  px_max = std::clamp(px_max, -1.0f, 1.0f);
  py_min = std::clamp(py_min, -1.0f, 1.0f);
  py_max = std::clamp(py_max, -1.0f, 1.0f);

  const float u_min = std::clamp((px_min + 1.0f) * 0.5f, 0.0f, 1.0f);
  const float u_max = std::clamp((px_max + 1.0f) * 0.5f, 0.0f, 1.0f);
  const float v_min = std::clamp((1.0f - py_max) * 0.5f, 0.0f, 1.0f);
  const float v_max = std::clamp((1.0f - py_min) * 0.5f, 0.0f, 1.0f);

  const float roi_factor_x = std::clamp(u_max - u_min, 1e-4f, 1.0f);
  const float roi_factor_y = std::clamp(v_max - v_min, 1e-4f, 1.0f);

  int roi_w = std::clamp(
      static_cast<int>(std::lround(static_cast<float>(base_width) * roi_factor_x)), 1, base_width);
  int roi_h = std::clamp(
      static_cast<int>(std::lround(static_cast<float>(base_height) * roi_factor_y)), 1, base_height);

  const float center_u = std::clamp((u_min + u_max) * 0.5f, 0.0f, 1.0f);
  const float center_v = std::clamp((v_min + v_max) * 0.5f, 0.0f, 1.0f);

  int x = static_cast<int>(std::lround(center_u * static_cast<float>(base_width) -
                                       static_cast<float>(roi_w) * 0.5f));
  int y = static_cast<int>(std::lround(center_v * static_cast<float>(base_height) -
                                       static_cast<float>(roi_h) * 0.5f));
  x = std::clamp(x, 0, std::max(0, base_width - roi_w));
  y = std::clamp(y, 0, std::max(0, base_height - roi_h));

  ViewportRenderRegion region;
  region.x_       = x;
  region.y_       = y;
  region.scale_x_ = std::clamp(static_cast<float>(roi_w) / static_cast<float>(base_width), 1e-4f,
                               1.0f);
  region.scale_y_ = std::clamp(static_cast<float>(roi_h) / static_cast<float>(base_height), 1e-4f,
                               1.0f);
  return region;
}

void QtEditViewer::UpdateViewportRenderRegionCache() {
  int image_width = 0;
  int image_height = 0;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    image_width  = buffers_[active_idx_].width;
    image_height = buffers_[active_idx_].height;
  }

  const auto region = ComputeViewportRenderRegion(image_width, image_height);
  std::lock_guard<std::mutex> view_lock(view_state_mutex_);
  viewport_render_region_cache_ = region;
}

auto QtEditViewer::WidgetPointToImageUv(const QPointF& widget_pos, int image_width, int image_height) const
    -> std::optional<QPointF> {
  if (image_width <= 0 || image_height <= 0) {
    return std::nullopt;
  }

  const float dpr = devicePixelRatioF();
  const float vw  = std::max(1.0f, static_cast<float>(width()) * dpr);
  const float vh  = std::max(1.0f, static_cast<float>(height()) * dpr);
  if (vw <= 0.0f || vh <= 0.0f) {
    return std::nullopt;
  }

  float     zoom = 1.0f;
  QVector2D pan(0.0f, 0.0f);
  {
    std::lock_guard<std::mutex> view_lock(view_state_mutex_);
    zoom = std::max(view_zoom_, 1e-4f);
    pan  = view_pan_;
  }

  const float imgW      = static_cast<float>(std::max(1, image_width));
  const float imgH      = static_cast<float>(std::max(1, image_height));
  const float winAspect = vw / vh;
  const float imgAspect = imgW / imgH;

  float sx = 1.0f;
  float sy = 1.0f;
  if (imgAspect > winAspect) {
    sy = winAspect / imgAspect;
  } else {
    sx = imgAspect / winAspect;
  }
  sx = std::max(sx, 1e-4f);
  sy = std::max(sy, 1e-4f);

  const float px   = static_cast<float>(widget_pos.x()) * dpr;
  const float py   = static_cast<float>(widget_pos.y()) * dpr;
  const float ndcX = (2.0f * px / vw) - 1.0f;
  const float ndcY = 1.0f - (2.0f * py / vh);

  const float imgX = (ndcX - pan.x()) / (sx * zoom);
  const float imgY = (ndcY - pan.y()) / (sy * zoom);

  const float u = (imgX + 1.0f) * 0.5f;
  const float v = (1.0f - imgY) * 0.5f;
  return QPointF(u, v);
}

auto QtEditViewer::ImageUvToWidgetPoint(const QPointF& uv, int image_width, int image_height) const
    -> std::optional<QPointF> {
  if (image_width <= 0 || image_height <= 0) {
    return std::nullopt;
  }

  const float dpr = devicePixelRatioF();
  const float vw  = std::max(1.0f, static_cast<float>(width()) * dpr);
  const float vh  = std::max(1.0f, static_cast<float>(height()) * dpr);
  if (vw <= 0.0f || vh <= 0.0f) {
    return std::nullopt;
  }

  float     zoom = 1.0f;
  QVector2D pan(0.0f, 0.0f);
  {
    std::lock_guard<std::mutex> view_lock(view_state_mutex_);
    zoom = std::max(view_zoom_, 1e-4f);
    pan  = view_pan_;
  }

  const float imgW      = static_cast<float>(std::max(1, image_width));
  const float imgH      = static_cast<float>(std::max(1, image_height));
  const float winAspect = vw / vh;
  const float imgAspect = imgW / imgH;

  float sx = 1.0f;
  float sy = 1.0f;
  if (imgAspect > winAspect) {
    sy = winAspect / imgAspect;
  } else {
    sx = imgAspect / winAspect;
  }
  sx = std::max(sx, 1e-4f);
  sy = std::max(sy, 1e-4f);

  const float u    = Clamp01(static_cast<float>(uv.x()));
  const float v    = Clamp01(static_cast<float>(uv.y()));
  const float imgX = (2.0f * u) - 1.0f;
  const float imgY = 1.0f - (2.0f * v);
  const float ndcX = (imgX * sx * zoom) + pan.x();
  const float ndcY = (imgY * sy * zoom) + pan.y();

  const float px = ((ndcX + 1.0f) * 0.5f) * vw;
  const float py = ((1.0f - ndcY) * 0.5f) * vh;
  return QPointF(px / dpr, py / dpr);
}

auto QtEditViewer::ShouldComputeHistogramNow() -> bool {
  if (histogram_update_interval_ms_ <= 0) {
    last_histogram_update_time_ = std::chrono::steady_clock::now();
    return true;
  }

  const auto now = std::chrono::steady_clock::now();
  if (last_histogram_update_time_.time_since_epoch().count() == 0) {
    last_histogram_update_time_ = now;
    return true;
  }

  const auto elapsed_ms =
      std::chrono::duration_cast<std::chrono::milliseconds>(now - last_histogram_update_time_)
          .count();
  if (elapsed_ms < histogram_update_interval_ms_) {
    return false;
  }
  last_histogram_update_time_ = now;
  return true;
}

auto QtEditViewer::ComputeHistogram(GLuint texture_id, int width, int height) -> bool {
  if (!histogram_resources_ready_ || texture_id == 0 || width <= 0 || height <= 0) {
    return false;
  }

  const int histogram_values = kHistogramBins * 3;
  glUseProgram(histogram_clear_program_);
  if (histogram_clear_count_loc_ >= 0) {
    glUniform1i(histogram_clear_count_loc_, histogram_values);
  }
  glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, histogram_count_ssbo_);
  const GLuint clear_groups = static_cast<GLuint>((histogram_values + 64 - 1) / 64);
  glDispatchCompute(clear_groups, 1, 1);
  glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

  glUseProgram(histogram_compute_program_);
  if (histogram_compute_tex_loc_ >= 0) {
    glUniform1i(histogram_compute_tex_loc_, 0);
  }
  if (histogram_compute_bins_loc_ >= 0) {
    glUniform1i(histogram_compute_bins_loc_, kHistogramBins);
  }
  if (histogram_compute_sample_loc_ >= 0) {
    glUniform1i(histogram_compute_sample_loc_, kHistogramSampleSize);
  }

  glActiveTexture(GL_TEXTURE0);
  glBindTexture(GL_TEXTURE_2D, texture_id);
  glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, histogram_count_ssbo_);

  const GLuint groups =
      static_cast<GLuint>((kHistogramSampleSize + 16 - 1) / 16);
  glDispatchCompute(groups, groups, 1);
  glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

  glUseProgram(histogram_normalize_program_);
  if (histogram_norm_bins_loc_ >= 0) {
    glUniform1i(histogram_norm_bins_loc_, kHistogramBins);
  }
  glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, histogram_count_ssbo_);
  glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, histogram_norm_ssbo_);
  glDispatchCompute(1, 1, 1);
  glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_VERTEX_ATTRIB_ARRAY_BARRIER_BIT);

  glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, 0);
  glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, 0);
  glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
  glUseProgram(0);
  glFlush();

  return true;
}

void QtEditViewer::paintGL() {
  std::lock_guard<std::mutex> lock(mutex_);

  // First, check if there is a pending frame to copy from staging buffer.
  // This must happen BEFORE we check the active buffer validity, because the
  // pending frame might switch active_idx_ to a newly initialized buffer.
  const int pending_idx = pending_frame_idx_.exchange(-1, std::memory_order_acq_rel);
  GLBuffer* target_buffer = nullptr;
  if (pending_idx >= 0 && pending_idx < static_cast<int>(buffers_.size())) {
    target_buffer = &buffers_[pending_idx];
  }

  if (target_buffer && target_buffer->cuda_resource && staging_ptr_ && staging_bytes_ > 0) {
    cudaError_t map_err = cudaGraphicsMapResources(1, &target_buffer->cuda_resource, 0);
    if (map_err != cudaSuccess) {
      qWarning("Failed to map CUDA resource (paintGL): %s", cudaGetErrorString(map_err));
    } else {
      cudaArray_t mapped_array = nullptr;
      cudaError_t array_err =
          cudaGraphicsSubResourceGetMappedArray(&mapped_array, target_buffer->cuda_resource, 0, 0);
      if (array_err != cudaSuccess || !mapped_array) {
        qWarning("Failed to map texture array (paintGL): %s", cudaGetErrorString(array_err));
      } else {
        const size_t row_bytes = static_cast<size_t>(target_buffer->width) * sizeof(float4);
        const size_t max_rows  = staging_bytes_ / row_bytes;
        const size_t copy_rows = std::min(max_rows, static_cast<size_t>(target_buffer->height));
        cudaError_t copy_err =
            cudaMemcpy2DToArray(mapped_array, 0, 0, staging_ptr_, row_bytes, row_bytes, copy_rows,
                                cudaMemcpyDeviceToDevice);
        if (copy_err != cudaSuccess) {
          qWarning("Failed to copy staging->texture: %s", cudaGetErrorString(copy_err));
        } else {
          active_idx_ = pending_idx;
          write_idx_  = 1 - active_idx_;
          if (pending_presentation_mode_valid_.exchange(false, std::memory_order_acq_rel)) {
            active_frame_presentation_mode_.store(
                pending_frame_presentation_mode_.load(std::memory_order_acquire),
                std::memory_order_release);
          }
        }
      }

      cudaError_t unmap_err = cudaGraphicsUnmapResources(1, &target_buffer->cuda_resource, 0);
      if (unmap_err != cudaSuccess) {
        qWarning("Failed to unmap CUDA resource (paintGL): %s", cudaGetErrorString(unmap_err));
      }
    }
  }

  // Now check if the active buffer is valid for rendering.
  GLBuffer& active_buffer = buffers_[active_idx_];
  if (!active_buffer.texture || !program_ || !program_->isLinked()) return;

  const float dpr = devicePixelRatioF();
  const float vw  = std::max(1.0f, float(width()) * dpr);
  const float vh  = std::max(1.0f, float(height()) * dpr);
  glViewport(0, 0, int(vw), int(vh));

  // Compute letterbox scale from IMAGE aspect vs WINDOW aspect
  const float imgW = float(std::max(1, active_buffer.width));
  const float imgH = float(std::max(1, active_buffer.height));
  const float winAspect = vw / vh;
  const float imgAspect = imgW / imgH;

  float sx = 1.0f, sy = 1.0f;
  if (imgAspect > winAspect) {
    sy = winAspect / imgAspect; // image wider -> reduce Y
  } else {
    sx = imgAspect / winAspect; // image taller -> reduce X
  }

  float     zoom = 1.0f;
  QVector2D pan(0.0f, 0.0f);
  {
    std::lock_guard<std::mutex> view_lock(view_state_mutex_);
    zoom = view_zoom_;
    pan  = view_pan_;
  }

  const auto presentation_mode = active_frame_presentation_mode_.load(std::memory_order_acquire);
  if (presentation_mode == FramePresentationMode::RoiFrame) {
    zoom = 1.0f;
    pan  = QVector2D(0.0f, 0.0f);
  } else {
    std::lock_guard<std::mutex> view_lock(view_state_mutex_);
    render_reference_width_  = active_buffer.width;
    render_reference_height_ = active_buffer.height;
  }

  const auto viewport_region = ComputeViewportRenderRegion(active_buffer.width, active_buffer.height);
  {
    std::lock_guard<std::mutex> view_lock(view_state_mutex_);
    viewport_render_region_cache_ = viewport_region;
  }

  glClearColor(0.f, 0.f, 0.f, 1.f);
  glClear(GL_COLOR_BUFFER_BIT);

  program_->bind();
  program_->setUniformValue("uScale", QVector2D(sx, sy));
  program_->setUniformValue("uPan", pan);
  program_->setUniformValue("uZoom", zoom);

  glActiveTexture(GL_TEXTURE0);
  program_->setUniformValue("textureSampler", 0);
  glBindTexture(GL_TEXTURE_2D, active_buffer.texture);

  glBindBuffer(GL_ARRAY_BUFFER, vbo_);
  int pos_loc = program_->attributeLocation("position");
  program_->enableAttributeArray(pos_loc);
  program_->setAttributeBuffer(pos_loc, GL_FLOAT, 0, 2, 2 * sizeof(float));

  glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
  program_->release();

  bool  draw_crop_overlay = false;
  QRectF crop_rect_uv;
  float crop_rotation_degrees = 0.0f;
  const float crop_metric_aspect = SafeAspect(active_buffer.width, active_buffer.height);
  {
    std::lock_guard<std::mutex> view_lock(view_state_mutex_);
    draw_crop_overlay            = crop_overlay_visible_;
    crop_rect_uv                 = crop_overlay_rect_;
    crop_rotation_degrees        = crop_overlay_rotation_degrees_;
    crop_overlay_metric_aspect_  = crop_metric_aspect;
  }
  if (draw_crop_overlay) {
    const auto image_top_left_opt =
        ImageUvToWidgetPoint(QPointF(0.0, 0.0), active_buffer.width, active_buffer.height);
    const auto image_bottom_right_opt =
        ImageUvToWidgetPoint(QPointF(1.0, 1.0), active_buffer.width, active_buffer.height);
    const auto crop_corners_uv =
        RotatedCropCornersUv(crop_rect_uv, crop_rotation_degrees, crop_metric_aspect);
    std::array<QPointF, 4> crop_corners_widget{};
    bool                   crop_corners_valid = true;
    for (size_t i = 0; i < crop_corners_uv.size(); ++i) {
      const auto corner_widget =
          ImageUvToWidgetPoint(crop_corners_uv[i], active_buffer.width, active_buffer.height);
      if (!corner_widget.has_value()) {
        crop_corners_valid = false;
        break;
      }
      crop_corners_widget[i] = *corner_widget;
    }

    if (crop_corners_valid && image_top_left_opt && image_bottom_right_opt) {
      const QRectF image_rect = QRectF(*image_top_left_opt, *image_bottom_right_opt).normalized();
      if (image_rect.isValid()) {
        QPolygonF crop_polygon;
        crop_polygon.reserve(static_cast<int>(crop_corners_widget.size()));
        for (const auto& p : crop_corners_widget) {
          crop_polygon.push_back(p);
        }

        QPainter painter(this);
        painter.setRenderHint(QPainter::Antialiasing, true);

        QPainterPath image_path;
        image_path.addRect(image_rect);
        QPainterPath crop_path;
        crop_path.addPolygon(crop_polygon);
        crop_path.closeSubpath();
        painter.fillPath(image_path.subtracted(crop_path), QColor(0, 0, 0, 110));

        painter.setPen(QPen(QColor(252, 199, 4, 220), 1.2));
        painter.setBrush(Qt::NoBrush);
        painter.drawPolygon(crop_polygon);

        painter.setPen(QPen(QColor(252, 199, 4, 150), 1.0, Qt::DashLine));
        for (const float t : {1.0f / 3.0f, 2.0f / 3.0f}) {
          painter.drawLine(LerpPoint(crop_corners_widget[0], crop_corners_widget[1], t),
                           LerpPoint(crop_corners_widget[3], crop_corners_widget[2], t));
          painter.drawLine(LerpPoint(crop_corners_widget[0], crop_corners_widget[3], t),
                           LerpPoint(crop_corners_widget[1], crop_corners_widget[2], t));
        }

        painter.setPen(QPen(QColor(18, 18, 18, 230), 1.0));
        painter.setBrush(QColor(252, 199, 4, 230));
        for (const auto& corner : crop_corners_widget) {
          painter.drawEllipse(corner, kCropCornerDrawRadiusPx, kCropCornerDrawRadiusPx);
        }
      }
    }
  }

  const bool histogram_requested = histogram_pending_frame_.exchange(false, std::memory_order_acq_rel);
  if (histogram_requested && ShouldComputeHistogramNow()) {
    if (histogram_resources_ready_ || InitHistogramResources()) {
      if (ComputeHistogram(active_buffer.texture, active_buffer.width, active_buffer.height)) {
        histogram_has_data_.store(true, std::memory_order_release);
        emit HistogramDataUpdated();
      }
    }
  }
}

void QtEditViewer::resizeGL(int w, int h) {
  if (w <= 0 || h <= 0) return;
  UpdateViewportRenderRegionCache();
}

void QtEditViewer::FreeBuffer(GLBuffer& buffer) {
  if (buffer.cuda_resource) {
    cudaGraphicsUnregisterResource(buffer.cuda_resource);
    buffer.cuda_resource = nullptr;
  }
  if (buffer.texture) {
    glDeleteTextures(1, &buffer.texture);
    buffer.texture = 0;
  }
  buffer.width  = 0;
  buffer.height = 0;
}

void QtEditViewer::FreeAllBuffers() {
  for (auto& buffer : buffers_) {
    FreeBuffer(buffer);
  }
}

void QtEditViewer::OnResizeGL(int w, int h) {
  bool skip_resize = false;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    const GLBuffer& target = buffers_[write_idx_];
    if (target.texture != 0 && target.cuda_resource && target.width == w && target.height == h) {
      render_target_idx_ = write_idx_;
      skip_resize        = true;
    }
  }
  if (skip_resize) {
    UpdateViewportRenderRegionCache();
    update();
    return;
  }

  makeCurrent();
  {
    std::lock_guard<std::mutex> lock(mutex_);
    GLBuffer& target = buffers_[write_idx_];
    FreeBuffer(target);
    if (InitBuffer(target, w, h)) {
      render_target_idx_ = write_idx_;
    }
  }
  // resizeGL(w, h);
  doneCurrent();
  UpdateViewportRenderRegionCache();

  // Ensure a repaint after resize
  update();
}

void QtEditViewer::wheelEvent(QWheelEvent* event) {
  // Allow Ctrl+Wheel zoom even when crop tool is active.
  if ((event->modifiers() & Qt::ControlModifier) == Qt::ControlModifier) {
    const int wheel_delta = event->angleDelta().y();
    if (wheel_delta != 0) {
      float current_zoom = kMinInteractiveZoom;
      QVector2D current_pan(0.0f, 0.0f);
      {
        std::lock_guard<std::mutex> view_lock(view_state_mutex_);
        current_zoom = view_zoom_;
        current_pan  = view_pan_;
      }

      const float steps =
          static_cast<float>(wheel_delta) / static_cast<float>(QWheelEvent::DefaultDeltasPerStep);
      const float target_zoom =
          std::clamp(current_zoom * std::pow(kWheelZoomStep, steps), kMinInteractiveZoom,
                     kMaxInteractiveZoom);
      const QVector2D target_pan = ComputeAnchoredPan(target_zoom, event->position(), current_pan);
      {
        std::lock_guard<std::mutex> view_lock(view_state_mutex_);
        double_click_zoom_target_  = target_zoom;
        // After explicit Ctrl+wheel zoom, the next double-click should reset to 100%.
        double_click_zoom_in_next_ = false;
      }
      click_zoom_toggle_active_ = false;
      StopZoomAnimation();
      ApplyViewTransform(target_zoom, target_pan, true);
    }
    event->accept();
    return;
  }

  event->accept();
}

void QtEditViewer::mousePressEvent(QMouseEvent* event) {
  if (event->button() == Qt::LeftButton) {
    bool   crop_tool_enabled = false;
    bool   crop_overlay_visible = false;
    QRectF crop_rect{};
    float  crop_rotation_degrees = 0.0f;
    int    image_width = 0;
    int    image_height = 0;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      image_width  = buffers_[active_idx_].width;
      image_height = buffers_[active_idx_].height;
    }
    {
      std::lock_guard<std::mutex> view_lock(view_state_mutex_);
      crop_tool_enabled     = crop_tool_enabled_;
      crop_overlay_visible  = crop_overlay_visible_;
      crop_rect             = crop_overlay_rect_;
      crop_rotation_degrees = crop_overlay_rotation_degrees_;
    }

    if (crop_tool_enabled && crop_overlay_visible) {
      const auto uv_opt = WidgetPointToImageUv(event->position(), image_width, image_height);
      if (uv_opt.has_value()) {
        const float  metric_aspect = SafeAspect(image_width, image_height);
        const QPointF uv_point(Clamp01(static_cast<float>(uv_opt->x())),
                               Clamp01(static_cast<float>(uv_opt->y())));

        int   hit_corner = -1;
        float best_dist2 = kCropCornerHitRadiusPx * kCropCornerHitRadiusPx;
        const auto crop_corners_uv =
            RotatedCropCornersUv(crop_rect, crop_rotation_degrees, metric_aspect);
        std::array<QPointF, 4> crop_corners_widget{};
        bool                   corners_valid = true;
        for (int i = 0; i < static_cast<int>(crop_corners_uv.size()); ++i) {
          const auto corner_widget =
              ImageUvToWidgetPoint(crop_corners_uv[static_cast<size_t>(i)], image_width, image_height);
          if (!corner_widget.has_value()) {
            corners_valid = false;
            break;
          }
          crop_corners_widget[static_cast<size_t>(i)] = *corner_widget;
          const float dx = static_cast<float>(corner_widget->x() - event->position().x());
          const float dy = static_cast<float>(corner_widget->y() - event->position().y());
          const float d2 = (dx * dx) + (dy * dy);
          if (d2 <= best_dist2) {
            best_dist2  = d2;
            hit_corner = i;
          }
        }

        CropEdge hit_edge = CropEdge::None;
        if (hit_corner < 0 && corners_valid) {
          const float edge_hit_dist2 = kCropEdgeHitRadiusPx * kCropEdgeHitRadiusPx;
          const float top_d2 = PointSegmentDistanceSquared(event->position(), crop_corners_widget[0],
                                                           crop_corners_widget[1]);
          const float right_d2 = PointSegmentDistanceSquared(event->position(), crop_corners_widget[1],
                                                             crop_corners_widget[2]);
          const float bottom_d2 = PointSegmentDistanceSquared(event->position(), crop_corners_widget[2],
                                                              crop_corners_widget[3]);
          const float left_d2 = PointSegmentDistanceSquared(event->position(), crop_corners_widget[3],
                                                            crop_corners_widget[0]);
          float min_edge_d2 = edge_hit_dist2;
          auto  try_edge = [&](float d2, CropEdge edge) {
            if (d2 <= min_edge_d2) {
              min_edge_d2 = d2;
              hit_edge    = edge;
            }
          };
          try_edge(top_d2, CropEdge::Top);
          try_edge(right_d2, CropEdge::Right);
          try_edge(bottom_d2, CropEdge::Bottom);
          try_edge(left_d2, CropEdge::Left);
        }

        const bool inside_crop =
            IsPointInsideRotatedCrop(uv_point, crop_rect, crop_rotation_degrees, metric_aspect);
        QRectF     emit_rect;
        {
          std::lock_guard<std::mutex> view_lock(view_state_mutex_);
          crop_overlay_metric_aspect_  = metric_aspect;
          crop_drag_anchor_uv_         = uv_point;
          crop_drag_anchor_widget_pos_ = event->position();
          crop_drag_origin_rect_       = crop_rect;
          crop_drag_rotation_degrees_  = crop_rotation_degrees;
          crop_drag_corner_            = CropCorner::None;
          crop_drag_edge_              = CropEdge::None;

          if (hit_corner >= 0) {
            crop_drag_mode_ = CropDragMode::RotateCorner;
            switch (hit_corner) {
              case 0:
                crop_drag_corner_ = CropCorner::TopLeft;
                break;
              case 1:
                crop_drag_corner_ = CropCorner::TopRight;
                break;
              case 2:
                crop_drag_corner_ = CropCorner::BottomRight;
                break;
              case 3:
                crop_drag_corner_ = CropCorner::BottomLeft;
                break;
              default:
                crop_drag_corner_ = CropCorner::None;
                break;
            }
            setCursor(Qt::SizeHorCursor);
          } else if (hit_edge != CropEdge::None) {
            crop_drag_mode_ = CropDragMode::ResizeEdge;
            crop_drag_edge_ = hit_edge;
            setCursor(Qt::SizeAllCursor);
          } else if (inside_crop) {
            crop_drag_mode_ = CropDragMode::Move;
            setCursor(Qt::SizeAllCursor);
          } else {
            crop_drag_mode_ = CropDragMode::Create;
            crop_overlay_rect_ = ClampCropRectForRotation(
                QRectF(uv_point, QSizeF(kCropMinSize, kCropMinSize)), crop_rotation_degrees,
                metric_aspect);
            crop_drag_origin_rect_ = crop_overlay_rect_;
            setCursor(Qt::CrossCursor);
          }
          emit_rect = crop_overlay_rect_;
        }

        emit CropOverlayRectChanged(static_cast<float>(emit_rect.x()),
                                    static_cast<float>(emit_rect.y()),
                                    static_cast<float>(emit_rect.width()),
                                    static_cast<float>(emit_rect.height()), false);
        update();
        event->accept();
        return;
      }
    }
  }

  if (event->button() == Qt::LeftButton || event->button() == Qt::MiddleButton) {
    bool crop_tool_enabled = false;
    bool crop_overlay_visible = false;
    {
      std::lock_guard<std::mutex> view_lock(view_state_mutex_);
      crop_tool_enabled   = crop_tool_enabled_;
      crop_overlay_visible = crop_overlay_visible_;
    }
    if (crop_tool_enabled && crop_overlay_visible) {
      event->accept();
      return;
    }
    pending_click_toggle_ = false;
    if (click_toggle_timer_ && click_toggle_timer_->isActive()) {
      click_toggle_timer_->stop();
    }
    StopZoomAnimation();
    dragging_              = true;
    dragged_since_press_   = false;
    last_mouse_pos_        = event->pos();
    drag_start_mouse_pos_  = event->pos();
    setCursor(Qt::ClosedHandCursor);
    event->accept();
    return;
  }
  QOpenGLWidget::mousePressEvent(event);
}

void QtEditViewer::mouseMoveEvent(QMouseEvent* event) {
  if ((event->buttons() & Qt::LeftButton) == Qt::LeftButton) {
    CropDragMode crop_mode = CropDragMode::None;
    CropCorner   crop_corner = CropCorner::None;
    CropEdge     crop_edge = CropEdge::None;
    QPointF      anchor_uv{};
    QPointF      anchor_widget_pos{};
    QRectF       origin_rect{};
    float        drag_rotation_degrees = 0.0f;
    bool         tool_enabled = false;
    bool         overlay_visible = false;
    int          image_width = 0;
    int          image_height = 0;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      image_width  = buffers_[active_idx_].width;
      image_height = buffers_[active_idx_].height;
    }
    {
      std::lock_guard<std::mutex> view_lock(view_state_mutex_);
      crop_mode              = crop_drag_mode_;
      crop_corner            = crop_drag_corner_;
      crop_edge              = crop_drag_edge_;
      anchor_uv              = crop_drag_anchor_uv_;
      anchor_widget_pos      = crop_drag_anchor_widget_pos_;
      origin_rect            = crop_drag_origin_rect_;
      drag_rotation_degrees  = crop_drag_rotation_degrees_;
      tool_enabled           = crop_tool_enabled_;
      overlay_visible        = crop_overlay_visible_;
    }

    if (tool_enabled && overlay_visible && crop_mode != CropDragMode::None) {
      const float metric_aspect = SafeAspect(image_width, image_height);
      QPointF     uv{};
      if (crop_mode != CropDragMode::RotateCorner) {
        const auto uv_opt = WidgetPointToImageUv(event->position(), image_width, image_height);
        if (!uv_opt.has_value()) {
          event->accept();
          return;
        }
        uv = QPointF(Clamp01(static_cast<float>(uv_opt->x())),
                     Clamp01(static_cast<float>(uv_opt->y())));
      }
      QRectF new_rect = origin_rect;
      float  new_rotation_degrees = drag_rotation_degrees;
      bool   rotation_changed = false;
      if (crop_mode == CropDragMode::Create) {
        new_rect =
            ClampCropRectForRotation(QRectF(anchor_uv, uv).normalized(), drag_rotation_degrees,
                                     metric_aspect);
      } else if (crop_mode == CropDragMode::Move) {
        const QPointF delta_metric = UvToMetric(uv, metric_aspect) - UvToMetric(anchor_uv, metric_aspect);
        const QPointF new_center_metric =
            UvToMetric(origin_rect.center(), metric_aspect) + delta_metric;
        const QPointF new_center_uv = MetricToUv(new_center_metric, metric_aspect);
        new_rect = ClampCropRectForRotation(
            MakeRectFromCenterSize(new_center_uv, static_cast<float>(origin_rect.width()),
                                   static_cast<float>(origin_rect.height())),
            drag_rotation_degrees, metric_aspect);
      } else if (crop_mode == CropDragMode::ResizeEdge) {
        const QPointF center_metric = UvToMetric(origin_rect.center(), metric_aspect);
        const QPointF cursor_metric = UvToMetric(uv, metric_aspect);
        const QPointF local =
            InverseRotateVector(cursor_metric - center_metric, drag_rotation_degrees);

        const float min_width_metric  = kCropMinSize * metric_aspect;
        const float min_height_metric = kCropMinSize;
        float       left = -std::max((kCropMinSize * metric_aspect) * 0.5f,
                                     static_cast<float>(origin_rect.width()) * metric_aspect * 0.5f);
        float right = std::max((kCropMinSize * metric_aspect) * 0.5f,
                               static_cast<float>(origin_rect.width()) * metric_aspect * 0.5f);
        float top = -std::max(kCropMinSize * 0.5f, static_cast<float>(origin_rect.height()) * 0.5f);
        float bottom = std::max(kCropMinSize * 0.5f, static_cast<float>(origin_rect.height()) * 0.5f);

        float center_local_x = 0.0f;
        float center_local_y = 0.0f;
        switch (crop_edge) {
          case CropEdge::Right:
            right          = std::max(left + min_width_metric, static_cast<float>(local.x()));
            center_local_x = (left + right) * 0.5f;
            break;
          case CropEdge::Left:
            left           = std::min(right - min_width_metric, static_cast<float>(local.x()));
            center_local_x = (left + right) * 0.5f;
            break;
          case CropEdge::Top:
            top            = std::min(bottom - min_height_metric, static_cast<float>(local.y()));
            center_local_y = (top + bottom) * 0.5f;
            break;
          case CropEdge::Bottom:
            bottom         = std::max(top + min_height_metric, static_cast<float>(local.y()));
            center_local_y = (top + bottom) * 0.5f;
            break;
          default:
            break;
        }

        const float   new_hw = std::max((kCropMinSize * metric_aspect) * 0.5f, (right - left) * 0.5f);
        const float   new_hh = std::max(kCropMinSize * 0.5f, (bottom - top) * 0.5f);
        const QPointF center_shift_local(center_local_x, center_local_y);
        const QPointF new_center_metric =
            center_metric + RotateVector(center_shift_local, drag_rotation_degrees);
        const QPointF new_center_uv = MetricToUv(new_center_metric, metric_aspect);
        const float   new_width_uv = std::max(kCropMinSize, (new_hw * 2.0f) / metric_aspect);
        const float   new_height_uv = std::max(kCropMinSize, new_hh * 2.0f);
        new_rect =
            ClampCropRectForRotation(MakeRectFromCenterSize(new_center_uv, new_width_uv, new_height_uv),
                                     drag_rotation_degrees, metric_aspect);
      } else if (crop_mode == CropDragMode::RotateCorner) {
        const float drag_dx =
            static_cast<float>(event->position().x() - anchor_widget_pos.x());
        const int corner_sign = (crop_corner == CropCorner::TopRight || crop_corner == CropCorner::BottomRight)
                                    ? 1
                                    : -1;
        new_rotation_degrees = NormalizeAngleDegrees(
            drag_rotation_degrees + (static_cast<float>(corner_sign) * drag_dx * kCropRotateDegreesPerPixel));
        new_rect = ClampCropRectForRotation(origin_rect, new_rotation_degrees, metric_aspect);
        rotation_changed = true;
      }

      {
        std::lock_guard<std::mutex> view_lock(view_state_mutex_);
        crop_overlay_metric_aspect_ = metric_aspect;
        crop_overlay_rect_          = new_rect;
        if (rotation_changed) {
          crop_overlay_rotation_degrees_ = new_rotation_degrees;
        }
      }
      emit CropOverlayRectChanged(static_cast<float>(new_rect.x()), static_cast<float>(new_rect.y()),
                                  static_cast<float>(new_rect.width()),
                                  static_cast<float>(new_rect.height()), false);
      if (rotation_changed) {
        emit CropOverlayRotationChanged(new_rotation_degrees, false);
      }
      update();
      event->accept();
      return;
    }
  }

  if (dragging_) {
    const QPoint total_delta = event->pos() - drag_start_mouse_pos_;
    if (!dragged_since_press_ &&
        total_delta.manhattanLength() >= kClickDragThresholdPixels) {
      dragged_since_press_      = true;
      click_zoom_toggle_active_ = false;
    }
    const QPoint delta = event->pos() - last_mouse_pos_;
    last_mouse_pos_    = event->pos();

    if (!dragged_since_press_) {
      event->accept();
      return;
    }

    const float dpr = devicePixelRatioF();
    const float vw  = std::max(1.0f, float(width()) * dpr);
    const float vh  = std::max(1.0f, float(height()) * dpr);

    // Convert pixel delta to normalized device coordinates
    QVector2D ndc_delta(2.0f * float(delta.x()) / vw, -2.0f * float(delta.y()) / vh);
    float     current_zoom = kMinInteractiveZoom;
    QVector2D current_pan(0.0f, 0.0f);
    {
      std::lock_guard<std::mutex> view_lock(view_state_mutex_);
      current_zoom = view_zoom_;
      current_pan  = view_pan_;
    }
    ApplyViewTransform(current_zoom, current_pan + ndc_delta, false);
    event->accept();
    return;
  }
  QOpenGLWidget::mouseMoveEvent(event);
}

void QtEditViewer::mouseReleaseEvent(QMouseEvent* event) {
  if (event->button() == Qt::LeftButton) {
    CropDragMode crop_mode = CropDragMode::None;
    QRectF       crop_rect{};
    float        crop_rotation_degrees = 0.0f;
    {
      std::lock_guard<std::mutex> view_lock(view_state_mutex_);
      crop_mode                   = crop_drag_mode_;
      crop_drag_mode_             = CropDragMode::None;
      crop_drag_corner_           = CropCorner::None;
      crop_drag_edge_             = CropEdge::None;
      crop_drag_rotation_degrees_ = 0.0f;
      crop_drag_fixed_corner_uv_  = QPointF();
      crop_drag_anchor_widget_pos_ = QPointF();
      crop_rect                   = crop_overlay_rect_;
      crop_rotation_degrees       = crop_overlay_rotation_degrees_;
    }
    if (crop_mode != CropDragMode::None) {
      unsetCursor();
      emit CropOverlayRectChanged(static_cast<float>(crop_rect.x()),
                                  static_cast<float>(crop_rect.y()),
                                  static_cast<float>(crop_rect.width()),
                                  static_cast<float>(crop_rect.height()), true);
      if (crop_mode == CropDragMode::RotateCorner) {
        emit CropOverlayRotationChanged(crop_rotation_degrees, true);
      }
      event->accept();
      return;
    }
  }

  if (dragging_ && (event->button() == Qt::LeftButton || event->button() == Qt::MiddleButton)) {
    const bool left_click_without_drag =
        event->button() == Qt::LeftButton && !dragged_since_press_;
    dragging_            = false;
    dragged_since_press_ = false;
    unsetCursor();
    if (left_click_without_drag) {
      bool crop_tool_enabled = false;
      bool crop_overlay_visible = false;
      {
        std::lock_guard<std::mutex> view_lock(view_state_mutex_);
        crop_tool_enabled    = crop_tool_enabled_;
        crop_overlay_visible = crop_overlay_visible_;
      }
      if (suppress_next_click_release_toggle_) {
        suppress_next_click_release_toggle_ = false;
      } else if (!(crop_tool_enabled && crop_overlay_visible)) {
        pending_click_toggle_pos_ = event->position();
        pending_click_toggle_     = true;
        if (click_toggle_timer_) {
          click_toggle_timer_->start(QApplication::doubleClickInterval());
        }
      }
    }
    event->accept();
    return;
  }
  QOpenGLWidget::mouseReleaseEvent(event);
}

void QtEditViewer::mouseDoubleClickEvent(QMouseEvent* event) {
  if (event->button() == Qt::LeftButton) {
    bool   reset_crop = false;
    QRectF crop_rect;
    float  aspect = 1.0f;
    {
      std::lock_guard<std::mutex> view_lock(view_state_mutex_);
      aspect = crop_overlay_metric_aspect_;
      if (crop_overlay_visible_) {
        crop_overlay_rect_ =
            ClampCropRectForRotation(QRectF(0.0, 0.0, 1.0, 1.0), crop_overlay_rotation_degrees_, aspect);
        crop_rect  = crop_overlay_rect_;
        reset_crop = true;
      }
    }
    if (reset_crop) {
      emit CropOverlayRectChanged(static_cast<float>(crop_rect.x()),
                                  static_cast<float>(crop_rect.y()),
                                  static_cast<float>(crop_rect.width()),
                                  static_cast<float>(crop_rect.height()), true);
      update();
      event->accept();
      return;
    }
    pending_click_toggle_ = false;
    if (click_toggle_timer_ && click_toggle_timer_->isActive()) {
      click_toggle_timer_->stop();
    }
    suppress_next_click_release_toggle_ = true;
    ToggleDoubleClickZoomAt(event->position());
    event->accept();
    return;
  }
  QOpenGLWidget::mouseDoubleClickEvent(event);
}
};  // namespace puerhlab
