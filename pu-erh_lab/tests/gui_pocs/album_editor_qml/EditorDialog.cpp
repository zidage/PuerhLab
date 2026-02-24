#include "EditorDialog.h"

#include <QAbstractItemView>
#include <QByteArray>
#include <QColor>
#include <QComboBox>
#include <QCoreApplication>
#include <QDateTime>
#include <QDialog>
#include <QFontDatabase>
#include <QFrame>
#include <QGridLayout>
#include <QHBoxLayout>
#include <QLabel>
#include <QListWidget>
#include <QMessageBox>
#include <QMouseEvent>
#include <QOpenGLContext>
#include <QOpenGLExtraFunctions>
#include <QOpenGLShaderProgram>
#include <QOpenGLWidget>
#include <QPainter>
#include <QPainterPath>
#include <QPushButton>
#include <QScrollArea>
#include <QSlider>
#include <QStackedWidget>
#include <QSplitter>
#include <QStyle>
#include <QSurfaceFormat>
#include <QTimer>
#include <QVBoxLayout>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <deque>
#include <filesystem>
#include <format>
#include <functional>
#include <future>
#include <json.hpp>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "app/render_service.hpp"
#include "edit/history/edit_transaction.hpp"
#include "edit/history/version.hpp"
#include "edit/pipeline/pipeline_cpu.hpp"
#include "image/image.hpp"
#include "io/image/image_loader.hpp"
#include "renderer/pipeline_task.hpp"
#include "ui/edit_viewer/edit_viewer.hpp"

namespace puerhlab::demo {
namespace {

using namespace std::chrono_literals;

auto ListCubeLutsInDir(const std::filesystem::path& dir) -> std::vector<std::filesystem::path> {
  std::vector<std::filesystem::path> files;
  std::error_code                    ec;
  if (!std::filesystem::exists(dir, ec) || ec) {
    return files;
  }

  for (const auto& entry : std::filesystem::directory_iterator(dir, ec)) {
    if (ec) {
      break;
    }
    if (!entry.is_regular_file()) {
      continue;
    }
    std::wstring ext = entry.path().extension().wstring();
    std::transform(ext.begin(), ext.end(), ext.begin(), ::towlower);
    if (ext == L".cube") {
      files.push_back(entry.path());
    }
  }

  std::sort(files.begin(), files.end(),
            [](const std::filesystem::path& a, const std::filesystem::path& b) {
              return a.filename().wstring() < b.filename().wstring();
            });
  return files;
}

constexpr float                kBlackSliderFromGlobalScale      = 1000.0f;
constexpr float                kWhiteSliderFromGlobalScale      = 300.0f;
constexpr float                kShadowsSliderFromGlobalScale    = 80.0f;
constexpr float                kHighlightsSliderFromGlobalScale = 50.0f;
constexpr float                kCurveEpsilon                    = 1e-6f;
constexpr float                kCurveMinPointSpacing            = 1e-3f;
constexpr int                  kCurveMaxControlPoints           = 12;
constexpr std::array<float, 8> kHlsCandidateHues                = {0.0f,   45.0f,  90.0f,  135.0f,
                                                                   180.0f, 225.0f, 270.0f, 315.0f};
constexpr float                kHlsFixedTargetLightness         = 0.5f;
constexpr float                kHlsFixedTargetSaturation        = 0.5f;
constexpr float                kHlsDefaultHueRange              = 15.0f;
constexpr float                kHlsFixedLightnessRange          = 1.0f;
constexpr float                kHlsFixedSaturationRange         = 1.0f;
constexpr float                kHlsMaxHueShiftDegrees           = 15.0f;
constexpr float                kHlsAdjUiMin                     = -100.0f;
constexpr float                kHlsAdjUiMax                     = 100.0f;
constexpr float                kHlsAdjUiToParamScale            = 1000.0f;
constexpr int                  kColorTempCctMin                 = 2000;
constexpr int                  kColorTempCctMax                 = 15000;
constexpr int                  kColorTempTintMin                = -150;
constexpr int                  kColorTempTintMax                = 150;
constexpr float                kColorTempRequestEpsilon         = 1e-3f;
constexpr int                  kColorTempSliderUiMin            = 0;
constexpr int                  kColorTempSliderUiMax            = 4096;
constexpr int                  kColorTempSliderUiMid            = 2048;
constexpr float                kColorTempPivotCct               = 6000.0f;
constexpr float                kRotationSliderScale             = 100.0f;
constexpr float                kCropRectSliderScale             = 1000.0f;
constexpr float                kCropRectMinSize                 = 1e-4f;

using HlsProfileArray = std::array<float, kHlsCandidateHues.size()>;

static_assert(kColorTempSliderUiMin < kColorTempSliderUiMid,
              "Color temp slider UI midpoint must be inside range.");
static_assert(kColorTempSliderUiMid < kColorTempSliderUiMax,
              "Color temp slider UI midpoint must be inside range.");
static_assert(kColorTempCctMin < static_cast<int>(kColorTempPivotCct),
              "Color temp pivot must be inside Kelvin range.");
static_assert(static_cast<int>(kColorTempPivotCct) < kColorTempCctMax,
              "Color temp pivot must be inside Kelvin range.");

auto ColorTempSliderPosToCct(int pos) -> float {
  const float min_cct     = static_cast<float>(kColorTempCctMin);
  const float max_cct     = static_cast<float>(kColorTempCctMax);
  const int   clamped_pos = std::clamp(pos, kColorTempSliderUiMin, kColorTempSliderUiMax);

  float cct = min_cct;
  if (clamped_pos <= kColorTempSliderUiMid) {
    const float t = static_cast<float>(clamped_pos - kColorTempSliderUiMin) /
                    static_cast<float>(kColorTempSliderUiMid - kColorTempSliderUiMin);
    cct = min_cct + t * (kColorTempPivotCct - min_cct);
  } else {
    const float t = static_cast<float>(clamped_pos - kColorTempSliderUiMid) /
                    static_cast<float>(kColorTempSliderUiMax - kColorTempSliderUiMid);
    cct = kColorTempPivotCct + t * (max_cct - kColorTempPivotCct);
  }

  return std::clamp(cct, min_cct, max_cct);
}

auto ColorTempCctToSliderPos(float cct) -> int {
  const float min_cct     = static_cast<float>(kColorTempCctMin);
  const float max_cct     = static_cast<float>(kColorTempCctMax);
  const float clamped_cct = std::clamp(cct, min_cct, max_cct);

  float pos = static_cast<float>(kColorTempSliderUiMin);
  if (clamped_cct <= kColorTempPivotCct) {
    const float t = (clamped_cct - min_cct) / (kColorTempPivotCct - min_cct);
    pos           = static_cast<float>(kColorTempSliderUiMin) +
          t * static_cast<float>(kColorTempSliderUiMid - kColorTempSliderUiMin);
  } else {
    const float t = (clamped_cct - kColorTempPivotCct) / (max_cct - kColorTempPivotCct);
    pos           = static_cast<float>(kColorTempSliderUiMid) +
          t * static_cast<float>(kColorTempSliderUiMax - kColorTempSliderUiMid);
  }

  return std::clamp(static_cast<int>(std::lround(pos)), kColorTempSliderUiMin,
                    kColorTempSliderUiMax);
}

auto MakeHlsFilledArray(float value) -> HlsProfileArray {
  HlsProfileArray out{};
  out.fill(value);
  return out;
}

auto WrapHueDegrees(float hue) -> float {
  hue = std::fmod(hue, 360.0f);
  if (hue < 0.0f) {
    hue += 360.0f;
  }
  return hue;
}

auto HueDistanceDegrees(float a, float b) -> float {
  const float diff = std::abs(WrapHueDegrees(a) - WrapHueDegrees(b));
  return std::min(diff, 360.0f - diff);
}

auto ClosestHlsCandidateHueIndex(float hue) -> int {
  int   best_idx  = 0;
  float best_dist = HueDistanceDegrees(hue, kHlsCandidateHues.front());
  for (int i = 1; i < static_cast<int>(kHlsCandidateHues.size()); ++i) {
    const float dist = HueDistanceDegrees(hue, kHlsCandidateHues[i]);
    if (dist < best_dist) {
      best_dist = dist;
      best_idx  = i;
    }
  }
  return best_idx;
}

auto HlsCandidateColor(float hue_degrees) -> QColor {
  const float wrapped = WrapHueDegrees(hue_degrees);
  return QColor::fromHslF(wrapped / 360.0f, 1.0f, 0.5f);
}

auto Clamp01(float v) -> float { return std::clamp(v, 0.0f, 1.0f); }

auto ClampCropRect(float x, float y, float w, float h) -> std::array<float, 4> {
  w = std::clamp(w, kCropRectMinSize, 1.0f);
  h = std::clamp(h, kCropRectMinSize, 1.0f);
  x = std::clamp(x, 0.0f, 1.0f - w);
  y = std::clamp(y, 0.0f, 1.0f - h);
  return {x, y, w, h};
}

auto DefaultCurveControlPoints() -> std::vector<QPointF> {
  return {QPointF(0.0, 0.0), QPointF(0.25, 0.25), QPointF(0.75, 0.75), QPointF(1.0, 1.0)};
}

auto NormalizeCurveControlPoints(const std::vector<QPointF>& in) -> std::vector<QPointF> {
  std::vector<QPointF> points;
  points.reserve(in.size());
  for (const auto& p : in) {
    if (!std::isfinite(p.x()) || !std::isfinite(p.y())) {
      continue;
    }
    points.emplace_back(Clamp01(static_cast<float>(p.x())), Clamp01(static_cast<float>(p.y())));
  }

  if (points.empty()) {
    return DefaultCurveControlPoints();
  }

  std::sort(points.begin(), points.end(), [](const QPointF& a, const QPointF& b) {
    if (std::abs(a.x() - b.x()) <= kCurveEpsilon) {
      return a.y() < b.y();
    }
    return a.x() < b.x();
  });

  std::vector<QPointF> deduped;
  deduped.reserve(points.size());
  for (const auto& p : points) {
    if (deduped.empty() || std::abs(p.x() - deduped.back().x()) > kCurveEpsilon) {
      deduped.push_back(p);
    } else {
      deduped.back().setY(p.y());
    }
  }

  if (deduped.front().x() > kCurveEpsilon) {
    deduped.insert(deduped.begin(), QPointF(0.0, Clamp01(static_cast<float>(deduped.front().y()))));
  }
  if (deduped.back().x() < (1.0 - kCurveEpsilon)) {
    deduped.push_back(QPointF(1.0, Clamp01(static_cast<float>(deduped.back().y()))));
  }

  deduped.front().setX(0.0);
  deduped.front().setY(Clamp01(static_cast<float>(deduped.front().y())));
  deduped.back().setX(1.0);
  deduped.back().setY(Clamp01(static_cast<float>(deduped.back().y())));

  std::vector<QPointF> normalized;
  normalized.reserve(deduped.size());
  normalized.push_back(deduped.front());
  for (size_t i = 1; i + 1 < deduped.size(); ++i) {
    const auto& p = deduped[i];
    if (p.x() <= normalized.back().x() + kCurveMinPointSpacing) {
      continue;
    }
    if (p.x() >= 1.0 - kCurveMinPointSpacing) {
      continue;
    }
    normalized.push_back(p);
    if (static_cast<int>(normalized.size()) >= (kCurveMaxControlPoints - 1)) {
      break;
    }
  }
  normalized.push_back(deduped.back());

  if (normalized.size() < 2) {
    return DefaultCurveControlPoints();
  }

  return normalized;
}

auto CurveControlPointsEqual(const std::vector<QPointF>& a, const std::vector<QPointF>& b,
                             float eps = 1e-4f) -> bool {
  if (a.size() != b.size()) {
    return false;
  }
  for (size_t i = 0; i < a.size(); ++i) {
    if (std::abs(a[i].x() - b[i].x()) > eps || std::abs(a[i].y() - b[i].y()) > eps) {
      return false;
    }
  }
  return true;
}

struct CurveHermiteCache {
  std::vector<float> h_;
  std::vector<float> m_;
};

auto BuildCurveHermiteCache(const std::vector<QPointF>& points) -> CurveHermiteCache {
  CurveHermiteCache cache;
  const size_t      n = points.size();
  if (n == 0) {
    return cache;
  }

  cache.m_.assign(n, 0.0f);
  if (n == 1) {
    return cache;
  }

  cache.h_.assign(n - 1, 0.0f);
  std::vector<float> delta(n - 1, 0.0f);
  for (size_t i = 0; i + 1 < n; ++i) {
    const float dx = static_cast<float>(points[i + 1].x() - points[i].x());
    cache.h_[i]    = dx;
    if (std::abs(dx) > kCurveEpsilon) {
      delta[i] = static_cast<float>(points[i + 1].y() - points[i].y()) / dx;
    }
  }

  cache.m_[0]     = delta[0];
  cache.m_[n - 1] = delta[n - 2];
  for (size_t i = 1; i + 1 < n; ++i) {
    if (delta[i - 1] * delta[i] <= 0.0f) {
      cache.m_[i] = 0.0f;
      continue;
    }
    const float w1    = 2.0f * cache.h_[i] + cache.h_[i - 1];
    const float w2    = cache.h_[i] + 2.0f * cache.h_[i - 1];
    const float denom = (w1 / delta[i - 1]) + (w2 / delta[i]);
    cache.m_[i] = ((w1 + w2) > 0.0f && std::abs(denom) > kCurveEpsilon) ? (w1 + w2) / denom : 0.0f;
  }

  return cache;
}

auto EvaluateCurveHermite(float x, const std::vector<QPointF>& points,
                          const CurveHermiteCache& cache) -> float {
  const size_t n = points.size();
  if (n == 0) {
    return Clamp01(x);
  }
  if (n == 1) {
    return Clamp01(static_cast<float>(points.front().y()));
  }
  if (x <= points.front().x()) {
    return Clamp01(static_cast<float>(points.front().y()));
  }
  if (x >= points.back().x()) {
    return Clamp01(static_cast<float>(points.back().y()));
  }

  int idx = static_cast<int>(n) - 2;
  for (int i = 0; i < static_cast<int>(n) - 1; ++i) {
    if (x < points[i + 1].x()) {
      idx = i;
      break;
    }
  }

  if (idx < 0 || static_cast<size_t>(idx) >= cache.h_.size() ||
      static_cast<size_t>(idx + 1) >= cache.m_.size()) {
    return Clamp01(x);
  }

  const float dx = cache.h_[idx];
  if (std::abs(dx) <= kCurveEpsilon) {
    return Clamp01(static_cast<float>(points[idx].y()));
  }

  const float t   = (x - static_cast<float>(points[idx].x())) / dx;
  const float h00 = 2.0f * t * t * t - 3.0f * t * t + 1.0f;
  const float h10 = t * t * t - 2.0f * t * t + t;
  const float h01 = -2.0f * t * t * t + 3.0f * t * t;
  const float h11 = t * t * t - t * t;

  const float y   = h00 * static_cast<float>(points[idx].y()) + h10 * dx * cache.m_[idx] +
                  h01 * static_cast<float>(points[idx + 1].y()) + h11 * dx * cache.m_[idx + 1];
  return Clamp01(y);
}

auto CurveControlPointsToParams(const std::vector<QPointF>& points) -> nlohmann::json {
  const auto     normalized = NormalizeCurveControlPoints(points);
  nlohmann::json pts        = nlohmann::json::array();
  for (const auto& p : normalized) {
    pts.push_back({{"x", static_cast<float>(p.x())}, {"y", static_cast<float>(p.y())}});
  }
  return {{"curve", {{"size", normalized.size()}, {"points", std::move(pts)}}}};
}

auto ParseCurveControlPointsFromParams(const nlohmann::json& params)
    -> std::optional<std::vector<QPointF>> {
  if (!params.is_object()) {
    return std::nullopt;
  }

  const nlohmann::json* curve_json = &params;
  if (params.contains("curve")) {
    curve_json = &params["curve"];
  }

  if (!curve_json->is_object()) {
    return std::nullopt;
  }

  std::vector<QPointF> points;

  if (curve_json->contains("points") && (*curve_json)["points"].is_array()) {
    for (const auto& p : (*curve_json)["points"]) {
      if (!p.is_object() || !p.contains("x") || !p.contains("y")) {
        continue;
      }
      try {
        points.emplace_back(p["x"].get<float>(), p["y"].get<float>());
      } catch (...) {
      }
    }
  }

  if (points.empty() && curve_json->contains("size")) {
    size_t point_count = 0;
    try {
      point_count = (*curve_json)["size"].get<size_t>();
    } catch (...) {
      point_count = 0;
    }
    for (size_t i = 0; i < point_count; ++i) {
      const std::string key = std::format("pts{}", i);
      if (!curve_json->contains(key)) {
        continue;
      }
      const auto& p = (*curve_json)[key];
      if (!p.is_object() || !p.contains("x") || !p.contains("y")) {
        continue;
      }
      try {
        points.emplace_back(p["x"].get<float>(), p["y"].get<float>());
      } catch (...) {
      }
    }
  }

  if (points.empty()) {
    return std::nullopt;
  }
  return NormalizeCurveControlPoints(points);
}

class ToneCurveWidget final : public QWidget {
 public:
  using CurveCallback = std::function<void(const std::vector<QPointF>&)>;

  explicit ToneCurveWidget(QWidget* parent = nullptr) : QWidget(parent) {
    setMinimumSize(270, 220);
    setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
    setMouseTracking(true);
    setCursor(Qt::CrossCursor);
    points_ = DefaultCurveControlPoints();
  }

  void SetControlPoints(const std::vector<QPointF>& points) {
    points_     = NormalizeCurveControlPoints(points);
    active_idx_ = -1;
    dragging_   = false;
    update();
  }

  auto GetControlPoints() const -> const std::vector<QPointF>& { return points_; }

  void SetCurveChangedCallback(CurveCallback cb) { on_curve_changed_ = std::move(cb); }

  void SetCurveReleasedCallback(CurveCallback cb) { on_curve_released_ = std::move(cb); }

 protected:
  void paintEvent(QPaintEvent*) override {
    QPainter painter(this);
    painter.setRenderHint(QPainter::Antialiasing, true);

    painter.fillRect(rect(), QColor(0x1A, 0x1A, 0x1A));
    painter.setPen(QPen(QColor(0x2A, 0x2A, 0x2A), 1.0));
    painter.drawRoundedRect(rect().adjusted(0, 0, -1, -1), 10.0, 10.0);

    const QRectF plot = PlotRect();
    painter.fillRect(plot, QColor(0x12, 0x12, 0x12));

    painter.setPen(QPen(QColor(0x2A, 0x2A, 0x2A), 1.0));
    for (int i = 1; i < 4; ++i) {
      const qreal t  = static_cast<qreal>(i) / 4.0;
      const qreal gx = plot.left() + t * plot.width();
      const qreal gy = plot.top() + t * plot.height();
      painter.drawLine(QPointF(gx, plot.top()), QPointF(gx, plot.bottom()));
      painter.drawLine(QPointF(plot.left(), gy), QPointF(plot.right(), gy));
    }

    painter.setPen(QPen(QColor(0x8C, 0x8C, 0x8C), 1.0, Qt::DashLine));
    painter.drawLine(QPointF(plot.left(), plot.bottom()), QPointF(plot.right(), plot.top()));

    const auto    cache = BuildCurveHermiteCache(points_);
    QPainterPath  curve_path;
    constexpr int kSamples = 240;
    for (int i = 0; i <= kSamples; ++i) {
      const float   x = static_cast<float>(i) / static_cast<float>(kSamples);
      const float   y = EvaluateCurveHermite(x, points_, cache);
      const QPointF p = ToWidgetPoint(QPointF(x, y));
      if (i == 0) {
        curve_path.moveTo(p);
      } else {
        curve_path.lineTo(p);
      }
    }

    painter.setPen(QPen(QColor(0xFC, 0xC7, 0x04), 2.0));
    painter.drawPath(curve_path);

    for (size_t i = 0; i < points_.size(); ++i) {
      const QPointF p       = ToWidgetPoint(points_[i]);
      const bool    active  = static_cast<int>(i) == active_idx_;
      const bool    pinned  = (i == 0 || i + 1 == points_.size());
      const QColor  fill    = active ? QColor(0xFC, 0xC7, 0x04) : QColor(0xE6, 0xE6, 0xE6);
      const QColor  outline = pinned ? QColor(0xFC, 0xC7, 0x04) : QColor(0x2A, 0x2A, 0x2A);

      painter.setPen(QPen(outline, 1.5));
      painter.setBrush(fill);
      painter.drawEllipse(p, active ? 5.5 : 4.5, active ? 5.5 : 4.5);
    }

    painter.setPen(QColor(0x8C, 0x8C, 0x8C));
    painter.drawText(QRectF(plot.left() - 2.0, plot.bottom() + 4.0, 32.0, 14.0), "0");
    painter.drawText(QRectF(plot.right() - 10.0, plot.bottom() + 4.0, 20.0, 14.0), "1");
    painter.drawText(QRectF(plot.left() - 16.0, plot.top() - 2.0, 14.0, 14.0), "1");
    painter.drawText(QRectF(plot.left() - 16.0, plot.bottom() - 10.0, 14.0, 14.0), "0");
  }

  void mousePressEvent(QMouseEvent* event) override {
    if (!event) {
      return;
    }

    const QPointF pos = event->position();
    if (event->button() == Qt::RightButton) {
      const int hit_idx = HitTestPoint(pos);
      if (hit_idx > 0 && hit_idx + 1 < static_cast<int>(points_.size())) {
        points_.erase(points_.begin() + hit_idx);
        points_     = NormalizeCurveControlPoints(points_);
        active_idx_ = -1;
        dragging_   = false;
        NotifyCurveChanged();
        NotifyCurveReleased();
        update();
      }
      return;
    }

    if (event->button() != Qt::LeftButton) {
      return;
    }

    const int hit_idx = HitTestPoint(pos);
    if (hit_idx >= 0) {
      active_idx_ = hit_idx;
      dragging_   = true;
      update();
      return;
    }

    if (!PlotRect().contains(pos) || static_cast<int>(points_.size()) >= kCurveMaxControlPoints) {
      return;
    }

    const QPointF normalized = ToNormalizedPoint(pos);
    if (normalized.x() <= kCurveMinPointSpacing ||
        normalized.x() >= (1.0 - kCurveMinPointSpacing)) {
      return;
    }

    points_.push_back(normalized);
    points_     = NormalizeCurveControlPoints(points_);
    active_idx_ = FindClosestPointIndex(normalized);
    dragging_   = true;
    NotifyCurveChanged();
    update();
  }

  void mouseMoveEvent(QMouseEvent* event) override {
    if (!event || !dragging_ || active_idx_ < 0 ||
        active_idx_ >= static_cast<int>(points_.size())) {
      return;
    }

    const int     last_idx   = static_cast<int>(points_.size()) - 1;
    const QPointF normalized = ToNormalizedPoint(event->position());
    if (active_idx_ == 0) {
      points_[0] = QPointF(0.0, Clamp01(static_cast<float>(normalized.y())));
      NotifyCurveChanged();
      update();
      return;
    }
    if (active_idx_ == last_idx) {
      points_[last_idx] = QPointF(1.0, Clamp01(static_cast<float>(normalized.y())));
      NotifyCurveChanged();
      update();
      return;
    }

    const float min_x    = static_cast<float>(points_[active_idx_ - 1].x()) + kCurveMinPointSpacing;
    const float max_x    = static_cast<float>(points_[active_idx_ + 1].x()) - kCurveMinPointSpacing;

    const float x        = std::clamp(static_cast<float>(normalized.x()), min_x, max_x);
    const float y        = Clamp01(static_cast<float>(normalized.y()));
    points_[active_idx_] = QPointF(x, y);

    NotifyCurveChanged();
    update();
  }

  void mouseReleaseEvent(QMouseEvent* event) override {
    if (!event || event->button() != Qt::LeftButton) {
      return;
    }
    if (!dragging_) {
      return;
    }
    dragging_ = false;
    NotifyCurveReleased();
  }

 private:
  auto PlotRect() const -> QRectF {
    constexpr qreal kLeft   = 22.0;
    constexpr qreal kTop    = 14.0;
    constexpr qreal kRight  = 12.0;
    constexpr qreal kBottom = 24.0;
    return QRectF(kLeft, kTop, std::max(30.0, width() - kLeft - kRight),
                  std::max(30.0, height() - kTop - kBottom));
  }

  auto ToWidgetPoint(const QPointF& normalized) const -> QPointF {
    const QRectF plot = PlotRect();
    const qreal  x    = plot.left() + normalized.x() * plot.width();
    const qreal  y    = plot.bottom() - normalized.y() * plot.height();
    return QPointF(x, y);
  }

  auto ToNormalizedPoint(const QPointF& widget_point) const -> QPointF {
    const QRectF plot = PlotRect();
    const qreal  nx =
        std::clamp((widget_point.x() - plot.left()) / std::max(1.0, plot.width()), 0.0, 1.0);
    const qreal ny =
        std::clamp((plot.bottom() - widget_point.y()) / std::max(1.0, plot.height()), 0.0, 1.0);
    return QPointF(nx, ny);
  }

  auto HitTestPoint(const QPointF& widget_point) const -> int {
    constexpr qreal kHitRadiusSq = 10.0 * 10.0;
    for (int i = static_cast<int>(points_.size()) - 1; i >= 0; --i) {
      const QPointF p       = ToWidgetPoint(points_[i]);
      const qreal   dx      = p.x() - widget_point.x();
      const qreal   dy      = p.y() - widget_point.y();
      const qreal   dist_sq = dx * dx + dy * dy;
      if (dist_sq <= kHitRadiusSq) {
        return i;
      }
    }
    return -1;
  }

  auto FindClosestPointIndex(const QPointF& normalized) const -> int {
    if (points_.empty()) {
      return -1;
    }
    int   best_idx = 0;
    qreal best_dist =
        std::abs(points_[0].x() - normalized.x()) + std::abs(points_[0].y() - normalized.y());
    for (int i = 1; i < static_cast<int>(points_.size()); ++i) {
      const qreal dist =
          std::abs(points_[i].x() - normalized.x()) + std::abs(points_[i].y() - normalized.y());
      if (dist < best_dist) {
        best_dist = dist;
        best_idx  = i;
      }
    }
    return best_idx;
  }

  void NotifyCurveChanged() {
    if (on_curve_changed_) {
      on_curve_changed_(points_);
    }
  }

  void NotifyCurveReleased() {
    if (on_curve_released_) {
      on_curve_released_(points_);
    }
  }

  std::vector<QPointF> points_{};
  int                  active_idx_ = -1;
  bool                 dragging_   = false;
  CurveCallback        on_curve_changed_{};
  CurveCallback        on_curve_released_{};
};

class SpinnerWidget final : public QWidget {
 public:
  explicit SpinnerWidget(QWidget* parent = nullptr) : QWidget(parent) {
    setFixedSize(22, 22);
    setAttribute(Qt::WA_TransparentForMouseEvents);
    setAttribute(Qt::WA_TranslucentBackground);

    timer_ = new QTimer(this);
    timer_->setInterval(16);
    QObject::connect(timer_, &QTimer::timeout, this, [this]() {
      angle_deg_ = (angle_deg_ + 18) % 360;
      update();
    });
    hide();
  }

  void Start() {
    show();
    raise();
    if (!timer_->isActive()) {
      timer_->start();
    }
  }

  void Stop() {
    if (timer_->isActive()) {
      timer_->stop();
    }
    hide();
  }

 protected:
  void paintEvent(QPaintEvent*) override {
    QPainter painter(this);
    painter.setRenderHint(QPainter::Antialiasing, true);

    const QRectF r = QRectF(2.5, 2.5, width() - 5.0, height() - 5.0);

    // Subtle background ring.
    {
      QPen pen(QColor(0x3A, 0x3A, 0x3A, 180));
      pen.setWidthF(2.0);
      pen.setCapStyle(Qt::RoundCap);
      painter.setPen(pen);
      painter.drawArc(r, 0 * 16, 360 * 16);
    }

    // Foreground arc.
    {
      QPen pen(QColor(0xFC, 0xC7, 0x04, 230));
      pen.setWidthF(2.2);
      pen.setCapStyle(Qt::RoundCap);
      painter.setPen(pen);
      painter.drawArc(r, (90 - angle_deg_) * 16, 100 * 16);
    }
  }

 private:
  QTimer* timer_     = nullptr;
  int     angle_deg_ = 0;
};

class HistogramWidget final : public QOpenGLWidget, protected QOpenGLExtraFunctions {
 public:
  explicit HistogramWidget(QtEditViewer* source_viewer, QWidget* parent = nullptr)
      : QOpenGLWidget(parent), source_viewer_(source_viewer) {
    setMinimumHeight(126);
    setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
    setAutoFillBackground(false);
  }

  ~HistogramWidget() override {
    if (!context()) {
      return;
    }
    makeCurrent();
    CleanupGl();
    doneCurrent();
  }

  void SetSourceViewer(QtEditViewer* source_viewer) {
    source_viewer_ = source_viewer;
    update();
  }

 protected:
  void initializeGL() override {
    initializeOpenGLFunctions();
    glDisable(GL_DEPTH_TEST);
    glGenVertexArrays(1, &vao_);
    glBindVertexArray(vao_);
    glBindVertexArray(0);
    gl_ready_ = InitPrograms();
  }

  void paintGL() override {
    const float dpr = devicePixelRatioF();
    const int   vw  = std::max(1, static_cast<int>(std::lround(width() * dpr)));
    const int   vh  = std::max(1, static_cast<int>(std::lround(height() * dpr)));
    glViewport(0, 0, vw, vh);
    glClearColor(0.07f, 0.07f, 0.07f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    if (!gl_ready_ || !source_viewer_ || !source_viewer_->HasHistogramData()) {
      return;
    }

    if (context() && source_viewer_->context() &&
        !QOpenGLContext::areSharing(context(), source_viewer_->context())) {
      if (!warned_context_sharing_) {
        qWarning("HistogramWidget disabled: OpenGL contexts are not sharing resources.");
        warned_context_sharing_ = true;
      }
      return;
    }

    const GLuint hist_buffer = source_viewer_->GetHistogramBufferId();
    const int    bins        = source_viewer_->GetHistogramBinCount();
    if (hist_buffer == 0 || bins <= 1 || !glIsBuffer(hist_buffer)) {
      return;
    }

    glBindVertexArray(vao_);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, hist_buffer);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    auto draw_fill = [&](int channel, const QVector4D& color) {
      if (!fill_program_) {
        return;
      }
      fill_program_->bind();
      fill_program_->setUniformValue("uBins", bins);
      fill_program_->setUniformValue("uChannel", channel);
      fill_program_->setUniformValue("uColor", color);
      glDrawArrays(GL_TRIANGLE_STRIP, 0, bins * 2);
      fill_program_->release();
    };

    auto draw_line = [&](int channel, const QVector4D& color) {
      if (!line_program_) {
        return;
      }
      line_program_->bind();
      line_program_->setUniformValue("uBins", bins);
      line_program_->setUniformValue("uChannel", channel);
      line_program_->setUniformValue("uColor", color);
      glLineWidth(1.0f);
      glDrawArrays(GL_LINE_STRIP, 0, bins);
      line_program_->release();
    };

    draw_fill(0, QVector4D(1.0f, 0.20f, 0.20f, 0.30f));
    draw_fill(1, QVector4D(0.20f, 1.0f, 0.20f, 0.28f));
    draw_fill(2, QVector4D(0.20f, 0.45f, 1.0f, 0.28f));

    draw_line(0, QVector4D(1.0f, 0.45f, 0.45f, 0.24f));
    draw_line(1, QVector4D(0.45f, 1.0f, 0.45f, 0.22f));
    draw_line(2, QVector4D(0.45f, 0.68f, 1.0f, 0.22f));

    glDisable(GL_BLEND);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, 0);
    glBindVertexArray(0);
  }

 private:
  auto InitPrograms() -> bool {
    if (!context()) {
      return false;
    }

    const auto format = context()->format();
    const bool has_compute_compatible_ssbo =
        (format.majorVersion() > 4 || (format.majorVersion() == 4 && format.minorVersion() >= 3)) ||
        context()->hasExtension(QByteArrayLiteral("GL_ARB_shader_storage_buffer_object"));
    if (!has_compute_compatible_ssbo) {
      qWarning("HistogramWidget disabled: OpenGL SSBO support is not available.");
      return false;
    }

    static const char* kFillVertex = R"(
#version 430 core
layout(std430, binding = 0) readonly buffer HistogramBuffer {
  float hist[];
};
uniform int uBins;
uniform int uChannel;
void main() {
  int bin = gl_VertexID >> 1;
  int top = gl_VertexID & 1;
  float x = (uBins > 1) ? float(bin) / float(uBins - 1) : 0.0;
  float y = (top == 0) ? 0.0 : clamp(hist[uChannel * uBins + bin], 0.0, 1.0);
  gl_Position = vec4(x * 2.0 - 1.0, y * 2.0 - 1.0, 0.0, 1.0);
}
)";

    static const char* kLineVertex = R"(
#version 430 core
layout(std430, binding = 0) readonly buffer HistogramBuffer {
  float hist[];
};
uniform int uBins;
uniform int uChannel;
void main() {
  int bin = gl_VertexID;
  float x = (uBins > 1) ? float(bin) / float(uBins - 1) : 0.0;
  float y = clamp(hist[uChannel * uBins + bin], 0.0, 1.0);
  gl_Position = vec4(x * 2.0 - 1.0, y * 2.0 - 1.0, 0.0, 1.0);
}
)";

    static const char* kFragment   = R"(
#version 430 core
uniform vec4 uColor;
out vec4 FragColor;
void main() {
  FragColor = uColor;
}
)";

    fill_program_                  = new QOpenGLShaderProgram();
    if (!fill_program_->addShaderFromSourceCode(QOpenGLShader::Vertex, kFillVertex) ||
        !fill_program_->addShaderFromSourceCode(QOpenGLShader::Fragment, kFragment) ||
        !fill_program_->link()) {
      qWarning("HistogramWidget fill program failed: %s",
               fill_program_->log().toUtf8().constData());
      CleanupGl();
      return false;
    }

    line_program_ = new QOpenGLShaderProgram();
    if (!line_program_->addShaderFromSourceCode(QOpenGLShader::Vertex, kLineVertex) ||
        !line_program_->addShaderFromSourceCode(QOpenGLShader::Fragment, kFragment) ||
        !line_program_->link()) {
      qWarning("HistogramWidget line program failed: %s",
               line_program_->log().toUtf8().constData());
      CleanupGl();
      return false;
    }
    return true;
  }

  void CleanupGl() {
    if (fill_program_) {
      delete fill_program_;
      fill_program_ = nullptr;
    }
    if (line_program_) {
      delete line_program_;
      line_program_ = nullptr;
    }
    if (vao_) {
      glDeleteVertexArrays(1, &vao_);
      vao_ = 0;
    }
    gl_ready_ = false;
  }

  QtEditViewer*         source_viewer_          = nullptr;
  QOpenGLShaderProgram* fill_program_           = nullptr;
  QOpenGLShaderProgram* line_program_           = nullptr;
  GLuint                vao_                    = 0;
  bool                  gl_ready_               = false;
  bool                  warned_context_sharing_ = false;
};

class HistogramRulerWidget final : public QWidget {
 public:
  explicit HistogramRulerWidget(int bins, QWidget* parent = nullptr)
      : QWidget(parent), bins_(std::max(2, bins)) {
    setMinimumHeight(28);
    setMaximumHeight(36);
    setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
    setAttribute(Qt::WA_TransparentForMouseEvents);
  }

  void SetBins(int bins) {
    bins_ = std::max(2, bins);
    update();
  }

 protected:
  void paintEvent(QPaintEvent*) override {
    QPainter painter(this);
    painter.setRenderHint(QPainter::Antialiasing, true);

    QFont ruler_font = painter.font();
    ruler_font.setPixelSize(9);
    painter.setFont(ruler_font);

    const QRectF area(10.0, 6.0, std::max(10.0, width() - 20.0), std::max(10.0, height() - 12.0));
    const qreal  baseline_y = area.top() + 2.0;
    const qreal  tick_h     = 7.0;

    painter.setPen(QPen(QColor(0x4A, 0x4A, 0x4A), 1.0));
    painter.drawLine(QPointF(area.left(), baseline_y), QPointF(area.right(), baseline_y));

    constexpr std::array<double, 5> stops = {0.0, 0.25, 0.50, 0.75, 1.0};
    painter.setPen(QPen(QColor(0x6F, 0x6F, 0x6F), 1.0));
    for (const double t : stops) {
      const qreal x = area.left() + t * area.width();
      painter.drawLine(QPointF(x, baseline_y), QPointF(x, baseline_y + tick_h));
    }

    painter.setPen(QColor(0x9A, 0x9A, 0x9A));
    for (const double t : stops) {
      const qreal   x    = area.left() + t * area.width();
      const QString text = QString::number(t, 'f', 2);
      const QRectF  text_rect(x - 20.0, baseline_y + tick_h + 1.0, 40.0, 14.0);
      painter.drawText(text_rect, Qt::AlignHCenter | Qt::AlignTop, text);
    }
  }

 private:
  int bins_ = 256;
};

class HistoryLaneWidget final : public QWidget {
 public:
  HistoryLaneWidget(QColor dot, QColor line, bool draw_top, bool draw_bottom,
                    QWidget* parent = nullptr)
      : QWidget(parent),
        dot_(std::move(dot)),
        line_(std::move(line)),
        draw_top_(draw_top),
        draw_bottom_(draw_bottom) {
    setFixedWidth(18);
    setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Expanding);
    setAttribute(Qt::WA_TransparentForMouseEvents);
  }

  void SetConnectors(bool draw_top, bool draw_bottom) {
    draw_top_    = draw_top;
    draw_bottom_ = draw_bottom;
    update();
  }

 protected:
  void paintEvent(QPaintEvent*) override {
    QPainter p(this);
    p.setRenderHint(QPainter::Antialiasing, true);

    const int cx = width() / 2;
    const int cy = height() / 2;

    // Vertical lane.
    {
      QPen pen(line_);
      pen.setWidthF(2.0);
      pen.setCapStyle(Qt::RoundCap);
      p.setPen(pen);

      if (draw_top_) {
        p.drawLine(QPointF(cx, 2.0), QPointF(cx, cy - 6.0));
      }
      if (draw_bottom_) {
        p.drawLine(QPointF(cx, cy + 6.0), QPointF(cx, height() - 2.0));
      }
    }

    // Node.
    {
      p.setPen(Qt::NoPen);
      p.setBrush(dot_);
      p.drawEllipse(QPointF(cx, cy), 4.4, 4.4);
      p.setBrush(QColor(0x12, 0x12, 0x12));
      p.drawEllipse(QPointF(cx, cy), 2.0, 2.0);
    }
  }

 private:
  QColor dot_;
  QColor line_;
  bool   draw_top_    = false;
  bool   draw_bottom_ = false;
};

class HistoryCardWidget final : public QFrame {
 public:
  explicit HistoryCardWidget(QWidget* parent = nullptr) : QFrame(parent) {
    setObjectName("HistoryCard");
    setAttribute(Qt::WA_StyledBackground, true);
    setAttribute(Qt::WA_Hover, true);
    setProperty("selected", false);

    setStyleSheet(
        "QFrame#HistoryCard {"
        "  background: #1A1A1A;"
        "  border: none;"
        "  border-radius: 10px;"
        "}"
        "QFrame#HistoryCard:hover {"
        "  background: #202020;"
        "}"
        "QFrame#HistoryCard[selected=\"true\"] {"
        "  background: rgba(252, 199, 4, 0.20);"
        "  border: 2px solid #FCC704;"
        "}");
  }

  void SetSelected(bool selected) {
    if (property("selected").toBool() == selected) {
      return;
    }
    setProperty("selected", selected);
    style()->unpolish(this);
    style()->polish(this);
    update();
  }
};

static QLabel* MakePillLabel(const QString& text, const QString& fg, const QString& bg,
                             const QString& border, QWidget* parent) {
  auto* l = new QLabel(text, parent);
  l->setStyleSheet(QString("QLabel {"
                           "  color: %1;"
                           "  background: %2;"
                           "  border: 1px solid %3;"
                           "  border-radius: 10px;"
                           "  padding: 1px 7px;"
                           "  font-size: 11px;"
                           "}")
                       .arg(fg, bg, border));
  return l;
}

class EditorDialog final : public QDialog {
 public:
  enum class WorkingMode : int { Incremental = 0, Plain = 1 };

  EditorDialog(std::shared_ptr<ImagePoolService>       image_pool,
               std::shared_ptr<PipelineGuard>          pipeline_guard,
               std::shared_ptr<EditHistoryMgmtService> history_service,
               std::shared_ptr<EditHistoryGuard> history_guard, sl_element_id_t element_id,
               image_id_t image_id, QWidget* parent = nullptr)
      : QDialog(parent),
        image_pool_(std::move(image_pool)),
        pipeline_guard_(std::move(pipeline_guard)),
        history_service_(std::move(history_service)),
        history_guard_(std::move(history_guard)),
        element_id_(element_id),
        image_id_(image_id),
        scheduler_(RenderService::GetPreviewScheduler()) {
    if (!image_pool_ || !pipeline_guard_ || !pipeline_guard_->pipeline_ || !history_service_ ||
        !history_guard_ || !history_guard_->history_ || !scheduler_) {
      throw std::runtime_error("EditorDialog: missing services");
    }

    setModal(true);
    setSizeGripEnabled(true);
    setWindowFlag(Qt::WindowMinMaxButtonsHint, true);
    setWindowFlag(Qt::MSWindowsFixedSizeDialogHint, false);
    setWindowTitle(QString("Editor - element #%1").arg(static_cast<qulonglong>(element_id_)));
    setMinimumSize(1080, 680);
    resize(1500, 1000);

    auto* root = new QHBoxLayout(this);
    root->setContentsMargins(10, 10, 10, 10);
    root->setSpacing(0);

    auto* main_splitter = new QSplitter(Qt::Horizontal, this);
    main_splitter->setObjectName("EditorMainSplitter");
    main_splitter->setChildrenCollapsible(false);
    main_splitter->setHandleWidth(8);
    main_splitter->setStyleSheet(
        "QSplitter#EditorMainSplitter::handle {"
        "  background: #2A2A2A;"
        "}"
        "QSplitter#EditorMainSplitter::handle:hover {"
        "  background: #FCC704;"
        "}");

    viewer_ = new QtEditViewer(this);
    viewer_->setMinimumSize(560, 360);
    viewer_->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    viewer_->setStyleSheet(
        "QOpenGLWidget {"
        "  background: #121212;"
        "  border: none;"
        "  border-radius: 12px;"
        "}");

    // Viewer container allows a small overlay spinner during rendering.
    viewer_container_ = new QWidget(this);
    auto* viewer_grid = new QGridLayout(viewer_container_);
    viewer_grid->setContentsMargins(0, 0, 0, 0);
    viewer_grid->setSpacing(0);
    viewer_grid->addWidget(viewer_, 0, 0);

    viewer_zoom_label_ = new QLabel(viewer_container_);
    viewer_zoom_label_->setAttribute(Qt::WA_TransparentForMouseEvents, true);
    viewer_zoom_label_->setStyleSheet(
        "QLabel {"
        "  color: #E6E6E6;"
        "  background: rgba(18, 18, 18, 180);"
        "  border: 1px solid rgba(42, 42, 42, 220);"
        "  border-radius: 8px;"
        "  padding: 4px 8px;"
        "  font-size: 11px;"
        "  font-weight: 600;"
        "}");
    viewer_grid->addWidget(viewer_zoom_label_, 0, 0, Qt::AlignLeft | Qt::AlignTop);

    spinner_ = new SpinnerWidget(viewer_container_);
    viewer_grid->addWidget(spinner_, 0, 0, Qt::AlignRight | Qt::AlignBottom);
    viewer_grid->setRowStretch(0, 1);
    viewer_grid->setColumnStretch(0, 1);

    if (viewer_) {
      QObject::connect(viewer_, &QtEditViewer::ViewZoomChanged, this,
                       [this](float zoom) { UpdateViewerZoomLabel(zoom); });
      UpdateViewerZoomLabel(viewer_->GetViewZoom());
    }

    auto* controls_panel = new QWidget(this);
    controls_panel->setMinimumWidth(320);
    controls_panel->setMaximumWidth(900);
    controls_panel->setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Expanding);
    controls_panel->setObjectName("EditorControlsPanel");
    controls_panel->setAttribute(Qt::WA_StyledBackground, true);
    controls_panel->setStyleSheet(
        "#EditorControlsPanel {"
        "  background: #1A1A1A;"
        "  border: none;"
        "  border-radius: 14px;"
        "}"
        "#EditorControlsPanel QFrame#EditorSection {"
        "  background: #121212;"
        "  border: none;"
        "  border-radius: 12px;"
        "}"
        "#EditorControlsPanel QLabel#EditorSectionTitle {"
        "  color: #E6E6E6;"
        "  font-size: 13px;"
        "  font-weight: 620;"
        "}"
        "#EditorControlsPanel QLabel#EditorSectionSub {"
        "  color: #A3A3A3;"
        "  font-size: 11px;"
        "}");
    auto* controls_panel_layout = new QVBoxLayout(controls_panel);
    controls_panel_layout->setContentsMargins(16, 16, 16, 16);
    controls_panel_layout->setSpacing(12);

    const QString scroll_style =
        "QScrollArea { background: transparent; border: none; }"
        "QScrollBar:vertical {"
        "  background: #121212;"
        "  width: 10px;"
        "  margin: 2px;"
        "  border-radius: 5px;"
        "}"
        "QScrollBar::handle:vertical {"
        "  background: #FCC704;"
        "  border-radius: 5px;"
        "}"
        "QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0px; }"
        "QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background: transparent; "
        "}";

    tone_controls_scroll_ = new QScrollArea(controls_panel);
    tone_controls_scroll_->setFrameShape(QFrame::NoFrame);
    tone_controls_scroll_->setWidgetResizable(true);
    tone_controls_scroll_->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    tone_controls_scroll_->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    tone_controls_scroll_->setStyleSheet(scroll_style);

    tone_controls_        = new QWidget(tone_controls_scroll_);
    controls_             = tone_controls_;
    auto* controls_layout = new QVBoxLayout(tone_controls_);
    controls_layout->setContentsMargins(0, 0, 0, 0);
    controls_layout->setSpacing(12);
    tone_controls_scroll_->setWidget(tone_controls_);
    controls_scroll_ = tone_controls_scroll_;

    geometry_controls_scroll_ = new QScrollArea(controls_panel);
    geometry_controls_scroll_->setFrameShape(QFrame::NoFrame);
    geometry_controls_scroll_->setWidgetResizable(true);
    geometry_controls_scroll_->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    geometry_controls_scroll_->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    geometry_controls_scroll_->setStyleSheet(scroll_style);

    geometry_controls_ = new QWidget(geometry_controls_scroll_);
    auto* geometry_controls_layout = new QVBoxLayout(geometry_controls_);
    geometry_controls_layout->setContentsMargins(0, 0, 0, 0);
    geometry_controls_layout->setSpacing(12);
    geometry_controls_layout->addStretch();
    geometry_controls_scroll_->setWidget(geometry_controls_);

    control_panels_stack_ = new QStackedWidget(controls_panel);
    control_panels_stack_->addWidget(tone_controls_scroll_);
    control_panels_stack_->addWidget(geometry_controls_scroll_);
    control_panels_stack_->setCurrentIndex(0);

    auto* panel_switch_row = new QWidget(controls_panel);
    auto* panel_switch_layout = new QHBoxLayout(panel_switch_row);
    panel_switch_layout->setContentsMargins(0, 0, 0, 0);
    panel_switch_layout->setSpacing(8);

    tone_panel_btn_     = new QPushButton("Tone", panel_switch_row);
    geometry_panel_btn_ = new QPushButton("Geometry", panel_switch_row);
    tone_panel_btn_->setCheckable(true);
    geometry_panel_btn_->setCheckable(true);
    tone_panel_btn_->setCursor(Qt::PointingHandCursor);
    geometry_panel_btn_->setCursor(Qt::PointingHandCursor);
    tone_panel_btn_->setFixedHeight(30);
    geometry_panel_btn_->setFixedHeight(30);

    panel_switch_layout->addWidget(tone_panel_btn_, 1);
    panel_switch_layout->addWidget(geometry_panel_btn_, 1);

    QObject::connect(tone_panel_btn_, &QPushButton::clicked, this,
                     [this]() { SetActiveControlPanel(ControlPanelKind::Tone); });
    QObject::connect(geometry_panel_btn_, &QPushButton::clicked, this,
                     [this]() { SetActiveControlPanel(ControlPanelKind::Geometry); });
    RefreshPanelSwitchUi();

    auto* shared_versioning_root = new QWidget(this);
    shared_versioning_root->setObjectName("EditorVersioningPanel");
    shared_versioning_root->setMinimumWidth(220);
    shared_versioning_root->setMaximumWidth(600);
    shared_versioning_root->setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Expanding);
    shared_versioning_root->setAttribute(Qt::WA_StyledBackground, true);
    shared_versioning_root->setStyleSheet(
        "#EditorVersioningPanel {"
        "  background: #1A1A1A;"
        "  border: none;"
        "  border-radius: 14px;"
        "}"
        "#EditorVersioningPanel QFrame#EditorSection {"
        "  background: #121212;"
        "  border: none;"
        "  border-radius: 12px;"
        "}"
        "#EditorVersioningPanel QLabel#EditorSectionTitle {"
        "  color: #E6E6E6;"
        "  font-size: 13px;"
        "  font-weight: 620;"
        "}"
        "#EditorVersioningPanel QLabel#EditorSectionSub {"
        "  color: #A3A3A3;"
        "  font-size: 11px;"
        "}");
    auto* shared_versioning_outer_layout = new QVBoxLayout(shared_versioning_root);
    shared_versioning_outer_layout->setContentsMargins(0, 0, 0, 0);
    shared_versioning_outer_layout->setSpacing(0);

    auto* versioning_scroll = new QScrollArea(shared_versioning_root);
    versioning_scroll->setFrameShape(QFrame::NoFrame);
    versioning_scroll->setWidgetResizable(true);
    versioning_scroll->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    versioning_scroll->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    versioning_scroll->setStyleSheet(
        "QScrollArea { background: transparent; border: none; }"
        "QScrollBar:vertical {"
        "  background: #121212;"
        "  width: 10px;"
        "  margin: 2px;"
        "  border-radius: 5px;"
        "}"
        "QScrollBar::handle:vertical {"
        "  background: #FCC704;"
        "  border-radius: 5px;"
        "}"
        "QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0px; }"
        "QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background: transparent; }");

    auto* versioning_content = new QWidget(versioning_scroll);
    versioning_content->setStyleSheet("background: transparent;");
    auto* shared_versioning_layout = new QVBoxLayout(versioning_content);
    shared_versioning_layout->setContentsMargins(16, 16, 16, 16);
    shared_versioning_layout->setSpacing(8);
    versioning_scroll->setWidget(versioning_content);
    shared_versioning_outer_layout->addWidget(versioning_scroll, 1);

    main_splitter->addWidget(shared_versioning_root);
    main_splitter->addWidget(viewer_container_);
    main_splitter->addWidget(controls_panel);
    main_splitter->setStretchFactor(0, 0);
    main_splitter->setStretchFactor(1, 1);
    main_splitter->setStretchFactor(2, 0);
    const int right_default_width =
        std::clamp(width() / 3, controls_panel->minimumWidth(), controls_panel->maximumWidth());
    const int left_default_width = 280;
    main_splitter->setSizes({left_default_width,
                             std::max(400, width() - right_default_width - left_default_width),
                             right_default_width});
    root->addWidget(main_splitter, 1);

    {
      auto* histogram_frame = new QFrame(controls_panel);
      histogram_frame->setObjectName("EditorSection");
      auto* histogram_layout = new QVBoxLayout(histogram_frame);
      histogram_layout->setContentsMargins(12, 10, 12, 10);
      histogram_layout->setSpacing(6);

      auto* histogram_title = new QLabel("Histogram", histogram_frame);
      histogram_title->setObjectName("EditorSectionTitle");

      histogram_widget_       = new HistogramWidget(viewer_, histogram_frame);
      histogram_ruler_widget_ = new HistogramRulerWidget(
          viewer_ ? viewer_->GetHistogramBinCount() : 256, histogram_frame);
      histogram_layout->addWidget(histogram_title, 0);
      histogram_layout->addWidget(histogram_widget_, 0);
      histogram_layout->addWidget(histogram_ruler_widget_, 0);
      controls_panel_layout->addWidget(histogram_frame, 0);

      if (viewer_ && histogram_widget_) {
        viewer_->SetHistogramUpdateIntervalMs(40);
        viewer_->SetHistogramFrameExpected(false);
        QObject::connect(viewer_, &QtEditViewer::HistogramDataUpdated, histogram_widget_,
                         QOverload<>::of(&QWidget::update), Qt::QueuedConnection);
      }
    }

    controls_panel_layout->addWidget(panel_switch_row, 0);
    controls_panel_layout->addWidget(control_panels_stack_, 1);

    auto* controls_header = new QLabel("Adjustments", controls_);
    controls_header->setObjectName("SectionTitle");
    controls_layout->addWidget(controls_header, 0);

    // Prefer LUTs next to the executable (installed layout), fall back to source tree.
    const auto app_luts_dir =
        std::filesystem::path(QCoreApplication::applicationDirPath().toStdWString()) / "LUTs";
    const auto src_luts_dir = std::filesystem::path(CONFIG_PATH) / "LUTs";
    const auto luts_dir = std::filesystem::is_directory(app_luts_dir) ? app_luts_dir : src_luts_dir;
    const auto lut_files = ListCubeLutsInDir(luts_dir);

    lut_paths_.push_back("");  // index 0 => None
    lut_names_.push_back("None");
    for (const auto& p : lut_files) {
      lut_paths_.push_back(p.generic_string());
      lut_names_.push_back(QString::fromStdString(p.filename().string()));
    }

    auto addSection = [&](const QString& title, const QString& subtitle) {
      auto* frame = new QFrame(controls_);
      frame->setObjectName("EditorSection");
      auto* v = new QVBoxLayout(frame);
      v->setContentsMargins(12, 10, 12, 10);
      v->setSpacing(2);

      auto* t = new QLabel(title, frame);
      t->setObjectName("EditorSectionTitle");
      auto* s = new QLabel(subtitle, frame);
      s->setObjectName("EditorSectionSub");
      s->setWordWrap(true);
      v->addWidget(t, 0);
      v->addWidget(s, 0);
      controls_layout->addWidget(frame, 0);
    };

    controls_layout->addStretch();

    int default_lut_index = 0;
    for (int i = 1; i < static_cast<int>(lut_paths_.size()); ++i) {
      if (std::filesystem::path(lut_paths_[i]).filename() == "5207.cube") {
        default_lut_index = i;
        break;
      }
    }

    // If the pipeline already has operator params (loaded from PipelineService/storage),
    // initialize UI state from those params rather than overwriting them.
    const bool loaded_state_from_pipeline = LoadStateFromPipelineIfPresent();
    if (!loaded_state_from_pipeline) {
      // Demo-friendly default: apply a LUT only for brand-new pipelines with no saved params.
      state_.lut_path_ = lut_paths_[default_lut_index];
    }
    committed_state_ = state_;

    // Seed a working version from the latest committed one (if any).
    try {
      const auto parent_id = history_guard_->history_->GetLatestVersion().ver_ref_.GetVersionID();
      working_version_     = Version(element_id_, parent_id);
    } catch (...) {
      working_version_ = Version(element_id_);
    }
    working_version_.SetBasePipelineExecutor(pipeline_guard_->pipeline_);

    int initial_lut_index = 0;
    if (!state_.lut_path_.empty()) {
      auto it = std::find(lut_paths_.begin(), lut_paths_.end(), state_.lut_path_);
      if (it != lut_paths_.end()) {
        initial_lut_index = static_cast<int>(std::distance(lut_paths_.begin(), it));
      } else {
        // Keep UI consistent even if LUT path comes from an external/custom location.
        lut_paths_.push_back(state_.lut_path_);
        lut_names_.push_back(
            QString::fromStdString(std::filesystem::path(state_.lut_path_).filename().string()));
        initial_lut_index = static_cast<int>(lut_paths_.size() - 1);
      }
    }

    auto addComboBox = [&](const QString& name, const QStringList& items, int initial_index,
                           auto&& onChange) {
      auto* label = new QLabel(name, controls_);
      label->setStyleSheet(
          "QLabel {"
          "  color: #E6E6E6;"
          "  font-size: 14px;"
          "  font-weight: 400;"
          "}");

      auto* combo = new QComboBox(controls_);
      combo->addItems(items);
      combo->setCurrentIndex(initial_index);
      combo->setMinimumWidth(240);
      combo->setFixedHeight(32);
      combo->setStyleSheet(
          "QComboBox {"
          "  background: #1A1A1A;"
          "  border: none;"
          "  border-radius: 8px;"
          "  padding: 4px 8px;"
          "}"
          "QComboBox::drop-down {"
          "  border: 0px;"
          "  width: 24px;"
          "}"
          "QComboBox QAbstractItemView {"
          "  background: #1A1A1A;"
          "  border: none;"
          "  selection-background-color: #FCC704;"
          "  selection-color: #121212;"
          "}"
          "QComboBox QAbstractItemView::item:hover {"
          "  background: #202020;"
          "  color: #E6E6E6;"
          "}"
          "QComboBox QAbstractItemView::item:selected {"
          "  background: #FCC704;"
          "  color: #121212;"
          "}");

      QObject::connect(combo, QOverload<int>::of(&QComboBox::currentIndexChanged), controls_,
                       [this, onChange = std::forward<decltype(onChange)>(onChange)](int idx) {
                         if (syncing_controls_) {
                           return;
                         }
                         onChange(idx);
                       });

      auto* row       = new QWidget(controls_);
      auto* rowLayout = new QHBoxLayout(row);
      rowLayout->setContentsMargins(0, 0, 0, 0);
      rowLayout->addWidget(label, /*stretch*/ 1);
      rowLayout->addWidget(combo);

      controls_layout->insertWidget(controls_layout->count() - 1, row);
      return combo;
    };

    auto addSlider = [&](const QString& name, int min, int max, int value, auto&& onChange,
                         auto&& onRelease, auto&& formatter) {
      auto* info = new QLabel(QString("%1: %2").arg(name).arg(formatter(value)), controls_);
      info->setStyleSheet(
          "QLabel {"
          "  color: #E6E6E6;"
          "  font-size: 14px;"
          "  font-weight: 400;"
          "}");

      auto* slider = new QSlider(Qt::Horizontal, controls_);
      slider->setRange(min, max);
      slider->setValue(value);
      slider->setSingleStep(1);
      slider->setPageStep(std::max(1, (max - min) / 20));
      slider->setMinimumWidth(240);
      slider->setFixedHeight(32);

      QObject::connect(slider, &QSlider::valueChanged, controls_,
                       [this, info, name, formatter,
                        onChange = std::forward<decltype(onChange)>(onChange)](int v) {
                         info->setText(QString("%1: %2").arg(name).arg(formatter(v)));
                         if (syncing_controls_) {
                           return;
                         }
                         onChange(v);
                       });

      QObject::connect(slider, &QSlider::sliderReleased, controls_,
                       [this, onRelease = std::forward<decltype(onRelease)>(onRelease)]() {
                         if (syncing_controls_) {
                           return;
                         }
                         onRelease();
                       });

      auto* row       = new QWidget(controls_);
      auto* rowLayout = new QHBoxLayout(row);
      rowLayout->setContentsMargins(0, 0, 0, 0);
      rowLayout->addWidget(info, /*stretch*/ 1);
      rowLayout->addWidget(slider);

      controls_layout->insertWidget(controls_layout->count() - 1, row);
      return slider;
    };

    lut_combo_ = addComboBox("LUT", lut_names_, initial_lut_index, [&](int idx) {
      if (idx < 0 || idx >= static_cast<int>(lut_paths_.size())) {
        return;
      }
      state_.lut_path_ = lut_paths_[idx];
      CommitAdjustment(AdjustmentField::Lut);
    });

    addSection("Tone", "Primary tonal shaping controls.");

    exposure_slider_ = addSlider(
        "Exposure", -1000, 1000, static_cast<int>(std::lround(state_.exposure_ * 100.0f)),
        [&](int v) {
          state_.exposure_ = static_cast<float>(v) / 100.0f;
          RequestRender();
        },
        [this]() { CommitAdjustment(AdjustmentField::Exposure); },
        [](int v) { return QString::number(v / 100.0, 'f', 2); });

    contrast_slider_ = addSlider(
        "Contrast", -100, 100, static_cast<int>(std::lround(state_.contrast_)),
        [&](int v) {
          state_.contrast_ = static_cast<float>(v);
          RequestRender();
        },
        [this]() { CommitAdjustment(AdjustmentField::Contrast); },
        [](int v) { return QString::number(v, 'f', 2); });

    highlights_slider_ = addSlider(
        "Highlights", -100, 100, static_cast<int>(std::lround(state_.highlights_)),
        [&](int v) {
          state_.highlights_ = static_cast<float>(v);
          RequestRender();
        },
        [this]() { CommitAdjustment(AdjustmentField::Highlights); },
        [](int v) { return QString::number(v, 'f', 2); });

    shadows_slider_ = addSlider(
        "Shadows", -100, 100, static_cast<int>(std::lround(state_.shadows_)),
        [&](int v) {
          state_.shadows_ = static_cast<float>(v);
          RequestRender();
        },
        [this]() { CommitAdjustment(AdjustmentField::Shadows); },
        [](int v) { return QString::number(v, 'f', 2); });

    whites_slider_ = addSlider(
        "Whites", -100, 100, static_cast<int>(std::lround(state_.whites_)),
        [&](int v) {
          state_.whites_ = static_cast<float>(v);
          RequestRender();
        },
        [this]() { CommitAdjustment(AdjustmentField::Whites); },
        [](int v) { return QString::number(v, 'f', 2); });

    blacks_slider_ = addSlider(
        "Blacks", -100, 100, static_cast<int>(std::lround(state_.blacks_)),
        [&](int v) {
          state_.blacks_ = static_cast<float>(v);
          RequestRender();
        },
        [this]() { CommitAdjustment(AdjustmentField::Blacks); },
        [](int v) { return QString::number(v, 'f', 2); });

    addSection("Curve", "Smooth tone curve mapped from input [0, 1] to output [0, 1].");
    {
      auto* frame  = new QFrame(controls_);
      auto* layout = new QVBoxLayout(frame);
      layout->setContentsMargins(0, 0, 0, 0);
      layout->setSpacing(8);

      curve_widget_ = new ToneCurveWidget(frame);
      curve_widget_->SetControlPoints(state_.curve_points_);
      curve_widget_->SetCurveChangedCallback([this](const std::vector<QPointF>& points) {
        if (syncing_controls_) {
          return;
        }
        state_.curve_points_ = NormalizeCurveControlPoints(points);
        RequestRender();
      });
      curve_widget_->SetCurveReleasedCallback([this](const std::vector<QPointF>& points) {
        if (syncing_controls_) {
          return;
        }
        state_.curve_points_ = NormalizeCurveControlPoints(points);
        CommitAdjustment(AdjustmentField::Curve);
      });

      auto* actions_row        = new QWidget(frame);
      auto* actions_row_layout = new QHBoxLayout(actions_row);
      actions_row_layout->setContentsMargins(0, 0, 0, 0);
      actions_row_layout->setSpacing(8);

      auto* curve_hint =
          new QLabel("Left click/drag to shape. Right click a point to remove.", actions_row);
      curve_hint->setStyleSheet(
          "QLabel {"
          "  color: #A3A3A3;"
          "  font-size: 11px;"
          "}");
      curve_hint->setWordWrap(true);

      auto* reset_curve_btn = new QPushButton("Reset Curve", actions_row);
      reset_curve_btn->setFixedHeight(28);
      reset_curve_btn->setStyleSheet(
          "QPushButton {"
          "  color: #121212;"
          "  background: #FCC704;"
          "  border: none;"
          "  border-radius: 8px;"
          "  padding: 4px 10px;"
          "}"
          "QPushButton:hover {"
          "  background: #FCC704;"
          "}");
      QObject::connect(reset_curve_btn, &QPushButton::clicked, this, [this]() {
        if (!curve_widget_) {
          return;
        }
        state_.curve_points_ = DefaultCurveControlPoints();
        curve_widget_->SetControlPoints(state_.curve_points_);
        RequestRender();
        CommitAdjustment(AdjustmentField::Curve);
      });

      actions_row_layout->addWidget(curve_hint, 1);
      actions_row_layout->addWidget(reset_curve_btn, 0);

      layout->addWidget(curve_widget_, 1);
      layout->addWidget(actions_row, 0);

      controls_layout->insertWidget(controls_layout->count() - 1, frame);
    }

    addSection("Color", "Color balance and saturation.");

    saturation_slider_ = addSlider(
        "Saturation", -100, 100, static_cast<int>(std::lround(state_.saturation_)),
        [&](int v) {
          state_.saturation_ = static_cast<float>(v);
          RequestRender();
        },
        [this]() { CommitAdjustment(AdjustmentField::Saturation); },
        [](int v) { return QString::number(v, 'f', 2); });

    color_temp_mode_combo_ = addComboBox(
        "White Balance", {"As Shot", "Custom"},
        ColorTempModeToComboIndex(state_.color_temp_mode_), [&](int idx) {
          const auto new_mode = ComboIndexToColorTempMode(idx);
          if (new_mode == state_.color_temp_mode_) {
            return;
          }
          if (state_.color_temp_mode_ == ColorTempMode::AS_SHOT &&
              new_mode == ColorTempMode::CUSTOM) {
            state_.color_temp_custom_cct_  = DisplayedColorTempCct(state_);
            state_.color_temp_custom_tint_ = DisplayedColorTempTint(state_);
          }
          state_.color_temp_mode_ = new_mode;
          SyncControlsFromState();
          RequestRender();
          CommitAdjustment(AdjustmentField::ColorTemp);
        });

    color_temp_cct_slider_ = addSlider(
        "Color Temp", kColorTempSliderUiMin, kColorTempSliderUiMax,
        ColorTempCctToSliderPos(DisplayedColorTempCct(state_)),
        [&](int v) {
          PromoteColorTempToCustomForEditing();
          state_.color_temp_custom_cct_ = ColorTempSliderPosToCct(v);
          RequestRender();
        },
        [this]() { CommitAdjustment(AdjustmentField::ColorTemp); },
        [](int v) { return QString("%1 K").arg(static_cast<int>(std::lround(ColorTempSliderPosToCct(v)))); });
    color_temp_cct_slider_->setStyleSheet(
        "QSlider::groove:horizontal {"
        "  border: 1px solid #2A2A2A;"
        "  height: 8px;"
        "  border-radius: 4px;"
        "  background: qlineargradient(x1:0, y1:0, x2:1, y2:0, "
        "stop:0 #9BD8FF, stop:0.5 #FFE8B0, stop:1 #FF8A3D);"
        "}"
        "QSlider::handle:horizontal {"
        "  background: #F2F2F2;"
        "  border: 1px solid #2A2A2A;"
        "  width: 14px;"
        "  margin: -4px 0;"
        "  border-radius: 7px;"
        "}");

    color_temp_tint_slider_ = addSlider(
        "Color Tint", kColorTempTintMin, kColorTempTintMax,
        static_cast<int>(std::lround(DisplayedColorTempTint(state_))),
        [&](int v) {
          PromoteColorTempToCustomForEditing();
          state_.color_temp_custom_tint_ = static_cast<float>(v);
          RequestRender();
        },
        [this]() { CommitAdjustment(AdjustmentField::ColorTemp); },
        [](int v) { return QString::number(v, 'f', 0); });
    color_temp_tint_slider_->setStyleSheet(
        "QSlider::groove:horizontal {"
        "  border: 1px solid #2A2A2A;"
        "  height: 8px;"
        "  border-radius: 4px;"
        "  background: qlineargradient(x1:0, y1:0, x2:1, y2:0, "
        "stop:0 #49C26D, stop:0.5 #E6E6E6, stop:1 #A85AE6);"
        "}"
        "QSlider::handle:horizontal {"
        "  background: #F2F2F2;"
        "  border: 1px solid #2A2A2A;"
        "  width: 14px;"
        "  margin: -4px 0;"
        "  border-radius: 7px;"
        "}");

    color_temp_unsupported_label_ =
        new QLabel("Color temperature/tint is unavailable for this image.", controls_);
    color_temp_unsupported_label_->setWordWrap(true);
    color_temp_unsupported_label_->setStyleSheet(
        "QLabel {"
        "  color: #FFB454;"
        "  background: rgba(255, 180, 84, 0.12);"
        "  border: 1px solid rgba(255, 180, 84, 0.35);"
        "  border-radius: 8px;"
        "  padding: 6px 8px;"
        "  font-size: 12px;"
        "}");
    controls_layout->insertWidget(controls_layout->count() - 1, color_temp_unsupported_label_);
    color_temp_unsupported_label_->setVisible(!state_.color_temp_supported_);

    {
      auto* frame = new QFrame(controls_);
      frame->setObjectName("EditorSection");
      auto* layout = new QVBoxLayout(frame);
      layout->setContentsMargins(12, 10, 12, 10);
      layout->setSpacing(8);

      hls_target_label_ = new QLabel(frame);
      hls_target_label_->setStyleSheet(
          "QLabel {"
          "  color: #E6E6E6;"
          "  font-size: 13px;"
          "  font-weight: 500;"
          "}");
      layout->addWidget(hls_target_label_, 0);

      auto* swatch_row        = new QWidget(frame);
      auto* swatch_row_layout = new QHBoxLayout(swatch_row);
      swatch_row_layout->setContentsMargins(0, 0, 0, 0);
      swatch_row_layout->setSpacing(6);

      hls_candidate_buttons_.clear();
      hls_candidate_buttons_.reserve(kHlsCandidateHues.size());
      for (int i = 0; i < static_cast<int>(kHlsCandidateHues.size()); ++i) {
        auto* btn = new QPushButton(swatch_row);
        btn->setFixedSize(22, 22);
        btn->setCursor(Qt::PointingHandCursor);
        btn->setToolTip(
            QString("Hue %1 deg").arg(kHlsCandidateHues[static_cast<size_t>(i)], 0, 'f', 0));
        QObject::connect(btn, &QPushButton::clicked, this, [this, i]() {
          if (syncing_controls_) {
            return;
          }
          SaveActiveHlsProfile(state_);
          state_.hls_target_hue_ = kHlsCandidateHues[static_cast<size_t>(i)];
          LoadActiveHlsProfile(state_);
          SyncControlsFromState();
        });
        hls_candidate_buttons_.push_back(btn);
        swatch_row_layout->addWidget(btn);
      }
      swatch_row_layout->addStretch();
      layout->addWidget(swatch_row, 0);

      controls_layout->insertWidget(controls_layout->count() - 1, frame);
      RefreshHlsTargetUi();
    }

    hls_hue_adjust_slider_ = addSlider(
        "Hue Shift", -15, 15, static_cast<int>(std::lround(state_.hls_hue_adjust_)),
        [&](int v) {
          state_.hls_hue_adjust_ =
              std::clamp(static_cast<float>(v), -kHlsMaxHueShiftDegrees, kHlsMaxHueShiftDegrees);
          SaveActiveHlsProfile(state_);
          RequestRender();
        },
        [this]() { CommitAdjustment(AdjustmentField::Hls); },
        [](int v) { return QString("%1 deg").arg(v); });

    hls_lightness_adjust_slider_ = addSlider(
        "Lightness", -100, 100, static_cast<int>(std::lround(state_.hls_lightness_adjust_)),
        [&](int v) {
          state_.hls_lightness_adjust_ =
              std::clamp(static_cast<float>(v), kHlsAdjUiMin, kHlsAdjUiMax);
          SaveActiveHlsProfile(state_);
          RequestRender();
        },
        [this]() { CommitAdjustment(AdjustmentField::Hls); },
        [](int v) { return QString::number(v, 'f', 0); });

    hls_saturation_adjust_slider_ = addSlider(
        "HSL Saturation", -100, 100, static_cast<int>(std::lround(state_.hls_saturation_adjust_)),
        [&](int v) {
          state_.hls_saturation_adjust_ =
              std::clamp(static_cast<float>(v), kHlsAdjUiMin, kHlsAdjUiMax);
          SaveActiveHlsProfile(state_);
          RequestRender();
        },
        [this]() { CommitAdjustment(AdjustmentField::Hls); },
        [](int v) { return QString::number(v, 'f', 0); });

    hls_hue_range_slider_ = addSlider(
        "Hue Range", 1, 180, static_cast<int>(std::lround(state_.hls_hue_range_)),
        [&](int v) {
          state_.hls_hue_range_ = static_cast<float>(v);
          SaveActiveHlsProfile(state_);
          RequestRender();
        },
        [this]() { CommitAdjustment(AdjustmentField::Hls); },
        [](int v) { return QString("%1 deg").arg(v); });

    addSection("Detail", "Micro-contrast and sharpen controls.");

    sharpen_slider_ = addSlider(
        "Sharpen", -100, 100, static_cast<int>(std::lround(state_.sharpen_)),
        [&](int v) {
          state_.sharpen_ = static_cast<float>(v);
          RequestRender();
        },
        [this]() { CommitAdjustment(AdjustmentField::Sharpen); },
        [](int v) { return QString::number(v, 'f', 2); });

    clarity_slider_ = addSlider(
        "Clarity", -100, 100, static_cast<int>(std::lround(state_.clarity_)),
        [&](int v) {
          state_.clarity_ = static_cast<float>(v);
          RequestRender();
        },
        [this]() { CommitAdjustment(AdjustmentField::Clarity); },
        [](int v) { return QString::number(v, 'f', 2); });

    auto addGeometrySection = [&](const QString& title, const QString& subtitle) {
      auto* frame = new QFrame(geometry_controls_);
      frame->setObjectName("EditorSection");
      auto* v = new QVBoxLayout(frame);
      v->setContentsMargins(12, 10, 12, 10);
      v->setSpacing(2);

      auto* t = new QLabel(title, frame);
      t->setObjectName("EditorSectionTitle");
      auto* s = new QLabel(subtitle, frame);
      s->setObjectName("EditorSectionSub");
      s->setWordWrap(true);
      v->addWidget(t, 0);
      v->addWidget(s, 0);
      geometry_controls_layout->insertWidget(geometry_controls_layout->count() - 1, frame);
    };

    auto addGeometrySlider = [&](const QString& name, int min, int max, int value, auto&& onChange,
                                 auto&& formatter) {
      auto* info = new QLabel(QString("%1: %2").arg(name).arg(formatter(value)), geometry_controls_);
      info->setStyleSheet(
          "QLabel {"
          "  color: #E6E6E6;"
          "  font-size: 14px;"
          "  font-weight: 400;"
          "}");

      auto* slider = new QSlider(Qt::Horizontal, geometry_controls_);
      slider->setRange(min, max);
      slider->setValue(value);
      slider->setSingleStep(1);
      slider->setPageStep(std::max(1, (max - min) / 20));
      slider->setMinimumWidth(240);
      slider->setFixedHeight(32);

      QObject::connect(slider, &QSlider::valueChanged, geometry_controls_,
                       [this, info, name, formatter,
                        onChange = std::forward<decltype(onChange)>(onChange)](int v) {
                         info->setText(QString("%1: %2").arg(name).arg(formatter(v)));
                         if (syncing_controls_) {
                           return;
                         }
                         onChange(v);
                       });

      auto* row       = new QWidget(geometry_controls_);
      auto* rowLayout = new QHBoxLayout(row);
      rowLayout->setContentsMargins(0, 0, 0, 0);
      rowLayout->addWidget(info, 1);
      rowLayout->addWidget(slider);
      geometry_controls_layout->insertWidget(geometry_controls_layout->count() - 1, row);
      return slider;
    };

    addGeometrySection("Geometry", "Rotate and crop workflow. Changes apply only when committed.");
    rotate_slider_ = addGeometrySlider(
        "Rotate", -18000, 18000, static_cast<int>(std::lround(state_.rotate_degrees_ * kRotationSliderScale)),
        [&](int v) {
          state_.rotate_degrees_ = static_cast<float>(v) / kRotationSliderScale;
          if (viewer_) {
            viewer_->SetCropOverlayRotationDegrees(state_.rotate_degrees_);
          }
        },
        [](int v) { return QString("%1 deg").arg(static_cast<double>(v) / kRotationSliderScale, 0, 'f', 2); });
    // Double-click rotate slider to reset to 0.
    rotate_slider_->installEventFilter(this);

    geometry_crop_x_slider_ = addGeometrySlider(
        "Crop X", 0, static_cast<int>(kCropRectSliderScale),
        static_cast<int>(std::lround(state_.crop_x_ * kCropRectSliderScale)),
        [&](int v) {
          const auto clamped = ClampCropRect(static_cast<float>(v) / kCropRectSliderScale, state_.crop_y_,
                                             state_.crop_w_, state_.crop_h_);
          state_.crop_x_      = clamped[0];
          state_.crop_y_      = clamped[1];
          state_.crop_w_      = clamped[2];
          state_.crop_h_      = clamped[3];
          state_.crop_enabled_ = true;
          UpdateGeometryCropRectLabel();
          if (viewer_) {
            viewer_->SetCropOverlayRectNormalized(state_.crop_x_, state_.crop_y_, state_.crop_w_,
                                                  state_.crop_h_);
          }
        },
        [](int v) { return QString::number(static_cast<double>(v) / kCropRectSliderScale, 'f', 3); });

    geometry_crop_y_slider_ = addGeometrySlider(
        "Crop Y", 0, static_cast<int>(kCropRectSliderScale),
        static_cast<int>(std::lround(state_.crop_y_ * kCropRectSliderScale)),
        [&](int v) {
          const auto clamped = ClampCropRect(state_.crop_x_, static_cast<float>(v) / kCropRectSliderScale,
                                             state_.crop_w_, state_.crop_h_);
          state_.crop_x_      = clamped[0];
          state_.crop_y_      = clamped[1];
          state_.crop_w_      = clamped[2];
          state_.crop_h_      = clamped[3];
          state_.crop_enabled_ = true;
          UpdateGeometryCropRectLabel();
          if (viewer_) {
            viewer_->SetCropOverlayRectNormalized(state_.crop_x_, state_.crop_y_, state_.crop_w_,
                                                  state_.crop_h_);
          }
        },
        [](int v) { return QString::number(static_cast<double>(v) / kCropRectSliderScale, 'f', 3); });

    geometry_crop_w_slider_ = addGeometrySlider(
        "Crop W", 1, static_cast<int>(kCropRectSliderScale),
        static_cast<int>(std::lround(state_.crop_w_ * kCropRectSliderScale)),
        [&](int v) {
          const auto clamped = ClampCropRect(state_.crop_x_, state_.crop_y_,
                                             static_cast<float>(v) / kCropRectSliderScale, state_.crop_h_);
          state_.crop_x_      = clamped[0];
          state_.crop_y_      = clamped[1];
          state_.crop_w_      = clamped[2];
          state_.crop_h_      = clamped[3];
          state_.crop_enabled_ = true;
          UpdateGeometryCropRectLabel();
          if (viewer_) {
            viewer_->SetCropOverlayRectNormalized(state_.crop_x_, state_.crop_y_, state_.crop_w_,
                                                  state_.crop_h_);
          }
        },
        [](int v) { return QString::number(static_cast<double>(v) / kCropRectSliderScale, 'f', 3); });

    geometry_crop_h_slider_ = addGeometrySlider(
        "Crop H", 1, static_cast<int>(kCropRectSliderScale),
        static_cast<int>(std::lround(state_.crop_h_ * kCropRectSliderScale)),
        [&](int v) {
          const auto clamped = ClampCropRect(state_.crop_x_, state_.crop_y_, state_.crop_w_,
                                             static_cast<float>(v) / kCropRectSliderScale);
          state_.crop_x_      = clamped[0];
          state_.crop_y_      = clamped[1];
          state_.crop_w_      = clamped[2];
          state_.crop_h_      = clamped[3];
          state_.crop_enabled_ = true;
          UpdateGeometryCropRectLabel();
          if (viewer_) {
            viewer_->SetCropOverlayRectNormalized(state_.crop_x_, state_.crop_y_, state_.crop_w_,
                                                  state_.crop_h_);
          }
        },
        [](int v) { return QString::number(static_cast<double>(v) / kCropRectSliderScale, 'f', 3); });

    {
      auto* frame = new QFrame(geometry_controls_);
      frame->setObjectName("EditorSection");
      auto* layout = new QVBoxLayout(frame);
      layout->setContentsMargins(12, 10, 12, 10);
      layout->setSpacing(8);

      geometry_crop_rect_label_ = new QLabel(frame);
      geometry_crop_rect_label_->setStyleSheet(
          "QLabel {"
          "  color: #A3A3A3;"
          "  font-size: 12px;"
          "}");
      layout->addWidget(geometry_crop_rect_label_, 0);

      auto* row       = new QWidget(frame);
      auto* rowLayout = new QHBoxLayout(row);
      rowLayout->setContentsMargins(0, 0, 0, 0);
      rowLayout->setSpacing(8);

      geometry_apply_btn_ = new QPushButton("Apply Crop", row);
      geometry_apply_btn_->setFixedHeight(30);
      geometry_apply_btn_->setCursor(Qt::PointingHandCursor);
      geometry_apply_btn_->setStyleSheet(
          "QPushButton {"
          "  color: #121212;"
          "  background: #FCC704;"
          "  border: none;"
          "  border-radius: 8px;"
          "  font-weight: 600;"
          "}"
          "QPushButton:hover {"
          "  background: #F5C200;"
          "}");
      QObject::connect(geometry_apply_btn_, &QPushButton::clicked, this, [this]() {
        state_.crop_enabled_ = true;
        CommitAdjustment(AdjustmentField::CropRotate);
      });
      rowLayout->addWidget(geometry_apply_btn_, 1);

      geometry_reset_btn_ = new QPushButton("Reset", row);
      geometry_reset_btn_->setFixedHeight(30);
      geometry_reset_btn_->setCursor(Qt::PointingHandCursor);
      geometry_reset_btn_->setToolTip("Reset crop & rotation (Ctrl+R)");
      geometry_reset_btn_->setShortcut(QKeySequence(Qt::CTRL | Qt::Key_R));
      geometry_reset_btn_->setStyleSheet(
          "QPushButton {"
          "  color: #E6E6E6;"
          "  background: #3A3A3A;"
          "  border: none;"
          "  border-radius: 8px;"
          "  font-weight: 600;"
          "}"
          "QPushButton:hover {"
          "  background: #505050;"
          "}");
      QObject::connect(geometry_reset_btn_, &QPushButton::clicked, this, [this]() {
        ResetCropAndRotation();
      });
      rowLayout->addWidget(geometry_reset_btn_, 0);
      layout->addWidget(row, 0);

      auto* hint = new QLabel(
          "Geometry panel edits only the crop frame overlay. Image pixels update only after Apply Crop. "
          "Double click viewer to restore full crop frame. "
          "Double click rotate slider to reset angle. Ctrl+R to reset all geometry.",
          frame);
      hint->setWordWrap(true);
      hint->setStyleSheet(
          "QLabel {"
          "  color: #A3A3A3;"
          "  font-size: 11px;"
          "}");
      layout->addWidget(hint, 0);

      geometry_controls_layout->insertWidget(geometry_controls_layout->count() - 1, frame);
    }

    if (viewer_) {
      QObject::connect(viewer_, &QtEditViewer::CropOverlayRectChanged, this,
                       [this](float x, float y, float w, float h, bool /*is_final*/) {
                         if (syncing_controls_) {
                           return;
                         }
                         const auto clamped = ClampCropRect(x, y, w, h);
                         state_.crop_x_     = clamped[0];
                         state_.crop_y_     = clamped[1];
                         state_.crop_w_     = clamped[2];
                         state_.crop_h_     = clamped[3];
                         state_.crop_enabled_ = true;
                         UpdateGeometryCropRectLabel();
                         const bool prev_sync = syncing_controls_;
                         syncing_controls_     = true;
                         if (geometry_crop_x_slider_) {
                           geometry_crop_x_slider_->setValue(
                               static_cast<int>(std::lround(state_.crop_x_ * kCropRectSliderScale)));
                         }
                         if (geometry_crop_y_slider_) {
                           geometry_crop_y_slider_->setValue(
                               static_cast<int>(std::lround(state_.crop_y_ * kCropRectSliderScale)));
                         }
                         if (geometry_crop_w_slider_) {
                           geometry_crop_w_slider_->setValue(
                               static_cast<int>(std::lround(state_.crop_w_ * kCropRectSliderScale)));
                         }
                         if (geometry_crop_h_slider_) {
                           geometry_crop_h_slider_->setValue(
                               static_cast<int>(std::lround(state_.crop_h_ * kCropRectSliderScale)));
                         }
                          syncing_controls_ = prev_sync;
                        });
      QObject::connect(viewer_, &QtEditViewer::CropOverlayRotationChanged, this,
                       [this](float angle_degrees, bool /*is_final*/) {
                         if (syncing_controls_) {
                           return;
                         }
                         state_.rotate_degrees_ = angle_degrees;
                         const bool prev_sync = syncing_controls_;
                         syncing_controls_     = true;
                         if (rotate_slider_) {
                           rotate_slider_->setValue(
                               static_cast<int>(std::lround(state_.rotate_degrees_ *
                                                            kRotationSliderScale)));
                         }
                         syncing_controls_ = prev_sync;
                       });
    }
    UpdateGeometryCropRectLabel();
    RefreshGeometryModeUi();

    {
      auto* section = new QFrame(shared_versioning_root);
      section->setObjectName("EditorSection");
      auto* v = new QVBoxLayout(section);
      v->setContentsMargins(12, 10, 12, 10);
      v->setSpacing(2);
      auto* t = new QLabel("Versioning", section);
      t->setObjectName("EditorSectionTitle");
      auto* s = new QLabel("Commit and inspect edit history.", section);
      s->setObjectName("EditorSectionSub");
      s->setWordWrap(true);
      v->addWidget(t, 0);
      v->addWidget(s, 0);
      shared_versioning_layout->addWidget(section, 0);
    }

    // Edit-history commit controls.
    {
      auto* row       = new QWidget(shared_versioning_root);
      auto* rowLayout = new QHBoxLayout(row);
      rowLayout->setContentsMargins(0, 0, 0, 0);
      rowLayout->setSpacing(10);

      version_status_ = new QLabel(row);
      version_status_->setStyleSheet(
          "QLabel {"
          "  color: #A3A3A3;"
          "  font-size: 12px;"
          "}");
      version_status_->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Preferred);
      version_status_->setMinimumWidth(0);

      undo_tx_btn_ = new QPushButton("Undo", row);
      undo_tx_btn_->setFixedHeight(32);
      undo_tx_btn_->setStyleSheet(
          "QPushButton {"
          "  color: #121212;"
          "  background: #FCC704;"
          "  border: none;"
          "  border-radius: 8px;"
          "  padding: 6px 10px;"
          "}"
          "QPushButton:hover {"
          "  background: #F5C200;"
          "}"
          "QPushButton:disabled {"
          "  color: #6A6A6A;"
          "  background: #1A1A1A;"
          "}");

      commit_version_btn_ = new QPushButton("Commit Version", row);
      commit_version_btn_->setFixedHeight(32);
      commit_version_btn_->setStyleSheet(
          "QPushButton {"
          "  color: #121212;"
          "  background: #FCC704;"
          "  border: none;"
          "  border-radius: 8px;"
          "  padding: 6px 10px;"
          "}"
          "QPushButton:hover {"
          "  background: #F5C200;"
          "}"
          "QPushButton:disabled {"
          "  color: #6A6A6A;"
          "  background: #1A1A1A;"
          "}");

      rowLayout->addWidget(version_status_, /*stretch*/ 1);
      rowLayout->addWidget(undo_tx_btn_, /*stretch*/ 0);
      rowLayout->addWidget(commit_version_btn_, /*stretch*/ 0);
      shared_versioning_layout->addWidget(row, 0);

      QObject::connect(undo_tx_btn_, &QPushButton::clicked, this,
                       [this]() { UndoLastTransaction(); });
      QObject::connect(commit_version_btn_, &QPushButton::clicked, this,
                       [this]() { CommitWorkingVersion(); });
    }

    // Edit-history visualization ("git log"-like) + working version mode.
    {
      auto* frame = new QFrame(shared_versioning_root);
      frame->setStyleSheet(
          "QFrame {"
          "  background: transparent;"
          "  border: none;"
          "  border-radius: 12px;"
          "}");
      auto* layout = new QVBoxLayout(frame);
      layout->setContentsMargins(10, 10, 10, 10);
      layout->setSpacing(8);

      auto* mode_row    = new QWidget(frame);
      auto* mode_layout = new QHBoxLayout(mode_row);
      mode_layout->setContentsMargins(0, 0, 0, 0);
      mode_layout->setSpacing(8);

      auto* mode_label = new QLabel("Working version:", mode_row);
      mode_label->setStyleSheet(
          "QLabel {"
          "  color: #A3A3A3;"
          "  font-size: 12px;"
          "}");

      working_mode_combo_ = new QComboBox(mode_row);
      working_mode_combo_->addItem("Plain", static_cast<int>(WorkingMode::Plain));
      working_mode_combo_->addItem("Incr", static_cast<int>(WorkingMode::Incremental));
      working_mode_combo_->setFixedHeight(28);
      working_mode_combo_->setStyleSheet(
          "QComboBox {"
          "  background: #1A1A1A;"
          "  border: none;"
          "  border-radius: 8px;"
          "  padding: 4px 8px;"
          "}"
          "QComboBox::drop-down {"
          "  border: 0px;"
          "  width: 24px;"
          "}"
          "QComboBox QAbstractItemView {"
          "  background: #1A1A1A;"
          "  border: none;"
          "  selection-background-color: #FCC704;"
          "  selection-color: #121212;"
          "}");

      new_working_btn_ = new QPushButton("New", mode_row);
      new_working_btn_->setFixedHeight(28);
      new_working_btn_->setStyleSheet(
          "QPushButton {"
          "  color: #121212;"
          "  background: #FCC704;"
          "  border: none;"
          "  border-radius: 8px;"
          "  padding: 4px 10px;"
          "}"
          "QPushButton:hover {"
          "  background: #F5C200;"
          "}");

      mode_layout->addWidget(mode_label, /*stretch*/ 0);
      mode_layout->addWidget(working_mode_combo_, /*stretch*/ 1);
      mode_layout->addWidget(new_working_btn_, /*stretch*/ 0);

      layout->addWidget(mode_row);

      auto* versions_label = new QLabel("Versions", frame);
      versions_label->setStyleSheet(
          "QLabel {"
          "  color: #E6E6E6;"
          "  font-size: 12px;"
          "  font-weight: 500;"
          "}");
      layout->addWidget(versions_label);

      version_log_ = new QListWidget(frame);
      version_log_->setSelectionMode(QAbstractItemView::SingleSelection);
      version_log_->setSpacing(6);
      version_log_->setVerticalScrollMode(QAbstractItemView::ScrollPerPixel);
      version_log_->setMinimumHeight(110);
      version_log_->setStyleSheet(
          "QListWidget {"
          "  background: #121212;"
          "  border: none;"
          "  border-radius: 10px;"
          "  padding: 6px;"
          "}"
          "QListWidget::item {"
          "  padding: 2px;"
          "}"
          "QListWidget::item:selected {"
          "  background: transparent;"
          "}");
      layout->addWidget(version_log_);

      QObject::connect(version_log_, &QListWidget::itemSelectionChanged, this,
                       [this]() { RefreshVersionLogSelectionStyles(); });
      QObject::connect(version_log_, &QListWidget::itemClicked, this,
                       [this](QListWidgetItem* item) { CheckoutSelectedVersion(item); });

      auto* tx_label = new QLabel("Uncommitted transactions (stack)", frame);
      tx_label->setStyleSheet(
          "QLabel {"
          "  color: #E6E6E6;"
          "  font-size: 12px;"
          "  font-weight: 500;"
          "}");
      layout->addWidget(tx_label);

      tx_stack_ = new QListWidget(frame);
      tx_stack_->setSelectionMode(QAbstractItemView::NoSelection);
      tx_stack_->setSpacing(6);
      tx_stack_->setVerticalScrollMode(QAbstractItemView::ScrollPerPixel);
      tx_stack_->setMinimumHeight(130);
      tx_stack_->setStyleSheet(
          "QListWidget {"
          "  background: #121212;"
          "  border: none;"
          "  border-radius: 10px;"
          "  padding: 6px;"
          "}"
          "QListWidget::item {"
          "  padding: 2px;"
          "}");
      layout->addWidget(tx_stack_, /*stretch*/ 1);

      shared_versioning_layout->addWidget(frame, 0);
      shared_versioning_layout->addStretch();

      QObject::connect(new_working_btn_, &QPushButton::clicked, this,
                       [this]() { StartNewWorkingVersionFromUi(); });
      QObject::connect(working_mode_combo_, QOverload<int>::of(&QComboBox::currentIndexChanged),
                       this, [this](int) { UpdateVersionUi(); });
    }

    UpdateVersionUi();

    SetupPipeline();
    pipeline_initialized_ = true;
    if (viewer_) {
      viewer_->SetCropOverlayRectNormalized(state_.crop_x_, state_.crop_y_, state_.crop_w_,
                                            state_.crop_h_);
      viewer_->SetCropOverlayRotationDegrees(state_.rotate_degrees_);
      viewer_->SetCropOverlayVisible(false);
      viewer_->SetCropToolEnabled(false);
    }

    // Load with a full-res preview first; scheduler transitions back to fast-preview baseline.
    QTimer::singleShot(0, this, [this]() {
      state_.type_ = RenderType::FULL_RES_PREVIEW;
      RequestRender();
      state_.type_ = RenderType::FAST_PREVIEW;
    });
  }

 private:
  enum class ControlPanelKind { Tone, Geometry };

  enum class AdjustmentField {
    Exposure,
    Contrast,
    Saturation,
    ColorTemp,
    Hls,
    Blacks,
    Whites,
    Shadows,
    Highlights,
    Curve,
    Sharpen,
    Clarity,
    Lut,
    CropRotate,
  };

  struct AdjustmentState {
    float                exposure_                    = 2.0f;
    float                contrast_                    = 0.0f;
    float                saturation_                  = 30.0f;
    ColorTempMode        color_temp_mode_             = ColorTempMode::AS_SHOT;
    float                color_temp_custom_cct_       = 6500.0f;
    float                color_temp_custom_tint_      = 0.0f;
    float                color_temp_resolved_cct_     = 6500.0f;
    float                color_temp_resolved_tint_    = 0.0f;
    bool                 color_temp_supported_         = true;
    float                hls_target_hue_              = 0.0f;
    float                hls_hue_adjust_              = 0.0f;
    float                hls_lightness_adjust_        = 0.0f;
    float                hls_saturation_adjust_       = 0.0f;
    float                hls_hue_range_               = kHlsDefaultHueRange;
    HlsProfileArray      hls_hue_adjust_table_        = {};
    HlsProfileArray      hls_lightness_adjust_table_  = {};
    HlsProfileArray      hls_saturation_adjust_table_ = {};
    HlsProfileArray      hls_hue_range_table_         = MakeHlsFilledArray(kHlsDefaultHueRange);
    float                blacks_                      = 0.0f;
    float                whites_                      = 0.0f;
    float                shadows_                     = 0.0f;
    float                highlights_                  = 0.0f;
    std::vector<QPointF> curve_points_                = DefaultCurveControlPoints();
    float                sharpen_                     = 0.0f;
    float                clarity_                     = 0.0f;
    float                rotate_degrees_              = 0.0f;
    bool                 crop_enabled_                = true;
    float                crop_x_                      = 0.0f;
    float                crop_y_                      = 0.0f;
    float                crop_w_                      = 1.0f;
    float                crop_h_                      = 1.0f;
    bool                 crop_expand_to_fit_          = true;
    std::string          lut_path_;
    RenderType           type_ = RenderType::FAST_PREVIEW;
  };

  struct PendingRenderRequest {
    AdjustmentState state_;
    bool            apply_state_ = true;
  };

  static bool NearlyEqual(float a, float b) { return std::abs(a - b) <= 1e-6f; }

  static auto ActiveHlsProfileIndex(const AdjustmentState& state) -> int {
    return std::clamp(ClosestHlsCandidateHueIndex(state.hls_target_hue_), 0,
                      static_cast<int>(kHlsCandidateHues.size()) - 1);
  }

  static void SaveActiveHlsProfile(AdjustmentState& state) {
    const int idx                           = ActiveHlsProfileIndex(state);
    state.hls_hue_adjust_table_[idx]        = state.hls_hue_adjust_;
    state.hls_lightness_adjust_table_[idx]  = state.hls_lightness_adjust_;
    state.hls_saturation_adjust_table_[idx] = state.hls_saturation_adjust_;
    state.hls_hue_range_table_[idx]         = state.hls_hue_range_;
  }

  static void LoadActiveHlsProfile(AdjustmentState& state) {
    const int idx                = ActiveHlsProfileIndex(state);
    state.hls_hue_adjust_        = state.hls_hue_adjust_table_[idx];
    state.hls_lightness_adjust_  = state.hls_lightness_adjust_table_[idx];
    state.hls_saturation_adjust_ = state.hls_saturation_adjust_table_[idx];
    state.hls_hue_range_         = state.hls_hue_range_table_[idx];
  }

  static auto ParseColorTempMode(const std::string& mode) -> ColorTempMode {
    if (mode == "custom") {
      return ColorTempMode::CUSTOM;
    }
    if (mode == "as-shot" || mode == "as_shot") {
      return ColorTempMode::AS_SHOT;
    }
    return ColorTempMode::AS_SHOT;
  }

  static auto ColorTempModeToString(ColorTempMode mode) -> std::string {
    switch (mode) {
      case ColorTempMode::CUSTOM:
        return "custom";
      case ColorTempMode::AS_SHOT:
      default:
        return "as_shot";
    }
  }

  static auto ColorTempModeToComboIndex(ColorTempMode mode) -> int {
    return mode == ColorTempMode::CUSTOM ? 1 : 0;
  }

  static auto ComboIndexToColorTempMode(int index) -> ColorTempMode {
    return index == 1 ? ColorTempMode::CUSTOM : ColorTempMode::AS_SHOT;
  }

  static auto DisplayedColorTempCct(const AdjustmentState& state) -> float {
    const float cct = (state.color_temp_mode_ == ColorTempMode::AS_SHOT)
                          ? state.color_temp_resolved_cct_
                          : state.color_temp_custom_cct_;
    return std::clamp(cct, static_cast<float>(kColorTempCctMin), static_cast<float>(kColorTempCctMax));
  }

  static auto DisplayedColorTempTint(const AdjustmentState& state) -> float {
    const float tint = (state.color_temp_mode_ == ColorTempMode::AS_SHOT)
                           ? state.color_temp_resolved_tint_
                           : state.color_temp_custom_tint_;
    return std::clamp(tint, static_cast<float>(kColorTempTintMin), static_cast<float>(kColorTempTintMax));
  }

  void RefreshHlsTargetUi() {
    if (!hls_target_label_ && hls_candidate_buttons_.empty()) {
      return;
    }

    const float hue = WrapHueDegrees(state_.hls_target_hue_);
    if (hls_target_label_) {
      hls_target_label_->setText(QString("Target Hue: %1 deg").arg(hue, 0, 'f', 0));
    }

    const int selected_idx = ClosestHlsCandidateHueIndex(hue);
    for (int i = 0; i < static_cast<int>(hls_candidate_buttons_.size()); ++i) {
      auto* btn = hls_candidate_buttons_[i];
      if (!btn) {
        continue;
      }
      const bool   selected    = (i == selected_idx);
      const QColor swatch      = HlsCandidateColor(kHlsCandidateHues[static_cast<size_t>(i)]);
      const auto   border_w_px = selected ? "3px" : "1px";
      const auto   border_col  = selected ? "#FCC704" : "#2A2A2A";
      btn->setStyleSheet(QString("QPushButton {"
                                 "  background: %1;"
                                 "  border: %2 solid %3;"
                                 "  border-radius: 11px;"
                                 "}"
                                 "QPushButton:hover {"
                                 "  border-color: #FCC704;"
                                 "}")
                             .arg(swatch.name(QColor::HexRgb), border_w_px, border_col));
    }
  }

  void UpdateGeometryCropRectLabel() {
    if (!geometry_crop_rect_label_) {
      return;
    }
    geometry_crop_rect_label_->setText(
        QString("Crop Rect: x=%1 y=%2 w=%3 h=%4")
            .arg(state_.crop_x_, 0, 'f', 3)
            .arg(state_.crop_y_, 0, 'f', 3)
            .arg(state_.crop_w_, 0, 'f', 3)
            .arg(state_.crop_h_, 0, 'f', 3));
  }

  void ResetCropAndRotation() {
    state_.crop_x_         = 0.0f;
    state_.crop_y_         = 0.0f;
    state_.crop_w_         = 1.0f;
    state_.crop_h_         = 1.0f;
    state_.crop_enabled_   = true;
    state_.rotate_degrees_ = 0.0f;

    const bool prev_sync = syncing_controls_;
    syncing_controls_     = true;
    if (geometry_crop_x_slider_) {
      geometry_crop_x_slider_->setValue(0);
    }
    if (geometry_crop_y_slider_) {
      geometry_crop_y_slider_->setValue(0);
    }
    if (geometry_crop_w_slider_) {
      geometry_crop_w_slider_->setValue(static_cast<int>(kCropRectSliderScale));
    }
    if (geometry_crop_h_slider_) {
      geometry_crop_h_slider_->setValue(static_cast<int>(kCropRectSliderScale));
    }
    if (rotate_slider_) {
      rotate_slider_->setValue(0);
    }
    syncing_controls_ = prev_sync;

    UpdateGeometryCropRectLabel();
    if (viewer_) {
      viewer_->SetCropOverlayRectNormalized(0.0f, 0.0f, 1.0f, 1.0f);
      viewer_->SetCropOverlayRotationDegrees(0.0f);
    }
  }

  bool eventFilter(QObject* obj, QEvent* event) override {
    if (obj == rotate_slider_ && event->type() == QEvent::MouseButtonDblClick) {
      state_.rotate_degrees_ = 0.0f;
      const bool prev_sync   = syncing_controls_;
      syncing_controls_       = true;
      if (rotate_slider_) {
        rotate_slider_->setValue(0);
      }
      syncing_controls_ = prev_sync;
      if (viewer_) {
        viewer_->SetCropOverlayRotationDegrees(0.0f);
      }
      return true;  // consume the event
    }
    return QDialog::eventFilter(obj, event);
  }

  void UpdateViewerZoomLabel(float zoom) {
    if (!viewer_zoom_label_) {
      return;
    }
    const float clamped = std::max(1.0f, zoom);
    viewer_zoom_label_->setText(
        QString("Zoom %1% (%2x)")
            .arg(clamped * 100.0f, 0, 'f', 0)
            .arg(clamped, 0, 'f', 2));
  }

  void RefreshGeometryModeUi() {
    // Geometry crop editing is always enabled when the geometry panel is active.
  }

  void RefreshPanelSwitchUi() {
    if (!tone_panel_btn_ || !geometry_panel_btn_) {
      return;
    }
    const bool tone_active = (active_panel_ == ControlPanelKind::Tone);
    tone_panel_btn_->setChecked(tone_active);
    geometry_panel_btn_->setChecked(!tone_active);

    const QString active_style =
        "QPushButton {"
        "  color: #121212;"
        "  background: #FCC704;"
        "  border: none;"
        "  border-radius: 8px;"
        "  font-weight: 600;"
        "}"
        "QPushButton:hover {"
        "  background: #F5C200;"
        "}";
    const QString inactive_style =
        "QPushButton {"
        "  color: #E6E6E6;"
        "  background: #121212;"
        "  border: 1px solid #2A2A2A;"
        "  border-radius: 8px;"
        "  font-weight: 500;"
        "}"
        "QPushButton:hover {"
        "  border-color: #FCC704;"
        "}";
    tone_panel_btn_->setStyleSheet(tone_active ? active_style : inactive_style);
    geometry_panel_btn_->setStyleSheet(tone_active ? inactive_style : active_style);
  }

  void SetActiveControlPanel(ControlPanelKind panel) {
    active_panel_ = panel;
    if (control_panels_stack_) {
      control_panels_stack_->setCurrentIndex(panel == ControlPanelKind::Tone ? 0 : 1);
    }

    const bool geometry_active = (panel == ControlPanelKind::Geometry);

    if (viewer_) {
      viewer_->SetCropOverlayRectNormalized(state_.crop_x_, state_.crop_y_, state_.crop_w_,
                                            state_.crop_h_);
      viewer_->SetCropOverlayRotationDegrees(state_.rotate_degrees_);
      viewer_->SetCropOverlayVisible(geometry_active);
      viewer_->SetCropToolEnabled(geometry_active);
    }
    RefreshGeometryModeUi();
    RefreshPanelSwitchUi();
    if (pipeline_initialized_) {
      RequestRender();
    }
  }

  void PromoteColorTempToCustomForEditing() {
    if (state_.color_temp_mode_ == ColorTempMode::CUSTOM) {
      return;
    }
    state_.color_temp_custom_cct_  = DisplayedColorTempCct(state_);
    state_.color_temp_custom_tint_ = DisplayedColorTempTint(state_);
    state_.color_temp_mode_        = ColorTempMode::CUSTOM;

    const bool prev_sync           = syncing_controls_;
    syncing_controls_              = true;
    if (color_temp_mode_combo_) {
      color_temp_mode_combo_->setCurrentIndex(ColorTempModeToComboIndex(state_.color_temp_mode_));
    }
    syncing_controls_ = prev_sync;
  }

  // Returns true if any resolved color temp value actually changed.
  auto RefreshColorTempRuntimeStateFromGlobalParams() -> bool {
    if (!pipeline_guard_ || !pipeline_guard_->pipeline_) {
      return false;
    }

    const auto& global = pipeline_guard_->pipeline_->GetGlobalParams();
    const float new_cct  = std::clamp(global.color_temp_resolved_cct_,
                                      static_cast<float>(kColorTempCctMin),
                                      static_cast<float>(kColorTempCctMax));
    const float new_tint = std::clamp(global.color_temp_resolved_tint_,
                                      static_cast<float>(kColorTempTintMin),
                                      static_cast<float>(kColorTempTintMax));
    const bool  new_sup  = global.color_temp_matrices_valid_;

    const bool changed = !NearlyEqual(state_.color_temp_resolved_cct_, new_cct) ||
                         !NearlyEqual(state_.color_temp_resolved_tint_, new_tint) ||
                         state_.color_temp_supported_ != new_sup;

    state_.color_temp_resolved_cct_  = new_cct;
    state_.color_temp_resolved_tint_ = new_tint;
    state_.color_temp_supported_     = new_sup;

    committed_state_.color_temp_resolved_cct_  = new_cct;
    committed_state_.color_temp_resolved_tint_ = new_tint;
    committed_state_.color_temp_supported_     = new_sup;

    return changed;
  }

  void SyncColorTempControlsFromState() {
    const bool prev_sync = syncing_controls_;
    syncing_controls_    = true;

    if (color_temp_mode_combo_) {
      color_temp_mode_combo_->setCurrentIndex(ColorTempModeToComboIndex(state_.color_temp_mode_));
    }
    if (color_temp_cct_slider_) {
      color_temp_cct_slider_->setValue(ColorTempCctToSliderPos(DisplayedColorTempCct(state_)));
      color_temp_cct_slider_->setEnabled(state_.color_temp_supported_);
    }
    if (color_temp_tint_slider_) {
      color_temp_tint_slider_->setValue(
          static_cast<int>(std::lround(DisplayedColorTempTint(state_))));
      color_temp_tint_slider_->setEnabled(state_.color_temp_supported_);
    }
    if (color_temp_unsupported_label_) {
      color_temp_unsupported_label_->setVisible(!state_.color_temp_supported_);
    }

    syncing_controls_ = prev_sync;
  }

  struct ColorTempRequestSnapshot {
    ColorTempMode mode_ = ColorTempMode::AS_SHOT;
    float         cct_  = 6500.0f;
    float         tint_ = 0.0f;
  };

  static auto BuildColorTempRequest(const AdjustmentState& state) -> ColorTempRequestSnapshot {
    return {state.color_temp_mode_, state.color_temp_custom_cct_, state.color_temp_custom_tint_};
  }

  static auto ColorTempRequestEqual(const ColorTempRequestSnapshot& a,
                                    const ColorTempRequestSnapshot& b) -> bool {
    return a.mode_ == b.mode_ && std::abs(a.cct_ - b.cct_) <= kColorTempRequestEpsilon &&
           std::abs(a.tint_ - b.tint_) <= kColorTempRequestEpsilon;
  }

  static void CopyFieldState(AdjustmentField field, const AdjustmentState& from, AdjustmentState& to) {
    switch (field) {
      case AdjustmentField::Exposure:
        to.exposure_ = from.exposure_;
        return;
      case AdjustmentField::Contrast:
        to.contrast_ = from.contrast_;
        return;
      case AdjustmentField::Saturation:
        to.saturation_ = from.saturation_;
        return;
      case AdjustmentField::ColorTemp:
        to.color_temp_mode_          = from.color_temp_mode_;
        to.color_temp_custom_cct_    = from.color_temp_custom_cct_;
        to.color_temp_custom_tint_   = from.color_temp_custom_tint_;
        to.color_temp_resolved_cct_  = from.color_temp_resolved_cct_;
        to.color_temp_resolved_tint_ = from.color_temp_resolved_tint_;
        to.color_temp_supported_     = from.color_temp_supported_;
        return;
      case AdjustmentField::Hls:
        to.hls_target_hue_              = from.hls_target_hue_;
        to.hls_hue_adjust_              = from.hls_hue_adjust_;
        to.hls_lightness_adjust_        = from.hls_lightness_adjust_;
        to.hls_saturation_adjust_       = from.hls_saturation_adjust_;
        to.hls_hue_range_               = from.hls_hue_range_;
        to.hls_hue_adjust_table_        = from.hls_hue_adjust_table_;
        to.hls_lightness_adjust_table_  = from.hls_lightness_adjust_table_;
        to.hls_saturation_adjust_table_ = from.hls_saturation_adjust_table_;
        to.hls_hue_range_table_         = from.hls_hue_range_table_;
        return;
      case AdjustmentField::Blacks:
        to.blacks_ = from.blacks_;
        return;
      case AdjustmentField::Whites:
        to.whites_ = from.whites_;
        return;
      case AdjustmentField::Shadows:
        to.shadows_ = from.shadows_;
        return;
      case AdjustmentField::Highlights:
        to.highlights_ = from.highlights_;
        return;
      case AdjustmentField::Curve:
        to.curve_points_ = from.curve_points_;
        return;
      case AdjustmentField::Sharpen:
        to.sharpen_ = from.sharpen_;
        return;
      case AdjustmentField::Clarity:
        to.clarity_ = from.clarity_;
        return;
      case AdjustmentField::Lut:
        to.lut_path_ = from.lut_path_;
        return;
      case AdjustmentField::CropRotate:
        to.rotate_degrees_     = from.rotate_degrees_;
        to.crop_enabled_       = from.crop_enabled_;
        to.crop_x_             = from.crop_x_;
        to.crop_y_             = from.crop_y_;
        to.crop_w_             = from.crop_w_;
        to.crop_h_             = from.crop_h_;
        to.crop_expand_to_fit_ = from.crop_expand_to_fit_;
        return;
    }
  }

  void RefreshVersionLogSelectionStyles() {
    if (!version_log_) {
      return;
    }
    for (int i = 0; i < version_log_->count(); ++i) {
      auto* item = version_log_->item(i);
      if (!item) {
        continue;
      }
      auto* w = version_log_->itemWidget(item);
      if (!w) {
        continue;
      }
      if (auto* card = dynamic_cast<HistoryCardWidget*>(w)) {
        card->SetSelected(item->isSelected());
      }
    }
  }

  void TriggerQualityPreviewRenderFromPipeline() {
    if (quality_preview_timer_ && quality_preview_timer_->isActive()) {
      quality_preview_timer_->stop();
    }
    state_.type_ = RenderType::FULL_RES_PREVIEW;
    RequestRenderWithoutApplyingState();
    state_.type_ = RenderType::FAST_PREVIEW;
  }

  void SyncControlsFromState() {
    if (!controls_) {
      return;
    }

    syncing_controls_ = true;
    LoadActiveHlsProfile(state_);

    if (lut_combo_) {
      int lut_index = 0;
      if (!state_.lut_path_.empty()) {
        auto it = std::find(lut_paths_.begin(), lut_paths_.end(), state_.lut_path_);
        if (it == lut_paths_.end()) {
          lut_paths_.push_back(state_.lut_path_);
          lut_names_.push_back(
              QString::fromStdString(std::filesystem::path(state_.lut_path_).filename().string()));
          lut_combo_->addItem(lut_names_.back());
          lut_index = static_cast<int>(lut_paths_.size() - 1);
        } else {
          lut_index = static_cast<int>(std::distance(lut_paths_.begin(), it));
        }
      }
      lut_combo_->setCurrentIndex(lut_index);
    }

    if (exposure_slider_) {
      exposure_slider_->setValue(static_cast<int>(std::lround(state_.exposure_ * 100.0f)));
    }
    if (contrast_slider_) {
      contrast_slider_->setValue(static_cast<int>(std::lround(state_.contrast_)));
    }
    if (saturation_slider_) {
      saturation_slider_->setValue(static_cast<int>(std::lround(state_.saturation_)));
    }
    if (color_temp_mode_combo_) {
      color_temp_mode_combo_->setCurrentIndex(ColorTempModeToComboIndex(state_.color_temp_mode_));
    }
    if (color_temp_cct_slider_) {
      color_temp_cct_slider_->setValue(ColorTempCctToSliderPos(DisplayedColorTempCct(state_)));
    }
    if (color_temp_tint_slider_) {
      color_temp_tint_slider_->setValue(
          static_cast<int>(std::lround(DisplayedColorTempTint(state_))));
    }
    if (color_temp_cct_slider_) {
      color_temp_cct_slider_->setEnabled(state_.color_temp_supported_);
    }
    if (color_temp_tint_slider_) {
      color_temp_tint_slider_->setEnabled(state_.color_temp_supported_);
    }
    if (color_temp_unsupported_label_) {
      color_temp_unsupported_label_->setVisible(!state_.color_temp_supported_);
    }
    if (hls_hue_adjust_slider_) {
      hls_hue_adjust_slider_->setValue(static_cast<int>(std::lround(state_.hls_hue_adjust_)));
    }
    if (hls_lightness_adjust_slider_) {
      hls_lightness_adjust_slider_->setValue(
          static_cast<int>(std::lround(state_.hls_lightness_adjust_)));
    }
    if (hls_saturation_adjust_slider_) {
      hls_saturation_adjust_slider_->setValue(
          static_cast<int>(std::lround(state_.hls_saturation_adjust_)));
    }
    if (hls_hue_range_slider_) {
      hls_hue_range_slider_->setValue(static_cast<int>(std::lround(state_.hls_hue_range_)));
    }
    if (blacks_slider_) {
      blacks_slider_->setValue(static_cast<int>(std::lround(state_.blacks_)));
    }
    if (whites_slider_) {
      whites_slider_->setValue(static_cast<int>(std::lround(state_.whites_)));
    }
    if (shadows_slider_) {
      shadows_slider_->setValue(static_cast<int>(std::lround(state_.shadows_)));
    }
    if (highlights_slider_) {
      highlights_slider_->setValue(static_cast<int>(std::lround(state_.highlights_)));
    }
    if (sharpen_slider_) {
      sharpen_slider_->setValue(static_cast<int>(std::lround(state_.sharpen_)));
    }
    if (clarity_slider_) {
      clarity_slider_->setValue(static_cast<int>(std::lround(state_.clarity_)));
    }
    if (rotate_slider_) {
      rotate_slider_->setValue(
          static_cast<int>(std::lround(state_.rotate_degrees_ * kRotationSliderScale)));
    }
    if (geometry_crop_x_slider_) {
      geometry_crop_x_slider_->setValue(static_cast<int>(std::lround(state_.crop_x_ * kCropRectSliderScale)));
    }
    if (geometry_crop_y_slider_) {
      geometry_crop_y_slider_->setValue(static_cast<int>(std::lround(state_.crop_y_ * kCropRectSliderScale)));
    }
    if (geometry_crop_w_slider_) {
      geometry_crop_w_slider_->setValue(static_cast<int>(std::lround(state_.crop_w_ * kCropRectSliderScale)));
    }
    if (geometry_crop_h_slider_) {
      geometry_crop_h_slider_->setValue(static_cast<int>(std::lround(state_.crop_h_ * kCropRectSliderScale)));
    }
    if (curve_widget_) {
      curve_widget_->SetControlPoints(state_.curve_points_);
    }
    UpdateGeometryCropRectLabel();
    RefreshGeometryModeUi();
    if (viewer_) {
      viewer_->SetCropOverlayRectNormalized(state_.crop_x_, state_.crop_y_, state_.crop_w_,
                                            state_.crop_h_);
      viewer_->SetCropOverlayRotationDegrees(state_.rotate_degrees_);
      const bool geometry_active = (active_panel_ == ControlPanelKind::Geometry);
      viewer_->SetCropOverlayVisible(geometry_active);
      viewer_->SetCropToolEnabled(geometry_active);
    }
    RefreshHlsTargetUi();

    syncing_controls_ = false;
  }

  auto ReconstructPipelineParamsForVersion(Version& version) -> std::optional<nlohmann::json> {
    if (const auto snapshot = version.GetFinalPipelineParams(); snapshot.has_value()) {
      return snapshot;
    }

    if (!history_guard_ || !history_guard_->history_) {
      return std::nullopt;
    }

    std::vector<Version*> lineage;
    lineage.push_back(&version);
    while (lineage.back()->HasParentVersion()) {
      try {
        lineage.push_back(
            &history_guard_->history_->GetVersion(lineage.back()->GetParentVersionID()));
      } catch (...) {
        return std::nullopt;
      }
    }
    std::reverse(lineage.begin(), lineage.end());

    auto   replay_exec = std::make_shared<CPUPipelineExecutor>();

    size_t replay_from = 0;
    for (size_t i = lineage.size(); i > 0; --i) {
      if (const auto snapshot = lineage[i - 1]->GetFinalPipelineParams(); snapshot.has_value()) {
        replay_exec->ImportPipelineParams(*snapshot);
        replay_exec->SetExecutionStages();
        replay_from = i;
        break;
      }
    }

    for (size_t i = replay_from; i < lineage.size(); ++i) {
      const auto& txs = lineage[i]->GetAllEditTransactions();
      for (auto it = txs.rbegin(); it != txs.rend(); ++it) {
        (void)it->ApplyTransaction(*replay_exec);
      }
    }
    return replay_exec->ExportPipelineParams();
  }

  auto ReloadUiStateFromPipeline(bool reset_to_defaults_if_missing) -> bool {
    const bool loaded = LoadStateFromPipelineIfPresent();
    if (!loaded && !reset_to_defaults_if_missing) {
      return false;
    }
    if (!loaded) {
      state_ = AdjustmentState{};
      last_submitted_color_temp_request_.reset();
    } else {
      last_submitted_color_temp_request_ = BuildColorTempRequest(state_);
    }
    committed_state_ = state_;
    SyncControlsFromState();
    TriggerQualityPreviewRenderFromPipeline();
    return true;
  }

  auto ApplyPipelineParamsToEditor(const nlohmann::json& params) -> bool {
    if (!pipeline_guard_ || !pipeline_guard_->pipeline_) {
      return false;
    }

    auto exec = pipeline_guard_->pipeline_;
    exec->ImportPipelineParams(params);
    exec->SetExecutionStages(viewer_);
    pipeline_guard_->dirty_ = true;
    last_applied_lut_path_.clear();

    return ReloadUiStateFromPipeline(/*reset_to_defaults_if_missing=*/true);
  }

  auto ReloadEditorFromHistoryVersion(Version& version, QString* error) -> bool {
    const auto selected_params = ReconstructPipelineParamsForVersion(version);
    if (!selected_params.has_value()) {
      if (error) {
        *error = "Could not reconstruct pipeline params for the selected version.";
      }
      return false;
    }

    if (!ApplyPipelineParamsToEditor(*selected_params)) {
      if (error) {
        *error = "Failed to apply selected version to the editor.";
      }
      return false;
    }
    return true;
  }

  void CheckoutSelectedVersion(QListWidgetItem* item) {
    if (!item || !history_guard_ || !history_guard_->history_) {
      return;
    }

    const auto version_id_str = item->data(Qt::UserRole).toString().toStdString();
    if (version_id_str.empty()) {
      return;
    }

    Hash128 version_id{};
    try {
      version_id = Hash128::FromString(version_id_str);
    } catch (const std::exception& e) {
      QMessageBox::warning(this, "History", QString("Invalid version ID: %1").arg(e.what()));
      return;
    }

    Version* selected_version = nullptr;
    try {
      selected_version = &history_guard_->history_->GetVersion(version_id);
    } catch (const std::exception& e) {
      QMessageBox::warning(this, "History",
                           QString("Failed to load selected version: %1").arg(e.what()));
      return;
    }

    QString reload_error;
    if (!ReloadEditorFromHistoryVersion(*selected_version, &reload_error)) {
      QMessageBox::warning(this, "History", reload_error);
      return;
    }

    if (CurrentWorkingMode() == WorkingMode::Plain) {
      working_version_ = Version(element_id_);
    } else {
      working_version_ = Version(element_id_, version_id);
    }
    working_version_.SetBasePipelineExecutor(pipeline_guard_->pipeline_);
    UpdateVersionUi();
  }

  void UndoLastTransaction() {
    if (!pipeline_guard_ || !pipeline_guard_->pipeline_) {
      return;
    }

    if (working_version_.GetAllEditTransactions().empty()) {
      QMessageBox::information(this, "History", "No transaction to undo.");
      return;
    }

    EditTransaction last_tx{TransactionType::_EDIT, OperatorType::UNKNOWN,
                            PipelineStageName::Basic_Adjustment, nlohmann::json::object()};
    try {
      last_tx = working_version_.RemoveLastEditTransaction();
    } catch (const std::exception& e) {
      QMessageBox::warning(this, "History", QString("Undo failed: %1").arg(e.what()));
      return;
    }

    auto            exec = pipeline_guard_->pipeline_;

    // Requirement (1): issue a delete transaction for the latest edit.
    EditTransaction undo_delete_tx(TransactionType::_DELETE, last_tx.GetTxOperatorType(),
                                   last_tx.GetTxOpStageName(), nlohmann::json::object());
    if (const auto prev = last_tx.GetLastOperatorParams(); prev.has_value()) {
      undo_delete_tx.SetLastOperatorParams(*prev);
    }
    (void)undo_delete_tx.ApplyTransaction(*exec);

    // Re-apply the latest previous transaction for the same operator if it exists.
    // This keeps undo fully in transaction space without resetting the whole pipeline.
    bool restored_from_stack = false;
    for (const auto& tx : working_version_.GetAllEditTransactions()) {
      if (tx.GetTxOpStageName() == last_tx.GetTxOpStageName() &&
          tx.GetTxOperatorType() == last_tx.GetTxOperatorType()) {
        (void)tx.ApplyTransaction(*exec);
        restored_from_stack = true;
        break;
      }
    }

    // No same-operator tx left in the working stack: restore from the popped tx's last params.
    if (!restored_from_stack) {
      if (const auto prev = last_tx.GetLastOperatorParams();
          prev.has_value() && prev->is_object() && !prev->empty()) {
        EditTransaction restore_tx(TransactionType::_EDIT, last_tx.GetTxOperatorType(),
                                   last_tx.GetTxOpStageName(), *prev);
        (void)restore_tx.ApplyTransaction(*exec);
      }
    }

    pipeline_guard_->dirty_ = true;
    if (!ReloadUiStateFromPipeline(/*reset_to_defaults_if_missing=*/false)) {
      QMessageBox::warning(this, "History", "Undo failed while reloading pipeline state.");
      return;
    }
    UpdateVersionUi();
  }

  void UpdateVersionUi() {
    if (!version_status_ || !commit_version_btn_) {
      return;
    }

    const size_t tx_count = working_version_.GetAllEditTransactions().size();
    QString      label    = QString("Uncommitted: %1 tx").arg(static_cast<qulonglong>(tx_count));

    if (working_version_.HasParentVersion()) {
      label += QString(" | parent: %1")
                   .arg(QString::fromStdString(
                       working_version_.GetParentVersionID().ToString().substr(0, 8)));
    } else {
      label += " | plain";
    }

    if (working_mode_combo_) {
      const auto mode = static_cast<WorkingMode>(working_mode_combo_->currentData().toInt());
      label += (mode == WorkingMode::Plain) ? " | mode: plain" : " | mode: incremental";
    }

    if (history_guard_ && history_guard_->history_) {
      try {
        const auto latest_id = history_guard_->history_->GetLatestVersion().ver_ref_.GetVersionID();
        label +=
            QString(" | Latest: %1").arg(QString::fromStdString(latest_id.ToString().substr(0, 8)));
      } catch (...) {
      }
    }

    version_status_->setText(label);
    version_status_->setToolTip(label);
    commit_version_btn_->setEnabled(tx_count > 0);
    if (undo_tx_btn_) {
      undo_tx_btn_->setEnabled(tx_count > 0);
    }

    if (tx_stack_) {
      tx_stack_->clear();
      const auto&  txs   = working_version_.GetAllEditTransactions();
      const size_t total = txs.size();
      size_t       i     = 0;
      for (const auto& tx : txs) {
        const QString title = QString::fromStdString(tx.Describe(true, 110));

        auto*         item  = new QListWidgetItem(tx_stack_);
        item->setToolTip(QString::fromStdString(tx.ToJSON().dump(2)));
        item->setSizeHint(QSize(0, 58));

        auto* card = new HistoryCardWidget(tx_stack_);
        auto* row  = new QHBoxLayout(card);
        row->setContentsMargins(10, 8, 10, 8);
        row->setSpacing(10);

        const QColor dot  = QColor(0xFC, 0xC7, 0x04);
        const QColor line = QColor(0x2A, 0x2A, 0x2A);
        auto*        lane = new HistoryLaneWidget(dot, line, /*draw_top*/ i > 0,
                                                  /*draw_bottom*/ (i + 1) < total, card);
        row->addWidget(lane, 0);

        auto* body = new QVBoxLayout();
        body->setContentsMargins(0, 0, 0, 0);
        body->setSpacing(2);

        auto* title_l = new QLabel(title, card);
        title_l->setWordWrap(true);
        title_l->setStyleSheet(
            "QLabel {"
            "  color: #E6E6E6;"
            "  font-size: 12px;"
            "  font-weight: 500;"
            "}");

        auto* meta_l =
            new QLabel(QString("uncommitted | #%1").arg(static_cast<qulonglong>(i + 1)), card);
        meta_l->setStyleSheet(
            "QLabel {"
            "  color: #A3A3A3;"
            "  font-size: 11px;"
            "}");

        body->addWidget(title_l);
        body->addWidget(meta_l);
        row->addLayout(body, 1);

        tx_stack_->setItemWidget(item, card);
        ++i;
      }
    }

    if (version_log_) {
      QString prev_selected_id;
      if (auto* cur = version_log_->currentItem()) {
        prev_selected_id = cur->data(Qt::UserRole).toString();
      }

      version_log_->clear();
      if (history_guard_ && history_guard_->history_) {
        const auto& tree = history_guard_->history_->GetCommitTree();
        Hash128     latest_id{};
        try {
          latest_id = history_guard_->history_->GetLatestVersion().ver_ref_.GetVersionID();
        } catch (...) {
        }

        const Hash128 base_parent = working_version_.GetParentVersionID();

        int           row_index   = 0;
        const int     total_rows  = static_cast<int>(tree.size());

        for (auto it = tree.rbegin(); it != tree.rend(); ++it, ++row_index) {
          const auto& ver      = it->ver_ref_;
          const auto  ver_id   = ver.GetVersionID();
          const auto  short_id = QString::fromStdString(ver_id.ToString().substr(0, 8));
          const auto  when =
              QDateTime::fromSecsSinceEpoch(static_cast<qint64>(ver.GetLastModifiedTime()))
                  .toString("yyyy-MM-dd HH:mm:ss");
          const auto committed_tx_count =
              static_cast<qulonglong>(ver.GetAllEditTransactions().size());

          QString     msg;
          const auto& txs = ver.GetAllEditTransactions();
          if (!txs.empty()) {
            msg = QString::fromStdString(txs.front().Describe(true, 70));
          } else {
            msg = "(empty)";
          }

          const bool is_head  = (ver_id == latest_id);
          const bool is_base  = (base_parent == ver_id && working_version_.HasParentVersion());
          const bool is_plain = !ver.HasParentVersion();

          auto*      item     = new QListWidgetItem(version_log_);
          item->setData(Qt::UserRole, QString::fromStdString(ver_id.ToString()));
          item->setToolTip(QString("version=%1\nparent=%2\ntx=%3")
                               .arg(QString::fromStdString(ver_id.ToString()))
                               .arg(QString::fromStdString(ver.GetParentVersionID().ToString()))
                               .arg(committed_tx_count));
          item->setSizeHint(QSize(0, 74));

          auto* card = new HistoryCardWidget(version_log_);
          auto* row  = new QHBoxLayout(card);
          row->setContentsMargins(10, 9, 10, 9);
          row->setSpacing(10);

          const QColor dot  = is_head
                                  ? QColor(0xFC, 0xC7, 0x04)
                                  : (is_base ? QColor(0xFC, 0xC7, 0x04) : QColor(0x8C, 0x8C, 0x8C));
          const QColor line = QColor(0x2A, 0x2A, 0x2A);
          auto*        lane = new HistoryLaneWidget(dot, line, /*draw_top*/ row_index > 0,
                                                    /*draw_bottom*/ (row_index + 1) < total_rows, card);
          row->addWidget(lane, 0);

          auto* body = new QVBoxLayout();
          body->setContentsMargins(0, 0, 0, 0);
          body->setSpacing(4);

          auto* top = new QHBoxLayout();
          top->setContentsMargins(0, 0, 0, 0);
          top->setSpacing(8);

          const QFont mono   = QFontDatabase::systemFont(QFontDatabase::FixedFont);
          auto*       hash_l = new QLabel(short_id, card);
          hash_l->setFont(mono);
          hash_l->setStyleSheet(
              "QLabel {"
              "  color: #E6E6E6;"
              "  font-size: 12px;"
              "  font-weight: 600;"
              "}");

          top->addWidget(hash_l, 0);

          if (is_head) {
            top->addWidget(MakePillLabel("HEAD", "#121212", "rgba(252, 199, 4, 0.95)",
                                         "rgba(252, 199, 4, 0.95)", card),
                           0);
          }
          if (is_base) {
            top->addWidget(MakePillLabel("BASE", "#121212", "rgba(252, 199, 4, 0.88)",
                                         "rgba(252, 199, 4, 0.88)", card),
                           0);
          }
          if (is_plain) {
            top->addWidget(MakePillLabel("PLAIN", "#1A1A1A", "rgba(252, 199, 4, 0.22)",
                                         "rgba(252, 199, 4, 0.40)", card),
                           0);
          } else {
            const auto parent_short =
                QString::fromStdString(ver.GetParentVersionID().ToString().substr(0, 8));
            top->addWidget(
                MakePillLabel(QString("PARENT %1").arg(parent_short), "#A3A3A3",
                              "rgba(252, 199, 4, 0.16)", "rgba(252, 199, 4, 0.32)", card),
                0);
          }

          top->addStretch(1);

          auto* tx_pill = MakePillLabel(QString("tx %1").arg(committed_tx_count), "#A3A3A3",
                                        "rgba(252, 199, 4, 0.16)", "rgba(252, 199, 4, 0.32)", card);
          top->addWidget(tx_pill, 0);

          auto* msg_l = new QLabel(msg, card);
          msg_l->setWordWrap(true);
          msg_l->setStyleSheet(
              "QLabel {"
              "  color: #E6E6E6;"
              "  font-size: 12px;"
              "}");

          auto* meta_l = new QLabel(when, card);
          meta_l->setStyleSheet(
              "QLabel {"
              "  color: #A3A3A3;"
              "  font-size: 11px;"
              "}");

          body->addLayout(top);
          body->addWidget(msg_l);
          body->addWidget(meta_l);
          row->addLayout(body, 1);

          version_log_->setItemWidget(item, card);

          const QString ver_id_str = QString::fromStdString(ver_id.ToString());
          if (!prev_selected_id.isEmpty() && ver_id_str == prev_selected_id) {
            version_log_->setCurrentItem(item);
            item->setSelected(true);
          } else if (prev_selected_id.isEmpty() && is_head) {
            version_log_->setCurrentItem(item);
            item->setSelected(true);
          }
        }
      }
      RefreshVersionLogSelectionStyles();
    }
  }

  void CommitWorkingVersion() {
    if (!history_service_ || !history_guard_ || !history_guard_->history_) {
      QMessageBox::warning(this, "History", "Edit history service not available.");
      return;
    }

    const size_t tx_count = working_version_.GetAllEditTransactions().size();
    if (tx_count == 0) {
      QMessageBox::information(this, "History", "No uncommitted transactions.");
      return;
    }

    history_id_t committed_id{};
    try {
      if (pipeline_guard_ && pipeline_guard_->pipeline_) {
        working_version_.SetFinalPipelineParams(pipeline_guard_->pipeline_->ExportPipelineParams());
      }
      committed_id = history_service_->CommitVersion(history_guard_, std::move(working_version_));
    } catch (const std::exception& e) {
      QMessageBox::warning(this, "History", QString("Commit failed: %1").arg(e.what()));
      working_version_ = Version(element_id_);
      working_version_.SetBasePipelineExecutor(pipeline_guard_->pipeline_);
      UpdateVersionUi();
      return;
    }

    // Start a fresh working version chained from the committed one.
    StartNewWorkingVersionFromCommit(committed_id);
    UpdateVersionUi();
  }

  auto CurrentWorkingMode() const -> WorkingMode {
    if (!working_mode_combo_) {
      return WorkingMode::Incremental;
    }
    return static_cast<WorkingMode>(working_mode_combo_->currentData().toInt());
  }

  void StartNewWorkingVersionFromUi() {
    if (CurrentWorkingMode() == WorkingMode::Plain) {
      working_version_ = Version(element_id_);
      working_version_.SetBasePipelineExecutor(pipeline_guard_->pipeline_);
      UpdateVersionUi();
      return;
    }

    // Incremental: seed from latest committed version (if any).
    try {
      if (history_guard_ && history_guard_->history_) {
        const auto parent_id = history_guard_->history_->GetLatestVersion().ver_ref_.GetVersionID();
        working_version_     = Version(element_id_, parent_id);
      } else {
        working_version_ = Version(element_id_);
      }
    } catch (...) {
      working_version_ = Version(element_id_);
    }
    working_version_.SetBasePipelineExecutor(pipeline_guard_->pipeline_);
    UpdateVersionUi();
  }

  void StartNewWorkingVersionFromCommit(const Hash128& committed_id) {
    if (CurrentWorkingMode() == WorkingMode::Plain) {
      working_version_ = Version(element_id_);
    } else {
      working_version_ = Version(element_id_, committed_id);
    }
    working_version_.SetBasePipelineExecutor(pipeline_guard_->pipeline_);
  }

  std::pair<PipelineStageName, OperatorType> FieldSpec(AdjustmentField field) const {
    switch (field) {
      case AdjustmentField::Exposure:
        return {PipelineStageName::Basic_Adjustment, OperatorType::EXPOSURE};
      case AdjustmentField::Contrast:
        return {PipelineStageName::Basic_Adjustment, OperatorType::CONTRAST};
      case AdjustmentField::Saturation:
        return {PipelineStageName::Color_Adjustment, OperatorType::SATURATION};
      case AdjustmentField::ColorTemp:
        return {PipelineStageName::To_WorkingSpace, OperatorType::COLOR_TEMP};
      case AdjustmentField::Hls:
        return {PipelineStageName::Color_Adjustment, OperatorType::HLS};
      case AdjustmentField::Blacks:
        return {PipelineStageName::Basic_Adjustment, OperatorType::BLACK};
      case AdjustmentField::Whites:
        return {PipelineStageName::Basic_Adjustment, OperatorType::WHITE};
      case AdjustmentField::Shadows:
        return {PipelineStageName::Basic_Adjustment, OperatorType::SHADOWS};
      case AdjustmentField::Highlights:
        return {PipelineStageName::Basic_Adjustment, OperatorType::HIGHLIGHTS};
      case AdjustmentField::Curve:
        return {PipelineStageName::Basic_Adjustment, OperatorType::CURVE};
      case AdjustmentField::Sharpen:
        return {PipelineStageName::Detail_Adjustment, OperatorType::SHARPEN};
      case AdjustmentField::Clarity:
        return {PipelineStageName::Detail_Adjustment, OperatorType::CLARITY};
      case AdjustmentField::Lut:
        return {PipelineStageName::Color_Adjustment, OperatorType::LMT};
      case AdjustmentField::CropRotate:
        return {PipelineStageName::Geometry_Adjustment, OperatorType::CROP_ROTATE};
    }
    return {PipelineStageName::Basic_Adjustment, OperatorType::EXPOSURE};
  }

  nlohmann::json ParamsForField(AdjustmentField field, const AdjustmentState& s) const {
    switch (field) {
      case AdjustmentField::Exposure:
        return {{"exposure", s.exposure_}};
      case AdjustmentField::Contrast:
        return {{"contrast", s.contrast_}};
      case AdjustmentField::Saturation:
        return {{"saturation", s.saturation_}};
      case AdjustmentField::ColorTemp:
        return {{"color_temp",
                 {{"mode", ColorTempModeToString(s.color_temp_mode_)},
                  {"cct", std::clamp(s.color_temp_custom_cct_,
                                     static_cast<float>(kColorTempCctMin),
                                     static_cast<float>(kColorTempCctMax))},
                  {"tint", std::clamp(s.color_temp_custom_tint_,
                                      static_cast<float>(kColorTempTintMin),
                                      static_cast<float>(kColorTempTintMax))},
                  {"resolved_cct", std::clamp(s.color_temp_resolved_cct_,
                                              static_cast<float>(kColorTempCctMin),
                                              static_cast<float>(kColorTempCctMax))},
                  {"resolved_tint", std::clamp(s.color_temp_resolved_tint_,
                                               static_cast<float>(kColorTempTintMin),
                                               static_cast<float>(kColorTempTintMax))}}}};
      case AdjustmentField::Hls: {
        nlohmann::json hue_bins      = nlohmann::json::array();
        nlohmann::json hls_adj_table = nlohmann::json::array();
        nlohmann::json h_range_table = nlohmann::json::array();
        for (size_t i = 0; i < kHlsCandidateHues.size(); ++i) {
          hue_bins.push_back(kHlsCandidateHues[i]);
          hls_adj_table.push_back(std::array<float, 3>{
              std::clamp(s.hls_hue_adjust_table_[i], -kHlsMaxHueShiftDegrees,
                         kHlsMaxHueShiftDegrees),
              std::clamp(s.hls_lightness_adjust_table_[i], kHlsAdjUiMin, kHlsAdjUiMax) /
                  kHlsAdjUiToParamScale,
              std::clamp(s.hls_saturation_adjust_table_[i], kHlsAdjUiMin, kHlsAdjUiMax) /
                  kHlsAdjUiToParamScale});
          h_range_table.push_back(std::max(s.hls_hue_range_table_[i], 1.0f));
        }
        const int active_idx = ActiveHlsProfileIndex(s);

        return {{"HLS",
                 {{"hue_bins", std::move(hue_bins)},
                  {"hls_adj_table", std::move(hls_adj_table)},
                  {"h_range_table", std::move(h_range_table)},
                  {"target_hls",
                   std::array<float, 3>{WrapHueDegrees(s.hls_target_hue_), kHlsFixedTargetLightness,
                                        kHlsFixedTargetSaturation}},
                  {"hls_adj",
                   std::array<float, 3>{std::clamp(s.hls_hue_adjust_table_[active_idx],
                                                   -kHlsMaxHueShiftDegrees, kHlsMaxHueShiftDegrees),
                                        std::clamp(s.hls_lightness_adjust_table_[active_idx],
                                                   kHlsAdjUiMin, kHlsAdjUiMax) /
                                            kHlsAdjUiToParamScale,
                                        std::clamp(s.hls_saturation_adjust_table_[active_idx],
                                                   kHlsAdjUiMin, kHlsAdjUiMax) /
                                            kHlsAdjUiToParamScale}},
                  {"h_range", std::max(s.hls_hue_range_table_[active_idx], 1.0f)},
                  {"l_range", kHlsFixedLightnessRange},
                  {"s_range", kHlsFixedSaturationRange}}}};
      }
      case AdjustmentField::Blacks:
        return {{"black", s.blacks_}};
      case AdjustmentField::Whites:
        return {{"white", s.whites_}};
      case AdjustmentField::Shadows:
        return {{"shadows", s.shadows_}};
      case AdjustmentField::Highlights:
        return {{"highlights", s.highlights_}};
      case AdjustmentField::Curve:
        return CurveControlPointsToParams(s.curve_points_);
      case AdjustmentField::Sharpen:
        return {{"sharpen", {{"offset", s.sharpen_}}}};
      case AdjustmentField::Clarity:
        return {{"clarity", s.clarity_}};
      case AdjustmentField::Lut:
        return {{"ocio_lmt", s.lut_path_}};
      case AdjustmentField::CropRotate: {
        const auto crop_rect = ClampCropRect(s.crop_x_, s.crop_y_, s.crop_w_, s.crop_h_);
        const bool has_rotation = std::abs(s.rotate_degrees_) > 1e-4f;
        const bool has_crop = s.crop_enabled_ &&
                              (std::abs(crop_rect[0]) > 1e-4f || std::abs(crop_rect[1]) > 1e-4f ||
                               std::abs(crop_rect[2] - 1.0f) > 1e-4f ||
                               std::abs(crop_rect[3] - 1.0f) > 1e-4f);
        return {{"crop_rotate",
                 {{"enabled", has_rotation || has_crop},
                  {"angle_degrees", s.rotate_degrees_},
                  {"enable_crop", s.crop_enabled_},
                  {"crop_rect",
                   {{"x", crop_rect[0]},
                    {"y", crop_rect[1]},
                    {"w", crop_rect[2]},
                    {"h", crop_rect[3]}}},
                  {"expand_to_fit", s.crop_expand_to_fit_}}}};
      }
    }
    return {};
  }

  bool FieldChanged(AdjustmentField field) const {
    switch (field) {
      case AdjustmentField::Exposure:
        return !NearlyEqual(state_.exposure_, committed_state_.exposure_);
      case AdjustmentField::Contrast:
        return !NearlyEqual(state_.contrast_, committed_state_.contrast_);
      case AdjustmentField::Saturation:
        return !NearlyEqual(state_.saturation_, committed_state_.saturation_);
      case AdjustmentField::ColorTemp:
        return state_.color_temp_mode_ != committed_state_.color_temp_mode_ ||
               !NearlyEqual(state_.color_temp_custom_cct_, committed_state_.color_temp_custom_cct_) ||
               !NearlyEqual(state_.color_temp_custom_tint_,
                            committed_state_.color_temp_custom_tint_);
      case AdjustmentField::Hls:
        for (size_t i = 0; i < kHlsCandidateHues.size(); ++i) {
          if (!NearlyEqual(state_.hls_hue_adjust_table_[i],
                           committed_state_.hls_hue_adjust_table_[i]) ||
              !NearlyEqual(state_.hls_lightness_adjust_table_[i],
                           committed_state_.hls_lightness_adjust_table_[i]) ||
              !NearlyEqual(state_.hls_saturation_adjust_table_[i],
                           committed_state_.hls_saturation_adjust_table_[i]) ||
              !NearlyEqual(state_.hls_hue_range_table_[i],
                           committed_state_.hls_hue_range_table_[i])) {
            return true;
          }
        }
        return false;
      case AdjustmentField::Blacks:
        return !NearlyEqual(state_.blacks_, committed_state_.blacks_);
      case AdjustmentField::Whites:
        return !NearlyEqual(state_.whites_, committed_state_.whites_);
      case AdjustmentField::Shadows:
        return !NearlyEqual(state_.shadows_, committed_state_.shadows_);
      case AdjustmentField::Highlights:
        return !NearlyEqual(state_.highlights_, committed_state_.highlights_);
      case AdjustmentField::Curve:
        return !CurveControlPointsEqual(state_.curve_points_, committed_state_.curve_points_);
      case AdjustmentField::Sharpen:
        return !NearlyEqual(state_.sharpen_, committed_state_.sharpen_);
      case AdjustmentField::Clarity:
        return !NearlyEqual(state_.clarity_, committed_state_.clarity_);
      case AdjustmentField::Lut:
        return state_.lut_path_ != committed_state_.lut_path_;
      case AdjustmentField::CropRotate: {
        const auto state_rect = ClampCropRect(state_.crop_x_, state_.crop_y_, state_.crop_w_, state_.crop_h_);
        const auto committed_rect = ClampCropRect(committed_state_.crop_x_, committed_state_.crop_y_,
                                                  committed_state_.crop_w_, committed_state_.crop_h_);
        return !NearlyEqual(state_.rotate_degrees_, committed_state_.rotate_degrees_) ||
               state_.crop_enabled_ != committed_state_.crop_enabled_ ||
               state_.crop_expand_to_fit_ != committed_state_.crop_expand_to_fit_ ||
               !NearlyEqual(state_rect[0], committed_rect[0]) ||
               !NearlyEqual(state_rect[1], committed_rect[1]) ||
               !NearlyEqual(state_rect[2], committed_rect[2]) ||
               !NearlyEqual(state_rect[3], committed_rect[3]);
      }
    }
    return false;
  }

  void CommitAdjustment(AdjustmentField field) {
    if (!FieldChanged(field) || !pipeline_guard_ || !pipeline_guard_->pipeline_) {
      // Still fulfill the "full res on release/change" behavior.
      ScheduleQualityPreviewRenderFromPipeline();
      return;
    }

    const auto [stage_name, op_type] = FieldSpec(field);
    const auto            old_params = ParamsForField(field, committed_state_);
    const auto            new_params = ParamsForField(field, state_);

    auto                  exec       = pipeline_guard_->pipeline_;
    auto&                 stage      = exec->GetStage(stage_name);
    const auto            op         = stage.GetOperator(op_type);
    const TransactionType tx_type =
        (op.has_value() && op.value() != nullptr) ? TransactionType::_EDIT : TransactionType::_ADD;

    EditTransaction tx{tx_type, op_type, stage_name, new_params};
    tx.SetLastOperatorParams(old_params);
    (void)tx.ApplyTransaction(*exec);

    working_version_.AppendEditTransaction(std::move(tx));
    pipeline_guard_->dirty_ = true;

    CopyFieldState(field, state_, committed_state_);
    UpdateVersionUi();

    ScheduleQualityPreviewRenderFromPipeline();
  }

  bool LoadStateFromPipelineIfPresent() {
    auto exec = pipeline_guard_ ? pipeline_guard_->pipeline_ : nullptr;
    if (!exec) {
      return false;
    }

    AdjustmentState loaded_state = state_;
    loaded_state.type_           = state_.type_;
    loaded_state.rotate_degrees_ = 0.0f;
    loaded_state.crop_enabled_   = true;
    loaded_state.crop_x_         = 0.0f;
    loaded_state.crop_y_         = 0.0f;
    loaded_state.crop_w_         = 1.0f;
    loaded_state.crop_h_         = 1.0f;
    loaded_state.crop_expand_to_fit_ = true;
    bool has_loaded_any          = false;

    auto IsOperatorEnabled       = [](const PipelineStage& stage,
                                OperatorType         type) -> std::optional<bool> {
      const auto op = stage.GetOperator(type);
      if (!op.has_value() || op.value() == nullptr) {
        return std::nullopt;
      }
      const auto j = op.value()->ExportOperatorParams();
      if (!j.contains("enable")) {
        return true;
      }
      try {
        return j["enable"].get<bool>();
      } catch (...) {
        return std::nullopt;
      }
    };

    auto ReadFloat = [](const PipelineStage& stage, OperatorType type,
                        const char* key) -> std::optional<float> {
      const auto op = stage.GetOperator(type);
      if (!op.has_value() || op.value() == nullptr) {
        return std::nullopt;
      }
      const auto j = op.value()->ExportOperatorParams();
      if (j.contains("enable") && !j["enable"].get<bool>()) {
        return std::nullopt;
      }
      if (!j.contains("params")) {
        return std::nullopt;
      }
      const auto& params = j["params"];
      if (!params.contains(key)) {
        return std::nullopt;
      }
      try {
        return params[key].get<float>();
      } catch (...) {
        return std::nullopt;
      }
    };

    auto ReadNestedFloat = [](const PipelineStage& stage, OperatorType type, const char* key1,
                              const char* key2) -> std::optional<float> {
      const auto op = stage.GetOperator(type);
      if (!op.has_value() || op.value() == nullptr) {
        return std::nullopt;
      }
      const auto j = op.value()->ExportOperatorParams();
      if (j.contains("enable") && !j["enable"].get<bool>()) {
        return std::nullopt;
      }
      if (!j.contains("params")) {
        return std::nullopt;
      }
      const auto& params = j["params"];
      if (!params.contains(key1)) {
        return std::nullopt;
      }
      const auto& inner = params[key1];
      if (!inner.contains(key2)) {
        return std::nullopt;
      }
      try {
        return inner[key2].get<float>();
      } catch (...) {
        return std::nullopt;
      }
    };

    auto ReadNestedObject = [](const PipelineStage& stage, OperatorType type,
                               const char* key) -> std::optional<nlohmann::json> {
      const auto op = stage.GetOperator(type);
      if (!op.has_value() || op.value() == nullptr) {
        return std::nullopt;
      }
      const auto j = op.value()->ExportOperatorParams();
      if (!j.contains("params")) {
        return std::nullopt;
      }
      const auto& params = j["params"];
      if (!params.contains(key) || !params[key].is_object()) {
        return std::nullopt;
      }
      return params[key];
    };

    auto ReadString = [](const PipelineStage& stage, OperatorType type,
                         const char* key) -> std::optional<std::string> {
      const auto op = stage.GetOperator(type);
      if (!op.has_value() || op.value() == nullptr) {
        return std::nullopt;
      }
      const auto j = op.value()->ExportOperatorParams();
      if (j.contains("enable") && !j["enable"].get<bool>()) {
        return std::nullopt;
      }
      if (!j.contains("params")) {
        return std::nullopt;
      }
      const auto& params = j["params"];
      if (!params.contains(key)) {
        return std::nullopt;
      }
      try {
        return params[key].get<std::string>();
      } catch (...) {
        return std::nullopt;
      }
    };

    auto ReadCurvePoints = [](const PipelineStage& stage,
                              OperatorType         type) -> std::optional<std::vector<QPointF>> {
      const auto op = stage.GetOperator(type);
      if (!op.has_value() || op.value() == nullptr) {
        return std::nullopt;
      }
      const auto j = op.value()->ExportOperatorParams();
      if (j.contains("enable")) {
        try {
          if (!j["enable"].get<bool>()) {
            return DefaultCurveControlPoints();
          }
        } catch (...) {
        }
      }
      if (!j.contains("params")) {
        return std::nullopt;
      }
      return ParseCurveControlPointsFromParams(j["params"]);
    };

    const auto& geometry = exec->GetStage(PipelineStageName::Geometry_Adjustment);
    const auto& to_ws    = exec->GetStage(PipelineStageName::To_WorkingSpace);
    const auto& basic    = exec->GetStage(PipelineStageName::Basic_Adjustment);
    const auto& color    = exec->GetStage(PipelineStageName::Color_Adjustment);
    const auto& detail   = exec->GetStage(PipelineStageName::Detail_Adjustment);

    if (const auto v = ReadFloat(basic, OperatorType::EXPOSURE, "exposure"); v.has_value()) {
      loaded_state.exposure_ = v.value();
      has_loaded_any         = true;
    }
    if (const auto v = ReadFloat(basic, OperatorType::CONTRAST, "contrast"); v.has_value()) {
      loaded_state.contrast_ = v.value();
      has_loaded_any         = true;
    }

    // Read tonal controls from global params to avoid operator-param representation drift.
    const auto black_enabled = IsOperatorEnabled(basic, OperatorType::BLACK);
    if (black_enabled.has_value() && black_enabled.value()) {
      loaded_state.blacks_ = exec->GetGlobalParams().black_point_ * kBlackSliderFromGlobalScale;
      has_loaded_any       = true;
    } else if (const auto v = ReadFloat(basic, OperatorType::BLACK, "black"); v.has_value()) {
      loaded_state.blacks_ = v.value();
      has_loaded_any       = true;
    }

    const auto white_enabled = IsOperatorEnabled(basic, OperatorType::WHITE);
    if (white_enabled.has_value() && white_enabled.value()) {
      loaded_state.whites_ =
          (exec->GetGlobalParams().white_point_ - 1.0f) * kWhiteSliderFromGlobalScale;
      has_loaded_any = true;
    } else if (const auto v = ReadFloat(basic, OperatorType::WHITE, "white"); v.has_value()) {
      loaded_state.whites_ = v.value();
      has_loaded_any       = true;
    }

    const auto shadows_enabled = IsOperatorEnabled(basic, OperatorType::SHADOWS);
    if (shadows_enabled.has_value() && shadows_enabled.value()) {
      loaded_state.shadows_ =
          exec->GetGlobalParams().shadows_offset_ * kShadowsSliderFromGlobalScale;
      has_loaded_any = true;
    } else if (const auto v = ReadFloat(basic, OperatorType::SHADOWS, "shadows"); v.has_value()) {
      loaded_state.shadows_ = v.value();
      has_loaded_any        = true;
    }

    const auto highlights_enabled = IsOperatorEnabled(basic, OperatorType::HIGHLIGHTS);
    if (highlights_enabled.has_value() && highlights_enabled.value()) {
      loaded_state.highlights_ =
          exec->GetGlobalParams().highlights_offset_ * kHighlightsSliderFromGlobalScale;
      has_loaded_any = true;
    } else if (const auto v = ReadFloat(basic, OperatorType::HIGHLIGHTS, "highlights");
               v.has_value()) {
      loaded_state.highlights_ = v.value();
      has_loaded_any           = true;
    }

    if (const auto curve_points = ReadCurvePoints(basic, OperatorType::CURVE);
        curve_points.has_value()) {
      loaded_state.curve_points_ = NormalizeCurveControlPoints(*curve_points);
      has_loaded_any             = true;
    }

    if (const auto v = ReadFloat(color, OperatorType::SATURATION, "saturation"); v.has_value()) {
      loaded_state.saturation_ = v.value();
      has_loaded_any           = true;
    }
    if (const auto color_temp_json =
            ReadNestedObject(to_ws, OperatorType::COLOR_TEMP, "color_temp");
        color_temp_json.has_value()) {
      const auto& color_temp = *color_temp_json;
      if (color_temp.contains("mode") && color_temp["mode"].is_string()) {
        loaded_state.color_temp_mode_ = ParseColorTempMode(color_temp["mode"].get<std::string>());
      }
      if (color_temp.contains("cct")) {
        try {
          loaded_state.color_temp_custom_cct_ =
              std::clamp(color_temp["cct"].get<float>(), static_cast<float>(kColorTempCctMin),
                         static_cast<float>(kColorTempCctMax));
        } catch (...) {
        }
      }
      if (color_temp.contains("tint")) {
        try {
          loaded_state.color_temp_custom_tint_ =
              std::clamp(color_temp["tint"].get<float>(), static_cast<float>(kColorTempTintMin),
                         static_cast<float>(kColorTempTintMax));
        } catch (...) {
        }
      }
      if (color_temp.contains("resolved_cct")) {
        try {
          loaded_state.color_temp_resolved_cct_ = std::clamp(
              color_temp["resolved_cct"].get<float>(), static_cast<float>(kColorTempCctMin),
              static_cast<float>(kColorTempCctMax));
        } catch (...) {
        }
      }
      if (color_temp.contains("resolved_tint")) {
        try {
          loaded_state.color_temp_resolved_tint_ = std::clamp(
              color_temp["resolved_tint"].get<float>(), static_cast<float>(kColorTempTintMin),
              static_cast<float>(kColorTempTintMax));
        } catch (...) {
        }
      }
      has_loaded_any     = true;
    }
    if (const auto hls_json = ReadNestedObject(color, OperatorType::HLS, "HLS");
        hls_json.has_value()) {
      auto ReadArray3 = [](const nlohmann::json& obj, const char* key,
                           std::array<float, 3>& out) -> bool {
        if (!obj.contains(key) || !obj[key].is_array() || obj[key].size() < 3) {
          return false;
        }
        try {
          out[0] = obj[key][0].get<float>();
          out[1] = obj[key][1].get<float>();
          out[2] = obj[key][2].get<float>();
          return true;
        } catch (...) {
          return false;
        }
      };

      const auto& hls = *hls_json;
      loaded_state.hls_hue_adjust_table_.fill(0.0f);
      loaded_state.hls_lightness_adjust_table_.fill(0.0f);
      loaded_state.hls_saturation_adjust_table_.fill(0.0f);
      loaded_state.hls_hue_range_table_  = MakeHlsFilledArray(kHlsDefaultHueRange);
      std::array<float, 3> target_hls    = {loaded_state.hls_target_hue_, kHlsFixedTargetLightness,
                                            kHlsFixedTargetSaturation};
      std::array<float, 3> hls_adj       = {};
      bool                 has_adj_table = false;
      bool                 has_range_table = false;

      if (hls.contains("hls_adj_table") && hls["hls_adj_table"].is_array()) {
        const auto& adj_tbl  = hls["hls_adj_table"];
        const bool  has_bins = hls.contains("hue_bins") && hls["hue_bins"].is_array();
        for (int i = 0; i < static_cast<int>(adj_tbl.size()); ++i) {
          if (!adj_tbl[i].is_array() || adj_tbl[i].size() < 3) {
            continue;
          }
          int idx = i;
          if (has_bins && i < static_cast<int>(hls["hue_bins"].size())) {
            try {
              idx = ClosestHlsCandidateHueIndex(hls["hue_bins"][i].get<float>());
            } catch (...) {
            }
          }
          if (idx < 0 || idx >= static_cast<int>(kHlsCandidateHues.size())) {
            continue;
          }
          try {
            loaded_state.hls_hue_adjust_table_[idx] = std::clamp(
                adj_tbl[i][0].get<float>(), -kHlsMaxHueShiftDegrees, kHlsMaxHueShiftDegrees);
            loaded_state.hls_lightness_adjust_table_[idx] = std::clamp(
                adj_tbl[i][1].get<float>() * kHlsAdjUiToParamScale, kHlsAdjUiMin, kHlsAdjUiMax);
            loaded_state.hls_saturation_adjust_table_[idx] = std::clamp(
                adj_tbl[i][2].get<float>() * kHlsAdjUiToParamScale, kHlsAdjUiMin, kHlsAdjUiMax);
            has_adj_table = true;
          } catch (...) {
          }
        }
      }

      if (hls.contains("h_range_table") && hls["h_range_table"].is_array()) {
        const auto& range_tbl = hls["h_range_table"];
        const bool  has_bins  = hls.contains("hue_bins") && hls["hue_bins"].is_array();
        for (int i = 0; i < static_cast<int>(range_tbl.size()); ++i) {
          int idx = i;
          if (has_bins && i < static_cast<int>(hls["hue_bins"].size())) {
            try {
              idx = ClosestHlsCandidateHueIndex(hls["hue_bins"][i].get<float>());
            } catch (...) {
            }
          }
          if (idx < 0 || idx >= static_cast<int>(kHlsCandidateHues.size())) {
            continue;
          }
          try {
            loaded_state.hls_hue_range_table_[idx] =
                std::clamp(range_tbl[i].get<float>(), 1.0f, 180.0f);
            has_range_table = true;
          } catch (...) {
          }
        }
      }

      (void)ReadArray3(hls, "target_hls", target_hls);
      (void)ReadArray3(hls, "hls_adj", hls_adj);

      loaded_state.hls_target_hue_ = WrapHueDegrees(target_hls[0]);
      const int active_idx         = ActiveHlsProfileIndex(loaded_state);
      loaded_state.hls_target_hue_ = kHlsCandidateHues[static_cast<size_t>(active_idx)];
      if (!has_adj_table) {
        loaded_state.hls_hue_adjust_table_[active_idx] =
            std::clamp(hls_adj[0], -kHlsMaxHueShiftDegrees, kHlsMaxHueShiftDegrees);
        loaded_state.hls_lightness_adjust_table_[active_idx] =
            std::clamp(hls_adj[1] * kHlsAdjUiToParamScale, kHlsAdjUiMin, kHlsAdjUiMax);
        loaded_state.hls_saturation_adjust_table_[active_idx] =
            std::clamp(hls_adj[2] * kHlsAdjUiToParamScale, kHlsAdjUiMin, kHlsAdjUiMax);
      }

      if (!has_range_table && hls.contains("h_range")) {
        try {
          loaded_state.hls_hue_range_table_[active_idx] =
              std::clamp(hls["h_range"].get<float>(), 1.0f, 180.0f);
        } catch (...) {
        }
      }
      LoadActiveHlsProfile(loaded_state);
      has_loaded_any = true;
    }

    if (const auto v = ReadNestedFloat(detail, OperatorType::SHARPEN, "sharpen", "offset");
        v.has_value()) {
      loaded_state.sharpen_ = v.value();
      has_loaded_any        = true;
    }
    if (const auto v = ReadFloat(detail, OperatorType::CLARITY, "clarity"); v.has_value()) {
      loaded_state.clarity_ = v.value();
      has_loaded_any        = true;
    }

    if (const auto crop_rotate_json =
            ReadNestedObject(geometry, OperatorType::CROP_ROTATE, "crop_rotate");
        crop_rotate_json.has_value()) {
      const auto& crop_rotate   = *crop_rotate_json;
      loaded_state.rotate_degrees_ = crop_rotate.value("angle_degrees", loaded_state.rotate_degrees_);
      loaded_state.crop_enabled_   = crop_rotate.value("enable_crop", loaded_state.crop_enabled_);
      loaded_state.crop_expand_to_fit_ =
          crop_rotate.value("expand_to_fit", loaded_state.crop_expand_to_fit_);
      bool has_non_full_crop_rect = false;
      if (crop_rotate.contains("crop_rect") && crop_rotate["crop_rect"].is_object()) {
        const auto& crop_rect = crop_rotate["crop_rect"];
        const auto clamped = ClampCropRect(crop_rect.value("x", loaded_state.crop_x_),
                                           crop_rect.value("y", loaded_state.crop_y_),
                                           crop_rect.value("w", loaded_state.crop_w_),
                                           crop_rect.value("h", loaded_state.crop_h_));
        loaded_state.crop_x_ = clamped[0];
        loaded_state.crop_y_ = clamped[1];
        loaded_state.crop_w_ = clamped[2];
        loaded_state.crop_h_ = clamped[3];
        has_non_full_crop_rect = std::abs(loaded_state.crop_x_) > 1e-4f ||
                                 std::abs(loaded_state.crop_y_) > 1e-4f ||
                                 std::abs(loaded_state.crop_w_ - 1.0f) > 1e-4f ||
                                 std::abs(loaded_state.crop_h_ - 1.0f) > 1e-4f;
      }
      loaded_state.crop_enabled_ = loaded_state.crop_enabled_ || has_non_full_crop_rect;
      has_loaded_any = true;
    }

    const auto lut = ReadString(color, OperatorType::LMT, "ocio_lmt");
    if (lut.has_value()) {
      loaded_state.lut_path_ = *lut;
      has_loaded_any         = true;
    } else if (const auto lmt_enabled = IsOperatorEnabled(color, OperatorType::LMT);
               lmt_enabled.has_value() && !lmt_enabled.value()) {
      loaded_state.lut_path_.clear();
      has_loaded_any = true;
    }

    loaded_state.color_temp_resolved_cct_ = std::clamp(
        exec->GetGlobalParams().color_temp_resolved_cct_, static_cast<float>(kColorTempCctMin),
        static_cast<float>(kColorTempCctMax));
    loaded_state.color_temp_resolved_tint_ = std::clamp(
        exec->GetGlobalParams().color_temp_resolved_tint_, static_cast<float>(kColorTempTintMin),
        static_cast<float>(kColorTempTintMax));
    loaded_state.color_temp_supported_ = exec->GetGlobalParams().color_temp_matrices_valid_;

    if (!has_loaded_any) {
      return false;
    }

    state_ = loaded_state;
    last_submitted_color_temp_request_ = BuildColorTempRequest(state_);
    return true;
  }

  void SetupPipeline() {
    auto img_desc = image_pool_->Read<std::shared_ptr<Image>>(
        image_id_, [](const std::shared_ptr<Image>& img) { return img; });
    auto bytes = ByteBufferLoader::LoadFromImage(img_desc);
    if (!bytes) {
      throw std::runtime_error("EditorDialog: failed to load image bytes");
    }

    base_task_.input_             = std::make_shared<ImageBuffer>(std::move(*bytes));
    base_task_.pipeline_executor_ = pipeline_guard_->pipeline_;

    auto           exec           = pipeline_guard_->pipeline_;
    // exec->SetPreviewMode(true);

    // auto& global_params = exec->GetGlobalParams();
    auto&          loading        = exec->GetStage(PipelineStageName::Image_Loading);
    nlohmann::json decode_params;
#ifdef HAVE_CUDA
    decode_params["raw"]["cuda"] = true;
#else
    decode_params["raw"]["cuda"] = false;
#endif
    decode_params["raw"]["highlights_reconstruct"] = true;
    decode_params["raw"]["use_camera_wb"]          = true;
    decode_params["raw"]["user_wb"]                = 7600.f;
    decode_params["raw"]["backend"]                = "puerh";
    loading.SetOperator(OperatorType::RAW_DECODE, decode_params);

    // auto& basic         = exec->GetStage(PipelineStageName::Basic_Adjustment);
    // basic.SetOperator(OperatorType::EXPOSURE, {{"exposure", 0.0f}}, global_params);
    // basic.SetOperator(OperatorType::CONTRAST, {{"contrast", 1.0f}}, global_params);
    // basic.SetOperator(OperatorType::BLACK, {{"black", 0.0f}}, global_params);
    // basic.SetOperator(OperatorType::WHITE, {{"white", 0.0f}}, global_params);
    // basic.SetOperator(OperatorType::SHADOWS, {{"shadows", 0.0f}}, global_params);
    // basic.SetOperator(OperatorType::HIGHLIGHTS, {{"highlights", 0.0f}}, global_params);

    // auto& color = exec->GetStage(PipelineStageName::Color_Adjustment);
    // color.SetOperator(OperatorType::SATURATION, {{"saturation", 0.0f}}, global_params);

    // auto& detail = exec->GetStage(PipelineStageName::Detail_Adjustment);
    // detail.SetOperator(OperatorType::SHARPEN, {{"sharpen", {{"offset", 0.0f}}}}, global_params);
    // detail.SetOperator(OperatorType::CLARITY, {{"clarity", 0.0f}}, global_params);

    exec->SetExecutionStages(viewer_);

    // Cached pipelines can clear transient GPU resources when returned to the service.
    // PipelineMgmtService now resyncs global params on load, so we no longer need a
    // per-dialog LMT rebind hack here.
    last_applied_lut_path_.clear();
  }

  void ApplyStateToPipeline(const AdjustmentState& render_state) {
    auto  exec          = pipeline_guard_->pipeline_;
    auto& global_params = exec->GetGlobalParams();
    auto& geometry      = exec->GetStage(PipelineStageName::Geometry_Adjustment);
    auto& to_ws         = exec->GetStage(PipelineStageName::To_WorkingSpace);

    const auto color_temp_request = BuildColorTempRequest(render_state);
    const bool color_temp_missing = !to_ws.GetOperator(OperatorType::COLOR_TEMP).has_value();
    if (color_temp_missing || !last_submitted_color_temp_request_.has_value() ||
        !ColorTempRequestEqual(*last_submitted_color_temp_request_, color_temp_request)) {
      to_ws.SetOperator(OperatorType::COLOR_TEMP,
                        ParamsForField(AdjustmentField::ColorTemp, render_state), global_params);
      to_ws.EnableOperator(OperatorType::COLOR_TEMP, true, global_params);
      last_submitted_color_temp_request_ = color_temp_request;
    } else {
      to_ws.EnableOperator(OperatorType::COLOR_TEMP, true, global_params);
    }

    // Geometry editing is overlay-only. While the geometry panel is active,
    // render the full pre-geometry frame so recropping can always expand back
    // to the original image bounds.
    nlohmann::json crop_rotate_params;
    bool           apply_crop = committed_state_.crop_enabled_;
    if (active_panel_ == ControlPanelKind::Geometry) {
      crop_rotate_params = {{"crop_rotate",
                             {{"enabled", false},
                              {"angle_degrees", 0.0f},
                              {"enable_crop", false},
                              {"crop_rect", {{"x", 0.0f}, {"y", 0.0f}, {"w", 1.0f}, {"h", 1.0f}}},
                              {"expand_to_fit", committed_state_.crop_expand_to_fit_}}}};
      apply_crop = false;
    } else {
      crop_rotate_params = ParamsForField(AdjustmentField::CropRotate, committed_state_);
    }

    crop_rotate_params["crop_rotate"]["enable_crop"] = apply_crop;
    const bool geometry_enabled = crop_rotate_params["crop_rotate"].value("enabled", false);
    geometry.SetOperator(OperatorType::CROP_ROTATE, crop_rotate_params, global_params);
    geometry.EnableOperator(OperatorType::CROP_ROTATE, geometry_enabled, global_params);

    auto& basic         = exec->GetStage(PipelineStageName::Basic_Adjustment);
    basic.SetOperator(OperatorType::EXPOSURE, {{"exposure", render_state.exposure_}}, global_params);
    basic.SetOperator(OperatorType::CONTRAST, {{"contrast", render_state.contrast_}}, global_params);
    basic.SetOperator(OperatorType::BLACK, {{"black", render_state.blacks_}}, global_params);
    basic.SetOperator(OperatorType::WHITE, {{"white", render_state.whites_}}, global_params);
    basic.SetOperator(OperatorType::SHADOWS, {{"shadows", render_state.shadows_}}, global_params);
    basic.SetOperator(OperatorType::HIGHLIGHTS, {{"highlights", render_state.highlights_}},
                      global_params);
    basic.SetOperator(OperatorType::CURVE, CurveControlPointsToParams(render_state.curve_points_),
                      global_params);

    auto& color = exec->GetStage(PipelineStageName::Color_Adjustment);
    color.SetOperator(OperatorType::SATURATION, {{"saturation", render_state.saturation_}},
                      global_params);
    color.EnableOperator(OperatorType::TINT, false, global_params);
    color.SetOperator(OperatorType::HLS, ParamsForField(AdjustmentField::Hls, render_state),
                      global_params);
    color.EnableOperator(OperatorType::HLS, true, global_params);

    // LUT (LMT): rebind only when the path changes. The operator's SetGlobalParams now
    // derives lmt_enabled_/dirty state from the path, and PipelineMgmtService resyncs on load.
    if (render_state.lut_path_ != last_applied_lut_path_) {
      color.SetOperator(OperatorType::LMT, {{"ocio_lmt", render_state.lut_path_}}, global_params);
      last_applied_lut_path_ = render_state.lut_path_;
    }

    auto& detail = exec->GetStage(PipelineStageName::Detail_Adjustment);
    detail.SetOperator(OperatorType::SHARPEN, {{"sharpen", {{"offset", render_state.sharpen_}}}},
                       global_params);
    detail.SetOperator(OperatorType::CLARITY, {{"clarity", render_state.clarity_}}, global_params);
  }

  static constexpr std::chrono::milliseconds kFastPreviewMinSubmitInterval{16};
  static constexpr std::chrono::milliseconds kQualityPreviewDebounceInterval{180};

  void EnsureQualityPreviewTimer() {
    if (quality_preview_timer_) {
      return;
    }
    quality_preview_timer_ = new QTimer(this);
    quality_preview_timer_->setSingleShot(true);
    QObject::connect(quality_preview_timer_, &QTimer::timeout, this,
                     [this]() { TriggerQualityPreviewRenderFromPipeline(); });
  }

  void ScheduleQualityPreviewRenderFromPipeline() {
    EnsureQualityPreviewTimer();
    quality_preview_timer_->start(static_cast<int>(kQualityPreviewDebounceInterval.count()));
  }

  auto CanSubmitFastPreviewNow() const -> bool {
    if (last_fast_preview_submit_time_.time_since_epoch().count() == 0) {
      return true;
    }
    const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - last_fast_preview_submit_time_);
    return elapsed >= kFastPreviewMinSubmitInterval;
  }

  void EnsureFastPreviewSubmitTimer() {
    if (fast_preview_submit_timer_) {
      return;
    }
    fast_preview_submit_timer_ = new QTimer(this);
    fast_preview_submit_timer_->setSingleShot(true);
    QObject::connect(fast_preview_submit_timer_, &QTimer::timeout, this, [this]() {
      if (!inflight_) {
        StartNext();
      }
    });
  }

  void ArmFastPreviewSubmitTimer() {
    EnsureFastPreviewSubmitTimer();

    const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - last_fast_preview_submit_time_);
    auto       delay   = kFastPreviewMinSubmitInterval - elapsed;
    if (delay <= 0ms) {
      delay = 1ms;
    }
    const int delay_ms = static_cast<int>(delay.count());

    if (!fast_preview_submit_timer_->isActive()) {
      fast_preview_submit_timer_->start(delay_ms);
      return;
    }

    const int current_remaining = fast_preview_submit_timer_->remainingTime();
    if (current_remaining < 0 || delay_ms < current_remaining) {
      fast_preview_submit_timer_->start(delay_ms);
    }
  }

  void EnqueueRenderRequest(const AdjustmentState& snapshot, bool apply_state) {
    PendingRenderRequest request{snapshot, apply_state};

    if (snapshot.type_ == RenderType::FAST_PREVIEW) {
      // Industry pattern for interactive rendering:
      // coalesce rapid slider updates and keep only the newest fast preview.
      if (quality_preview_timer_ && quality_preview_timer_->isActive()) {
        quality_preview_timer_->stop();
      }
      pending_fast_preview_request_ = std::move(request);
    } else {
      // Keep quality requests ordered and drop stale fast previews.
      pending_quality_render_requests_.push_back(std::move(request));
      pending_fast_preview_request_.reset();
      if (fast_preview_submit_timer_ && fast_preview_submit_timer_->isActive()) {
        fast_preview_submit_timer_->stop();
      }
    }

    if (!inflight_) {
      StartNext();
    }
  }

  void RequestRender() {
    EnqueueRenderRequest(state_, true);
  }

  void RequestRenderWithoutApplyingState() {
    EnqueueRenderRequest(state_, false);
  }

  void EnsurePollTimer() {
    if (poll_timer_) {
      return;
    }
    poll_timer_ = new QTimer(this);
    poll_timer_->setInterval(4);
    QObject::connect(poll_timer_, &QTimer::timeout, this, [this]() { PollInflight(); });
  }

  void PollInflight() {
    if (!inflight_future_.has_value()) {
      if (poll_timer_ && poll_timer_->isActive() && !inflight_) {
        poll_timer_->stop();
      }
      return;
    }

    if (inflight_future_->wait_for(0ms) != std::future_status::ready) {
      return;
    }

    try {
      (void)inflight_future_->get();
    } catch (...) {
    }
    inflight_future_.reset();
    OnRenderFinished();
  }

  void StartNext() {
    if (inflight_) {
      return;
    }

    std::optional<PendingRenderRequest> request;
    if (!pending_quality_render_requests_.empty()) {
      request = pending_quality_render_requests_.front();
      pending_quality_render_requests_.pop_front();
    } else if (pending_fast_preview_request_.has_value()) {
      if (!CanSubmitFastPreviewNow()) {
        ArmFastPreviewSubmitTimer();
        return;
      }
      request = pending_fast_preview_request_;
      pending_fast_preview_request_.reset();
      last_fast_preview_submit_time_ = std::chrono::steady_clock::now();
      if (fast_preview_submit_timer_ && fast_preview_submit_timer_->isActive()) {
        fast_preview_submit_timer_->stop();
      }
    }

    if (!request.has_value()) {
      return;
    }
    const PendingRenderRequest next_request = *request;

    if (spinner_) {
      spinner_->Start();
    }

    if (next_request.apply_state_) {
      ApplyStateToPipeline(next_request.state_);
      pipeline_guard_->dirty_ = true;
    }

    PipelineTask task                       = base_task_;
    task.options_.render_desc_.render_type_ = next_request.state_.type_;
    task.options_.is_callback_              = false;
    task.options_.is_seq_callback_          = false;
    task.options_.is_blocking_              = true;

    if (viewer_) {
      const auto render_type = task.options_.render_desc_.render_type_;
      viewer_->SetHistogramFrameExpected(render_type == RenderType::FAST_PREVIEW ||
                                         render_type == RenderType::FULL_RES_PREVIEW);
    }

    auto promise = std::make_shared<std::promise<std::shared_ptr<ImageBuffer>>>();
    auto fut     = promise->get_future();
    task.result_ = promise;

    inflight_    = true;
    scheduler_->ScheduleTask(std::move(task));

    inflight_future_ = std::move(fut);
    EnsurePollTimer();
    if (poll_timer_ && !poll_timer_->isActive()) {
      poll_timer_->start();
    }
  }

  void OnRenderFinished() {
    inflight_ = false;

    if (spinner_) {
      spinner_->Stop();
    }

    if (RefreshColorTempRuntimeStateFromGlobalParams()) {
      SyncColorTempControlsFromState();
    }

    if (!pending_quality_render_requests_.empty() || pending_fast_preview_request_.has_value()) {
      StartNext();
    } else if (poll_timer_ && poll_timer_->isActive()) {
      poll_timer_->stop();
    }
  }

  std::shared_ptr<ImagePoolService>                        image_pool_;
  std::shared_ptr<PipelineGuard>                           pipeline_guard_;
  std::shared_ptr<EditHistoryMgmtService>                  history_service_;
  std::shared_ptr<EditHistoryGuard>                        history_guard_;
  sl_element_id_t                                          element_id_ = 0;
  image_id_t                                               image_id_   = 0;

  std::shared_ptr<PipelineScheduler>                       scheduler_;
  PipelineTask                                             base_task_{};

  QtEditViewer*                                            viewer_                 = nullptr;
  QWidget*                                                 viewer_container_       = nullptr;
  QLabel*                                                  viewer_zoom_label_      = nullptr;
  QScrollArea*                                             controls_scroll_        = nullptr;
  QScrollArea*                                             tone_controls_scroll_   = nullptr;
  QScrollArea*                                             geometry_controls_scroll_ = nullptr;
  QStackedWidget*                                          control_panels_stack_   = nullptr;
  SpinnerWidget*                                           spinner_                = nullptr;
  QWidget*                                                 controls_               = nullptr;
  QWidget*                                                 tone_controls_          = nullptr;
  QWidget*                                                 geometry_controls_      = nullptr;
  QPushButton*                                             tone_panel_btn_         = nullptr;
  QPushButton*                                             geometry_panel_btn_     = nullptr;
  HistogramWidget*                                         histogram_widget_       = nullptr;
  HistogramRulerWidget*                                    histogram_ruler_widget_ = nullptr;
  QComboBox*                                               lut_combo_              = nullptr;
  QSlider*                                                 exposure_slider_        = nullptr;
  QSlider*                                                 contrast_slider_        = nullptr;
  QSlider*                                                 saturation_slider_      = nullptr;
  QComboBox*                                               color_temp_mode_combo_  = nullptr;
  QSlider*                                                 color_temp_cct_slider_  = nullptr;
  QSlider*                                                 color_temp_tint_slider_ = nullptr;
  QLabel*                                                  color_temp_unsupported_label_ = nullptr;
  QLabel*                                                  hls_target_label_       = nullptr;
  std::vector<QPushButton*>                                hls_candidate_buttons_{};
  QSlider*                                                 hls_hue_adjust_slider_        = nullptr;
  QSlider*                                                 hls_lightness_adjust_slider_  = nullptr;
  QSlider*                                                 hls_saturation_adjust_slider_ = nullptr;
  QSlider*                                                 hls_hue_range_slider_         = nullptr;
  QSlider*                                                 blacks_slider_                = nullptr;
  QSlider*                                                 whites_slider_                = nullptr;
  QSlider*                                                 shadows_slider_               = nullptr;
  QSlider*                                                 highlights_slider_            = nullptr;
  ToneCurveWidget*                                         curve_widget_                 = nullptr;
  QSlider*                                                 sharpen_slider_               = nullptr;
  QSlider*                                                 clarity_slider_               = nullptr;
  QSlider*                                                 rotate_slider_                = nullptr;
  QSlider*                                                 geometry_crop_x_slider_       = nullptr;
  QSlider*                                                 geometry_crop_y_slider_       = nullptr;
  QSlider*                                                 geometry_crop_w_slider_       = nullptr;
  QSlider*                                                 geometry_crop_h_slider_       = nullptr;
  QLabel*                                                  geometry_crop_rect_label_     = nullptr;
  QPushButton*                                             geometry_apply_btn_           = nullptr;
  QPushButton*                                             geometry_reset_btn_           = nullptr;
  QLabel*                                                  version_status_               = nullptr;
  QPushButton*                                             undo_tx_btn_                  = nullptr;
  QPushButton*                                             commit_version_btn_           = nullptr;
  QComboBox*                                               working_mode_combo_           = nullptr;
  QPushButton*                                             new_working_btn_              = nullptr;
  QListWidget*                                             version_log_                  = nullptr;
  QListWidget*                                             tx_stack_                     = nullptr;
  QTimer*                                                  poll_timer_                   = nullptr;
  std::optional<std::future<std::shared_ptr<ImageBuffer>>> inflight_future_{};

  std::vector<std::string>                                 lut_paths_{};
  QStringList                                              lut_names_{};

  std::string                                              last_applied_lut_path_{};
  std::optional<ColorTempRequestSnapshot>                  last_submitted_color_temp_request_{};
  AdjustmentState                                          state_{};
  AdjustmentState                                          committed_state_{};
  Version                                                  working_version_{};
  std::deque<PendingRenderRequest>                         pending_quality_render_requests_{};
  std::optional<PendingRenderRequest>                      pending_fast_preview_request_{};
  ControlPanelKind                                         active_panel_               = ControlPanelKind::Tone;
  bool                                                     pipeline_initialized_       = false;
  bool                                                     inflight_                   = false;
  QTimer*                                                  quality_preview_timer_      = nullptr;
  QTimer*                                                  fast_preview_submit_timer_  = nullptr;
  std::chrono::steady_clock::time_point                    last_fast_preview_submit_time_{};
  bool                                                     syncing_controls_           = false;
};
}  // namespace

auto OpenEditorDialog(std::shared_ptr<ImagePoolService>       image_pool,
                      std::shared_ptr<PipelineGuard>          pipeline_guard,
                      std::shared_ptr<EditHistoryMgmtService> history_service,
                      std::shared_ptr<EditHistoryGuard> history_guard, sl_element_id_t element_id,
                      image_id_t image_id, QWidget* parent) -> bool {
  EditorDialog dlg(std::move(image_pool), std::move(pipeline_guard), std::move(history_service),
                   std::move(history_guard), element_id, image_id, parent);
  dlg.exec();
  return true;
}

}  // namespace puerhlab::demo
