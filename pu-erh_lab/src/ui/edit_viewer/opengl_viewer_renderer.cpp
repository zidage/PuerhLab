//  Copyright 2026 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#include "opengl_viewer_renderer.hpp"

#include <GL/gl.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>

#include <QByteArray>
#include <QDebug>
#include <QOpenGLContext>
#include <QPainter>
#include <QPainterPath>
#include <QPen>
#include <QPolygonF>
#include <QSurfaceFormat>

#include <algorithm>
#include <vector>

namespace puerhlab {
namespace {

static const char* kVertexShaderSource = R"(
#version 330 core
layout(location = 0) in vec2 position;

uniform vec2 uScale;
uniform vec2 uPan;
uniform float uZoom;

out vec2 vTexCoord;

void main() {
  vec2 pos = position * uScale * uZoom + uPan;
  gl_Position = vec4(pos, 0.0, 1.0);

  vec2 uv = (position + 1.0) * 0.5;
  vTexCoord = vec2(uv.x, 1.0 - uv.y);
}
)";

static const char* kFragmentShaderSource = R"(
#version 330 core
uniform sampler2D textureSampler;
in vec2 vTexCoord;
out vec4 FragColor;
void main() {
  FragColor = texture(textureSampler, vTexCoord);
}
)";

static const char* kHistogramClearShaderSource = R"(
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

static const char* kHistogramComputeShaderSource = R"(
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

static const char* kHistogramNormalizeShaderSource = R"(
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

}  // namespace

void OpenGLViewerRenderer::Initialize() {
  initializeOpenGLFunctions();

  program_ = new QOpenGLShaderProgram();
  if (!program_->addShaderFromSourceCode(QOpenGLShader::Vertex, kVertexShaderSource)) {
    qWarning("Vertex shader compile failed: %s", program_->log().toUtf8().constData());
  }
  if (!program_->addShaderFromSourceCode(QOpenGLShader::Fragment, kFragmentShaderSource)) {
    qWarning("Fragment shader compile failed: %s", program_->log().toUtf8().constData());
  }
  if (!program_->link()) {
    qWarning("Shader program link failed: %s", program_->log().toUtf8().constData());
  }

  const float vertices[] = {-1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f};
  glGenBuffers(1, &vbo_);
  glBindBuffer(GL_ARRAY_BUFFER, vbo_);
  glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

  InitHistogramResources();
}

void OpenGLViewerRenderer::Shutdown() {
  FreeAllBuffers();
  FreeHistogramResources();
  if (vbo_) {
    glDeleteBuffers(1, &vbo_);
    vbo_ = 0;
  }
  delete program_;
  program_ = nullptr;
}

auto OpenGLViewerRenderer::EnsureSlot(int slot_index, int width, int height) -> bool {
  if (slot_index < 0 || slot_index >= static_cast<int>(buffers_.size())) {
    return false;
  }
  return InitBuffer(buffers_[slot_index], width, height);
}

auto OpenGLViewerRenderer::HasSlot(int slot_index, int width, int height) const -> bool {
  if (slot_index < 0 || slot_index >= static_cast<int>(buffers_.size())) {
    return false;
  }
  const auto& buffer = buffers_[slot_index];
  return buffer.texture != 0 && buffer.cuda_resource != nullptr && buffer.width == width &&
         buffer.height == height;
}

auto OpenGLViewerRenderer::UploadPendingFrame(const FrameMailbox::PendingFrame& pending_frame) -> bool {
  if (pending_frame.slot_index < 0 ||
      pending_frame.slot_index >= static_cast<int>(buffers_.size())) {
    return false;
  }

  GLBuffer& buffer = buffers_[pending_frame.slot_index];
  if (!buffer.cuda_resource || !pending_frame.staging_ptr || pending_frame.staging_bytes == 0) {
    return false;
  }

  const cudaError_t map_err = cudaGraphicsMapResources(1, &buffer.cuda_resource, 0);
  if (map_err != cudaSuccess) {
    qWarning("Failed to map CUDA resource (paintGL): %s", cudaGetErrorString(map_err));
    return false;
  }

  bool success = false;
  cudaArray_t mapped_array = nullptr;
  const cudaError_t array_err =
      cudaGraphicsSubResourceGetMappedArray(&mapped_array, buffer.cuda_resource, 0, 0);
  if (array_err != cudaSuccess || !mapped_array) {
    qWarning("Failed to map texture array (paintGL): %s", cudaGetErrorString(array_err));
  } else {
    const size_t row_bytes = static_cast<size_t>(buffer.width) * sizeof(float4);
    const size_t max_rows  = pending_frame.staging_bytes / row_bytes;
    const size_t copy_rows = std::min(max_rows, static_cast<size_t>(buffer.height));
    const cudaError_t copy_err =
        cudaMemcpy2DToArray(mapped_array, 0, 0, pending_frame.staging_ptr, row_bytes, row_bytes,
                            copy_rows, cudaMemcpyDeviceToDevice);
    if (copy_err != cudaSuccess) {
      qWarning("Failed to copy staging->texture: %s", cudaGetErrorString(copy_err));
    } else {
      success = true;
    }
  }

  const cudaError_t unmap_err = cudaGraphicsUnmapResources(1, &buffer.cuda_resource, 0);
  if (unmap_err != cudaSuccess) {
    qWarning("Failed to unmap CUDA resource (paintGL): %s", cudaGetErrorString(unmap_err));
    return false;
  }
  return success;
}

auto OpenGLViewerRenderer::Render(QOpenGLWidget& widget, const FrameMailbox::ActiveFrame& active_frame,
                                  const ViewerStateSnapshot& state_snapshot,
                                  bool histogram_requested) -> RenderResult {
  RenderResult result;
  if (active_frame.slot_index < 0 ||
      active_frame.slot_index >= static_cast<int>(buffers_.size())) {
    return result;
  }

  GLBuffer& active_buffer = buffers_[active_frame.slot_index];
  if (!active_buffer.texture || !program_ || !program_->isLinked()) {
    return result;
  }

  const ViewportWidgetInfo widget_info{
      widget.width(), widget.height(), static_cast<float>(widget.devicePixelRatioF())};
  const ViewportImageInfo  image_info{active_frame.width, active_frame.height};
  const auto               scale = ViewportMapper::ComputeLetterboxScale(widget_info, image_info);
  const float dpr = std::max(widget_info.device_pixel_ratio, 1e-4f);
  const float vw = std::max(1.0f, static_cast<float>(widget_info.widget_width) * dpr);
  const float vh = std::max(1.0f, static_cast<float>(widget_info.widget_height) * dpr);
  glViewport(0, 0, static_cast<int>(vw), static_cast<int>(vh));

  float     zoom = state_snapshot.view_transform.zoom;
  QVector2D pan  = state_snapshot.view_transform.pan;
  if (active_frame.presentation_mode == FramePresentationMode::RoiFrame) {
    zoom = 1.0f;
    pan  = QVector2D(0.0f, 0.0f);
  }

  glClearColor(0.f, 0.f, 0.f, 1.f);
  glClear(GL_COLOR_BUFFER_BIT);

  program_->bind();
  program_->setUniformValue("uScale", QVector2D(scale.x, scale.y));
  program_->setUniformValue("uPan", pan);
  program_->setUniformValue("uZoom", zoom);
  glActiveTexture(GL_TEXTURE0);
  program_->setUniformValue("textureSampler", 0);
  glBindTexture(GL_TEXTURE_2D, active_buffer.texture);
  glBindBuffer(GL_ARRAY_BUFFER, vbo_);
  const int pos_loc = program_->attributeLocation("position");
  program_->enableAttributeArray(pos_loc);
  program_->setAttributeBuffer(pos_loc, GL_FLOAT, 0, 2, 2 * sizeof(float));
  glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
  program_->release();

  if (state_snapshot.crop_overlay.overlay_visible) {
    const auto image_top_left_opt = ViewportMapper::ImageUvToWidgetPoint(
        QPointF(0.0, 0.0), widget_info, image_info, zoom, pan);
    const auto image_bottom_right_opt = ViewportMapper::ImageUvToWidgetPoint(
        QPointF(1.0, 1.0), widget_info, image_info, zoom, pan);
    const auto crop_corners_uv = CropGeometry::RotatedCropCornersUv(
        state_snapshot.crop_overlay.rect, state_snapshot.crop_overlay.rotation_degrees,
        state_snapshot.crop_overlay.metric_aspect);
    std::array<QPointF, 4> crop_corners_widget{};
    bool                   crop_corners_valid = true;
    for (size_t i = 0; i < crop_corners_uv.size(); ++i) {
      const auto corner_widget =
          ViewportMapper::ImageUvToWidgetPoint(crop_corners_uv[i], widget_info, image_info, zoom, pan);
      if (!corner_widget.has_value()) {
        crop_corners_valid = false;
        break;
      }
      crop_corners_widget[i] = *corner_widget;
    }

    if (crop_corners_valid && image_top_left_opt.has_value() && image_bottom_right_opt.has_value()) {
      const QRectF image_rect = QRectF(*image_top_left_opt, *image_bottom_right_opt).normalized();
      if (image_rect.isValid()) {
        const auto [rotate_stem_widget, rotate_handle_widget] =
            CropGeometry::CropRotateHandleWidgetPoint(crop_corners_widget);
        QPolygonF crop_polygon;
        crop_polygon.reserve(static_cast<int>(crop_corners_widget.size()));
        for (const auto& point : crop_corners_widget) {
          crop_polygon.push_back(point);
        }

        QPainter painter(&widget);
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
        painter.drawLine(rotate_stem_widget, rotate_handle_widget);

        painter.setPen(QPen(QColor(252, 199, 4, 150), 1.0, Qt::DashLine));
        for (const float t : {1.0f / 3.0f, 2.0f / 3.0f}) {
          painter.drawLine(CropGeometry::LerpPoint(crop_corners_widget[0], crop_corners_widget[1], t),
                           CropGeometry::LerpPoint(crop_corners_widget[3], crop_corners_widget[2], t));
          painter.drawLine(CropGeometry::LerpPoint(crop_corners_widget[0], crop_corners_widget[3], t),
                           CropGeometry::LerpPoint(crop_corners_widget[1], crop_corners_widget[2], t));
        }

        painter.setPen(QPen(QColor(18, 18, 18, 230), 1.0));
        painter.setBrush(QColor(252, 199, 4, 230));
        for (const auto& corner : crop_corners_widget) {
          painter.drawEllipse(corner, CropGeometry::kCropCornerDrawRadiusPx,
                              CropGeometry::kCropCornerDrawRadiusPx);
        }
        painter.setBrush(QColor(252, 199, 4, 245));
        painter.drawEllipse(rotate_handle_widget, CropGeometry::kCropRotateHandleDrawRadiusPx,
                            CropGeometry::kCropRotateHandleDrawRadiusPx);
      }
    }
  }

  if (histogram_requested && ShouldComputeHistogramNow()) {
    if ((histogram_resources_ready_ || InitHistogramResources()) &&
        ComputeHistogram(active_buffer.texture, active_buffer.width, active_buffer.height)) {
      histogram_has_data_           = true;
      result.histogram_data_updated = true;
    }
  }

  return result;
}

void OpenGLViewerRenderer::SetHistogramUpdateIntervalMs(int interval_ms) {
  histogram_update_interval_ms_ = std::max(0, interval_ms);
}

auto OpenGLViewerRenderer::GetHistogramBufferId() const -> GLuint {
  return histogram_resources_ready_ ? histogram_norm_ssbo_ : 0;
}

auto OpenGLViewerRenderer::GetHistogramBinCount() const -> int { return kHistogramBins; }

auto OpenGLViewerRenderer::HasHistogramData() const -> bool { return histogram_has_data_; }

bool OpenGLViewerRenderer::InitBuffer(GLBuffer& buffer, int width, int height) {
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

  const cudaError_t err =
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

void OpenGLViewerRenderer::FreeBuffer(GLBuffer& buffer) {
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

void OpenGLViewerRenderer::FreeAllBuffers() {
  for (auto& buffer : buffers_) {
    FreeBuffer(buffer);
  }
}

auto OpenGLViewerRenderer::ShouldComputeHistogramNow() -> bool {
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
      std::chrono::duration_cast<std::chrono::milliseconds>(now - last_histogram_update_time_).count();
  if (elapsed_ms < histogram_update_interval_ms_) {
    return false;
  }
  last_histogram_update_time_ = now;
  return true;
}

auto OpenGLViewerRenderer::ComputeHistogram(GLuint texture_id, int width, int height) -> bool {
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

  const GLuint groups = static_cast<GLuint>((kHistogramSampleSize + 16 - 1) / 16);
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

auto OpenGLViewerRenderer::BuildComputeProgram(const char* source, const char* debug_name,
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

bool OpenGLViewerRenderer::InitHistogramResources() {
  if (histogram_resources_ready_) {
    return true;
  }

  auto* gl_context = QOpenGLContext::currentContext();
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

  if (!BuildComputeProgram(kHistogramClearShaderSource, "HistogramClear",
                           histogram_clear_program_) ||
      !BuildComputeProgram(kHistogramComputeShaderSource, "HistogramCompute",
                           histogram_compute_program_) ||
      !BuildComputeProgram(kHistogramNormalizeShaderSource, "HistogramNormalize",
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
  histogram_has_data_        = false;
  last_histogram_update_time_ = {};
  return histogram_resources_ready_;
}

void OpenGLViewerRenderer::FreeHistogramResources() {
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
  histogram_compute_tex_loc_    = -1;
  histogram_compute_bins_loc_   = -1;
  histogram_compute_sample_loc_ = -1;
  histogram_norm_bins_loc_      = -1;
  histogram_resources_ready_    = false;
  histogram_has_data_           = false;
}

}  // namespace puerhlab
