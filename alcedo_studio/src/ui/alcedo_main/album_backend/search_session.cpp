//  Copyright 2026 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#include "ui/alcedo_main/album_backend/search_session.hpp"

#include <algorithm>
#include <utility>

namespace alcedo::ui {

SearchSession::SearchSession(SearchSessionPort port, QObject* parent)
    : QObject(parent), port_(std::move(port)) {}

auto SearchSession::RouteFor(const QString& query) const -> QString {
  const QString trimmed = query.trimmed();
  if (trimmed.isEmpty()) {
    return QStringLiteral("empty");
  }
  if (port_.natural_language_enabled() &&
      port_.classify_query(trimmed) == QStringLiteral("semantic")) {
    return QStringLiteral("semantic");
  }
  return QStringLiteral("fuzzy");
}

void SearchSession::Preview(const QString& query) {
  const QString trimmed = query.trimmed();
  const QString route   = RouteFor(trimmed);
  if (route == QStringLiteral("fuzzy")) {
    if (ClearWindow()) {
      emit ResultsChanged();
    }
    Request(trimmed, 0, kPageSize, QStringLiteral("replace"), false, true);
    return;
  }
  // An empty query shows the recommendations; a semantic query runs only on an explicit
  // submit, so typing does not reach the semantic provider.
  DropActiveRequest();
  const bool window_changed = ClearWindow();
  ClearStatus();
  query_           = trimmed;
  route_           = route;
  semantic_active_ = false;
  awaiting_submit_ = route == QStringLiteral("semantic");
  if (window_changed) {
    emit ResultsChanged();
  }
  emit StateChanged();
}

void SearchSession::SubmitSemantic(const QString& query) {
  const QString trimmed = query.trimmed();
  if (trimmed.isEmpty()) {
    return;
  }
  if (ClearWindow()) {
    emit ResultsChanged();
  }
  Request(trimmed, 0, kPageSize, QStringLiteral("replace"), true, true);
}

auto SearchSession::Replace(const QString& query, int offset, int limit, bool submit,
                            bool allow_semantic) -> qulonglong {
  const QString trimmed = query.trimmed();
  if (trimmed.isEmpty()) {
    Preview(trimmed);
    return 0;
  }
  if (ClearWindow()) {
    emit ResultsChanged();
  }
  Request(trimmed, std::max(0, offset), limit, QStringLiteral("replace"), submit, allow_semantic);
  return active_request_id_;
}

void SearchSession::LoadMore() {
  if (Loading() || !has_more_ || query_.isEmpty()) {
    return;
  }
  Request(query_, window_start_ + static_cast<int>(results_.size()), kPageSize,
          QStringLiteral("append"), semantic_active_, allow_semantic_);
}

void SearchSession::LoadPrevious() {
  if (Loading() || window_start_ <= 0 || query_.isEmpty()) {
    return;
  }
  const int offset = std::max(0, window_start_ - kPageSize);
  Request(query_, offset, window_start_ - offset, QStringLiteral("prepend"), semantic_active_,
          allow_semantic_);
}

void SearchSession::Reset() {
  DropActiveRequest();
  const bool window_changed = ClearWindow();
  ClearStatus();
  query_.clear();
  route_.clear();
  semantic_active_ = false;
  allow_semantic_  = true;
  if (window_changed) {
    emit ResultsChanged();
  }
  emit StateChanged();
}

void SearchSession::Request(const QString& query, int offset, int limit, const QString& mode,
                            bool submit, bool allow_semantic) {
  DropActiveRequest();
  ClearStatus();
  query_             = query;
  semantic_active_   = submit && allow_semantic && RouteFor(query) == QStringLiteral("semantic");
  allow_semantic_    = allow_semantic;
  active_request_id_ = port_.request_page(query, offset, limit, mode, submit, allow_semantic);
  emit StateChanged();
}

void SearchSession::ApplyResponse(qulonglong request_id, const QString& mode,
                                  const QVariantMap& response) {
  if (active_request_id_ == 0 || request_id != active_request_id_) {
    return;
  }
  active_request_id_ = 0;

  route_    = response.value(QStringLiteral("route"), QStringLiteral("semantic")).toString();
  too_long_ = response.value(QStringLiteral("tooLong")).toBool();
  semantic_unavailable_            = response.value(QStringLiteral("semanticUnavailable")).toBool();
  semantic_error_text_             = response.value(QStringLiteral("semanticErrorText")).toString();
  search_error_text_               = response.value(QStringLiteral("searchErrorText")).toString();
  awaiting_submit_                 = response.value(QStringLiteral("awaitingSubmit")).toBool();

  const QVariantList rows          = response.value(QStringLiteral("rows")).toList();
  const int          offset        = std::max(0, response.value(QStringLiteral("offset")).toInt());
  int                dropped_count = 0;
  int                prepended_count = 0;
  if (mode == QStringLiteral("append")) {
    results_.append(rows);
    if (results_.size() > kWindowCapacity) {
      dropped_count = static_cast<int>(results_.size()) - kWindowCapacity;
      results_.remove(0, dropped_count);
      window_start_ += dropped_count;
    }
  } else if (mode == QStringLiteral("prepend")) {
    QVariantList next = rows;
    next.append(results_);
    if (next.size() > kWindowCapacity) {
      next.resize(kWindowCapacity);
    }
    results_        = std::move(next);
    window_start_   = offset;
    prepended_count = static_cast<int>(rows.size());
  } else {
    results_      = rows;
    window_start_ = offset;
    if (results_.size() > kWindowCapacity) {
      dropped_count = static_cast<int>(results_.size()) - kWindowCapacity;
      results_.remove(0, dropped_count);
      window_start_ += dropped_count;
    }
  }
  const int window_end = window_start_ + static_cast<int>(results_.size());
  total_               = response.contains(QStringLiteral("total"))
                             ? response.value(QStringLiteral("total")).toInt()
                             : static_cast<int>(results_.size());
  has_more_            = response.contains(QStringLiteral("hasMore"))
                             ? response.value(QStringLiteral("hasMore")).toBool()
                             : window_end < total_;
  emit ResultsChanged();
  emit StateChanged();
  emit ResponseApplied(request_id, mode, dropped_count, prepended_count);
}

void SearchSession::DropActiveRequest() {
  if (active_request_id_ == 0) {
    return;
  }
  const qulonglong dropped = active_request_id_;
  active_request_id_       = 0;
  emit RequestDropped(dropped);
}

auto SearchSession::ClearWindow() -> bool {
  const bool changed = !results_.isEmpty() || window_start_ != 0 || total_ != 0 || has_more_;
  results_.clear();
  window_start_ = 0;
  total_        = 0;
  has_more_     = false;
  return changed;
}

void SearchSession::ClearStatus() {
  awaiting_submit_      = false;
  too_long_             = false;
  semantic_unavailable_ = false;
  semantic_error_text_.clear();
  search_error_text_.clear();
}

}  // namespace alcedo::ui
