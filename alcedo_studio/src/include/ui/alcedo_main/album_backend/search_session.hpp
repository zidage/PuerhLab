//  Copyright 2026 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#pragma once

#include <QObject>
#include <QString>
#include <QVariantList>
#include <QVariantMap>
#include <functional>

namespace alcedo::ui {

/// Calls of SearchSession into its owner (SearchController). Tests give fakes.
struct SearchSessionPort {
  /// Route name of ClassifyQuery: "empty", "traditional", "label", or "semantic".
  std::function<QString(const QString& query)> classify_query;
  /// True when natural language search is on.
  std::function<bool()>                        natural_language_enabled;
  /// Queues one page request on the search worker and returns its request id. @p submit is
  /// an explicit submit (the only request that reaches the semantic provider). When
  /// @p allow_semantic is false the query uses the SQL path for each route.
  std::function<qulonglong(const QString& query, int offset, int limit, const QString& mode,
                           bool submit, bool allow_semantic)>
      request_page;
};

/**
 * @brief Routing, response matching, and the sliding result window of the global search.
 *
 * @details GlobalSearchDialog.qml and the search.query command read the search through this
 * object. A request replaces the active request; only the response whose request id is the
 * active request id changes the window. The window holds at most kWindowCapacity rows: an
 * append drops leading rows and a prepend drops trailing rows. GUI thread only.
 */
class SearchSession final : public QObject {
  Q_OBJECT
  Q_PROPERTY(int pageSize READ PageSize CONSTANT)
  Q_PROPERTY(int windowCapacity READ WindowCapacity CONSTANT)
  Q_PROPERTY(QString query READ Query NOTIFY StateChanged)
  Q_PROPERTY(QString route READ Route NOTIFY StateChanged)
  Q_PROPERTY(QVariantList results READ Results NOTIFY ResultsChanged)
  Q_PROPERTY(int windowStart READ WindowStart NOTIFY ResultsChanged)
  Q_PROPERTY(int total READ Total NOTIFY ResultsChanged)
  Q_PROPERTY(bool hasMore READ HasMore NOTIFY ResultsChanged)
  Q_PROPERTY(bool hasPrevious READ HasPrevious NOTIFY ResultsChanged)
  Q_PROPERTY(bool loading READ Loading NOTIFY StateChanged)
  Q_PROPERTY(qulonglong activeRequestId READ ActiveRequestId NOTIFY StateChanged)
  Q_PROPERTY(bool semanticActive READ SemanticActive NOTIFY StateChanged)
  Q_PROPERTY(bool awaitingSubmit READ AwaitingSubmit NOTIFY StateChanged)
  Q_PROPERTY(bool tooLong READ TooLong NOTIFY StateChanged)
  Q_PROPERTY(bool semanticUnavailable READ SemanticUnavailable NOTIFY StateChanged)
  Q_PROPERTY(QString semanticErrorText READ SemanticErrorText NOTIFY StateChanged)

 public:
  static constexpr int kPageSize       = 24;
  static constexpr int kWindowCapacity = 48;

  explicit SearchSession(SearchSessionPort port, QObject* parent = nullptr);

  [[nodiscard]] static int          PageSize() { return kPageSize; }
  [[nodiscard]] static int          WindowCapacity() { return kWindowCapacity; }
  [[nodiscard]] const QString&      Query() const { return query_; }
  [[nodiscard]] const QString&      Route() const { return route_; }
  [[nodiscard]] const QVariantList& Results() const { return results_; }
  [[nodiscard]] int                 WindowStart() const { return window_start_; }
  [[nodiscard]] int                 Total() const { return total_; }
  [[nodiscard]] bool                HasMore() const { return has_more_; }
  [[nodiscard]] bool                HasPrevious() const { return window_start_ > 0; }
  [[nodiscard]] bool                Loading() const { return active_request_id_ != 0; }
  [[nodiscard]] bool                SemanticActive() const { return semantic_active_; }
  [[nodiscard]] bool                AwaitingSubmit() const { return awaiting_submit_; }
  [[nodiscard]] bool                TooLong() const { return too_long_; }
  [[nodiscard]] bool                SemanticUnavailable() const { return semantic_unavailable_; }
  [[nodiscard]] const QString&      SemanticErrorText() const { return semantic_error_text_; }
  [[nodiscard]] const QString&      SearchErrorText() const { return search_error_text_; }
  [[nodiscard]] qulonglong          ActiveRequestId() const { return active_request_id_; }

  /// "empty" for an empty query; "semantic" when ClassifyQuery returns `semantic` and natural
  /// language search is on; else "fuzzy".
  Q_INVOKABLE QString               RouteFor(const QString& query) const;
  /// Typing preview. An empty query clears the window; a semantic query only waits for an
  /// explicit submit; any other query requests the first page.
  Q_INVOKABLE void                  Preview(const QString& query);
  /// Explicit semantic submit of the first page.
  Q_INVOKABLE void                  SubmitSemantic(const QString& query);
  /// Requests the page after the window and appends it.
  Q_INVOKABLE void                  LoadMore();
  /// Requests the page before the window and prepends it.
  Q_INVOKABLE void                  LoadPrevious();
  /// Drops the active request and clears the window.
  Q_INVOKABLE void                  Reset();
  /// Requests one page that replaces the window. @p allow_semantic false keeps a semantic
  /// query on the SQL path. Returns the request id, or 0 for an empty query.
  auto Replace(const QString& query, int offset, int limit, bool submit, bool allow_semantic)
      -> qulonglong;

  /// Response of the search worker (SearchController::SearchResponseReady). Applied only when
  /// @p request_id is the active request id.
  void ApplyResponse(qulonglong request_id, const QString& mode, const QVariantMap& response);

 signals:
  void StateChanged();
  /// The rows or the paging values of the window changed.
  void ResultsChanged();
  /// The response of @p requestId changed the window. @p droppedCount rows left the start of
  /// the window (append); @p prependedCount rows were added before it (prepend).
  void ResponseApplied(qulonglong requestId, const QString& mode, int droppedCount,
                       int prependedCount);
  /// @p requestId stopped being the active request before its response arrived.
  void RequestDropped(qulonglong requestId);

 private:
  void Request(const QString& query, int offset, int limit, const QString& mode, bool submit,
               bool allow_semantic);
  void DropActiveRequest();
  /// Returns true when the window had rows or paging values.
  auto ClearWindow() -> bool;
  void ClearStatus();

  SearchSessionPort port_;
  QString           query_{};
  QString           route_{};
  QVariantList      results_{};
  int               window_start_         = 0;
  int               total_                = 0;
  bool              has_more_             = false;
  bool              semantic_active_      = false;
  bool              allow_semantic_       = true;
  bool              awaiting_submit_      = false;
  bool              too_long_             = false;
  bool              semantic_unavailable_ = false;
  QString           semantic_error_text_{};
  QString           search_error_text_{};
  qulonglong        active_request_id_ = 0;
};

}  // namespace alcedo::ui
