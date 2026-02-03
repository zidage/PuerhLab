#include <qfontdatabase.h>

#include <QAbstractItemView>
#include <QApplication>
#include <QComboBox>
#include <QCoreApplication>
#include <QDateTime>
#include <QDialogButtonBox>
#include <QDir>
#include <QDoubleValidator>
#include <QEventLoop>
#include <QFileDialog>
#include <QFontDatabase>
#include <QFormLayout>
#include <QFrame>
#include <QHBoxLayout>
#include <QIcon>
#include <QIntValidator>
#include <QLabel>
#include <QLineEdit>
#include <QListWidget>
#include <QMessageBox>
#include <QMetaObject>
#include <QPainter>
#include <QPixmap>
#include <QPointer>
#include <QProgressDialog>
#include <QPushButton>
#include <QScrollArea>
#include <QSignalBlocker>
#include <QSlider>
#include <QSpinBox>
#include <QStandardPaths>
#include <QStyle>
#include <QStyleFactory>
#include <QTimer>
#include <QToolButton>
#include <QVBoxLayout>
#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <ctime>
#include <exiv2/exiv2.hpp>
#include <filesystem>
#include <functional>
#include <future>
#include <iostream>
#include <json.hpp>
#include <memory>
#include <opencv2/opencv.hpp>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "app/export_service.hpp"
#include "app/import_service.hpp"
#include "app/history_mgmt_service.hpp"
#include "app/pipeline_service.hpp"
#include "app/project_service.hpp"
#include "app/render_service.hpp"
#include "app/sleeve_filter_service.hpp"
#include "app/thumbnail_service.hpp"
#include "edit/operators/operator_registeration.hpp"
#include "image/image_buffer.hpp"
#include "io/image/image_loader.hpp"
#include "renderer/pipeline_task.hpp"
#include "sleeve/sleeve_filter/filter_combo.hpp"
#include "type/supported_file_type.hpp"
#include "ui/edit_viewer/edit_viewer.hpp"

namespace puerhlab {
namespace {
using namespace std::chrono_literals;

static QString FsPathToQString(const std::filesystem::path& path) {
#if defined(_WIN32)
  return QString::fromStdWString(path.wstring());
#else
  return QString::fromUtf8(path.string().c_str());
#endif
}

static std::optional<std::string_view> FindArgValue(int argc, char** argv,
                                                    std::string_view opt_name) {
  const std::string opt_eq = std::string(opt_name) + "=";
  for (int i = 1; i < argc; ++i) {
    const std::string_view arg(argv[i] ? argv[i] : "");
    if (arg == opt_name) {
      if (i + 1 < argc && argv[i + 1]) {
        return std::string_view(argv[i + 1]);
      }
      return std::nullopt;
    }
    if (arg.rfind(opt_eq, 0) == 0) {
      return arg.substr(opt_eq.size());
    }
  }
  return std::nullopt;
}

static void ApplyExternalAppFont(QApplication& app, int argc, char** argv) {
  std::vector<std::filesystem::path> candidates;

  if (const auto arg = FindArgValue(argc, argv, "--font"); arg.has_value()) {
    candidates.emplace_back(std::string(arg.value()));
  }
  if (const auto env = qEnvironmentVariable("PUERHLAB_FONT_PATH"); !env.isEmpty()) {
    candidates.emplace_back(env.toStdString());
  }

  const auto app_dir = std::filesystem::path(QCoreApplication::applicationDirPath().toStdWString());
  candidates.emplace_back(app_dir / "fonts" / "main_IBM.ttf");

#if defined(PUERHLAB_SOURCE_DIR)
  candidates.emplace_back(std::filesystem::path(PUERHLAB_SOURCE_DIR) / "pu-erh_lab" / "src" /
                          "config" / "fonts" / "main_IBM.ttf");
#endif

  for (const auto& path : candidates) {
    std::error_code ec;
    if (!std::filesystem::exists(path, ec) || ec) {
      continue;
    }

    const int font_id = QFontDatabase::addApplicationFont(FsPathToQString(path));
    if (font_id < 0) {
      continue;
    }

    const auto families = QFontDatabase::applicationFontFamilies(font_id);
    if (families.isEmpty()) {
      continue;
    }

    app.setFont(QFont(families.front()));
    return;
  }
}

static void ApplyMaterialLikeTheme(QApplication& app) {
  app.setStyle(QStyleFactory::create("Fusion"));

  QPalette p;
  p.setColor(QPalette::Window, QColor(0x12, 0x12, 0x12));
  p.setColor(QPalette::WindowText, QColor(0xE8, 0xEA, 0xED));
  p.setColor(QPalette::Base, QColor(0x1E, 0x1E, 0x1E));
  p.setColor(QPalette::AlternateBase, QColor(0x20, 0x20, 0x20));
  p.setColor(QPalette::ToolTipBase, QColor(0x20, 0x20, 0x20));
  p.setColor(QPalette::ToolTipText, QColor(0xE8, 0xEA, 0xED));
  p.setColor(QPalette::Text, QColor(0xE8, 0xEA, 0xED));
  p.setColor(QPalette::Button, QColor(0x1E, 0x1E, 0x1E));
  p.setColor(QPalette::ButtonText, QColor(0xE8, 0xEA, 0xED));
  p.setColor(QPalette::Link, QColor(0x5F, 0xA2, 0xFF));
  p.setColor(QPalette::Highlight, QColor(0x5F, 0xA2, 0xFF));
  p.setColor(QPalette::HighlightedText, QColor(0x08, 0x0A, 0x0C));
  app.setPalette(p);

  // Match the ImagePreview demo's simple global QSS for consistent control look.
  app.setStyleSheet(
      "QWidget {"
      "  color: #E8EAED;"
      "  font-size: 12px;"
      "}"
      "QSlider {"
      "  font-size: 14px;"
      "}"
      "QLabel {"
      "  color: #E8EAED;"
      "}"
      "QToolTip {"
      "  background-color: #202124;"
      "  color: #E8EAED;"
      "  border: 1px solid #303134;"
      "  padding: 6px 8px;"
      "  border-radius: 8px;"
      "}"
      // Material-like slider (horizontal)
      "QSlider::groove:horizontal {"
      "  height: 4px;"
      "  background: #3C4043;"
      "  border-radius: 2px;"
      "}"
      "QSlider::sub-page:horizontal {"
      "  background: #8ab4f8;"
      "  border-radius: 2px;"
      "}"
      "QSlider::add-page:horizontal {"
      "  background: #3C4043;"
      "  border-radius: 2px;"
      "}"
      "QSlider::handle:horizontal {"
      "  background: #8ab4f8;"
      "  width: 18px;"
      "  height: 18px;"
      "  margin: -7px 0;"
      "  border-radius: 9px;"
      "}"
      "QSlider::handle:horizontal:hover {"
      "  background: #8ab4f8;"
      "}"
      "QSlider::handle:horizontal:pressed {"
      "  background: #8ab4f8;"
      "}"
      "QSlider::tick-mark {"
      "  background: transparent;"
      "}");
}

enum class FilterValueKind { String, Int64, Double, DateTime };

static FilterValueKind KindForField(FilterField field) {
  switch (field) {
    case FilterField::ExifISO:
    case FilterField::Rating:
      return FilterValueKind::Int64;
    case FilterField::ExifFocalLength:
    case FilterField::ExifAperture:
      return FilterValueKind::Double;
    case FilterField::CaptureDate:
    case FilterField::ImportDate:
      return FilterValueKind::DateTime;
    default:
      return FilterValueKind::String;
  }
}

static void PopulateCompareOps(QComboBox* combo, FilterField field) {
  combo->clear();

  const auto kind   = KindForField(field);
  const auto add_op = [combo](CompareOp op, const char* label) {
    combo->addItem(QString::fromUtf8(label), static_cast<int>(op));
  };

  if (kind == FilterValueKind::String) {
    add_op(CompareOp::CONTAINS, "contains");
    add_op(CompareOp::NOT_CONTAINS, "not contains");
    add_op(CompareOp::EQUALS, "=");
    add_op(CompareOp::NOT_EQUALS, "!=");
    add_op(CompareOp::STARTS_WITH, "starts with");
    add_op(CompareOp::ENDS_WITH, "ends with");
    add_op(CompareOp::REGEX, "regex");
  } else if (kind == FilterValueKind::Int64 || kind == FilterValueKind::Double) {
    add_op(CompareOp::EQUALS, "=");
    add_op(CompareOp::NOT_EQUALS, "!=");
    add_op(CompareOp::GREATER_THAN, "> ");
    add_op(CompareOp::LESS_THAN, "< ");
    add_op(CompareOp::GREATER_EQUAL, ">=");
    add_op(CompareOp::LESS_EQUAL, "<=");
    add_op(CompareOp::BETWEEN, "between");
  } else {
    // Date/time
    add_op(CompareOp::GREATER_THAN, "> ");
    add_op(CompareOp::LESS_THAN, "< ");
    add_op(CompareOp::GREATER_EQUAL, ">=");
    add_op(CompareOp::LESS_EQUAL, "<=");
    add_op(CompareOp::BETWEEN, "between");
  }
}

static std::optional<std::tm> ParseDateTimeYmd(const QString& text) {
  // Accept: YYYY-MM-DD
  const auto trimmed = text.trimmed();
  const auto parts   = trimmed.split('-', Qt::SkipEmptyParts);
  if (parts.size() != 3) {
    return std::nullopt;
  }
  bool      ok_y = false, ok_m = false, ok_d = false;
  const int year  = parts[0].toInt(&ok_y);
  const int month = parts[1].toInt(&ok_m);
  const int day   = parts[2].toInt(&ok_d);
  if (!ok_y || !ok_m || !ok_d) {
    return std::nullopt;
  }
  if (year < 1900 || month < 1 || month > 12 || day < 1 || day > 31) {
    return std::nullopt;
  }

  std::tm t{};
  t.tm_year = year - 1900;
  t.tm_mon  = month - 1;
  t.tm_mday = day;
  return t;
}

static std::optional<FilterValue> ParseFilterValue(FilterField field, const QString& text) {
  const auto kind = KindForField(field);
  if (kind == FilterValueKind::String) {
    return FilterValue{text.trimmed().toStdWString()};
  }
  if (kind == FilterValueKind::Int64) {
    bool       ok = false;
    const auto v  = text.trimmed().toLongLong(&ok);
    if (!ok) {
      return std::nullopt;
    }
    return FilterValue{static_cast<int64_t>(v)};
  }
  if (kind == FilterValueKind::Double) {
    bool       ok = false;
    const auto v  = text.trimmed().toDouble(&ok);
    if (!ok) {
      return std::nullopt;
    }
    return FilterValue{v};
  }

  const auto tm_opt = ParseDateTimeYmd(text);
  if (!tm_opt.has_value()) {
    return std::nullopt;
  }
  return FilterValue{tm_opt.value()};
}

class FilterDrawer final : public QWidget {
 public:
  explicit FilterDrawer(QWidget* parent = nullptr) : QWidget(parent) {
    setObjectName("FilterDrawer");
    setMinimumWidth(360);
    setMaximumWidth(420);
    setStyleSheet(DrawerStyleSheet());

    auto* root = new QVBoxLayout(this);
    root->setContentsMargins(0, 0, 0, 0);
    root->setSpacing(10);

    auto* header = new QHBoxLayout();
    header->setContentsMargins(12, 12, 12, 0);
    header->setSpacing(8);

    auto* title = new QLabel("Filters", this);
    title->setObjectName("DrawerTitle");

    collapse_btn_ = new QToolButton(this);
    collapse_btn_->setObjectName("CollapseButton");
    collapse_btn_->setCheckable(true);
    collapse_btn_->setChecked(true);
    collapse_btn_->setToolTip("Collapse / expand filter builder");
    collapse_btn_->setArrowType(Qt::DownArrow);
    collapse_btn_->setAutoRaise(true);

    header->addWidget(title, 0);
    header->addStretch(1);
    header->addWidget(collapse_btn_, 0);
    root->addLayout(header, 0);

    content_             = new QWidget(this);
    auto* content_layout = new QVBoxLayout(content_);
    content_layout->setContentsMargins(12, 0, 12, 12);
    content_layout->setSpacing(10);

    // Quick search card.
    auto* quick_card = new QFrame(this);
    quick_card->setObjectName("Card");
    auto* quick = new QVBoxLayout(quick_card);
    quick->setContentsMargins(12, 12, 12, 12);
    quick->setSpacing(6);

    auto* quick_label = new QLabel("Quick search", this);
    quick_label->setObjectName("CardTitle");

    auto* quick_row = new QHBoxLayout();
    quick_row->setSpacing(6);

    quick_search_ = new QLineEdit(this);
    quick_search_->setObjectName("QuickSearch");
    quick_search_->setPlaceholderText("Search filename, camera model, tags…");
    quick_search_->setClearButtonEnabled(true);

    inline_apply_ = new QToolButton(this);
    inline_apply_->setObjectName("InlineApply");
    inline_apply_->setAutoRaise(true);
    inline_apply_->setToolTip("Apply filters");
    inline_apply_->setIcon(style()->standardIcon(QStyle::SP_DialogApplyButton));

    quick_row->addWidget(quick_search_, 1);
    quick_row->addWidget(inline_apply_, 0);

    quick->addWidget(quick_label, 0);
    quick->addLayout(quick_row, 0);
    content_layout->addWidget(quick_card, 0);

    // Rules card.
    auto* rules_card = new QFrame(this);
    rules_card->setObjectName("Card");
    auto* rules = new QVBoxLayout(rules_card);
    rules->setContentsMargins(12, 12, 12, 12);
    rules->setSpacing(8);

    auto* rules_top   = new QHBoxLayout();
    auto* rules_title = new QLabel("Rules", this);
    rules_title->setObjectName("CardTitle");

    join_op_ = new QComboBox(this);
    join_op_->setObjectName("JoinOp");
    join_op_->addItem("Match all rules", static_cast<int>(FilterOp::AND));
    join_op_->addItem("Match any rule", static_cast<int>(FilterOp::OR));
    join_op_->setToolTip("How multiple rules are combined");

    rules_top->addWidget(rules_title, 0);
    rules_top->addStretch(1);
    rules_top->addWidget(join_op_, 0);
    rules->addLayout(rules_top, 0);

    rows_scroll_ = new QScrollArea(this);
    rows_scroll_->setObjectName("RowsScroll");
    rows_scroll_->setWidgetResizable(true);
    rows_scroll_->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    rows_scroll_->setFrameShape(QFrame::NoFrame);

    rows_widget_    = new QWidget(this);
    rows_container_ = new QVBoxLayout(rows_widget_);
    rows_container_->setContentsMargins(0, 0, 0, 0);
    rows_container_->setSpacing(8);
    rows_container_->addStretch(1);
    rows_scroll_->setWidget(rows_widget_);
    rules->addWidget(rows_scroll_, 1);

    btn_add_ = new QPushButton("Add rule", this);
    btn_add_->setObjectName("AddRule");
    btn_add_->setIcon(style()->standardIcon(QStyle::SP_FileDialogNewFolder));
    rules->addWidget(btn_add_, 0, Qt::AlignLeft);

    content_layout->addWidget(rules_card, 1);

    // Footer.
    auto* actions = new QHBoxLayout();
    actions->setSpacing(8);
    btn_clear_ = new QPushButton("Clear", this);
    btn_clear_->setObjectName("ClearButton");
    btn_clear_->setIcon(style()->standardIcon(QStyle::SP_DialogResetButton));
    btn_apply_ = new QPushButton("Apply", this);
    btn_apply_->setObjectName("ApplyButton");
    btn_apply_->setIcon(style()->standardIcon(QStyle::SP_DialogApplyButton));
    actions->addWidget(btn_clear_, 0);
    actions->addStretch(1);
    actions->addWidget(btn_apply_, 0);
    content_layout->addLayout(actions, 0);

    info_ = new QLabel("No images loaded.", this);
    info_->setObjectName("FilterInfo");
    content_layout->addWidget(info_, 0);

    sql_preview_ = new QLabel("", this);
    sql_preview_->setObjectName("SqlPreview");
    sql_preview_->setWordWrap(true);
    sql_preview_->setTextInteractionFlags(Qt::TextSelectableByMouse);
    sql_preview_->setVisible(false);
    content_layout->addWidget(sql_preview_, 0);

    root->addWidget(content_, 1);

    connect(collapse_btn_, &QToolButton::toggled, this, [this](bool expanded) {
      content_->setVisible(expanded);
      collapse_btn_->setArrowType(expanded ? Qt::DownArrow : Qt::RightArrow);
    });

    connect(btn_add_, &QPushButton::clicked, this, [this]() { AddRuleRow(); });
    connect(btn_clear_, &QPushButton::clicked, this, [this]() {
      ClearAllUi();
      if (on_clear_) {
        on_clear_();
      }
    });

    const auto apply_now = [this]() {
      auto node_opt = BuildFilterNode(true);
      if (!node_opt.has_value()) {
        return;
      }
      SetSqlPreview(QString::fromStdWString(FilterSQLCompiler::Compile(node_opt.value())));
      if (on_apply_) {
        on_apply_(node_opt.value());
      }
    };

    connect(btn_apply_, &QPushButton::clicked, this, apply_now);
    connect(inline_apply_, &QToolButton::clicked, this, apply_now);
    connect(quick_search_, &QLineEdit::returnPressed, this, apply_now);

    AddRuleRow();
  }

  void SetOnApply(std::function<void(const FilterNode&)> fn) { on_apply_ = std::move(fn); }
  void SetOnClear(std::function<void()> fn) { on_clear_ = std::move(fn); }

  void SetResultsSummary(int shown, int total) {
    if (total <= 0) {
      info_->setText("No images loaded.");
      return;
    }
    if (shown == total) {
      info_->setText(QString("Showing %1 images").arg(total));
      return;
    }
    info_->setText(QString("Showing %1 of %2").arg(shown).arg(total));
  }

 private:
  struct Row {
    QWidget*     container_ = nullptr;
    QComboBox*   field_     = nullptr;
    QComboBox*   op_        = nullptr;
    QLineEdit*   value_     = nullptr;
    QLabel*      between_   = nullptr;
    QLineEdit*   value2_    = nullptr;
    QToolButton* remove_    = nullptr;
  };

  static QString DrawerStyleSheet() {
    return QString::fromUtf8(R"QSS(
#FilterDrawer { background: #121212; }
#FilterDrawer QLabel#DrawerTitle { font-size: 16px; font-weight: 600; }
#FilterDrawer QFrame#Card { background: #1E1E1E; border: 1px solid #303134; border-radius: 12px; }
#FilterDrawer QLabel#CardTitle { font-weight: 600; }
#FilterDrawer QLineEdit#QuickSearch { padding: 8px 10px; border: 1px solid #3C4043; border-radius: 10px; background: #202124; }
#FilterDrawer QComboBox, #FilterDrawer QLineEdit { padding: 6px 8px; border: 1px solid #3C4043; border-radius: 8px; background: #202124; }
#FilterDrawer QPushButton { padding: 7px 12px; border-radius: 10px; background: #202124; border: 1px solid #3C4043; }
#FilterDrawer QPushButton#ApplyButton { background: #8ab4f8; color: #080A0C; border: 1px solid #8ab4f8; }
#FilterDrawer QPushButton#ApplyButton:hover { background: #a3c2ff; }
#FilterDrawer QToolButton#RemoveRule { background: transparent; border: none; padding: 6px; }
#FilterDrawer QToolButton#RemoveRule:hover { background: #2a2b2e; border-radius: 10px; }
#FilterDrawer QLabel#FilterInfo { color: #9aa0a6; }
#FilterDrawer QLabel#SqlPreview { color: #cdd0d4; font-family: 'Consolas','Courier New',monospace; font-size: 11px; padding: 8px 10px; background: #1a1b1e; border: 1px solid #303134; border-radius: 10px; }
)QSS");
  }

  static QString PlaceholderForField(FilterField field) {
    switch (field) {
      case FilterField::CaptureDate:
      case FilterField::ImportDate:
        return "YYYY-MM-DD";
      case FilterField::ExifISO:
      case FilterField::Rating:
        return "number";
      case FilterField::ExifAperture:
      case FilterField::ExifFocalLength:
        return "number";
      default:
        return "type to filter…";
    }
  }

  Row* FindRow(QWidget* container) {
    for (auto& r : rows_) {
      if (r.container_ == container) {
        return &r;
      }
    }
    return nullptr;
  }

  void ConfigureEditorsForRow(Row& row, FilterField field) {
    PopulateCompareOps(row.op_, field);
    row.value_->setPlaceholderText(PlaceholderForField(field));
    row.value2_->setPlaceholderText(PlaceholderForField(field));

    row.value_->setValidator(nullptr);
    row.value2_->setValidator(nullptr);

    const auto kind = KindForField(field);
    if (kind == FilterValueKind::Int64) {
      row.value_->setValidator(new QIntValidator(row.value_));
      row.value2_->setValidator(new QIntValidator(row.value2_));
    } else if (kind == FilterValueKind::Double) {
      auto* v1 = new QDoubleValidator(row.value_);
      v1->setNotation(QDoubleValidator::StandardNotation);
      row.value_->setValidator(v1);
      auto* v2 = new QDoubleValidator(row.value2_);
      v2->setNotation(QDoubleValidator::StandardNotation);
      row.value2_->setValidator(v2);
    }
  }

  void UpdateRowBetweenUi(Row& row) {
    const auto op         = static_cast<CompareOp>(row.op_->currentData().toInt());
    const bool is_between = (op == CompareOp::BETWEEN);
    row.between_->setVisible(is_between);
    row.value2_->setVisible(is_between);
  }

  void AddRuleRow() {
    Row r;
    r.container_ = new QWidget(this);
    auto* h      = new QHBoxLayout(r.container_);
    h->setContentsMargins(0, 0, 0, 0);
    h->setSpacing(6);

    r.field_ = new QComboBox(this);
    r.field_->addItem("Filename", static_cast<int>(FilterField::FileName));
    r.field_->addItem("Camera Model", static_cast<int>(FilterField::ExifCameraModel));
    r.field_->addItem("File Extension", static_cast<int>(FilterField::FileExtension));
    r.field_->addItem("ISO", static_cast<int>(FilterField::ExifISO));
    r.field_->addItem("Aperture", static_cast<int>(FilterField::ExifAperture));
    r.field_->addItem("Focal Length", static_cast<int>(FilterField::ExifFocalLength));
    r.field_->addItem("Capture Date", static_cast<int>(FilterField::CaptureDate));
    r.field_->addItem("Import Date", static_cast<int>(FilterField::ImportDate));
    r.field_->addItem("Rating", static_cast<int>(FilterField::Rating));
    r.field_->addItem("Tags", static_cast<int>(FilterField::SemanticTags));

    r.op_      = new QComboBox(this);

    r.value_   = new QLineEdit(this);
    r.value2_  = new QLineEdit(this);
    r.between_ = new QLabel("to", this);

    r.remove_  = new QToolButton(this);
    r.remove_->setObjectName("RemoveRule");
    r.remove_->setAutoRaise(true);
    r.remove_->setIcon(style()->standardIcon(QStyle::SP_DockWidgetCloseButton));
    r.remove_->setToolTip("Remove rule");

    h->addWidget(r.field_, 1);
    h->addWidget(r.op_, 0);
    h->addWidget(r.value_, 1);
    h->addWidget(r.between_, 0);
    h->addWidget(r.value2_, 1);
    h->addWidget(r.remove_, 0);

    const auto initial_field = static_cast<FilterField>(r.field_->currentData().toInt());
    ConfigureEditorsForRow(r, initial_field);
    UpdateRowBetweenUi(r);

    connect(r.field_, &QComboBox::currentIndexChanged, this, [this, c = r.container_]() {
      auto* row = FindRow(c);
      if (!row) {
        return;
      }
      const auto field = static_cast<FilterField>(row->field_->currentData().toInt());
      {
        QSignalBlocker block(*row->op_);
        ConfigureEditorsForRow(*row, field);
      }
      UpdateRowBetweenUi(*row);
    });
    connect(r.op_, &QComboBox::currentIndexChanged, this, [this, c = r.container_]() {
      auto* row = FindRow(c);
      if (!row) {
        return;
      }
      UpdateRowBetweenUi(*row);
    });

    connect(r.remove_, &QToolButton::clicked, this, [this, c = r.container_]() {
      auto it = std::find_if(rows_.begin(), rows_.end(),
                             [c](const Row& rr) { return rr.container_ == c; });
      if (it == rows_.end()) {
        return;
      }
      delete it->container_;
      rows_.erase(it);
    });

    const int stretch_index = rows_container_->count() - 1;
    rows_container_->insertWidget(stretch_index, r.container_);
    rows_.push_back(r);
  }

  void ClearAllUi() {
    quick_search_->clear();
    SetSqlPreview(QString());

    for (auto& row : rows_) {
      delete row.container_;
    }
    rows_.clear();
    AddRuleRow();
  }

  void SetSqlPreview(const QString& sql) {
    const auto trimmed = sql.trimmed();
    if (trimmed.isEmpty()) {
      sql_preview_->clear();
      sql_preview_->setVisible(false);
      return;
    }
    sql_preview_->setText(trimmed);
    sql_preview_->setVisible(true);
  }

  std::optional<FilterNode> BuildFilterNode(bool show_dialogs) {
    std::optional<FilterNode> quick_node;
    const auto                q = quick_search_->text().trimmed();
    if (!q.isEmpty()) {
      const FilterValue       v{q.toStdWString()};
      std::vector<FilterNode> clauses;
      clauses.reserve(4);
      clauses.push_back(FilterNode{FilterNode::Type::Condition,
                                   {},
                                   {},
                                   FieldCondition{.field_        = FilterField::FileName,
                                                  .op_           = CompareOp::CONTAINS,
                                                  .value_        = v,
                                                  .second_value_ = std::nullopt},
                                   std::nullopt});
      clauses.push_back(FilterNode{FilterNode::Type::Condition,
                                   {},
                                   {},
                                   FieldCondition{.field_        = FilterField::ExifCameraModel,
                                                  .op_           = CompareOp::CONTAINS,
                                                  .value_        = v,
                                                  .second_value_ = std::nullopt},
                                   std::nullopt});
      clauses.push_back(FilterNode{FilterNode::Type::Condition,
                                   {},
                                   {},
                                   FieldCondition{.field_        = FilterField::SemanticTags,
                                                  .op_           = CompareOp::CONTAINS,
                                                  .value_        = v,
                                                  .second_value_ = std::nullopt},
                                   std::nullopt});
      clauses.push_back(FilterNode{FilterNode::Type::Condition,
                                   {},
                                   {},
                                   FieldCondition{.field_        = FilterField::FileExtension,
                                                  .op_           = CompareOp::CONTAINS,
                                                  .value_        = v,
                                                  .second_value_ = std::nullopt},
                                   std::nullopt});

      quick_node =
          FilterNode{FilterNode::Type::Logical, FilterOp::OR, std::move(clauses), {}, std::nullopt};
    }

    std::optional<FilterNode> rules_node;
    std::vector<FilterNode>   conditions;
    conditions.reserve(rows_.size());

    for (const auto& row : rows_) {
      const auto field   = static_cast<FilterField>(row.field_->currentData().toInt());
      const auto op      = static_cast<CompareOp>(row.op_->currentData().toInt());

      const auto v1_text = row.value_->text().trimmed();
      if (v1_text.isEmpty()) {
        continue;
      }

      const auto v1_opt = ParseFilterValue(field, v1_text);
      if (!v1_opt.has_value()) {
        if (show_dialogs) {
          QMessageBox::warning(this, "Filter", "Invalid value. Check field type and input.");
        }
        return std::nullopt;
      }

      FieldCondition cond{
          .field_ = field, .op_ = op, .value_ = v1_opt.value(), .second_value_ = std::nullopt};

      if (op == CompareOp::BETWEEN) {
        const auto v2_text = row.value2_->text().trimmed();
        if (v2_text.isEmpty()) {
          if (show_dialogs) {
            QMessageBox::warning(this, "Filter", "BETWEEN needs two values.");
          }
          return std::nullopt;
        }
        const auto v2_opt = ParseFilterValue(field, v2_text);
        if (!v2_opt.has_value()) {
          if (show_dialogs) {
            QMessageBox::warning(this, "Filter", "Invalid second value for BETWEEN.");
          }
          return std::nullopt;
        }
        cond.second_value_ = v2_opt.value();
      }

      conditions.push_back(
          FilterNode{FilterNode::Type::Condition, {}, {}, std::move(cond), std::nullopt});
    }

    if (!conditions.empty()) {
      if (conditions.size() == 1) {
        rules_node = conditions.front();
      } else {
        const auto join = static_cast<FilterOp>(join_op_->currentData().toInt());
        rules_node =
            FilterNode{FilterNode::Type::Logical, join, std::move(conditions), {}, std::nullopt};
      }
    }

    if (!quick_node.has_value() && !rules_node.has_value()) {
      if (show_dialogs) {
        QMessageBox::information(this, "Filter", "No filters set.");
      }
      return std::nullopt;
    }
    if (quick_node.has_value() && !rules_node.has_value()) {
      return quick_node;
    }
    if (!quick_node.has_value() && rules_node.has_value()) {
      return rules_node;
    }

    std::vector<FilterNode> children;
    children.reserve(2);
    children.push_back(std::move(quick_node.value()));
    children.push_back(std::move(rules_node.value()));
    return FilterNode{
        FilterNode::Type::Logical, FilterOp::AND, std::move(children), {}, std::nullopt};
  }

  QToolButton*                           collapse_btn_   = nullptr;
  QWidget*                               content_        = nullptr;

  QLineEdit*                             quick_search_   = nullptr;
  QToolButton*                           inline_apply_   = nullptr;

  QComboBox*                             join_op_        = nullptr;
  QScrollArea*                           rows_scroll_    = nullptr;
  QWidget*                               rows_widget_    = nullptr;
  QVBoxLayout*                           rows_container_ = nullptr;
  QPushButton*                           btn_add_        = nullptr;
  QPushButton*                           btn_apply_      = nullptr;
  QPushButton*                           btn_clear_      = nullptr;

  QLabel*                                info_           = nullptr;
  QLabel*                                sql_preview_    = nullptr;

  std::vector<Row>                       rows_;

  std::function<void(const FilterNode&)> on_apply_;
  std::function<void()>                  on_clear_;
};

static std::vector<std::filesystem::path> ListCubeLutsInDir(const std::filesystem::path& dir) {
  std::vector<std::filesystem::path> results;
  std::error_code                    ec;
  if (!std::filesystem::exists(dir, ec) || ec || !std::filesystem::is_directory(dir, ec)) {
    return results;
  }

  for (const auto& entry : std::filesystem::directory_iterator(dir, ec)) {
    if (ec) {
      break;
    }
    if (!entry.is_regular_file(ec) || ec) {
      ec.clear();
      continue;
    }

    const auto  ext = entry.path().extension().string();
    std::string ext_lower;
    ext_lower.reserve(ext.size());
    for (const char c : ext) {
      ext_lower.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
    }
    if (ext_lower == ".cube") {
      results.push_back(entry.path());
    }
  }

  std::sort(results.begin(), results.end(),
            [](const std::filesystem::path& a, const std::filesystem::path& b) {
              return a.filename().string() < b.filename().string();
            });

  return results;
}

static QImage MatRgba32fToQImageCopy(const cv::Mat& rgba32f_or_u8) {
  if (rgba32f_or_u8.empty()) {
    return {};
  }

  cv::Mat rgba8;
  if (rgba32f_or_u8.type() == CV_32FC4) {
    rgba32f_or_u8.convertTo(rgba8, CV_8UC4, 255.0);
  } else if (rgba32f_or_u8.type() == CV_8UC4) {
    rgba8 = rgba32f_or_u8;
  } else {
    cv::Mat tmp;
    rgba32f_or_u8.convertTo(tmp, CV_8UC4);
    rgba8 = tmp;
  }

  if (!rgba8.isContinuous()) {
    rgba8 = rgba8.clone();
  }

  QImage img(rgba8.data, rgba8.cols, rgba8.rows, static_cast<int>(rgba8.step),
             QImage::Format_RGBA8888);
  return img.copy();
}

static std::string ExtensionForExportFormat(ImageFormatType fmt) {
  switch (fmt) {
    case ImageFormatType::JPEG:
      return ".jpg";
    case ImageFormatType::PNG:
      return ".png";
    case ImageFormatType::TIFF:
      return ".tiff";
    case ImageFormatType::WEBP:
      return ".webp";
    case ImageFormatType::EXR:
      return ".exr";
    default:
      return ".jpg";
  }
}

static QString FormatName(ImageFormatType fmt) {
  switch (fmt) {
    case ImageFormatType::JPEG:
      return "JPEG";
    case ImageFormatType::PNG:
      return "PNG";
    case ImageFormatType::TIFF:
      return "TIFF";
    case ImageFormatType::WEBP:
      return "WEBP";
    case ImageFormatType::EXR:
      return "EXR";
    default:
      return "JPEG";
  }
}

static ImageFormatType FormatFromName(const QString& s) {
  const auto u = s.trimmed().toUpper();
  if (u == "PNG") {
    return ImageFormatType::PNG;
  }
  if (u == "TIFF") {
    return ImageFormatType::TIFF;
  }
  if (u == "WEBP") {
    return ImageFormatType::WEBP;
  }
  if (u == "EXR") {
    return ImageFormatType::EXR;
  }
  return ImageFormatType::JPEG;
}

class ExportDialog final : public QDialog {
 public:
  struct Item {
    sl_element_id_t sleeve_id_ = 0;
    image_id_t      image_id_  = 0;
  };

  ExportDialog(std::shared_ptr<ImagePoolService> image_pool,
               std::shared_ptr<ExportService> export_service, std::vector<Item> items,
               QWidget* parent = nullptr)
      : QDialog(parent),
        image_pool_(std::move(image_pool)),
        export_service_(std::move(export_service)),
        items_(std::move(items)) {
    if (!image_pool_ || !export_service_) {
      throw std::runtime_error("ExportDialog: missing services");
    }

    setModal(true);
    setWindowTitle("Export");
    resize(520, 320);

    auto* root = new QVBoxLayout(this);
    root->setContentsMargins(16, 16, 16, 16);
    root->setSpacing(12);

    auto* info = new QLabel(QString("Queue %1 item(s)").arg(static_cast<int>(items_.size())), this);
    root->addWidget(info);

    auto* form = new QFormLayout();
    form->setLabelAlignment(Qt::AlignRight);
    form->setFormAlignment(Qt::AlignTop);
    root->addLayout(form);

    // Output directory
    auto* out_row = new QWidget(this);
    auto* out_h   = new QHBoxLayout(out_row);
    out_h->setContentsMargins(0, 0, 0, 0);
    out_dir_ = new QLineEdit(this);
    out_dir_->setPlaceholderText("Select output directory");
    const auto default_dir = QStandardPaths::writableLocation(QStandardPaths::PicturesLocation);
    out_dir_->setText(default_dir.isEmpty() ? QDir::currentPath() : default_dir);
    auto* browse = new QPushButton("Browse…", this);
    out_h->addWidget(out_dir_, 1);
    out_h->addWidget(browse, 0);
    form->addRow("Output", out_row);

    connect(browse, &QPushButton::clicked, this, [this]() {
      const auto d =
          QFileDialog::getExistingDirectory(this, "Select export folder", out_dir_->text());
      if (!d.isEmpty()) {
        out_dir_->setText(d);
      }
    });

    // Format
    format_ = new QComboBox(this);
    for (auto f : {ImageFormatType::JPEG, ImageFormatType::PNG, ImageFormatType::TIFF,
                   ImageFormatType::WEBP, ImageFormatType::EXR}) {
      format_->addItem(FormatName(f));
    }
    form->addRow("Format", format_);

    // Resize
    resize_enabled_ = new QComboBox(this);
    resize_enabled_->addItems({"No", "Yes"});
    form->addRow("Resize", resize_enabled_);

    max_side_ = new QSpinBox(this);
    max_side_->setRange(256, 16384);
    max_side_->setValue(4096);
    form->addRow("Max side", max_side_);

    // Quality
    quality_ = new QSpinBox(this);
    quality_->setRange(1, 100);
    quality_->setValue(95);
    form->addRow("Quality", quality_);

    // Bit depth
    bit_depth_ = new QComboBox(this);
    bit_depth_->addItems({"8", "16", "32"});
    bit_depth_->setCurrentText("16");
    form->addRow("Bit depth", bit_depth_);

    // PNG compression
    png_compress_ = new QSpinBox(this);
    png_compress_->setRange(0, 9);
    png_compress_->setValue(5);
    form->addRow("PNG level", png_compress_);

    // TIFF compression
    tiff_compress_ = new QComboBox(this);
    tiff_compress_->addItems({"NONE", "LZW", "ZIP"});
    form->addRow("TIFF comp", tiff_compress_);

    status_ = new QLabel("", this);
    status_->setWordWrap(true);
    root->addWidget(status_);

    auto* buttons = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel, this);
    buttons->button(QDialogButtonBox::Ok)->setText("Export");
    root->addWidget(buttons);

    connect(buttons, &QDialogButtonBox::rejected, this, &QDialog::reject);
    connect(buttons, &QDialogButtonBox::accepted, this, [this]() { StartExport(); });
  }

 private:
  ExportFormatOptions BuildOptionsForOne(const std::filesystem::path& src_path,
                                         sl_element_id_t sleeve_id, image_id_t image_id) const {
    ExportFormatOptions opt;

    const auto          fmt = FormatFromName(format_->currentText());
    opt.format_             = fmt;

    opt.resize_enabled_     = (resize_enabled_->currentIndex() == 1);
    opt.max_length_side_    = opt.resize_enabled_ ? max_side_->value() : 0;

    opt.quality_            = quality_->value();

    const auto bd           = bit_depth_->currentText().trimmed();
    if (bd == "8") {
      opt.bit_depth_ = ExportFormatOptions::BIT_DEPTH::BIT_8;
    } else if (bd == "32") {
      opt.bit_depth_ = ExportFormatOptions::BIT_DEPTH::BIT_32;
    } else {
      opt.bit_depth_ = ExportFormatOptions::BIT_DEPTH::BIT_16;
    }

    opt.compression_level_ = png_compress_->value();

    const auto tc          = tiff_compress_->currentText().trimmed().toUpper();
    if (tc == "LZW") {
      opt.tiff_compress_ = ExportFormatOptions::TIFF_COMPRESS::LZW;
    } else if (tc == "ZIP") {
      opt.tiff_compress_ = ExportFormatOptions::TIFF_COMPRESS::ZIP;
    } else {
      opt.tiff_compress_ = ExportFormatOptions::TIFF_COMPRESS::NONE;
    }

    std::filesystem::path out_dir(out_dir_->text().toStdWString());
    std::filesystem::path stem = src_path.stem();
    if (stem.empty()) {
      stem = std::filesystem::path("image");
    }
    const auto suffix = std::string("_") + std::to_string(static_cast<uint64_t>(sleeve_id)) + "_" +
                        std::to_string(static_cast<uint64_t>(image_id));
    const auto ext   = ExtensionForExportFormat(fmt);
    opt.export_path_ = out_dir / (stem.wstring() + std::wstring(suffix.begin(), suffix.end()) +
                                  std::wstring(ext.begin(), ext.end()));

    return opt;
  }

  void StartExport() {
    if (items_.empty()) {
      status_->setText("Nothing to export.");
      return;
    }

    const auto out_dir = out_dir_->text().trimmed();
    if (out_dir.isEmpty()) {
      QMessageBox::warning(this, "Export", "Please select an output directory.");
      return;
    }

    // Enqueue
    for (const auto& it : items_) {
      const auto src_path = image_pool_->Read<std::filesystem::path>(
          it.image_id_, [](std::shared_ptr<Image> img) { return img->image_path_; });
      ExportTask task;
      task.sleeve_id_ = it.sleeve_id_;
      task.image_id_  = it.image_id_;
      task.options_   = BuildOptionsForOne(src_path, it.sleeve_id_, it.image_id_);
      export_service_->EnqueueExportTask(task);
    }

    // UI: busy
    setEnabled(false);
    status_->setText("Exporting…");

    QPointer<ExportDialog> self(this);
    export_service_->ExportAll([self](std::shared_ptr<std::vector<ExportResult>> results) {
      if (!self) {
        return;
      }
      QMetaObject::invokeMethod(
          self,
          [self, results]() {
            if (!self) {
              return;
            }
            int         ok   = 0;
            int         fail = 0;
            QStringList errors;
            if (results) {
              for (const auto& r : *results) {
                if (r.success_) {
                  ok++;
                } else {
                  fail++;
                  if (!r.message_.empty()) {
                    errors << QString::fromUtf8(r.message_.c_str());
                  }
                }
              }
            }

            if (fail == 0) {
              QMessageBox::information(self, "Export", QString("Done: %1 file(s)").arg(ok));
            } else {
              QMessageBox::warning(
                  self, "Export",
                  QString("Done: %1 ok, %2 failed\n\n%3").arg(ok).arg(fail).arg(errors.join("\n")));
            }

            self->accept();
          },
          Qt::QueuedConnection);
    });
  }

  std::shared_ptr<ImagePoolService> image_pool_;
  std::shared_ptr<ExportService>    export_service_;
  std::vector<Item>                 items_;

  QLineEdit*                        out_dir_        = nullptr;
  QComboBox*                        format_         = nullptr;
  QComboBox*                        resize_enabled_ = nullptr;
  QSpinBox*                         max_side_       = nullptr;
  QSpinBox*                         quality_        = nullptr;
  QComboBox*                        bit_depth_      = nullptr;
  QSpinBox*                         png_compress_   = nullptr;
  QComboBox*                        tiff_compress_  = nullptr;
  QLabel*                           status_         = nullptr;
};

[[maybe_unused]] static void SetPipelineTemplateToGlobalParams(
    std::shared_ptr<PipelineExecutor> executor) {
  auto&          global_params = executor->GetGlobalParams();

  nlohmann::json to_ws_params;
  to_ws_params["ocio"] = {
      {"src", "ACES2065-1"}, {"dst", "ACEScc"}, {"normalize", true}, {"transform_type", 0}};
  (void)executor->GetStage(PipelineStageName::Basic_Adjustment);
  // basic.SetOperator(OperatorType::TO_WS, to_ws_params, global_params);

  nlohmann::json output_params;
  auto&          output_stage = executor->GetStage(PipelineStageName::Output_Transform);
  output_params["ocio"]       = {
      {"src", "ACEScc"}, {"dst", "Camera Rec.709"}, {"limit", true}, {"transform_type", 1}};
  output_params["aces_odt"] = {{"encoding_space", "rec709"},
                               {"encoding_etof", "gamma_2_2"},
                               {"limiting_space", "rec709"},
                               {"peak_luminance", 100.0f}};
  output_stage.SetOperator(OperatorType::ODT, output_params, global_params);
}

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
      QPen pen(QColor(0x30, 0x31, 0x34, 200));
      pen.setWidthF(2.0);
      pen.setCapStyle(Qt::RoundCap);
      painter.setPen(pen);
      painter.drawArc(r, 0 * 16, 360 * 16);
    }

    // Foreground arc.
    {
      QPen pen(QColor(0x8a, 0xb4, 0xf8, 230));
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

class EditorDialog final : public QDialog {
 public:
  enum class WorkingMode : int { Incremental = 0, Plain = 1 };

  EditorDialog(std::shared_ptr<ImagePoolService> image_pool,
               std::shared_ptr<PipelineGuard> pipeline_guard,
               std::shared_ptr<EditHistoryMgmtService> history_service,
               std::shared_ptr<EditHistoryGuard>       history_guard, sl_element_id_t element_id,
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
    setWindowTitle(QString("Editor - element #%1").arg(static_cast<qulonglong>(element_id_)));
    resize(1500, 1000);

    auto* root = new QHBoxLayout(this);

    viewer_    = new QtEditViewer(this);
    viewer_->setMinimumSize(800, 600);
    viewer_->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    viewer_->setStyleSheet(
        "QOpenGLWidget {"
        "  background: #121212;"
        "  border: 1px solid #303134;"
        "  border-radius: 12px;"
        "}");

    // Viewer container allows a small overlay spinner during rendering.
    viewer_container_ = new QWidget(this);
    auto* viewer_grid = new QGridLayout(viewer_container_);
    viewer_grid->setContentsMargins(0, 0, 0, 0);
    viewer_grid->setSpacing(0);
    viewer_grid->addWidget(viewer_, 0, 0);

    spinner_ = new SpinnerWidget(viewer_container_);
    viewer_grid->addWidget(spinner_, 0, 0, Qt::AlignRight | Qt::AlignBottom);
    viewer_grid->setRowStretch(0, 1);
    viewer_grid->setColumnStretch(0, 1);

    controls_             = new QWidget(this);
    auto* controls_layout = new QVBoxLayout(controls_);
    controls_layout->setContentsMargins(16, 16, 16, 16);
    controls_layout->setSpacing(12);
    controls_->setFixedWidth(420);
    controls_->setStyleSheet(
        "QWidget {"
        "  background: #1E1E1E;"
        "  border: 1px solid #303134;"
        "  border-radius: 12px;"
        "}");

    root->addWidget(viewer_container_, 1);
    root->addWidget(controls_, 0);

    const auto luts_dir  = std::filesystem::path(CONFIG_PATH) / "LUTs";
    const auto lut_files = ListCubeLutsInDir(luts_dir);

    lut_paths_.push_back("");  // index 0 => None
    lut_names_.push_back("None");
    for (const auto& p : lut_files) {
      lut_paths_.push_back(p.generic_string());
      lut_names_.push_back(QString::fromStdString(p.filename().string()));
    }

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
          "  color: #E8EAED;"
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
          "  background: #202124;"
          "  border: 1px solid #303134;"
          "  border-radius: 8px;"
          "  padding: 4px 8px;"
          "}"
          "QComboBox::drop-down {"
          "  border: 0px;"
          "  width: 24px;"
          "}"
          "QComboBox QAbstractItemView {"
          "  background: #202124;"
          "  border: 1px solid #303134;"
          "  selection-background-color: #8ab4f8;"
          "  selection-color: #080A0C;"
          "}"
          "QComboBox QAbstractItemView::item:hover {"
          "  background: #2B2F33;"
          "  color: #E8EAED;"
          "}"
          "QComboBox QAbstractItemView::item:selected {"
          "  background: #8ab4f8;"
          "  color: #080A0C;"
          "}");

      QObject::connect(
          combo, QOverload<int>::of(&QComboBox::currentIndexChanged), controls_,
          [onChange = std::forward<decltype(onChange)>(onChange)](int idx) { onChange(idx); });

      auto* row       = new QWidget(controls_);
      auto* rowLayout = new QHBoxLayout(row);
      rowLayout->setContentsMargins(0, 0, 0, 0);
      rowLayout->addWidget(label, /*stretch*/ 1);
      rowLayout->addWidget(combo);

      controls_layout->insertWidget(controls_layout->count() - 1, row);
      return combo;
    };

    auto addSlider = [&](const QString& name, int min, int max, int value, auto&& onChange,
                         auto&& onRelease,
                         auto&& formatter) {
      auto* info = new QLabel(QString("%1: %2").arg(name).arg(formatter(value)), controls_);
      info->setStyleSheet(
          "QLabel {"
          "  color: #E8EAED;"
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

      QObject::connect(
          slider, &QSlider::valueChanged, controls_,
          [info, name, formatter, onChange = std::forward<decltype(onChange)>(onChange)](int v) {
            info->setText(QString("%1: %2").arg(name).arg(formatter(v)));
            onChange(v);
          });

      QObject::connect(
          slider, &QSlider::sliderReleased, controls_,
          [onRelease = std::forward<decltype(onRelease)>(onRelease)]() { onRelease(); });

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

    saturation_slider_ = addSlider(
        "Saturation", -100, 100, static_cast<int>(std::lround(state_.saturation_)),
        [&](int v) {
          state_.saturation_ = static_cast<float>(v);
          RequestRender();
        },
        [this]() { CommitAdjustment(AdjustmentField::Saturation); },
        [](int v) { return QString::number(v, 'f', 2); });

    tint_slider_ = addSlider(
        "Tint", -100, 100, static_cast<int>(std::lround(state_.tint_)),
        [&](int v) {
          state_.tint_ = static_cast<float>(v);
          RequestRender();
        },
        [this]() { CommitAdjustment(AdjustmentField::Tint); },
        [](int v) { return QString::number(v, 'f', 2); });

    blacks_slider_ = addSlider(
        "Blacks", -100, 100, static_cast<int>(std::lround(state_.blacks_)),
        [&](int v) {
          state_.blacks_ = static_cast<float>(v);
          RequestRender();
        },
        [this]() { CommitAdjustment(AdjustmentField::Blacks); },
        [](int v) { return QString::number(v, 'f', 2); });

    whites_slider_ = addSlider(
        "Whites", -100, 100, static_cast<int>(std::lround(state_.whites_)),
        [&](int v) {
          state_.whites_ = static_cast<float>(v);
          RequestRender();
        },
        [this]() { CommitAdjustment(AdjustmentField::Whites); },
        [](int v) { return QString::number(v, 'f', 2); });

    shadows_slider_ = addSlider(
        "Shadows", -100, 100, static_cast<int>(std::lround(state_.shadows_)),
        [&](int v) {
          state_.shadows_ = static_cast<float>(v);
          RequestRender();
        },
        [this]() { CommitAdjustment(AdjustmentField::Shadows); },
        [](int v) { return QString::number(v, 'f', 2); });

    highlights_slider_ = addSlider(
        "Highlights", -100, 100, static_cast<int>(std::lround(state_.highlights_)),
        [&](int v) {
          state_.highlights_ = static_cast<float>(v);
          RequestRender();
        },
        [this]() { CommitAdjustment(AdjustmentField::Highlights); },
        [](int v) { return QString::number(v, 'f', 2); });

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

    // Edit-history commit controls.
    {
      auto* row = new QWidget(controls_);
      auto* rowLayout = new QHBoxLayout(row);
      rowLayout->setContentsMargins(0, 0, 0, 0);
      rowLayout->setSpacing(10);

      version_status_ = new QLabel(row);
      version_status_->setStyleSheet(
          "QLabel {"
          "  color: #AAB0B6;"
          "  font-size: 12px;"
          "}");

      commit_version_btn_ = new QPushButton("Commit Version", row);
      commit_version_btn_->setFixedHeight(32);
      commit_version_btn_->setStyleSheet(
          "QPushButton {"
          "  background: #2B2F33;"
          "  border: 1px solid #303134;"
          "  border-radius: 8px;"
          "  padding: 6px 10px;"
          "}"
          "QPushButton:hover {"
          "  background: #34383C;"
          "}"
          "QPushButton:disabled {"
          "  color: #6B7075;"
          "  background: #1E1E1E;"
          "}");

      rowLayout->addWidget(version_status_, /*stretch*/ 1);
      rowLayout->addWidget(commit_version_btn_, /*stretch*/ 0);
      controls_layout->insertWidget(controls_layout->count() - 1, row);

      QObject::connect(commit_version_btn_, &QPushButton::clicked, this,
                       [this]() { CommitWorkingVersion(); });
    }

    // Edit-history visualization ("git log"-like) + working version mode.
    {
      auto* frame = new QFrame(controls_);
      frame->setStyleSheet(
          "QFrame {"
          "  background: transparent;"
          "  border: 1px solid #303134;"
          "  border-radius: 12px;"
          "}");

      auto* layout = new QVBoxLayout(frame);
      layout->setContentsMargins(10, 10, 10, 10);
      layout->setSpacing(8);

      const QFont mono = QFontDatabase::systemFont(QFontDatabase::FixedFont);

      auto* mode_row     = new QWidget(frame);
      auto* mode_layout  = new QHBoxLayout(mode_row);
      mode_layout->setContentsMargins(0, 0, 0, 0);
      mode_layout->setSpacing(8);

      auto* mode_label = new QLabel("Working version:", mode_row);
      mode_label->setStyleSheet(
          "QLabel {"
          "  color: #AAB0B6;"
          "  font-size: 12px;"
          "}");

      working_mode_combo_ = new QComboBox(mode_row);
      working_mode_combo_->addItem("Incremental (from latest)", static_cast<int>(WorkingMode::Incremental));
      working_mode_combo_->addItem("Plain (no parent)", static_cast<int>(WorkingMode::Plain));
      working_mode_combo_->setFixedHeight(28);
      working_mode_combo_->setStyleSheet(
          "QComboBox {"
          "  background: #202124;"
          "  border: 1px solid #303134;"
          "  border-radius: 8px;"
          "  padding: 4px 8px;"
          "}"
          "QComboBox::drop-down {"
          "  border: 0px;"
          "  width: 24px;"
          "}"
          "QComboBox QAbstractItemView {"
          "  background: #202124;"
          "  border: 1px solid #303134;"
          "  selection-background-color: #8ab4f8;"
          "  selection-color: #080A0C;"
          "}");

      new_working_btn_ = new QPushButton("New", mode_row);
      new_working_btn_->setFixedHeight(28);
      new_working_btn_->setStyleSheet(
          "QPushButton {"
          "  background: #2B2F33;"
          "  border: 1px solid #303134;"
          "  border-radius: 8px;"
          "  padding: 4px 10px;"
          "}"
          "QPushButton:hover {"
          "  background: #34383C;"
          "}");

      mode_layout->addWidget(mode_label, /*stretch*/ 0);
      mode_layout->addWidget(working_mode_combo_, /*stretch*/ 1);
      mode_layout->addWidget(new_working_btn_, /*stretch*/ 0);

      layout->addWidget(mode_row);

      auto* versions_label = new QLabel("Versions", frame);
      versions_label->setStyleSheet(
          "QLabel {"
          "  color: #E8EAED;"
          "  font-size: 12px;"
          "  font-weight: 500;"
          "}");
      layout->addWidget(versions_label);

      version_log_ = new QListWidget(frame);
      version_log_->setFont(mono);
      version_log_->setSelectionMode(QAbstractItemView::SingleSelection);
      version_log_->setUniformItemSizes(true);
      version_log_->setMinimumHeight(150);
      version_log_->setStyleSheet(
          "QListWidget {"
          "  background: #121212;"
          "  border: 1px solid #303134;"
          "  border-radius: 10px;"
          "  padding: 4px;"
          "}"
          "QListWidget::item {"
          "  padding: 4px 6px;"
          "  border-radius: 6px;"
          "}"
          "QListWidget::item:selected {"
          "  background: rgba(138, 180, 248, 0.22);"
          "}");
      layout->addWidget(version_log_);

      auto* tx_label = new QLabel("Uncommitted transactions (stack)", frame);
      tx_label->setStyleSheet(
          "QLabel {"
          "  color: #E8EAED;"
          "  font-size: 12px;"
          "  font-weight: 500;"
          "}");
      layout->addWidget(tx_label);

      tx_stack_ = new QListWidget(frame);
      tx_stack_->setFont(mono);
      tx_stack_->setSelectionMode(QAbstractItemView::NoSelection);
      tx_stack_->setUniformItemSizes(true);
      tx_stack_->setMinimumHeight(170);
      tx_stack_->setStyleSheet(
          "QListWidget {"
          "  background: #121212;"
          "  border: 1px solid #303134;"
          "  border-radius: 10px;"
          "  padding: 4px;"
          "}"
          "QListWidget::item {"
          "  padding: 4px 6px;"
          "  border-radius: 6px;"
          "}");
      layout->addWidget(tx_stack_, /*stretch*/ 1);

      controls_layout->insertWidget(controls_layout->count() - 1, frame);

      QObject::connect(new_working_btn_, &QPushButton::clicked, this,
                       [this]() { StartNewWorkingVersionFromUi(); });
      QObject::connect(working_mode_combo_, QOverload<int>::of(&QComboBox::currentIndexChanged),
                       this, [this](int) { UpdateVersionUi(); });
    }

    UpdateVersionUi();

    SetupPipeline();

    // Requirement (3): init state render via singleShot.
    QTimer::singleShot(0, this, [this]() {
      state_.type_ = RenderType::FULL_RES_PREVIEW;
      RequestRender();
      state_.type_ = RenderType::FAST_PREVIEW;
    });
  }

 private:
  enum class AdjustmentField {
    Exposure,
    Contrast,
    Saturation,
    Tint,
    Blacks,
    Whites,
    Shadows,
    Highlights,
    Sharpen,
    Clarity,
    Lut,
  };

  struct AdjustmentState {
    float       exposure_   = 1.0f;
    float       contrast_   = 1.0f;
    float       saturation_ = 0.0f;
    float       tint_       = 0.0f;
    float       blacks_     = 0.0f;
    float       whites_     = 0.0f;
    float       shadows_    = 0.0f;
    float       highlights_ = 0.0f;
    float       sharpen_    = 0.0f;
    float       clarity_    = 0.0f;
    std::string lut_path_;
    RenderType  type_ = RenderType::FAST_PREVIEW;
  };

  static bool NearlyEqual(float a, float b) {
    return std::abs(a - b) <= 1e-6f;
  }

  void UpdateVersionUi() {
    if (!version_status_ || !commit_version_btn_) {
      return;
    }

    const size_t tx_count = working_version_.GetAllEditTransactions().size();
    QString      label    = QString("Uncommitted: %1 tx").arg(static_cast<qulonglong>(tx_count));

    if (working_version_.HasParentVersion()) {
      label += QString(" • parent: %1").arg(QString::fromStdString(
          working_version_.GetParentVersionID().ToString().substr(0, 8)));
    } else {
      label += " • plain";
    }

    if (working_mode_combo_) {
      const auto mode = static_cast<WorkingMode>(working_mode_combo_->currentData().toInt());
      label += (mode == WorkingMode::Plain) ? " • mode: plain" : " • mode: incremental";
    }

    if (history_guard_ && history_guard_->history_) {
      try {
        const auto latest_id = history_guard_->history_->GetLatestVersion().ver_ref_.GetVersionID();
        label += QString(" • Latest: %1").arg(QString::fromStdString(latest_id.ToString().substr(
            0, 8)));
      } catch (...) {
      }
    }

    version_status_->setText(label);
    commit_version_btn_->setEnabled(tx_count > 0);

    if (tx_stack_) {
      tx_stack_->clear();
      for (const auto& tx : working_version_.GetAllEditTransactions()) {
        auto* item = new QListWidgetItem(QString::fromStdString(tx.Describe(true, 110)), tx_stack_);
        item->setToolTip(QString::fromStdString(tx.ToJSON().dump(2)));
      }
    }

    if (version_log_) {
      version_log_->clear();
      if (history_guard_ && history_guard_->history_) {
        Hash128 latest_id{};
        try {
          latest_id = history_guard_->history_->GetLatestVersion().ver_ref_.GetVersionID();
        } catch (...) {
        }

        const Hash128 base_parent = working_version_.GetParentVersionID();

        for (auto it = history_guard_->history_->GetCommitTree().rbegin();
             it != history_guard_->history_->GetCommitTree().rend(); ++it) {
          const auto& ver = it->ver_ref_;
          const auto  ver_id = ver.GetVersionID();
          const auto  short_id = QString::fromStdString(ver_id.ToString().substr(0, 8));
          const auto  when = QDateTime::fromSecsSinceEpoch(static_cast<qint64>(ver.GetLastModifiedTime()))
                                .toString("yyyy-MM-dd HH:mm:ss");
          const auto  committed_tx_count =
              static_cast<qulonglong>(ver.GetAllEditTransactions().size());

          QString tags;
          if (ver_id == latest_id) {
            tags += " HEAD";
          }
          if (base_parent == ver_id && working_version_.HasParentVersion()) {
            tags += " base";
          }
          if (!ver.HasParentVersion()) {
            tags += " plain";
          } else {
            tags += QString(" p:%1").arg(QString::fromStdString(
                ver.GetParentVersionID().ToString().substr(0, 8)));
          }

          QString msg;
          const auto& txs = ver.GetAllEditTransactions();
          if (!txs.empty()) {
            msg = QString::fromStdString(txs.front().Describe(true, 70));
          } else {
            msg = "(empty)";
          }

          const QString line =
              QString("* %1  %2  tx:%3  %4%5").arg(short_id).arg(when).arg(committed_tx_count).arg(
                  msg, tags);

          auto* item = new QListWidgetItem(line, version_log_);
          item->setToolTip(QString("version=%1\nparent=%2\ntx=%3")
                               .arg(QString::fromStdString(ver_id.ToString()))
                               .arg(QString::fromStdString(ver.GetParentVersionID().ToString()))
                               .arg(committed_tx_count));
        }
      }
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
      case AdjustmentField::Tint:
        return {PipelineStageName::Color_Adjustment, OperatorType::TINT};
      case AdjustmentField::Blacks:
        return {PipelineStageName::Basic_Adjustment, OperatorType::BLACK};
      case AdjustmentField::Whites:
        return {PipelineStageName::Basic_Adjustment, OperatorType::WHITE};
      case AdjustmentField::Shadows:
        return {PipelineStageName::Basic_Adjustment, OperatorType::SHADOWS};
      case AdjustmentField::Highlights:
        return {PipelineStageName::Basic_Adjustment, OperatorType::HIGHLIGHTS};
      case AdjustmentField::Sharpen:
        return {PipelineStageName::Detail_Adjustment, OperatorType::SHARPEN};
      case AdjustmentField::Clarity:
        return {PipelineStageName::Detail_Adjustment, OperatorType::CLARITY};
      case AdjustmentField::Lut:
        return {PipelineStageName::Color_Adjustment, OperatorType::LMT};
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
      case AdjustmentField::Tint:
        return {{"tint", s.tint_}};
      case AdjustmentField::Blacks:
        return {{"black", s.blacks_}};
      case AdjustmentField::Whites:
        return {{"white", s.whites_}};
      case AdjustmentField::Shadows:
        return {{"shadows", s.shadows_}};
      case AdjustmentField::Highlights:
        return {{"highlights", s.highlights_}};
      case AdjustmentField::Sharpen:
        return {{"sharpen", {{"offset", s.sharpen_}}}};
      case AdjustmentField::Clarity:
        return {{"clarity", s.clarity_}};
      case AdjustmentField::Lut:
        return {{"ocio_lmt", s.lut_path_}};
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
      case AdjustmentField::Tint:
        return !NearlyEqual(state_.tint_, committed_state_.tint_);
      case AdjustmentField::Blacks:
        return !NearlyEqual(state_.blacks_, committed_state_.blacks_);
      case AdjustmentField::Whites:
        return !NearlyEqual(state_.whites_, committed_state_.whites_);
      case AdjustmentField::Shadows:
        return !NearlyEqual(state_.shadows_, committed_state_.shadows_);
      case AdjustmentField::Highlights:
        return !NearlyEqual(state_.highlights_, committed_state_.highlights_);
      case AdjustmentField::Sharpen:
        return !NearlyEqual(state_.sharpen_, committed_state_.sharpen_);
      case AdjustmentField::Clarity:
        return !NearlyEqual(state_.clarity_, committed_state_.clarity_);
      case AdjustmentField::Lut:
        return state_.lut_path_ != committed_state_.lut_path_;
    }
    return false;
  }

  void CommitAdjustment(AdjustmentField field) {
    if (!FieldChanged(field) || !pipeline_guard_ || !pipeline_guard_->pipeline_) {
      // Still fulfill the "full res on release/change" behavior.
      state_.type_ = RenderType::FULL_RES_PREVIEW;
      RequestRender();
      state_.type_ = RenderType::FAST_PREVIEW;
      return;
    }

    const auto [stage_name, op_type] = FieldSpec(field);
    const auto old_params            = ParamsForField(field, committed_state_);
    const auto new_params            = ParamsForField(field, state_);

    auto exec = pipeline_guard_->pipeline_;
    auto& stage = exec->GetStage(stage_name);
    const auto op = stage.GetOperator(op_type);
    const TransactionType tx_type =
        (op.has_value() && op.value() != nullptr) ? TransactionType::_EDIT : TransactionType::_ADD;

    EditTransaction tx{tx_type, op_type, stage_name, new_params};
    tx.SetLastOperatorParams(old_params);
    (void)tx.ApplyTransaction(*exec);

    working_version_.AppendEditTransaction(std::move(tx));
    pipeline_guard_->dirty_ = true;

    committed_state_        = state_;
    UpdateVersionUi();

    state_.type_ = RenderType::FULL_RES_PREVIEW;
    RequestRender();
    state_.type_ = RenderType::FAST_PREVIEW;
  }

  bool LoadStateFromPipelineIfPresent() {
    auto exec = pipeline_guard_ ? pipeline_guard_->pipeline_ : nullptr;
    if (!exec) {
      return false;
    }

    auto ReadFloat = [](const PipelineStage& stage, OperatorType type,
                        const char* key) -> std::optional<float> {
      const auto op = stage.GetOperator(type);
      if (!op.has_value() || op.value() == nullptr) {
        return std::nullopt;
      }
      const auto j = op.value()->ExportOperatorParams();
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

    auto ReadString = [](const PipelineStage& stage, OperatorType type,
                         const char* key) -> std::optional<std::string> {
      const auto op = stage.GetOperator(type);
      if (!op.has_value() || op.value() == nullptr) {
        return std::nullopt;
      }
      const auto j = op.value()->ExportOperatorParams();
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

    const auto& basic  = exec->GetStage(PipelineStageName::Basic_Adjustment);
    const auto& color  = exec->GetStage(PipelineStageName::Color_Adjustment);
    const auto& detail = exec->GetStage(PipelineStageName::Detail_Adjustment);

    // If there is no exposure operator, treat this as a fresh pipeline without saved params.
    if (!basic.GetOperator(OperatorType::EXPOSURE).has_value()) {
      return false;
    }

    if (const auto v = ReadFloat(basic, OperatorType::EXPOSURE, "exposure"); v.has_value()) {
      state_.exposure_ = v.value();
    }
    if (const auto v = ReadFloat(basic, OperatorType::CONTRAST, "contrast"); v.has_value()) {
      state_.contrast_ = v.value();
    }
    if (const auto v = ReadFloat(basic, OperatorType::BLACK, "black"); v.has_value()) {
      state_.blacks_ = v.value();
    }
    if (const auto v = ReadFloat(basic, OperatorType::WHITE, "white"); v.has_value()) {
      state_.whites_ = v.value();
    }
    if (const auto v = ReadFloat(basic, OperatorType::SHADOWS, "shadows"); v.has_value()) {
      state_.shadows_ = v.value();
    }
    if (const auto v = ReadFloat(basic, OperatorType::HIGHLIGHTS, "highlights"); v.has_value()) {
      state_.highlights_ = v.value();
    }

    if (const auto v = ReadFloat(color, OperatorType::SATURATION, "saturation"); v.has_value()) {
      state_.saturation_ = v.value();
    }
    if (const auto v = ReadFloat(color, OperatorType::TINT, "tint"); v.has_value()) {
      state_.tint_ = v.value();
    }

    if (const auto v = ReadNestedFloat(detail, OperatorType::SHARPEN, "sharpen", "offset");
        v.has_value()) {
      state_.sharpen_ = v.value();
    }
    if (const auto v = ReadFloat(detail, OperatorType::CLARITY, "clarity"); v.has_value()) {
      state_.clarity_ = v.value();
    }

    // LUT (LMT): if a path exists, treat it as enabled.
    const auto lut = ReadString(color, OperatorType::LMT, "ocio_lmt");
    if (lut.has_value() && !lut->empty()) {
      state_.lut_path_ = *lut;
    } else {
      // Fall back to global flag, but keep empty path if none is specified.
      state_.lut_path_.clear();
    }

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
    decode_params["raw"]["cuda"] = false;
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
    // color.SetOperator(OperatorType::TINT, {{"tint", 0.0f}}, global_params);

    // auto& detail = exec->GetStage(PipelineStageName::Detail_Adjustment);
    // detail.SetOperator(OperatorType::SHARPEN, {{"sharpen", {{"offset", 0.0f}}}}, global_params);
    // detail.SetOperator(OperatorType::CLARITY, {{"clarity", 0.0f}}, global_params);

    exec->SetExecutionStages(viewer_);

    // Cached pipelines can clear transient GPU resources when returned to the service.
    // PipelineMgmtService now resyncs global params on load, so we no longer need a
    // per-dialog LMT rebind hack here.
    last_applied_lut_path_.clear();
  }

  void ApplyStateToPipeline() {
    auto  exec          = pipeline_guard_->pipeline_;
    auto& global_params = exec->GetGlobalParams();

    auto& basic         = exec->GetStage(PipelineStageName::Basic_Adjustment);
    basic.SetOperator(OperatorType::EXPOSURE, {{"exposure", state_.exposure_}}, global_params);
    basic.SetOperator(OperatorType::CONTRAST, {{"contrast", state_.contrast_}}, global_params);
    basic.SetOperator(OperatorType::BLACK, {{"black", state_.blacks_}}, global_params);
    basic.SetOperator(OperatorType::WHITE, {{"white", state_.whites_}}, global_params);
    basic.SetOperator(OperatorType::SHADOWS, {{"shadows", state_.shadows_}}, global_params);
    basic.SetOperator(OperatorType::HIGHLIGHTS, {{"highlights", state_.highlights_}},
                      global_params);

    auto& color = exec->GetStage(PipelineStageName::Color_Adjustment);
    color.SetOperator(OperatorType::SATURATION, {{"saturation", state_.saturation_}},
                      global_params);
    color.SetOperator(OperatorType::TINT, {{"tint", state_.tint_}}, global_params);

    // LUT (LMT): rebind only when the path changes. The operator's SetGlobalParams now
    // derives lmt_enabled_/dirty state from the path, and PipelineMgmtService resyncs on load.
    if (state_.lut_path_ != last_applied_lut_path_) {
      color.SetOperator(OperatorType::LMT, {{"ocio_lmt", state_.lut_path_}}, global_params);
      last_applied_lut_path_ = state_.lut_path_;
    }

    auto& detail = exec->GetStage(PipelineStageName::Detail_Adjustment);
    detail.SetOperator(OperatorType::SHARPEN, {{"sharpen", {{"offset", state_.sharpen_}}}},
                       global_params);
    detail.SetOperator(OperatorType::CLARITY, {{"clarity", state_.clarity_}}, global_params);
  }

  void RequestRender() {
    pending_     = state_;
    has_pending_ = true;
    if (!inflight_) {
      StartNext();
    }
  }

  void EnsurePollTimer() {
    if (poll_timer_) {
      return;
    }
    poll_timer_ = new QTimer(this);
    poll_timer_->setInterval(16);
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
    if (!has_pending_) {
      return;
    }

    state_       = pending_;
    has_pending_ = false;

    if (spinner_) {
      spinner_->Start();
      // Ensure the spinner paints before any potentially blocking work.
      QCoreApplication::processEvents(QEventLoop::ExcludeUserInputEvents);
    }

    ApplyStateToPipeline();
    pipeline_guard_->dirty_                 = true;

    PipelineTask task                       = base_task_;
    task.options_.render_desc_.render_type_ = state_.type_;
    task.options_.is_callback_              = false;
    task.options_.is_seq_callback_          = false;
    task.options_.is_blocking_              = true;

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

    if (has_pending_) {
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

  QtEditViewer*                                            viewer_           = nullptr;
  QWidget*                                                 viewer_container_ = nullptr;
  SpinnerWidget*                                           spinner_          = nullptr;
  QWidget*                                                 controls_         = nullptr;
  QComboBox*                                               lut_combo_        = nullptr;
  QSlider*                                                 exposure_slider_  = nullptr;
  QSlider*                                                 contrast_slider_  = nullptr;
  QSlider*                                                 saturation_slider_ = nullptr;
  QSlider*                                                 tint_slider_      = nullptr;
  QSlider*                                                 blacks_slider_    = nullptr;
  QSlider*                                                 whites_slider_    = nullptr;
  QSlider*                                                 shadows_slider_   = nullptr;
  QSlider*                                                 highlights_slider_ = nullptr;
  QSlider*                                                 sharpen_slider_   = nullptr;
  QSlider*                                                 clarity_slider_   = nullptr;
  QLabel*                                                  version_status_   = nullptr;
  QPushButton*                                             commit_version_btn_ = nullptr;
  QComboBox*                                               working_mode_combo_ = nullptr;
  QPushButton*                                             new_working_btn_ = nullptr;
  QListWidget*                                             version_log_ = nullptr;
  QListWidget*                                             tx_stack_ = nullptr;
  QTimer*                                                  poll_timer_       = nullptr;
  std::optional<std::future<std::shared_ptr<ImageBuffer>>> inflight_future_{};

  std::vector<std::string>                                 lut_paths_{};
  QStringList                                              lut_names_{};

  std::string                                              last_applied_lut_path_{};
  AdjustmentState                                          state_{};
  AdjustmentState                                          committed_state_{};
  Version                                                  working_version_{};
  AdjustmentState                                          pending_{};
  bool                                                     inflight_    = false;
  bool                                                     has_pending_ = false;
};

class AlbumWindow final : public QWidget {
 public:
  AlbumWindow(std::shared_ptr<ProjectService> project, std::filesystem::path meta_path,
              std::shared_ptr<ThumbnailService>    thumbnail_service,
              std::shared_ptr<ImagePoolService>    image_pool,
              std::shared_ptr<PipelineMgmtService> pipeline_service, QWidget* parent = nullptr)
      : QWidget(parent),
        project_(std::move(project)),
        meta_path_(std::move(meta_path)),
        thumbnails_(std::move(thumbnail_service)),
        image_pool_(std::move(image_pool)),
        pipeline_service_(std::move(pipeline_service)),
        import_service_(project_ ? project_->GetSleeveService() : nullptr,
                        project_ ? project_->GetImagePoolService() : nullptr) {
    if (!project_ || !thumbnails_ || !image_pool_ || !pipeline_service_) {
      throw std::runtime_error("AlbumWindow: missing services");
    }

    export_service_ = std::make_shared<ExportService>(project_->GetSleeveService(), image_pool_,
                                                      pipeline_service_);
    filter_service_ = std::make_unique<SleeveFilterService>(project_->GetStorageService());
    history_service_ = std::make_shared<EditHistoryMgmtService>(project_->GetStorageService());

    auto* outer     = new QHBoxLayout(this);
    outer->setContentsMargins(10, 10, 10, 10);
    outer->setSpacing(12);

    filter_drawer_ = new FilterDrawer(this);
    outer->addWidget(filter_drawer_, 0);

    auto* root = new QVBoxLayout();
    root->setSpacing(8);
    outer->addLayout(root, 1);

    auto* top   = new QHBoxLayout();
    import_btn_ = new QPushButton("Import…", this);
    export_btn_ = new QPushButton("Export…", this);
    status_     = new QLabel("No images. Click Import…", this);
    top->addWidget(import_btn_, 0);
    top->addWidget(export_btn_, 0);

    filters_btn_ = new QToolButton(this);
    filters_btn_->setText("Filters");
    filters_btn_->setToolButtonStyle(Qt::ToolButtonTextBesideIcon);
    filters_btn_->setCheckable(true);
    filters_btn_->setChecked(true);
    filters_btn_->setIcon(style()->standardIcon(QStyle::SP_FileDialogContentsView));
    filters_btn_->setToolTip("Show/hide filter drawer");
    top->addWidget(filters_btn_, 0);

    top->addWidget(status_, 1);
    root->addLayout(top);

    list_ = new QListWidget(this);
    list_->setViewMode(QListView::IconMode);
    list_->setResizeMode(QListWidget::Adjust);
    list_->setMovement(QListView::Static);
    list_->setIconSize(QSize(220, 160));
    list_->setGridSize(QSize(240, 190));
    list_->setSpacing(8);
    list_->setSelectionMode(QAbstractItemView::ExtendedSelection);
    root->addWidget(list_, 1);

    connect(import_btn_, &QPushButton::clicked, this, [this]() { BeginImport(); });
    connect(export_btn_, &QPushButton::clicked, this, [this]() { OpenExport(); });
    connect(filters_btn_, &QToolButton::toggled, this, [this](bool on) {
      if (filter_drawer_) {
        filter_drawer_->setVisible(on);
      }
    });
    connect(list_, &QListWidget::itemClicked, this, [this](QListWidgetItem* item) {
      if (!item) {
        return;
      }
      const auto element_id = static_cast<sl_element_id_t>(item->data(Qt::UserRole + 1).toUInt());
      const auto image_id   = static_cast<image_id_t>(item->data(Qt::UserRole + 2).toUInt());
      OpenEditor(element_id, image_id);
    });

    if (filter_drawer_) {
      filter_drawer_->SetOnApply([this](const FilterNode& node) { ApplyFilter(node, true); });
      filter_drawer_->SetOnClear([this]() { ClearFilter(); });
      filter_drawer_->SetResultsSummary(/*shown=*/0, /*total=*/0);
    }
  }

 private:
  int VisibleItemCount() const {
    if (!list_) {
      return 0;
    }
    int shown = 0;
    for (int i = 0; i < list_->count(); ++i) {
      auto* it = list_->item(i);
      if (it && !it->isHidden()) {
        ++shown;
      }
    }
    return shown;
  }

  void UpdateStatusText() {
    if (!status_) {
      return;
    }
    const int total = list_ ? list_->count() : 0;
    if (total <= 0) {
      status_->setText("No images. Click Import…");
      if (filter_drawer_) {
        filter_drawer_->SetResultsSummary(0, 0);
      }
      return;
    }

    const int shown = VisibleItemCount();
    status_->setText(QString("Showing %1 of %2 images").arg(shown).arg(total));
    if (filter_drawer_) {
      filter_drawer_->SetResultsSummary(shown, total);
    }
  }

  void SetBusyUi(bool busy, const QString& label) {
    import_btn_->setEnabled(!busy);
    export_btn_->setEnabled(!busy);
    if (filters_btn_) {
      filters_btn_->setEnabled(!busy);
    }
    if (filter_drawer_) {
      filter_drawer_->setEnabled(!busy);
    }
    list_->setEnabled(!busy);
    if (busy) {
      if (!busy_) {
        busy_ = new QProgressDialog(label, QString(), 0, 0, this);
        busy_->setWindowModality(Qt::ApplicationModal);
        busy_->setCancelButton(nullptr);
        busy_->setMinimumDuration(0);
        busy_->show();
      } else {
        busy_->setLabelText(label);
        busy_->show();
      }
    } else {
      if (busy_) {
        busy_->close();
        busy_->deleteLater();
        busy_ = nullptr;
      }
    }
  }

  void BeginImport() {
    if (import_inflight_) {
      return;
    }

    const QStringList files = QFileDialog::getOpenFileNames(
        this, "Import images", QString(),
        "Images (*.dng *.nef *.cr2 *.cr3 *.arw *.rw2 *.raf *.tif *.tiff *.jpg *.jpeg *.png);;All "
        "Files (*)");
    if (files.isEmpty()) {
      return;
    }

    std::vector<image_path_t> paths;
    paths.reserve(static_cast<size_t>(files.size()));
    for (const auto& f : files) {
      std::filesystem::path p = std::filesystem::path(f.toStdWString());
      if (is_supported_file(p)) {
        paths.push_back(std::move(p));
      }
    }

    if (paths.empty()) {
      QMessageBox::information(this, "Import", "No supported images selected.");
      return;
    }

    import_inflight_ = true;
    SetBusyUi(true, "Importing…");

    auto job            = std::make_shared<ImportJob>();
    current_import_job_ = job;

    QPointer<AlbumWindow> self(this);
    job->on_progress_ = [self](const ImportProgress& p) {
      if (!self) {
        return;
      }
      const uint32_t total        = p.total_;
      const uint32_t placeholders = p.placeholders_created_.load();
      const uint32_t meta_done    = p.metadata_done_.load();
      const uint32_t failed       = p.failed_.load();

      QMetaObject::invokeMethod(
          self,
          [self, total, placeholders, meta_done, failed]() {
            if (!self) {
              return;
            }
            self->SetBusyUi(true, QString("Importing… %1/%2 (meta %3, failed %4)")
                                      .arg(placeholders)
                                      .arg(total)
                                      .arg(meta_done)
                                      .arg(failed));
          },
          Qt::QueuedConnection);
    };

    job->on_finished_ = [self](const ImportResult& r) {
      if (!self) {
        return;
      }
      QMetaObject::invokeMethod(
          self,
          [self, r]() {
            if (!self) {
              return;
            }
            self->FinishImport(r);
          },
          Qt::QueuedConnection);
    };

    current_import_job_ = import_service_.ImportToFolder(paths, image_path_t{}, {}, job);
  }

  void FinishImport(const ImportResult& result) {
    import_inflight_ = false;
    SetBusyUi(false, "");

    if (!current_import_job_ || !current_import_job_->import_log_) {
      QMessageBox::warning(this, "Import", "Import finished but no log snapshot available.");
      return;
    }

    const auto snapshot = current_import_job_->import_log_->Snapshot();

    import_service_.SyncImports(snapshot, image_path_t{});
    project_->GetSleeveService()->Sync();
    project_->GetImagePoolService()->SyncWithStorage();
    project_->SaveProject(meta_path_);

    if (result.failed_ != 0) {
      QMessageBox::warning(this, "Import",
                           QString("Import finished: %1 imported, %2 failed.")
                               .arg(result.imported_)
                               .arg(result.failed_));
    }

    for (const auto& c : snapshot.created_) {
      AddAlbumItem(c.element_id_, c.image_id_);
    }

    if (active_filter_node_.has_value()) {
      ApplyFilter(active_filter_node_.value(), /*show_dialogs=*/false);
    } else {
      UpdateStatusText();
    }
  }

  void AddAlbumItem(sl_element_id_t element_id, image_id_t image_id) {
    auto* item = new QListWidgetItem();
    item->setText(QString("#%1").arg(static_cast<qulonglong>(element_id)));
    item->setData(Qt::UserRole + 1, static_cast<uint32_t>(element_id));
    item->setData(Qt::UserRole + 2, static_cast<uint32_t>(image_id));
    item->setSizeHint(QSize(240, 190));
    list_->addItem(item);

    items_by_element_[element_id]    = item;

    // Requirement (1): clickable cells; thumbnails load async.
    CallbackDispatcher ui_dispatcher = [](std::function<void()> fn) {
      auto* obj = QCoreApplication::instance();
      if (!obj) {
        fn();
        return;
      }
      QMetaObject::invokeMethod(obj, std::move(fn), Qt::QueuedConnection);
    };

    RefreshThumbnailForItem(element_id, image_id, item, /*invalidate=*/false);
  }

  void ClearFilter() {
    if (active_filter_id_.has_value() && filter_service_) {
      try {
        filter_service_->RemoveFilterCombo(active_filter_id_.value());
      } catch (...) {
      }
      active_filter_id_.reset();
    }
    active_filter_node_.reset();

    if (!list_) {
      return;
    }
    for (int i = 0; i < list_->count(); ++i) {
      auto* it = list_->item(i);
      if (it) {
        it->setHidden(false);
      }
    }
    list_->clearSelection();
    UpdateStatusText();
  }

  void ApplyFilter(const FilterNode& node, bool show_dialogs) {
    if (!filter_service_ || !list_) {
      return;
    }
    if (list_->count() == 0) {
      if (show_dialogs) {
        QMessageBox::information(this, "Filter", "No images loaded.");
      }
      return;
    }

    // Filters are immutable and results are cached by filter_id_. Always create a new filter.
    const filter_id_t new_filter_id = filter_service_->CreateFilterCombo(node);
    const auto        filter_opt    = filter_service_->GetFilterCombo(new_filter_id);
    if (!filter_opt.has_value() || !filter_opt.value()) {
      if (show_dialogs) {
        QMessageBox::warning(this, "Filter", "Filter creation failed.");
      }
      return;
    }

    // Replace previous filter to avoid unbounded growth of in-memory caches.
    if (active_filter_id_.has_value()) {
      try {
        filter_service_->RemoveFilterCombo(active_filter_id_.value());
      } catch (...) {
      }
    }
    active_filter_id_   = new_filter_id;
    active_filter_node_ = node;

    std::vector<sl_element_id_t> filtered_ids;
    try {
      // Root folder id is 0.
      const auto ids_opt = filter_service_->ApplyFilterOn(new_filter_id, /*parent_id=*/0);
      if (!ids_opt.has_value()) {
        if (show_dialogs) {
          QMessageBox::warning(this, "Filter", "Unknown filter id.");
        }
        return;
      }
      filtered_ids = std::move(ids_opt.value());
    } catch (const std::exception& e) {
      if (show_dialogs) {
        QMessageBox::warning(this, "Filter", QString("Filter failed: %1").arg(e.what()));
      }
      return;
    }

    std::unordered_set<sl_element_id_t> allow;
    allow.reserve(filtered_ids.size() * 2 + 1);
    for (const auto id : filtered_ids) {
      allow.insert(id);
    }

    int shown = 0;
    for (int i = 0; i < list_->count(); ++i) {
      auto* it = list_->item(i);
      if (!it) {
        continue;
      }
      const auto element_id = static_cast<sl_element_id_t>(it->data(Qt::UserRole + 1).toUInt());
      const bool keep       = allow.contains(element_id);
      it->setHidden(!keep);
      if (keep) {
        ++shown;
      }
    }

    if (show_dialogs && shown == 0) {
      QMessageBox::information(this, "Filter", "No matches.");
    }

    list_->clearSelection();
    list_->scrollToTop();
    UpdateStatusText();
  }

  void RefreshThumbnailForItem(sl_element_id_t element_id, image_id_t image_id,
                               QListWidgetItem* item, bool invalidate) {
    if (!item || !thumbnails_) {
      return;
    }

    // Requirement: allow redraw after editor closes.
    if (invalidate) {
      try {
        thumbnails_->InvalidateThumbnail(element_id);
      } catch (...) {
      }
    }

    CallbackDispatcher ui_dispatcher = [](std::function<void()> fn) {
      auto* obj = QCoreApplication::instance();
      if (!obj) {
        fn();
        return;
      }
      QMetaObject::invokeMethod(obj, std::move(fn), Qt::QueuedConnection);
    };

    auto                  svc = thumbnails_;
    QPointer<AlbumWindow> self(this);
    thumbnails_->GetThumbnail(
        element_id, image_id,
        [self, svc, element_id, item, ui_dispatcher](std::shared_ptr<ThumbnailGuard> guard) {
          if (!guard || !guard->thumbnail_buffer_) {
            return;
          }

          std::thread([self, svc, element_id, item, ui_dispatcher,
                       guard = std::move(guard)]() mutable {
            QImage scaled;
            try {
              auto* buf = guard->thumbnail_buffer_.get();
              if (!buf) {
                throw std::runtime_error("null buffer");
              }
              if (!buf->cpu_data_valid_ && buf->gpu_data_valid_) {
                buf->SyncToCPU();
              }
              if (buf->cpu_data_valid_) {
                QImage img = MatRgba32fToQImageCopy(buf->GetCPUData());
                scaled     = img.scaled(220, 160, Qt::KeepAspectRatio, Qt::SmoothTransformation);
              }
            } catch (...) {
            }

            ui_dispatcher([self, item, scaled]() mutable {
              if (!self || !item) {
                return;
              }
              if (!self->list_ || self->list_->row(item) < 0) {
                return;
              }
              if (!scaled.isNull()) {
                item->setIcon(QIcon(QPixmap::fromImage(scaled)));
              }
            });

            try {
              if (svc) {
                svc->ReleaseThumbnail(element_id);
              }
            } catch (...) {
            }
          }).detach();
        },
        /*pin_if_found=*/true, ui_dispatcher);
  }

  void OpenEditor(sl_element_id_t element_id, image_id_t image_id) {
    if (element_id == 0) {
      return;
    }

    // Requirement (2): create a new preview scheduler (in dialog ctor) and pass pipeline guard.
    std::shared_ptr<PipelineGuard> guard;
    try {
      guard = pipeline_service_->LoadPipeline(element_id);
    } catch (const std::exception& e) {
      QMessageBox::warning(this, "Editor", QString("LoadPipeline failed: %1").arg(e.what()));
      return;
    }

    std::shared_ptr<EditHistoryGuard> history_guard;
    try {
      history_guard = history_service_ ? history_service_->LoadHistory(element_id) : nullptr;
    } catch (const std::exception& e) {
      QMessageBox::warning(this, "Editor", QString("LoadHistory failed: %1").arg(e.what()));
      return;
    }

    try {
      EditorDialog dlg(image_pool_, guard, history_service_, history_guard, element_id, image_id,
                       this);
      dlg.exec();
    } catch (const std::exception& e) {
      QMessageBox::warning(this, "Editor", QString("Editor failed: %1").arg(e.what()));
    }

    // Requirement (4): return PipelineGuard back to cell / persist to DB.
    pipeline_service_->SavePipeline(guard);
    pipeline_service_->Sync();

    if (history_service_) {
      history_service_->SaveHistory(history_guard);
      history_service_->Sync();
    }
    project_->SaveProject(meta_path_);

    // Refresh thumbnail for this cell after edits.
    auto it = items_by_element_.find(element_id);
    if (it != items_by_element_.end()) {
      RefreshThumbnailForItem(element_id, image_id, it->second, /*invalidate=*/true);
    }
  }

  void OpenExport() {
    if (!export_service_) {
      QMessageBox::warning(this, "Export", "ExportService not available.");
      return;
    }

    std::vector<ExportDialog::Item> items;

    const auto                      selected = list_->selectedItems();
    if (!selected.isEmpty()) {
      items.reserve(static_cast<size_t>(selected.size()));
      for (auto* it : selected) {
        if (!it) {
          continue;
        }
        ExportDialog::Item e;
        e.sleeve_id_ = static_cast<sl_element_id_t>(it->data(Qt::UserRole + 1).toUInt());
        e.image_id_  = static_cast<image_id_t>(it->data(Qt::UserRole + 2).toUInt());
        if (e.sleeve_id_ != 0 && e.image_id_ != 0) {
          items.push_back(e);
        }
      }
    } else {
      items.reserve(static_cast<size_t>(list_->count()));
      for (int i = 0; i < list_->count(); ++i) {
        auto* it = list_->item(i);
        if (!it) {
          continue;
        }
        ExportDialog::Item e;
        e.sleeve_id_ = static_cast<sl_element_id_t>(it->data(Qt::UserRole + 1).toUInt());
        e.image_id_  = static_cast<image_id_t>(it->data(Qt::UserRole + 2).toUInt());
        if (e.sleeve_id_ != 0 && e.image_id_ != 0) {
          items.push_back(e);
        }
      }
    }

    if (items.empty()) {
      QMessageBox::information(this, "Export", "No images to export.");
      return;
    }

    try {
      ExportDialog dlg(image_pool_, export_service_, std::move(items), this);
      dlg.exec();
    } catch (const std::exception& e) {
      QMessageBox::warning(this, "Export", QString("Export dialog failed: %1").arg(e.what()));
    }
  }

  std::shared_ptr<ProjectService>                       project_;
  std::filesystem::path                                 meta_path_;
  std::shared_ptr<ThumbnailService>                     thumbnails_;
  std::shared_ptr<ImagePoolService>                     image_pool_;
  std::shared_ptr<PipelineMgmtService>                  pipeline_service_;
  std::shared_ptr<EditHistoryMgmtService>              history_service_;
  std::shared_ptr<ExportService>                        export_service_;
  ImportServiceImpl                                     import_service_;

  std::unique_ptr<SleeveFilterService>                  filter_service_;
  std::optional<filter_id_t>                            active_filter_id_;
  std::optional<FilterNode>                             active_filter_node_;

  QPushButton*                                          import_btn_    = nullptr;
  QPushButton*                                          export_btn_    = nullptr;
  QToolButton*                                          filters_btn_   = nullptr;
  QLabel*                                               status_        = nullptr;
  QListWidget*                                          list_          = nullptr;
  FilterDrawer*                                         filter_drawer_ = nullptr;

  std::unordered_map<sl_element_id_t, QListWidgetItem*> items_by_element_{};

  bool                                                  import_inflight_ = false;
  std::shared_ptr<ImportJob>                            current_import_job_{};
  QProgressDialog*                                      busy_ = nullptr;
};

}  // namespace
}  // namespace puerhlab

int main(int argc, char* argv[]) {
  Exiv2::LogMsg::setLevel(Exiv2::LogMsg::Level::mute);
  puerhlab::RegisterAllOperators();

  QCoreApplication::setAttribute(Qt::AA_UseDesktopOpenGL);

  QApplication app(argc, argv);
  puerhlab::ApplyExternalAppFont(app, argc, argv);
  puerhlab::ApplyMaterialLikeTheme(app);

  try {
    const auto db_path   = std::filesystem::temp_directory_path() / "album_editor_demo.db";
    const auto meta_path = std::filesystem::temp_directory_path() / "album_editor_demo.json";

    if (std::filesystem::exists(db_path)) {
      std::filesystem::remove(db_path);
    }
    if (std::filesystem::exists(meta_path)) {
      std::filesystem::remove(meta_path);
    }

    auto project = std::make_shared<puerhlab::ProjectService>(db_path, meta_path);
    auto pipeline_service =
        std::make_shared<puerhlab::PipelineMgmtService>(project->GetStorageService());
    auto thumb_service = std::make_shared<puerhlab::ThumbnailService>(
        project->GetSleeveService(), project->GetImagePoolService(), pipeline_service);

    auto* w = new puerhlab::AlbumWindow(project, meta_path, thumb_service,
                                        project->GetImagePoolService(), pipeline_service);
    w->setWindowTitle("pu-erh_lab - Album + Editor (Qt Demo)");
    w->resize(1400, 900);
    w->show();

    const int rc = app.exec();

    pipeline_service->Sync();
    project->GetImagePoolService()->SyncWithStorage();
    project->SaveProject(meta_path);
    return rc;
  } catch (const std::exception& e) {
    std::cerr << "[AlbumEditorQtDemo] Fatal: " << e.what() << std::endl;
    return 1;
  }
}
