#include "ui/puerhlab_main/album_backend/filter_rule_model.hpp"

#include <algorithm>

namespace puerhlab::ui {
namespace {

auto ToOption(const QString& text, int value) -> QVariantMap {
  return QVariantMap{{"text", text}, {"value", value}};
}

}  // namespace

FilterRuleModel::FilterRuleModel(QObject* parent) : QAbstractListModel(parent) {
  rules_.push_back(Rule{});
}

auto FilterRuleModel::rowCount(const QModelIndex& parent) const -> int {
  if (parent.isValid()) {
    return 0;
  }
  return static_cast<int>(rules_.size());
}

auto FilterRuleModel::data(const QModelIndex& index, int role) const -> QVariant {
  if (!index.isValid() || index.row() < 0 || index.row() >= rowCount()) {
    return {};
  }

  const Rule& rule = rules_.at(static_cast<size_t>(index.row()));
  switch (role) {
    case FieldRole:
      return static_cast<int>(rule.field);
    case OpRole:
      return static_cast<int>(rule.op);
    case ValueRole:
      return rule.value;
    case Value2Role:
      return rule.value2;
    case ShowSecondValueRole:
      return IsBetween(rule.op);
    case PlaceholderRole:
      return PlaceholderForField(rule.field);
    case OpOptionsRole:
      return CompareOptionsForField(rule.field);
    default:
      return {};
  }
}

auto FilterRuleModel::setData(const QModelIndex& index, const QVariant& value, int role) -> bool {
  if (!index.isValid() || index.row() < 0 || index.row() >= rowCount()) {
    return false;
  }

  switch (role) {
    case FieldRole:
      SetField(index.row(), value.toInt());
      return true;
    case OpRole:
      SetOp(index.row(), value.toInt());
      return true;
    case ValueRole:
      SetValue(index.row(), value.toString());
      return true;
    case Value2Role:
      SetValue2(index.row(), value.toString());
      return true;
    default:
      return false;
  }
}

auto FilterRuleModel::flags(const QModelIndex& index) const -> Qt::ItemFlags {
  if (!index.isValid()) {
    return Qt::NoItemFlags;
  }
  return Qt::ItemIsEnabled | Qt::ItemIsSelectable | Qt::ItemIsEditable;
}

auto FilterRuleModel::roleNames() const -> QHash<int, QByteArray> {
  return {
      {FieldRole, "fieldValue"},
      {OpRole, "opValue"},
      {ValueRole, "valueText"},
      {Value2Role, "value2Text"},
      {ShowSecondValueRole, "showSecondValue"},
      {PlaceholderRole, "placeholder"},
      {OpOptionsRole, "opOptions"},
  };
}

void FilterRuleModel::AddRule() {
  const int row = rowCount();
  beginInsertRows(QModelIndex(), row, row);
  rules_.push_back(Rule{});
  endInsertRows();
}

void FilterRuleModel::RemoveRule(int index) {
  if (index < 0 || index >= rowCount()) {
    return;
  }
  beginRemoveRows(QModelIndex(), index, index);
  rules_.erase(rules_.begin() + index);
  endRemoveRows();
}

void FilterRuleModel::ClearAndReset() {
  beginResetModel();
  rules_.clear();
  rules_.push_back(Rule{});
  endResetModel();
}

void FilterRuleModel::SetField(int index, int fieldValue) {
  if (index < 0 || index >= rowCount()) {
    return;
  }

  auto& rule      = rules_[static_cast<size_t>(index)];
  rule.field      = static_cast<FilterField>(fieldValue);
  if (!OpAllowedForField(rule.field, rule.op)) {
    rule.op = DefaultOpForField(rule.field);
  }

  emit dataChanged(createIndex(index, 0), createIndex(index, 0),
                   {FieldRole, OpRole, ShowSecondValueRole, PlaceholderRole, OpOptionsRole});
}

void FilterRuleModel::SetOp(int index, int opValue) {
  if (index < 0 || index >= rowCount()) {
    return;
  }

  auto& rule = rules_[static_cast<size_t>(index)];
  rule.op    = static_cast<CompareOp>(opValue);
  if (!OpAllowedForField(rule.field, rule.op)) {
    rule.op = DefaultOpForField(rule.field);
  }

  emit dataChanged(createIndex(index, 0), createIndex(index, 0), {OpRole, ShowSecondValueRole});
}

void FilterRuleModel::SetValue(int index, const QString& value) {
  if (index < 0 || index >= rowCount()) {
    return;
  }

  auto& rule = rules_[static_cast<size_t>(index)];
  if (rule.value == value) {
    return;
  }
  rule.value = value;
  emit dataChanged(createIndex(index, 0), createIndex(index, 0), {ValueRole});
}

void FilterRuleModel::SetValue2(int index, const QString& value) {
  if (index < 0 || index >= rowCount()) {
    return;
  }

  auto& rule = rules_[static_cast<size_t>(index)];
  if (rule.value2 == value) {
    return;
  }
  rule.value2 = value;
  emit dataChanged(createIndex(index, 0), createIndex(index, 0), {Value2Role});
}

auto FilterRuleModel::FieldOptions() const -> QVariantList {
  return {
      ToOption("Camera Model", static_cast<int>(FilterField::ExifCameraModel)),
      ToOption("ISO", static_cast<int>(FilterField::ExifISO)),
      ToOption("Aperture", static_cast<int>(FilterField::ExifAperture)),
      ToOption("Focal Length", static_cast<int>(FilterField::ExifFocalLength)),
      ToOption("Capture Date", static_cast<int>(FilterField::CaptureDate)),
      ToOption("Import Date", static_cast<int>(FilterField::ImportDate)),
      ToOption("Rating", static_cast<int>(FilterField::Rating)),
  };
}

auto FilterRuleModel::CompareOptionsForField(FilterField field) -> QVariantList {
  const auto kind = KindForField(field);
  if (kind == FilterValueKind::String) {
    return {
        ToOption("contains", static_cast<int>(CompareOp::CONTAINS)),
        ToOption("not contains", static_cast<int>(CompareOp::NOT_CONTAINS)),
        ToOption("=", static_cast<int>(CompareOp::EQUALS)),
        ToOption("!=", static_cast<int>(CompareOp::NOT_EQUALS)),
        ToOption("starts with", static_cast<int>(CompareOp::STARTS_WITH)),
        ToOption("ends with", static_cast<int>(CompareOp::ENDS_WITH)),
        ToOption("regex", static_cast<int>(CompareOp::REGEX)),
    };
  }
  if (kind == FilterValueKind::Int64 || kind == FilterValueKind::Double) {
    return {
        ToOption("=", static_cast<int>(CompareOp::EQUALS)),
        ToOption("!=", static_cast<int>(CompareOp::NOT_EQUALS)),
        ToOption(">", static_cast<int>(CompareOp::GREATER_THAN)),
        ToOption("<", static_cast<int>(CompareOp::LESS_THAN)),
        ToOption(">=", static_cast<int>(CompareOp::GREATER_EQUAL)),
        ToOption("<=", static_cast<int>(CompareOp::LESS_EQUAL)),
        ToOption("between", static_cast<int>(CompareOp::BETWEEN)),
    };
  }

  return {
      ToOption("=", static_cast<int>(CompareOp::EQUALS)),
      ToOption("!=", static_cast<int>(CompareOp::NOT_EQUALS)),
      ToOption(">", static_cast<int>(CompareOp::GREATER_THAN)),
      ToOption("<", static_cast<int>(CompareOp::LESS_THAN)),
      ToOption(">=", static_cast<int>(CompareOp::GREATER_EQUAL)),
      ToOption("<=", static_cast<int>(CompareOp::LESS_EQUAL)),
      ToOption("between", static_cast<int>(CompareOp::BETWEEN)),
  };
}

auto FilterRuleModel::PlaceholderForField(FilterField field) -> QString {
  switch (field) {
    case FilterField::CaptureDate:
    case FilterField::ImportDate:
      return "YYYY-MM-DD";
    case FilterField::ExifISO:
    case FilterField::Rating:
    case FilterField::ExifAperture:
    case FilterField::ExifFocalLength:
      return "number";
    default:
      return "type to filter...";
  }
}

auto FilterRuleModel::KindForField(FilterField field) -> FilterValueKind {
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

auto FilterRuleModel::IsBetween(CompareOp op) -> bool {
  return op == CompareOp::BETWEEN;
}

auto FilterRuleModel::AllowedOpsForKind(FilterValueKind kind) -> std::vector<CompareOp> {
  if (kind == FilterValueKind::String) {
    return {CompareOp::CONTAINS,     CompareOp::NOT_CONTAINS, CompareOp::EQUALS,
            CompareOp::NOT_EQUALS,   CompareOp::STARTS_WITH,  CompareOp::ENDS_WITH,
            CompareOp::REGEX};
  }
  if (kind == FilterValueKind::Int64 || kind == FilterValueKind::Double) {
    return {CompareOp::EQUALS,       CompareOp::NOT_EQUALS,  CompareOp::GREATER_THAN,
            CompareOp::LESS_THAN,    CompareOp::GREATER_EQUAL, CompareOp::LESS_EQUAL,
            CompareOp::BETWEEN};
  }
  return {CompareOp::EQUALS,       CompareOp::NOT_EQUALS,  CompareOp::GREATER_THAN,
          CompareOp::LESS_THAN,    CompareOp::GREATER_EQUAL, CompareOp::LESS_EQUAL,
          CompareOp::BETWEEN};
}

auto FilterRuleModel::DefaultOpForField(FilterField field) -> CompareOp {
  const auto ops = AllowedOpsForKind(KindForField(field));
  return ops.empty() ? CompareOp::CONTAINS : ops.front();
}

auto FilterRuleModel::OpAllowedForField(FilterField field, CompareOp op) -> bool {
  const auto ops = AllowedOpsForKind(KindForField(field));
  return std::find(ops.begin(), ops.end(), op) != ops.end();
}

}  // namespace puerhlab::ui
