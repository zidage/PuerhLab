#include "FilterRuleModel.h"

#include <algorithm>

namespace puerhlab::demo {
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
      return isBetween(rule.op);
    case PlaceholderRole:
      return placeholderForField(rule.field);
    case OpOptionsRole:
      return compareOptionsForField(rule.field);
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
      setField(index.row(), value.toInt());
      return true;
    case OpRole:
      setOp(index.row(), value.toInt());
      return true;
    case ValueRole:
      setValue(index.row(), value.toString());
      return true;
    case Value2Role:
      setValue2(index.row(), value.toString());
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

void FilterRuleModel::addRule() {
  const int row = rowCount();
  beginInsertRows(QModelIndex(), row, row);
  rules_.push_back(Rule{});
  endInsertRows();
}

void FilterRuleModel::removeRule(int index) {
  if (index < 0 || index >= rowCount()) {
    return;
  }
  beginRemoveRows(QModelIndex(), index, index);
  rules_.erase(rules_.begin() + index);
  endRemoveRows();
}

void FilterRuleModel::clearAndReset() {
  beginResetModel();
  rules_.clear();
  rules_.push_back(Rule{});
  endResetModel();
}

void FilterRuleModel::setField(int index, int fieldValue) {
  if (index < 0 || index >= rowCount()) {
    return;
  }

  auto& rule      = rules_[static_cast<size_t>(index)];
  rule.field      = static_cast<FilterField>(fieldValue);
  if (!opAllowedForField(rule.field, rule.op)) {
    rule.op = defaultOpForField(rule.field);
  }

  emit dataChanged(createIndex(index, 0), createIndex(index, 0),
                   {FieldRole, OpRole, ShowSecondValueRole, PlaceholderRole, OpOptionsRole});
}

void FilterRuleModel::setOp(int index, int opValue) {
  if (index < 0 || index >= rowCount()) {
    return;
  }

  auto& rule = rules_[static_cast<size_t>(index)];
  rule.op    = static_cast<CompareOp>(opValue);
  if (!opAllowedForField(rule.field, rule.op)) {
    rule.op = defaultOpForField(rule.field);
  }

  emit dataChanged(createIndex(index, 0), createIndex(index, 0), {OpRole, ShowSecondValueRole});
}

void FilterRuleModel::setValue(int index, const QString& value) {
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

void FilterRuleModel::setValue2(int index, const QString& value) {
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

auto FilterRuleModel::fieldOptions() const -> QVariantList {
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

auto FilterRuleModel::compareOptionsForField(FilterField field) -> QVariantList {
  const auto kind = kindForField(field);
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

auto FilterRuleModel::placeholderForField(FilterField field) -> QString {
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

auto FilterRuleModel::kindForField(FilterField field) -> FilterValueKind {
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

auto FilterRuleModel::isBetween(CompareOp op) -> bool {
  return op == CompareOp::BETWEEN;
}

auto FilterRuleModel::allowedOpsForKind(FilterValueKind kind) -> std::vector<CompareOp> {
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

auto FilterRuleModel::defaultOpForField(FilterField field) -> CompareOp {
  const auto ops = allowedOpsForKind(kindForField(field));
  return ops.empty() ? CompareOp::CONTAINS : ops.front();
}

auto FilterRuleModel::opAllowedForField(FilterField field, CompareOp op) -> bool {
  const auto ops = allowedOpsForKind(kindForField(field));
  return std::find(ops.begin(), ops.end(), op) != ops.end();
}

}  // namespace puerhlab::demo
