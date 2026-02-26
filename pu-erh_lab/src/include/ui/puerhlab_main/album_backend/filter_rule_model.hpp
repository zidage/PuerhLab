#pragma once

#include <QAbstractListModel>
#include <QString>
#include <QVariantList>

#include <vector>

#include "sleeve/sleeve_filter/filter_combo.hpp"

namespace puerhlab::ui {

enum class FilterValueKind { String, Int64, Double, DateTime };

class FilterRuleModel final : public QAbstractListModel {
  Q_OBJECT

 public:
  enum Role {
    FieldRole = Qt::UserRole + 1,
    OpRole,
    ValueRole,
    Value2Role,
    ShowSecondValueRole,
    PlaceholderRole,
    OpOptionsRole,
  };
  Q_ENUM(Role)

  struct Rule {
    FilterField field = FilterField::ExifCameraModel;
    CompareOp   op    = CompareOp::CONTAINS;
    QString     value{};
    QString     value2{};
  };

  explicit FilterRuleModel(QObject* parent = nullptr);

  auto rowCount(const QModelIndex& parent = QModelIndex()) const -> int override;
  auto data(const QModelIndex& index, int role = Qt::DisplayRole) const -> QVariant override;
  auto setData(const QModelIndex& index, const QVariant& value, int role) -> bool override;
  auto flags(const QModelIndex& index) const -> Qt::ItemFlags override;
  auto roleNames() const -> QHash<int, QByteArray> override;

  Q_INVOKABLE void AddRule();
  Q_INVOKABLE void RemoveRule(int index);
  Q_INVOKABLE void ClearAndReset();
  Q_INVOKABLE void SetField(int index, int fieldValue);
  Q_INVOKABLE void SetOp(int index, int opValue);
  Q_INVOKABLE void SetValue(int index, const QString& value);
  Q_INVOKABLE void SetValue2(int index, const QString& value);

  auto FieldOptions() const -> QVariantList;
  static auto CompareOptionsForField(FilterField field) -> QVariantList;
  static auto PlaceholderForField(FilterField field) -> QString;
  static auto KindForField(FilterField field) -> FilterValueKind;
  static auto IsBetween(CompareOp op) -> bool;

  auto Rules() const -> const std::vector<Rule>& { return rules_; }

 private:
  static auto AllowedOpsForKind(FilterValueKind kind) -> std::vector<CompareOp>;
  static auto DefaultOpForField(FilterField field) -> CompareOp;
  static auto OpAllowedForField(FilterField field, CompareOp op) -> bool;

  std::vector<Rule> rules_{};
};

}  // namespace puerhlab::ui
