#pragma once

#include <QAbstractListModel>
#include <QString>
#include <QVariantList>

#include <vector>

#include "sleeve/sleeve_filter/filter_combo.hpp"

namespace puerhlab::demo {

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
    FilterField field = FilterField::FileName;
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

  Q_INVOKABLE void addRule();
  Q_INVOKABLE void removeRule(int index);
  Q_INVOKABLE void clearAndReset();
  Q_INVOKABLE void setField(int index, int fieldValue);
  Q_INVOKABLE void setOp(int index, int opValue);
  Q_INVOKABLE void setValue(int index, const QString& value);
  Q_INVOKABLE void setValue2(int index, const QString& value);

  auto fieldOptions() const -> QVariantList;
  static auto compareOptionsForField(FilterField field) -> QVariantList;
  static auto placeholderForField(FilterField field) -> QString;
  static auto kindForField(FilterField field) -> FilterValueKind;
  static auto isBetween(CompareOp op) -> bool;

  auto rules() const -> const std::vector<Rule>& { return rules_; }

 private:
  static auto allowedOpsForKind(FilterValueKind kind) -> std::vector<CompareOp>;
  static auto defaultOpForField(FilterField field) -> CompareOp;
  static auto opAllowedForField(FilterField field, CompareOp op) -> bool;

  std::vector<Rule> rules_{};
};

}  // namespace puerhlab::demo
