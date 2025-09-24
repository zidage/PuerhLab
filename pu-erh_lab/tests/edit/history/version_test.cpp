#include <iostream>

#include "edit/history/version.hpp"
#include <gtest/gtest.h>

#include "edit/history/edit_transaction.hpp"
#include "edit/pipeline/pipeline.hpp"
#include "edit/pipeline/pipeline_cpu.hpp"
#include "history_test_fixation.hpp"
#include "type/type.hpp"


namespace puerhlab {
TEST_F(EditHistoryTests, VersionIDGenerationTest) {
  {
    Version v{0};
    v.SetBasePipelineExecutor(std::make_shared<CPUPipelineExecutor>());
    EditTransaction tx1{0,
                        TransactionType::_ADD,
                        OperatorType::EXPOSURE,
                        PipelineStageName::Basic_Adjustment,
                        {{"exposure", 1.0f}}};
    v.AppendEditTransaction(std::move(tx1));

    v.CalculateVersionID();
    auto id1 = v.GetVersionID();

    EditTransaction tx2{1,
                        TransactionType::_ADD,
                        OperatorType::CONTRAST,
                        PipelineStageName::Basic_Adjustment,
                        {{"contrast", 2.2f}}};
    v.AppendEditTransaction(std::move(tx2));
    v.CalculateVersionID();
    auto id2 = v.GetVersionID();

    std::cout << "Version ID after first transaction: 0x" << id1.ToString() << std::endl;
    std::cout << "Version ID after second transaction: 0x" << id2.ToString() << std::endl;
    EXPECT_NE(id1, id2) << "Version IDs (hashes) should differ after adding a new transaction.";
  }
}

TEST_F(EditHistoryTests, RemoveTransactionTest) {
  {
    Version v{0};
    v.SetBasePipelineExecutor(std::make_shared<CPUPipelineExecutor>());

    EditTransaction tx1{0,
                        TransactionType::_ADD,
                        OperatorType::EXPOSURE,
                        PipelineStageName::Basic_Adjustment,
                        {{"exposure", 1.0f}}};
    v.AppendEditTransaction(std::move(tx1));

    EditTransaction tx2{1,
                        TransactionType::_ADD,
                        OperatorType::CONTRAST,
                        PipelineStageName::Basic_Adjustment,
                        {{"contrast", 2.2f}}};
    v.AppendEditTransaction(std::move(tx2));

    auto all_txs = v.GetAllEditTransactions();
    EXPECT_EQ(all_txs.size(), 2) << "There should be 2 transactions before removal.";

    v.CalculateVersionID();
    auto        id_before_removal = v.GetVersionID();
    std::cout << "Version ID before removing transaction: 0x" << id_before_removal.ToString()
              << std::endl;

    EditTransaction removed_tx = v.RemoveLastEditTransaction();

    v.CalculateVersionID();
    auto id_after_removal = v.GetVersionID();
    std::cout << "Version ID after removing transaction: 0x" << id_after_removal.ToString()
              << std::endl;

    EXPECT_NE(id_before_removal, id_after_removal)
        << "Version IDs (hashes) should differ after removing a transaction.";

    all_txs = v.GetAllEditTransactions();
    EXPECT_EQ(all_txs.size(), 1) << "There should be 1 transaction after removal.";
  }
}

TEST_F(EditHistoryTests, FuzzTest) {
  {
    Version v{0};
    v.SetBasePipelineExecutor(std::make_shared<CPUPipelineExecutor>());

    const int num_transactions = 10000;
    auto id = v.GetVersionID();
    for (int i = 0; i < num_transactions; ++i) {
      EditTransaction tx{i,
                          TransactionType::_ADD,
                          OperatorType::EXPOSURE,
                          PipelineStageName::Basic_Adjustment,
                          {{"exposure", static_cast<float>(i) * 0.1f}}};
      v.AppendEditTransaction(std::move(tx));
      v.CalculateVersionID();
      EXPECT_NE(id, v.GetVersionID())
          << "Version ID should change after adding transaction " << i;
      id = v.GetVersionID();
    }

    EXPECT_EQ(2048, v.GetAllEditTransactions().size())
        << "Version should only keep the last 2048 transactions.";

  }
}

TEST_F(EditHistoryTests, JSONSerializationTest) {
  Version version(1);
  version.SetBoundImage(1001);

  EditTransaction tx1(1, TransactionType::_ADD, OperatorType::EXPOSURE,
                      PipelineStageName::Basic_Adjustment, {{"exposure", 0.5}});
  version.AppendEditTransaction(std::move(tx1));

  EditTransaction tx2(2, TransactionType::_ADD, OperatorType::CONTRAST,
                      PipelineStageName::Basic_Adjustment, {{"contrast", 50}}, &tx1);
  version.AppendEditTransaction(std::move(tx2));

  version.CalculateVersionID();

  nlohmann::json j = version.ToJSON();
  
  // std::cout << j.dump(2) << std::endl;

  Version version2;
  version2.FromJSON(j);
  version2.CalculateVersionID();

  nlohmann::json j2 = version2.ToJSON();
  // std::cout << j2.dump(2) << std::endl;

  EXPECT_EQ(j.dump(), j2.dump());

  // Check parent-child relationships
  auto txs = version2.GetAllEditTransactions();
  EXPECT_EQ(txs.size(), 2);

  auto& tx1_reloaded = version2.GetTransactionByID(1);
  auto& tx2_reloaded = version2.GetTransactionByID(2);

  EXPECT_NE(tx2_reloaded.GetParentTransactionID(), -1);
  EXPECT_EQ(tx2_reloaded.GetParentTransactionID(), tx1_reloaded.GetTransactionID());
  EXPECT_EQ(tx1_reloaded.GetParentTransactionID(), -1);
  
}
}  // namespace puerhlab