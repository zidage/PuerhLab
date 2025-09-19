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
    p_hash_t        id1 = v.GetVersionID();

    EditTransaction tx2{1,
                        TransactionType::_ADD,
                        OperatorType::CONTRAST,
                        PipelineStageName::Basic_Adjustment,
                        {{"contrast", 2.2f}}};
    v.AppendEditTransaction(std::move(tx2));
    v.CalculateVersionID();
    p_hash_t id2 = v.GetVersionID();

    std::cout << "Version ID after first transaction: 0x" << std::hex << id1 << std::endl;
    std::cout << "Version ID after second transaction: 0x" << std::hex << id2 << std::endl;
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
    p_hash_t        id_before_removal = v.GetVersionID();
    std::cout << "Version ID before removing transaction: 0x" << std::hex << id_before_removal
              << std::endl;

    EditTransaction removed_tx = v.RemoveLastEditTransaction();

    v.CalculateVersionID();
    p_hash_t id_after_removal = v.GetVersionID();
    std::cout << "Version ID after removing transaction: 0x" << std::hex << id_after_removal
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
    p_hash_t id = v.GetVersionID();
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
}  // namespace puerhlab