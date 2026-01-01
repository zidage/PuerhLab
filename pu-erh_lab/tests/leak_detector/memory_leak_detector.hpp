/*
 * @file        pu-erh_lab/tests/leak_detector/memory_leak_detector.hpp
 * @brief       A memory leakage detector
 * @author      Copied from https://stackoverflow.com/questions/29174938/googletest-and-memory-leaks
 * @date        2025-03-28
 * @license     CC BY-SA 4.0
 *
 * @copyright   Copyright (c) 2025 Muperman
 */

// Copyright (c) 2025 Muperman
//
// https://creativecommons.org/licenses/by-sa/4.0/


#include "gtest/gtest.h"
#include <crtdbg.h>

class MemoryLeakDetector {
public:
    MemoryLeakDetector() {
        _CrtMemCheckpoint(&mem_state_);
    }

    ~MemoryLeakDetector() {
        _CrtMemState stateNow, stateDiff;
        _CrtMemCheckpoint(&stateNow);
        int diffResult = _CrtMemDifference(&stateDiff, &mem_state_, &stateNow);
        if (diffResult)
            reportFailure(stateDiff.lSizes[1]);
    }
private:
    void reportFailure(unsigned int unfreedBytes) {
        FAIL() << "Memory leak of " << unfreedBytes << " byte(s) detected.";
    }
    _CrtMemState mem_state_;
};