add_executable(SampleTest sample_test.cpp)
target_link_libraries(SampleTest PRIVATE GTest::gtest_main)

include(GoogleTest)
gtest_discover_tests(SampleTest)