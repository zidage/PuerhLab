add_library(MemLeakDect INTERFACE leak_detector/memory_leak_detector.hpp)
target_link_libraries(MemLeakDect INTERFACE GTest::gtest_main)

add_executable(SampleTest sample_test.cpp)
target_link_libraries(SampleTest PRIVATE GTest::gtest_main)

add_executable(SingleRawLoad raw/single_raw_decode_test.cpp)
target_include_directories(SingleRawLoad PUBLIC ${CMAKE_SOURCE_DIR}/pu-erh_lab/src/include)
target_link_libraries(SingleRawLoad PRIVATE GTest::gtest_main LibRaw ${OpenCV_LIBS})
target_compile_definitions(SingleRawLoad PRIVATE LIBRAW_NODLL)

add_executable(SingleThumbnailLoad raw/single_raw_thumbnail_test.cpp)
target_include_directories(SingleThumbnailLoad PUBLIC ${CMAKE_SOURCE_DIR}/pu-erh_lab/src/include)
target_link_libraries(SingleThumbnailLoad PRIVATE GTest::gtest_main LibRaw ${OpenCV_LIBS})
target_compile_definitions(SingleThumbnailLoad PRIVATE LIBRAW_NODLL)

add_executable(ImageDecoderTest decoders/image_decoder_test.cpp)
target_include_directories(ImageDecoderTest PUBLIC ${CMAKE_SOURCE_DIR}/pu-erh_lab/src/include)
target_link_libraries(ImageDecoderTest PRIVATE GTest::gtest_main ImageDecoder Exiv2)

add_executable(ImageLoaderTest image/image_loader_test.cpp)
target_include_directories(ImageLoaderTest PUBLIC ${CMAKE_SOURCE_DIR}/pu-erh_lab/src/include)
target_link_libraries(ImageLoaderTest PRIVATE GTest::gtest_main ImageDecoder ImageLoader Exiv2)

# add_executable(SleeveOperationTest sleeve/sleeve_operation_test.cpp)
# target_include_directories(SleeveOperationTest PUBLIC ${CMAKE_SOURCE_DIR}/pu-erh_lab/src/include)
# target_link_libraries(SleeveOperationTest PRIVATE GTest::gtest_main ImageDecoder ImageLoader Exiv2 Sleeve TimeProvider SleeveFS)

add_executable(ImagePoolTest storage/image_pool_test.cpp)
target_include_directories(ImagePoolTest PUBLIC ${CMAKE_SOURCE_DIR}/pu-erh_lab/src/include)
target_link_libraries(ImagePoolTest PRIVATE GTest::gtest_main Image ImagePool)

# add_executable(SleeveViewTest sleeve/sleeve_view_test.cpp)
# target_include_directories(SleeveViewTest PUBLIC ${CMAKE_SOURCE_DIR}/pu-erh_lab/src/include)
# target_link_libraries(SleeveViewTest PRIVATE GTest::gtest_main Image ImagePool Sleeve TimeProvider SleeveView SleeveManager)

# add_executable(SleeveMapperTest storage/sleeve_mapper_test.cpp)
# target_include_directories(SleeveMapperTest PUBLIC ${CMAKE_SOURCE_DIR}/pu-erh_lab/src/include)
# target_link_libraries(SleeveMapperTest PRIVATE GTest::gtest_main Image ImagePool DuckORM SleeveManager SleeveMapper SleeveController)

add_executable(SleeveFSTest sleeve/sleeve_fs_test.cpp)
target_include_directories(SleeveFSTest PUBLIC ${CMAKE_SOURCE_DIR}/pu-erh_lab/src/include)
target_link_libraries(SleeveFSTest PRIVATE GTest::gtest_main SleeveFS)


include(GoogleTest)
# set(CMAKE_GTEST_DISCOVER_TESTS_DISCOVERY_MODE PRE_TEST)
gtest_discover_tests(SampleTest)
gtest_discover_tests(SingleRawLoad)
gtest_discover_tests(SingleThumbnailLoad)
gtest_discover_tests(ImageDecoderTest)
gtest_discover_tests(ImageLoaderTest)
# gtest_discover_tests(SleeveOperationTest)
gtest_discover_tests(ImagePoolTest)
# gtest_discover_tests(SleeveViewTest)
# gtest_discover_tests(SleeveMapperTest)
gtest_discover_tests(SleeveFSTest)
