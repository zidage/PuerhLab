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

# add_executable(ImageDecoderTest decoders/image_decoder_test.cpp)
# target_include_directories(ImageDecoderTest PUBLIC ${CMAKE_SOURCE_DIR}/pu-erh_lab/src/include)
# target_link_libraries(ImageDecoderTest PRIVATE GTest::gtest_main ImageDecoder Exiv2)

add_executable(ImageLoaderTest image/image_loader_test.cpp)
target_include_directories(ImageLoaderTest PUBLIC ${CMAKE_SOURCE_DIR}/pu-erh_lab/src/include)
target_link_libraries(ImageLoaderTest PRIVATE GTest::gtest_main ImageDecoder ImageLoader Exiv2)

add_executable(ImagePoolTest storage/image_pool_test.cpp)
target_include_directories(ImagePoolTest PUBLIC ${CMAKE_SOURCE_DIR}/pu-erh_lab/src/include)
target_link_libraries(ImagePoolTest PRIVATE GTest::gtest_main Image ImagePool)


add_executable(SleeveFSTest sleeve/sleeve_fs_test.cpp)
target_include_directories(SleeveFSTest PUBLIC ${CMAKE_SOURCE_DIR}/pu-erh_lab/src/include)
target_link_libraries(SleeveFSTest PRIVATE GTest::gtest_main SleeveFS StrConv)

add_executable(SleeveViewTest sleeve/sleeve_view_test.cpp)
target_include_directories(SleeveViewTest PUBLIC ${CMAKE_SOURCE_DIR}/pu-erh_lab/src/include)
target_link_libraries(SleeveViewTest PRIVATE GTest::gtest_main Image ImagePool TimeProvider SleeveView SleeveFS SleeveManager)

# Operation tests
# add_executable(ExposureOPTest edit/operators/basic/exposure_op_test.cpp)
# target_include_directories(ExposureOPTest PUBLIC ${CMAKE_SOURCE_DIR}/pu-erh_lab/src/include)
# target_link_libraries(ExposureOPTest GTest::gtest_main Operators SleeveManager)

# add_executable(ContrastOPTest edit/operators/basic/contrast_op_test.cpp)
# target_include_directories(ContrastOPTest PUBLIC ${CMAKE_SOURCE_DIR}/pu-erh_lab/src/include)
# target_link_libraries(ContrastOPTest GTest::gtest_main Operators SleeveManager)

# add_executable(TintOPTest edit/operators/color/tint_op_test.cpp)
# target_include_directories(TintOPTest PUBLIC ${CMAKE_SOURCE_DIR}/pu-erh_lab/src/include)
# target_link_libraries(TintOPTest GTest::gtest_main Operators SleeveManager)

# add_executable(SaturationOPTest edit/operators/color/saturation_op_test.cpp)
# target_include_directories(SaturationOPTest PUBLIC ${CMAKE_SOURCE_DIR}/pu-erh_lab/src/include)
# target_link_libraries(SaturationOPTest GTest::gtest_main Operators SleeveManager)

# add_executable(VibranceOPTest edit/operators/color/vibrance_op_test.cpp)
# target_include_directories(VibranceOPTest PUBLIC ${CMAKE_SOURCE_DIR}/pu-erh_lab/src/include)
# target_link_libraries(VibranceOPTest GTest::gtest_main Operators SleeveManager)

# add_executable(HLSOPTest edit/operators/color/hls_op_test.cpp)
# target_include_directories(HLSOPTest PUBLIC ${CMAKE_SOURCE_DIR}/pu-erh_lab/src/include)
# target_link_libraries(HLSOPTest GTest::gtest_main Operators SleeveManager)

# add_executable(CWOPTest edit/operators/wheel/color_wheel_op_test.cpp)
# target_include_directories(CWOPTest PUBLIC ${CMAKE_SOURCE_DIR}/pu-erh_lab/src/include)
# target_link_libraries(CWOPTest GTest::gtest_main Operators SleeveManager)

# add_executable(CurveOPTest edit/operators/curve/curve_op_test.cpp)
# target_include_directories(CurveOPTest PUBLIC ${CMAKE_SOURCE_DIR}/pu-erh_lab/src/include)
# target_link_libraries(CurveOPTest GTest::gtest_main Operators SleeveManager)
# if (WIN32)
# target_link_libraries(CurveOPTest easy_profiler)
# endif()

# add_executable(ClarityOPTest edit/operators/detail/clarity_op_test.cpp)
# target_include_directories(ClarityOPTest PUBLIC ${CMAKE_SOURCE_DIR}/pu-erh_lab/src/include)
# target_link_libraries(ClarityOPTest GTest::gtest_main Operators SleeveManager)

# add_executable(SharpenOPTest edit/operators/detail/sharpen_op_test.cpp)
# target_include_directories(SharpenOPTest PUBLIC ${CMAKE_SOURCE_DIR}/pu-erh_lab/src/include)
# target_link_libraries(SharpenOPTest GTest::gtest_main Operators SleeveManager)

# add_executable(CSTTest edit/operators/cst/cst_op_test.cpp)
# target_include_directories(CSTTest PUBLIC ${CMAKE_SOURCE_DIR}/pu-erh_lab/src/include)
# target_link_libraries(CSTTest GTest::gtest_main Operators SleeveManager ImageDecoder easy_profiler)

add_executable(TileTest edit/pipeline/tile_test.cpp)
target_include_directories(TileTest PUBLIC ${CMAKE_SOURCE_DIR}/pu-erh_lab/src/include)
target_link_libraries(TileTest GTest::gtest_main Operators SleeveManager ImageDecoder EditPipeline OpenColorIO lcms2::lcms2)

add_executable(PipelineTest edit/pipeline/cpu_pipeline_test.cpp)
target_include_directories(PipelineTest PUBLIC ${CMAKE_SOURCE_DIR}/pu-erh_lab/src/include)
target_link_libraries(PipelineTest GTest::gtest_main Operators SleeveManager ImageDecoder EditPipeline OpenColorIO lcms2::lcms2)
if (WIN32)
target_link_libraries(PipelineTest easy_profiler)
endif()

add_executable(PipelineSchedulerTest edit/pipeline/pipeline_scheduler_test.cpp)
target_include_directories(PipelineSchedulerTest PUBLIC ${CMAKE_SOURCE_DIR}/pu-erh_lab/src/include)
target_link_libraries(PipelineSchedulerTest GTest::gtest_main Operators SleeveManager ImageDecoder EditPipeline OpenColorIO lcms2::lcms2)
if (WIN32)
target_link_libraries(PipelineSchedulerTest easy_profiler)
endif()

add_executable(EditTransactionTest edit/history/transaction_test.cpp)
target_include_directories(EditTransactionTest PUBLIC ${CMAKE_SOURCE_DIR}/pu-erh_lab/src/include)
target_link_libraries(EditTransactionTest GTest::gtest_main Operators SleeveManager ImageDecoder EditPipeline OpenColorIO lcms2::lcms2)
if (WIN32)
target_link_libraries(EditTransactionTest easy_profiler)
endif()


add_executable(EditVersionTest edit/history/version_test.cpp)
target_include_directories(EditVersionTest PUBLIC ${CMAKE_SOURCE_DIR}/pu-erh_lab/src/include)
target_link_libraries(EditVersionTest GTest::gtest_main Operators SleeveManager ImageDecoder EditPipeline OpenColorIO lcms2::lcms2)
if (WIN32)
target_link_libraries(EditVersionTest easy_profiler)
endif()

add_executable(EditHistoryTest edit/history/history_test.cpp)
target_include_directories(EditHistoryTest PUBLIC ${CMAKE_SOURCE_DIR}/pu-erh_lab/src/include)
target_link_libraries(EditHistoryTest GTest::gtest_main Operators SleeveManager ImageDecoder EditPipeline OpenColorIO lcms2::lcms2)
if (WIN32)
target_link_libraries(EditHistoryTest easy_profiler)
endif()

add_executable(ImagePreview gui_pocs/image_preview_cpu.cpp)
target_include_directories(ImagePreview PUBLIC ${CMAKE_SOURCE_DIR}/pu-erh_lab/src/include)
target_link_libraries(ImagePreview Image ImageLoader ImageDecoder EditPipeline Operators SleeveManager OpenColorIO lcms2::lcms2)
if (WIN32)
target_link_libraries(ImagePreview easy_profiler Qt6::Widgets)
endif()


include(GoogleTest)
# set(CMAKE_GTEST_DISCOVER_TESTS_DISCOVERY_MODE PRE_TEST)
# gtest_discover_tests(SampleTest)
# gtest_discover_tests(SingleRawLoad)
# gtest_discover_tests(SingleThumbnailLoad)
# gtest_discover_tests(ImageDecoderTest)
# gtest_discover_tests(ImageLoaderTest)
# # gtest_discover_tests(SleeveOperationTest)
# gtest_discover_tests(ImagePoolTest)
# # gtest_discover_tests(SleeveViewTest)
# # gtest_discover_tests(SleeveMapperTest)
# gtest_discover_tests(SleeveFSTest)
# gtest_discover_tests(ExposureOPTest)
# gtest_discover_tests(ContrastOPTest)
# gtest_discover_tests(TintOPTest)
# gtest_discover_tests(SaturationOPTest)
# gtest_discover_tests(VibranceOPTest)
# gtest_discover_tests(ClarityOPTest)
# gtest_discover_tests(SharpenOPTest)
# gtest_discover_tests(PipelineTest)
