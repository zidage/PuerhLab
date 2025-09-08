add_library(ThreadPool concurrency/thread_pool.cpp)
target_include_directories(ThreadPool PUBLIC include)

add_library(TimeProvider utils/clock/time_provider.cpp)
target_include_directories(TimeProvider PUBLIC include)

add_library(StrConv utils/string/converter.cpp)
target_include_directories(StrConv PUBLIC include)
target_link_libraries(StrConv PUBLIC utfcpp)

add_library(Image 
    image/image_buffer.cpp
    image/image.cpp
)
target_include_directories(Image PUBLIC include)
target_link_libraries(Image PUBLIC Exiv2 ${OpenCV_LIBS} LibRaw TimeProvider JSON xxHash easy_profiler OpenMP::OpenMP_CXX)

add_library(RawProcessorOp
    decoders/processor/operators/cpu/color_space_conv.cpp
    decoders/processor/operators/cpu/debayer_ahd.cpp
    decoders/processor/operators/cpu/debayer_amaze.cpp
    decoders/processor/operators/cpu/highlight_reconstruct.cpp
    decoders/processor/operators/cpu/white_balance.cpp
    decoders/processor/operators/cpu/raw_proc_utils.cpp
    decoders/processor/operators/gpu/cuda_color_space_conv.cu
    decoders/processor/operators/gpu/cuda_debayer_ahd.cu
    decoders/processor/operators/gpu/cuda_highlight_reconstruct.cu
    decoders/processor/operators/gpu/cuda_white_balance.cu
)
target_include_directories(RawProcessorOp PUBLIC include)
target_link_libraries(RawProcessorOp PUBLIC Image LibRaw Exiv2 hwy::hwy)
set_target_properties(RawProcessorOp PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

add_library(RawProcessor
    decoders/processor/raw_processor.cpp
)
target_include_directories(RawProcessor PUBLIC include)
target_link_libraries(RawProcessor PUBLIC Image LibRaw Exiv2 RawProcessorOp)

add_library(ImageDecoder 
    decoders/decoder_scheduler.cpp
    decoders/raw_decoder.cpp
    decoders/thumbnail_decoder.cpp    
    decoders/metadata_decoder.cpp
)
target_include_directories(ImageDecoder PUBLIC include)
target_link_libraries(ImageDecoder PUBLIC Image ThreadPool ${OpenCV_LIBS} LibRaw RawProcessor easy_profiler) 

add_library(ImageLoader io/image/image_loader.cpp)
target_include_directories(ImageLoader PUBLIC include)
target_link_libraries(ImageLoader PUBLIC ImageDecoder)

add_library(EditHistory edit/history/edit_history.cpp edit/history/version.cpp)
target_include_directories(EditHistory PUBLIC include)
target_link_libraries(EditHistory PUBLIC xxHash Image TimeProvider)

add_library(SleeveFilter
    sleeve/sleeve_filter/filters/exif_filter.cpp
    sleeve/sleeve_filter/filters/datetime_filter.cpp
    sleeve/sleeve_filter/filter_combo.cpp
)
target_include_directories(SleeveFilter PUBLIC include)
target_link_libraries(SleeveFilter PRIVATE Image EditHistory JSON)

add_library(ImagePool
    storage/image_pool/image_pool_manager.cpp
)
target_include_directories(ImagePool PUBLIC include)
target_link_libraries(ImagePool PRIVATE Image)



add_library(SleeveElement 
    sleeve/sleeve_element/sleeve_element_factory.cpp
    sleeve/sleeve_element/sleeve_element.cpp
    sleeve/sleeve_element/sleeve_file.cpp
    sleeve/sleeve_element/sleeve_folder.cpp
)
target_include_directories(SleeveElement PUBLIC include)
target_link_libraries(SleeveElement PUBLIC Image ImageLoader SleeveFilter EditHistory ImagePool)



add_library(DuckORM     
    storage/mapper/duckorm/duckdb_orm.cpp
    storage/mapper/duckorm/duckdb_types.cpp)
target_include_directories(DuckORM PUBLIC include)
target_link_libraries(DuckORM PUBLIC DuckDB)

add_library(SleeveMapper     
    storage/mapper/sleeve/element/file_mapper.cpp
    storage/mapper/sleeve/element/folder_mapper.cpp
    storage/mapper/sleeve/element/element_mapper.cpp
    storage/mapper/sleeve/base/base_mapper.cpp
    storage/mapper/image/image_mapper.cpp
    storage/mapper/sleeve/filter/filter_mapper.cpp
    )
target_include_directories(SleeveMapper PUBLIC include)
target_link_libraries(SleeveMapper PUBLIC DuckDB DuckORM Image SleeveElement ImagePool easy_profiler)

add_library(SleeveService
    storage/service/image/image_service.cpp
    storage/service/sleeve/base/base_service.cpp
    storage/service/sleeve/base/root_service.cpp
    storage/service/sleeve/element/element_service.cpp
    storage/service/sleeve/element/file_service.cpp
    storage/service/sleeve/element/folder_service.cpp
)
target_include_directories(SleeveService PUBLIC include)
target_link_libraries(SleeveService PUBLIC DuckDB DuckORM SleeveMapper StrConv)

add_library(SleeveController
    storage/controller/controller_types.cpp
    storage/controller/db_controller.cpp
    storage/controller/image/image_controller.cpp
    storage/controller/sleeve/element_controller.cpp
)
target_include_directories(SleeveController PUBLIC include)
target_link_libraries(SleeveController PUBLIC DuckDB SleeveService StrConv)

add_library(SleeveFS
    sleeve/sleeve_base.cpp
    sleeve/path_resolver.cpp
    sleeve/storage_service.cpp
    sleeve/dentry_cache_manager.cpp
    sleeve/sleeve_filesystem.cpp
)
target_include_directories(SleeveFS PUBLIC include)
target_link_libraries(SleeveFS PUBLIC SleeveElement SleeveController)

add_library(SleeveView sleeve/sleeve_view.cpp)
target_include_directories(SleeveView PUBLIC include)
target_link_libraries(SleeveView PUBLIC Image SleeveFS ImagePool easy_profiler)

add_library(SleeveManager 
    sleeve/sleeve_manager.cpp)
target_include_directories(SleeveManager PUBLIC include)
target_link_libraries(SleeveManager PUBLIC SleeveFS SleeveView xxHash)

add_library(ImageWriter
    io/image/image_writer.cpp
)
target_include_directories(ImageWriter PUBLIC include)
target_link_libraries(ImageWriter PUBLIC SleeveManager DuckORM)


add_library(Operators
    edit/operators/operator_factory.cpp
    edit/operators/basic/exposure_op.cpp
    edit/operators/basic/contrast_op.cpp
    edit/operators/basic/black_op.cpp
    edit/operators/basic/highlight_op.cpp
    edit/operators/basic/shadow_op.cpp
    edit/operators/basic/white_op.cpp
    edit/operators/color/tint_op.cpp
    edit/operators/color/conversion/Oklab_cvt.cpp
    edit/operators/color/saturation_op.cpp
    edit/operators/color/vibrance_op.cpp
    edit/operators/color/HLS_op.cpp
    edit/operators/detail/clarity_op.cpp
    edit/operators/detail/sharpen_op.cpp
    edit/operators/wheel/color_wheel_op.cpp
    edit/operators/curve/curve_op.cpp
    edit/operators/tone_mapping/ACES_tone_mapping_op.cpp
    edit/operators/basic/auto_exposure_op.cpp
    edit/operators/cst/cst_op.cpp
    edit/operators/cst/lmt_op.cpp
    edit/operators/operator_registeration.cpp
)
target_include_directories(Operators PUBLIC include)
target_link_libraries(Operators PUBLIC Image SleeveFS JSON OpenColorIO easy_profiler hwy::hwy)

add_library(EditPipeline
    edit/pipeline/pipeline_cpu.cpp
    edit/pipeline/pipeline_utils.cpp
    edit/scheduler/pipeline_scheduler.cpp)
target_include_directories(EditPipeline PUBLIC include)
target_link_libraries(EditPipeline PUBLIC Image OpenColorIO Operators)
