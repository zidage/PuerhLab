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
target_link_libraries(Image PUBLIC Exiv2 ${OpenCV_LIBS} LibRaw TimeProvider JSON xxHash easy_profiler)


add_library(ImageDecoder 
    decoders/decoder_scheduler.cpp
    decoders/raw_decoder.cpp
    decoders/thumbnail_decoder.cpp    
    decoders/metadata_decoder.cpp
)
target_include_directories(ImageDecoder PUBLIC include)
target_link_libraries(ImageDecoder PUBLIC Image ThreadPool ${OpenCV_LIBS} LibRaw easy_profiler) 

add_library(IO io/image/image_loader.cpp)
target_include_directories(IO PUBLIC include)
target_link_libraries(IO PUBLIC ImageDecoder)

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
target_link_libraries(SleeveElement PUBLIC Image IO SleeveFilter EditHistory ImagePool)



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

add_library(Operators
    edit/operators/basic/exposure_op.cpp
    edit/operators/basic/contrast_op.cpp
    edit/operators/basic/tone_region_op.cpp
    edit/operators/color/tint_op.cpp
    edit/operators/color/conversion/Oklab_cvt.cpp
    edit/operators/color/saturation_op.cpp
    edit/operators/color/vibrance_op.cpp
    edit/operators/color/HLS_op.cpp
    edit/operators/detail/clarity_op.cpp
    edit/operators/detail/sharpen_op.cpp
    edit/operators/wheel/color_wheel_op.cpp
    edit/operators/curve/curve_op.cpp
    edit/operators/cst/cst_op.cpp
)
target_include_directories(Operators PUBLIC include)
target_link_libraries(Operators PUBLIC Image SleeveFS JSON OpenColorIO::OpenColorIO)