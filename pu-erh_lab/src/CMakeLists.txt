add_library(ThreadPool concurrency/thread_pool.cpp)
target_include_directories(ThreadPool PUBLIC include)

add_library(TimeProvider utils/clock/time_provider.cpp)
target_include_directories(TimeProvider PUBLIC include)

add_library(Image 
    image/image.cpp
)
target_include_directories(Image PUBLIC include)
target_link_libraries(Image PUBLIC Exiv2 ${OpenCV_LIBS} LibRaw TimeProvider JSON xxHash)


add_library(ImageDecoder 
    decoders/decoder_scheduler.cpp
    decoders/raw_decoder.cpp
    decoders/thumbnail_decoder.cpp    
    decoders/metadata_decoder.cpp
)
target_include_directories(ImageDecoder PUBLIC include)
target_link_libraries(ImageDecoder PUBLIC Image ThreadPool ${OpenCV_LIBS} LibRaw) 

add_library(ImageLoader image/image_loader.cpp)
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

add_library(Sleeve 
    sleeve/sleeve_base.cpp    
    sleeve/dentry_cache_manager.cpp
    sleeve/sleeve_element/sleeve_element_factory.cpp
    sleeve/sleeve_element/sleeve_element.cpp
    sleeve/sleeve_element/sleeve_file.cpp
    sleeve/sleeve_element/sleeve_folder.cpp
)
target_include_directories(Sleeve PUBLIC include)
target_link_libraries(Sleeve PUBLIC Image ImageLoader SleeveFilter EditHistory ImagePool)

add_library(SleeveView sleeve/sleeve_view.cpp)
target_include_directories(SleeveView PUBLIC include)
target_link_libraries(SleeveView PUBLIC Image Sleeve ImagePool)

add_library(SleeveMapper     
    storage/mapper/sleeve/query_prepare.cpp
    storage/mapper/sleeve/sleeve_mapper.cpp)
target_include_directories(SleeveMapper PUBLIC include)
target_link_libraries(SleeveMapper PUBLIC DuckDB Image Sleeve ImagePool )

add_library(SleeveManager 
    sleeve/sleeve_manager.cpp)
target_include_directories(SleeveManager PUBLIC include)
target_link_libraries(SleeveManager PRIVATE Sleeve SleeveView)






