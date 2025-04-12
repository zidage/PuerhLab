add_library(ThreadPool concurrency/thread_pool.cpp)
target_include_directories(ThreadPool PUBLIC include)

add_library(Image 
    image/image.cpp
)
target_include_directories(Image PUBLIC include)
target_link_libraries(Image PUBLIC Exiv2 ${OpenCV_LIBS} LibRaw)


add_library(ImageDecoder 
    decoders/decoder_scheduler.cpp
    decoders/raw_decoder.cpp
    decoders/thumbnail_decoder.cpp    
    decoders/metadata_decoder.cpp
)
target_include_directories(ImageDecoder PUBLIC include)
target_link_libraries(ImageDecoder PUBLIC Image ThreadPool Exiv2 ${OpenCV_LIBS} LibRaw)

add_library(ImageLoader image/image_loader.cpp)
target_include_directories(ImageLoader PUBLIC include)
target_link_libraries(ImageLoader PUBLIC ImageDecoder)

add_library(Sleeve 
    sleeve/sleeve_base.cpp    
    sleeve/sleeve.cpp
    mapper/sleeve/sleeve_mapper.cpp
    sleeve/sleeve_filter.cpp
    sleeve/sleeve_element/sleeve_element.cpp
    sleeve/sleeve_element/sleeve_file.cpp
    sleeve/sleeve_element/sleeve_folder.cpp
)
target_include_directories(Sleeve PUBLIC include)
target_link_libraries(Sleeve PRIVATE Image ImageLoader DuckDB)

add_library(SleeveManager 
    sleeve/sleeve_manager.cpp)
target_include_directories(SleeveManager PUBLIC include)
target_link_libraries(SleeveManager PRIVATE Sleeve ImageLoader)

add_library(SleeveBuffer
    sleeve/sleeve_buffer/sleeve_buffer_manager.cpp
    sleeve/sleeve_buffer/sleeve_buffer.cpp
)
target_include_directories(SleeveBuffer PUBLIC include)
target_link_libraries(SleeveBuffer PRIVATE Image)


