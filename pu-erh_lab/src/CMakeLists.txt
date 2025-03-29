# include_directories(${OpenCV_INCLUDE_DIRS} ${Exiv2_INCLUDE_DIRS})

add_library(ThreadPool concurrency/thread_pool.cpp)
target_include_directories(ThreadPool PUBLIC include)

add_library(Image 
    image/image.cpp 
    image/image_loader.cpp
)
target_include_directories(Image PUBLIC include)
target_link_libraries(Image PUBLIC Exiv2 ${OpenCV_LIBS} LibRaw)


add_library(ImageDecoder decoders/image_decoder.cpp decoders/raw_decoder.cpp)
target_include_directories(ImageDecoder PUBLIC include)
target_link_libraries(ImageDecoder PRIVATE Image ThreadPool)

add_library(ImageLoader image/image_loader.cpp)
target_include_directories(ImageLoader PUBLIC include)
target_link_libraries(ImageLoader PRIVATE Image ImageDecoder)

add_library(SleeveBuffer
    sleeve/sleeve_buffer/sleeve_buffer_manager.cpp
    sleeve/sleeve_buffer/sleeve_buffer.cpp
)
target_include_directories(SleeveBuffer PUBLIC include)
target_link_libraries(SleeveBuffer PRIVATE Image)

add_library(Sleeve 
    sleeve/sleeve_base.cpp    
    sleeve/sleeve_manager.cpp
    sleeve/sleeve.cpp
    mapper/sleeve/sleeve_mapper.cpp
)
target_include_directories(Sleeve PUBLIC include)
target_link_libraries(Sleeve PRIVATE Image SleeveBuffer DuckDB)

