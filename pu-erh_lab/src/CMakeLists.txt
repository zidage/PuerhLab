include_directories(${OpenCV_INCLUDE_DIRS})

add_library(Image INTERFACE)
target_include_directories(Image INTERFACE "include\\image")
target_link_libraries(Image INTERFACE Exiv2 ${OpenCV_LIBS} LibRaw)


add_library(RawDecoder raw/raw_decoder.cpp)
target_include_directories(RawDecoder PUBLIC include)
target_link_libraries(RawDecoder PRIVATE LibRaw Exiv2 Image)

