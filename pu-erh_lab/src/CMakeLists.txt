add_library(RawDecoder raw/raw_decoder.cpp)
target_include_directories(RawDecoder PUBLIC include)
target_link_libraries(RawDecoder PRIVATE LibRaw)

