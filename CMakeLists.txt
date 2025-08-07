cmake_minimum_required(VERSION 3.19)

set(CMAKE_CXX_STANDARD 20)


#set(CMAKE_C_COMPILER "clang")
#set(CMAKE_CXX_COMPILER "clang++")
# Compiler flags.
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4 /external:anglebrackets /external:W0 /D_DEBUG /EHsc")
set(CMAKE_LINKER_FLAGS_DEBUG "/DEBUG")
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(BUILD_STATIC_LIB ON)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_VS_USE_DEBUG_LIBRARIES "$<CONFIG:Debug,Custom>")
set(DCMAKE_TOOLCHAIN_FILE "D:\\misc\\vcpkg\\scripts\\buildsystems\\vcpkg.cmake")

set(CMAKE_PREFIX_PATH "D:/Projects/pu-erh_lab/pu-erh_lab/third_party/easy_profiler-v2.1.0-msvc15-win64/lib/cmake/easy_profiler")

project(pu-erh_lab VERSION 0.1.0 LANGUAGES C CXX)


# Dependencies
# CTest
include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/6910c9d9165801d8827d628cb72eb7ea9dd538c5.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
set(GOOGLETEST_ENABLE_FUZZTEST ON CACHE BOOL "Enable fuzztest support")

FetchContent_MakeAvailable(googletest)
enable_testing() 

# LibRaw
add_library(LibRaw SHARED IMPORTED)
set_target_properties(LibRaw PROPERTIES
    cpu->apply(desc_idt);// // // // // // // // 
    IMPORTED_LOCATION "${CMAKE_SOURCE_DIR}\\pu-erh_lab\\third_party\\LibRaw-0.21.4_openmp\\buildfiles\\debug-x86_64\\libraw.dll"
    IMPORTED_IMPLIB "${CMAKE_SOURCE_DIR}\\pu-erh_lab\\third_party\\LibRaw-0.21.4_openmp\\buildfiles\\debug-x86_64\\libraw.lib"
    INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_SOURCE_DIR}\\pu-erh_lab\\third_party\\LibRaw-0.21.4_openmp"
)

# DuckDB API
add_library(DuckDB SHARED IMPORTED)
set_target_properties(DuckDB PROPERTIES
    IMPORTED_LOCATION "${CMAKE_SOURCE_DIR}/pu-erh_lab/third_party/libduckdb-windows/duckdb.dll"
    IMPORTED_IMPLIB "${CMAKE_SOURCE_DIR}/pu-erh_lab/third_party/libduckdb-windows/duckdb.lib"
    INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_SOURCE_DIR}/pu-erh_lab/third_party/libduckdb-windows"
)

# Cereal
add_library(Cereal INTERFACE "${CMAKE_SOURCE_DIR}/pu-erh_lab/third_party/cereal-1.3.2/include")

# UUID
add_library(stduuid INTERFACE)
target_include_directories(stduuid INTERFACE "${CMAKE_SOURCE_DIR}/pu-erh_lab/third_party/stduuid")

# UUID
add_library(uuid_v4 INTERFACE)
target_include_directories(uuid_v4 INTERFACE "${CMAKE_SOURCE_DIR}/pu-erh_lab/third_party/uuid_v4")

# Exiv2
add_library(Exiv2 SHARED IMPORTED)
set_target_properties(Exiv2 PROPERTIES
    IMPORTED_LOCATION "${CMAKE_SOURCE_DIR}/pu-erh_lab/third_party/exiv2_x64-windows/bin/exiv2.dll"
    IMPORTED_IMPLIB "${CMAKE_SOURCE_DIR}/pu-erh_lab/third_party/exiv2_x64-windows/lib/exiv2.lib"
    INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_SOURCE_DIR}/pu-erh_lab/third_party/exiv2_x64-windows/include"
)

# MurmurHash3
add_library(MurmurHash3 STATIC "${CMAKE_SOURCE_DIR}/pu-erh_lab/third_party/murmurhash3/MurmurHash3.cpp")
set_target_properties(MurmurHash3 PROPERTIES 
    INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_SOURCE_DIR}/pu-erh_lab/third_party/murmurhash3"
)

# utfcpp 
add_library(utfcpp INTERFACE)
target_include_directories(utfcpp INTERFACE "${CMAKE_SOURCE_DIR}/pu-erh_lab/third_party/utfcpp")

# OpenCV
set(OpenCV_DIR "D:\\Projects\\opencv_selfbuild\\opencv-4.11.0\\build")
find_package(OpenCV REQUIRED)

# xxHash
add_library(xxHash INTERFACE)
target_include_directories(xxHash INTERFACE "${CMAKE_SOURCE_DIR}/pu-erh_lab/third_party/xxhash")

# JSON  
add_library(JSON INTERFACE)
target_include_directories(JSON INTERFACE "${CMAKE_SOURCE_DIR}/pu-erh_lab/third_party/nlohmann_json")

# Libexif
add_library(Libexif SHARED IMPORTED)
set_target_properties(Libexif PROPERTIES
    IMPORTED_LOCATION "${CMAKE_SOURCE_DIR}/pu-erh_lab/third_party/libexif_x64-windows/debug/bin/exif-12.dll"
    IMPORTED_IMPLIB "${CMAKE_SOURCE_DIR}/pu-erh_lab/third_party/libexif_x64-windows/debug/lib/exif.lib"
    INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_SOURCE_DIR}/pu-erh_lab/third_party/libexif_x64-windows/include")

# Profiler
find_package(easy_profiler REQUIRED)

# OpenColorIO
# find_package(OpenColorIO CONFIG REQUIRED)
add_library(OpenColorIO SHARED IMPORTED)
set_target_properties(OpenColorIO PROPERTIES
    IMPORTED_LOCATION "${CMAKE_SOURCE_DIR}/pu-erh_lab/third_party/ocio/bin/OpenColorIO_2_4.dll"
    IMPORTED_IMPLIB "${CMAKE_SOURCE_DIR}/pu-erh_lab/third_party/ocio/lib/OpenColorIO.lib"
    INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_SOURCE_DIR}/pu-erh_lab/third_party/ocio/include")

    # Modules
add_subdirectory(${CMAKE_SOURCE_DIR}/pu-erh_lab/src)
add_subdirectory(${CMAKE_SOURCE_DIR}/pu-erh_lab/tests)

file(GLOB_RECURSE ALL_SOURCE_FILES
    "${CMAKE_SOURCE_DIR}/pu-erh_lab/src/*.c"
    "${CMAKE_SOURCE_DIR}/pu-erh_lab/src/*.h"
    "${CMAKE_SOURCE_DIR}/pu-erh_lab/src/*.cpp"
    "${CMAKE_SOURCE_DIR}/pu-erh_lab/src/*.hpp")
add_custom_target(format
    COMMAND clang-format
    -i
    -style=file
    ${ALL_SOURCE_FILES}
    COMMENT "Running clang-format on source files"
)