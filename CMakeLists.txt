cmake_minimum_required(VERSION 3.19)

set(CMAKE_CXX_STANDARD 20)


#set(CMAKE_C_COMPILER "clang")
#set(CMAKE_CXX_COMPILER "clang++")
# Compiler flags.
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4 /external:anglebrackets /external:W0 /D_DEBUG /EHsc ")
set(CMAKE_LINKER_FLAGS_DEBUG "/DEBUG")
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(BUILD_STATIC_LIB OFF)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)



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

FetchContent_MakeAvailable(googletest)
enable_testing() 

# LibRaw
add_library(LibRaw SHARED IMPORTED)
set_target_properties(LibRaw PROPERTIES
    IMPORTED_LOCATION "${CMAKE_SOURCE_DIR}\\pu-erh_lab\\third_party\\LibRaw-0.21.3-windows\\bin\\libraw.dll"
    IMPORTED_IMPLIB "${CMAKE_SOURCE_DIR}\\pu-erh_lab\\third_party\\LibRaw-0.21.3-windows\\lib\\libraw.lib"
    INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_SOURCE_DIR}\\pu-erh_lab\\third_party\\LibRaw-0.21.3-windows"
)

# DuckDB API
add_library(DuckDB SHARED IMPORTED)
set_target_properties(DuckDB PROPERTIES
    IMPORTED_LOCATION "${CMAKE_SOURCE_DIR}\\pu-erh_lab\\third_party\\libduckdb-windows\\duckdb.dll"
    IMPORTED_IMPLIB "${CMAKE_SOURCE_DIR}\\pu-erh_lab\\third_party\\libduckdb-windows\\duckdb.lib"
    INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_SOURCE_DIR}\\pu-erh_lab\\third_party\\libduckdb-windows"
)

# Cereal
add_library(Cereal INTERFACE "${CMAKE_SOURCE_DIR}/pu-erh_lab/third_party/cereal-1.3.2/include")

# UUID
add_library(stduuid INTERFACE)
target_include_directories(stduuid INTERFACE "${CMAKE_SOURCE_DIR}/pu-erh_lab/third_party/stduuid")

# Exiv2
add_library(Exiv2 SHARED IMPORTED)
set_target_properties(Exiv2 PROPERTIES
    IMPORTED_LOCATION "${CMAKE_SOURCE_DIR}/pu-erh_lab/third_party/exiv2-0.28.5-2022msvc-AMD64/bin/exiv2.dll"
    IMPORTED_IMPLIB "${CMAKE_SOURCE_DIR}/pu-erh_lab/third_party/exiv2-0.28.5-2022msvc-AMD64/lib/exiv2.lib"
    INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_SOURCE_DIR}/pu-erh_lab/third_party/exiv2-0.28.5-2022msvc-AMD64/include"
)

# MurmurHash3
add_library(MurmurHash3 STATIC "${CMAKE_SOURCE_DIR}/pu-erh_lab/third_party/murmurhash3/MurmurHash3.cpp")
set_target_properties(MurmurHash3 PROPERTIES 
    INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_SOURCE_DIR}/pu-erh_lab/third_party/murmurhash3"
)

# OpenCV
set(OpenCV_DIR "D:/misc/opencv/build")
find_package(OpenCV REQUIRED)




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