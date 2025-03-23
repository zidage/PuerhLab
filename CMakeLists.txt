cmake_minimum_required(VERSION 3.10.0)

set(CMAKE_CXX_STANDARD 20)

set(CMAKE_C_COMPILER "clang")
set(CMAKE_CXX_COMPILER "clang++")
# Compiler flags.
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Werror")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-parameter -Wno-attributes") # TODO: remove
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 -ggdb")
#set(CMAKE_POSITION_INDEPENDENT_CODE ON)


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
add_library(LibRaw STATIC IMPORTED)
set_target_properties(LibRaw PROPERTIES
    IMPORTED_LOCATION "${CMAKE_SOURCE_DIR}/pu-erh_lab/third_party/LibRaw-0.21.3-windows/lib/libraw.lib"
    INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_SOURCE_DIR}/pu-erh_lab/third_party/LibRaw-0.21.3-windows/libraw"
)

# DuckDB API
add_library(DuckDB STATIC IMPORTED)
set_target_properties(DuckDB PROPERTIES
    IMPORTED_LOCATION "${CMAKE_SOURCE_DIR}/pu-erh_lab/third_party/libduckdb-windows/duckdb.lib"
    INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_SOURCE_DIR}/pu-erh_lab/third_party/libduckdb-windows"
)

# Modules
add_subdirectory(${CMAKE_SOURCE_DIR}/pu-erh_lab/tests)
add_subdirectory(${CMAKE_SOURCE_DIR}/pu-erh_lab/src)