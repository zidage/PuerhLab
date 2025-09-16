cmake_minimum_required(VERSION 3.19)



#set(CMAKE_C_COMPILER "clang")
#set(CMAKE_CXX_COMPILER "clang++")
# Compiler flags.
if (NOT CMAKE_TOOLCHAIN_FILE AND DEFINED ENV{VCPKG_ROOT} AND NOT DEFINED CMAKE_VCPKG_ROOT)
    set(CMAKE_VCPKG_ROOT $ENV{VCPKG_ROOT})     
endif()

project(pu-erh_lab VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(BUILD_STATIC_LIBS ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if (MSVC)
    set(CMAKE_CXX_FLAGS_INIT "/W4 /external:anglebrackets /external:W0 /D_DEBUG /EHsc /arch:AVX2 /openmp")
    add_link_options(/DEBUG)
    set(CMAKE_VS_USE_DEBUG_LIBRARIES "$<CONFIG:Debug,Custom>")
else()
    set(CMAKE_CXX_FLAGS_INIT "-Wall -Wextra -Wpedantic")
    if(APPLE)
        add_compile_definitions(__MACOSX_CORE__)
    endif()
endif()

find_package(CUDAToolkit)
if (CUDAToolkit_FOUND)
    enable_language(CUDA)
    add_definitions(-DHAVE_CUDA)
    
    if(MSVC)
        list(APPEND CMAKE_CUDA_HOST_COMPILER_OPTIONS /W4 /external:anglebrackets /external:W0 /D_DEBUG /EHsc /arch:AVX2 /openmp)
    endif()
endif()

# Dependencies
# GTest
include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/6910c9d9165801d8827d628cb72eb7ea9dd538c5.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)
enable_testing() 

# OpenMP
find_package(OpenMP REQUIRED)

# Headers-only libraries
# UUID
add_library(stduuid INTERFACE)
target_include_directories(stduuid INTERFACE "${CMAKE_SOURCE_DIR}/pu-erh_lab/third_party/stduuid")

# UUID
add_library(uuid_v4 INTERFACE)
target_include_directories(uuid_v4 INTERFACE "${CMAKE_SOURCE_DIR}/pu-erh_lab/third_party/uuid_v4")

# utfcpp 
add_library(utfcpp INTERFACE)
target_include_directories(utfcpp INTERFACE "${CMAKE_SOURCE_DIR}/pu-erh_lab/third_party/utfcpp")

# MurmurHash3
add_library(MurmurHash3 STATIC "${CMAKE_SOURCE_DIR}/pu-erh_lab/third_party/murmurhash3/MurmurHash3.cpp")
set_target_properties(MurmurHash3 PROPERTIES 
    INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_SOURCE_DIR}/pu-erh_lab/third_party/murmurhash3"
)

# xxHash
add_library(xxHash INTERFACE)
target_include_directories(xxHash INTERFACE "${CMAKE_SOURCE_DIR}/pu-erh_lab/third_party/xxhash")

# JSON  
add_library(JSON INTERFACE)
target_include_directories(JSON INTERFACE "${CMAKE_SOURCE_DIR}/pu-erh_lab/third_party/nlohmann_json")


# Other libraries
# OpenCV
find_package(OpenCV REQUIRED)

# Profiler/hwy/lcms2
find_package(easy_profiler REQUIRED)
find_package(hwy CONFIG REQUIRED)
find_package(lcms2 CONFIG REQUIRED)

# OpenColorIO
find_package(OpenColorIO CONFIG QUIET)
if (TARGET OpenColorIO::OpenColorIO)
    add_library(OpenColorIO ALIAS OpenColorIO::OpenColorIO)
elseif(WIN32)
    add_library(OpenColorIO SHARED IMPORTED)
    set_target_properties(OpenColorIO PROPERTIES
        IMPORTED_LOCATION "${CMAKE_SOURCE_DIR}/pu-erh_lab/third_party/ocio/Release/bin/OpenColorIO_2_4.dll"
        IMPORTED_IMPLIB "${CMAKE_SOURCE_DIR}/pu-erh_lab/third_party/ocio/Release/lib/OpenColorIO.lib"
        INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_SOURCE_DIR}/pu-erh_lab/third_party/ocio/Release/include")
else()
    find_library(OCIO_LIB NAMES OpenColorIO PATHS /opt/homebrew/lib /usr/local/lib)
    find_path(OCIO_INC NAMES OpenColorIO/OpenColorIO.h PATHS /opt/homebrew/include /usr/local/include)
    if(OCIO_LIB AND OCIO_INC)
        add_library(OpenColorIO SHARED IMPORTED)
        set_target_properties(OpenColorIO PROPERTIES IMPORTED_LOCATION "${OCIO_LIB}" INTERFACE_INCLUDE_DIRECTORIES "${OCIO_INC}")
    else()
        message(FATAL_ERROR "OpenColorIO not found. Install via Homebrew (opencolorio) or vcpkg.")
    endif()
endif()
   
# Exiv2
find_package(Exiv2 QUIET CONFIG)
if (TARGET Exiv2::Exiv2 AND NOT TARGET Exiv2)
    add_library(Exiv2 ALIAS Exiv2::Exiv2)
elseif(WIN32)
    add_library(Exiv2 SHARED IMPORTED)
    set_target_properties(Exiv2 PROPERTIES
        IMPORTED_LOCATION "${CMAKE_SOURCE_DIR}/pu-erh_lab/third_party/exiv2_x64-windows/bin/exiv2.dll"
        IMPORTED_IMPLIB "${CMAKE_SOURCE_DIR}/pu-erh_lab/third_party/exiv2_x64-windows/lib/exiv2.lib"
        INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_SOURCE_DIR}/pu-erh_lab/third_party/exiv2_x64-windows/include"
    )
else()
  find_library(EXIV2_LIB NAMES exiv2 PATHS /opt/homebrew/lib /usr/local/lib)
  find_path(EXIV2_INC NAMES exiv2/exiv2.hpp PATHS /opt/homebrew/include /usr/local/include)
  if(EXIV2_LIB AND EXIV2_INC)
    add_library(Exiv2 SHARED IMPORTED)
    set_target_properties(Exiv2 PROPERTIES IMPORTED_LOCATION "${EXIV2_LIB}" INTERFACE_INCLUDE_DIRECTORIES "${EXIV2_INC}")
  else()
    message(FATAL_ERROR "Exiv2 not found. Install via Homebrew (exiv2) or vcpkg.")
  endif()
endif()

# LibRaw
if (WIN32)
    add_library(LibRaw SHARED IMPORTED)
    set_target_properties(LibRaw PROPERTIES
        IMPORTED_LOCATION "${CMAKE_SOURCE_DIR}\\pu-erh_lab\\third_party\\LibRaw-0.21.4_openmp\\buildfiles\\release-x86_64\\libraw.dll"
        IMPORTED_IMPLIB "${CMAKE_SOURCE_DIR}\\pu-erh_lab\\third_party\\LibRaw-0.21.4_openmp\\buildfiles\\release-x86_64\\libraw.lib"
        INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_SOURCE_DIR}\\pu-erh_lab\\third_party\\LibRaw-0.21.4_openmp")
else()
    find_path(LIBRAW_INC NAMES libraw/libraw.h PATHS /opt/homebrew/include /usr/local/include PATH_SUFFIXES libraw)
    find_library(LIBRAW_LIB NAMES raw libraw PATHS /opt/homebrew/lib /usr/local/lib)
    if(LIBRAW_INC AND LIBRAW_LIB)
        add_library(LibRaw SHARED IMPORTED)
        set_target_properties(LibRaw PROPERTIES IMPORTED_LOCATION "${LIBRAW_LIB}" INTERFACE_INCLUDE_DIRECTORIES "${LIBRAW_INC}")
    else()
        message(FATAL_ERROR "LibRaw not found. Install via Homebrew (libraw) or vcpkg.")
    endif()
endif()

# DuckDB API
if (WIN32)
    add_library(DuckDB SHARED IMPORTED)
    set_target_properties(DuckDB PROPERTIES
        IMPORTED_LOCATION "${CMAKE_SOURCE_DIR}/pu-erh_lab/third_party/libduckdb-windows/duckdb.dll"
        IMPORTED_IMPLIB "${CMAKE_SOURCE_DIR}/pu-erh_lab/third_party/libduckdb-windows/duckdb.lib"
        INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_SOURCE_DIR}/pu-erh_lab/third_party/libduckdb-windows")
else()        
  find_path(DUCKDB_INC NAMES duckdb.h PATHS /opt/homebrew/include /usr/local/include)
  find_library(DUCKDB_LIB NAMES duckdb PATHS /opt/homebrew/lib /usr/local/lib)
  if(DUCKDB_INC AND DUCKDB_LIB)
    add_library(DuckDB SHARED IMPORTED)
    set_target_properties(DuckDB PROPERTIES IMPORTED_LOCATION "${DUCKDB_LIB}" INTERFACE_INCLUDE_DIRECTORIES "${DUCKDB_INC}")
  else()
    message(FATAL_ERROR "DuckDB not found. Install via Homebrew (duckdb) or vcpkg.")
  endif()
endif()

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