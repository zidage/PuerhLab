cmake_minimum_required(VERSION 3.19)


#set(CMAKE_C_COMPILER "clang")
#set(CMAKE_CXX_COMPILER "clang++")
# Compiler flags.

project(pu-erh_lab VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
# set(BUILD_STATIC_LIBS ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)



if (MSVC)
    set(CMAKE_CXX_FLAGS_DEBUG "/Zi /arch:AVX /bigobj /W4 /external:anglebrackets /external:W0 /D_DEBUG /EHsc /openmp")
    set(CMAKE_CXX_FLAGS_RELEASE "/O2 /Ob3 /arch:AVX2 /bigobj /W4 /external:anglebrackets /external:W0 /EHsc /openmp /GL")
    # add_link_options(/DEBUG)
    set(CMAKE_VS_USE_DEBUG_LIBRARIES "$<CONFIG:Debug,Custom>")
    # FIXME: Hardcoded Qt path for Windows, consider making it configurable
    set(CMAKE_PREFIX_PATH "D:/misc/Qt/6.9.3/msvc2022_64/lib/cmake;${CMAKE_PREFIX_PATH}")
else()
    #set(CMAKE_CXX_FLAGS_INIT "-Wall -Wextra -Wpedantic -D_DEBUG -fopenmp -O2 -fsanitize=address -g")

    add_compile_options(-Wall -Wextra -Wpedantic)

    add_compile_options(
        $<$<CONFIG:Debug>:-g>
        $<$<CONFIG:Debug>:-O2>
        $<$<CONFIG:Debug>:-fsanitize=address>
        $<$<CONFIG:Debug>:-fno-omit-frame-pointer>
        $<$<CONFIG:Release>:-O3>
        $<$<CONFIG:Release>:-DNDEBUG>
    )

    add_link_options(
        $<$<CONFIG:Debug>:-fsanitize=address>
    )
    if(APPLE)
        add_compile_definitions(__MACOSX_CORE__)
    # FIXME: Hardcoded Qt path for macOS, consider making it configurable
    set(CMAKE_PREFIX_PATH "/Users/zidage/Qt/6.9.3/macos/lib/cmake;${CMAKE_PREFIX_PATH}")
    endif()
endif()

# Qt6
find_package(Qt6 REQUIRED COMPONENTS Widgets Quick OpenGL OpenGLWidgets)

set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTORCC ON)

# CUDA
find_package(CUDAToolkit)
if (CUDAToolkit_FOUND)
    enable_language(CUDA)
    add_definitions(-DHAVE_CUDA)
    add_compile_options($<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Release>>:-O3>)
    add_compile_options($<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Release>>:--use_fast_math>)
    add_compile_options($<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Release>>:--extra-device-vectorization>) 
    add_compile_options($<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Release>>:-gencode=arch=compute_75,code=sm_75>) 
    add_compile_options($<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Debug>>:-G>)
    if(MSVC)
        list(APPEND CMAKE_CUDA_HOST_COMPILER_OPTIONS /W4 /external:anglebrackets /external:W0 /D_DEBUG /EHsc /arch:AVX2 /openmp)
    endif()
endif()

# Dependencies
# GTest
include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.17.0.zip
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
if (WIN32)
    find_package(xxHash CONFIG REQUIRED)
else()
    find_package(xxHash CONFIG REQUIRED)
    # Create a unified alias target 'xxHash' so the rest of the project can link to it.
    add_library(xxHash INTERFACE)
    if (TARGET xxHash::xxhash)
        target_link_libraries(xxHash INTERFACE xxHash::xxhash)
    elseif (TARGET xxhash)
        target_link_libraries(xxHash INTERFACE xxhash)
    else()
        message(FATAL_ERROR "xxHash package found but no known CMake target (xxHash::xxhash or xxhash) was defined.")
    endif()
endif()

# JSON  
add_library(JSON INTERFACE)
target_include_directories(JSON INTERFACE "${CMAKE_SOURCE_DIR}/pu-erh_lab/third_party/nlohmann_json")


# Other libraries
# OpenCV
find_package(OpenCV REQUIRED)

# Eigen3
find_package(Eigen3 CONFIG REQUIRED)

# OpenGL
find_package(OpenGL REQUIRED)
find_package(glad CONFIG REQUIRED)

# Profiler/hwy/lcms2
if (WIN32)
find_package(easy_profiler REQUIRED)
add_compile_definitions(HAS_EASY_PROFILER)
endif()
find_package(hwy CONFIG REQUIRED)

if (WIN32)
find_package(lcms2 CONFIG REQUIRED)
elseif(APPLE)
    find_library(LCMS2_LIB NAMES lcms2 PATHS /opt/homebrew/lib /usr/local/lib)
    find_path(LCMS2_INC NAMES lcms2.h PATHS /opt/homebrew/include /usr/local/include)
    if(LCMS2_LIB AND LCMS2_INC)
        add_library(lcms2::lcms2 SHARED IMPORTED)
        set_target_properties(lcms2::lcms2 PROPERTIES IMPORTED_LOCATION "${LCMS2_LIB}" INTERFACE_INCLUDE_DIRECTORIES "${LCMS2_INC}")
    else()
        message(FATAL_ERROR "lcms2 not found. Install via Homebrew (lcms2) or vcpkg.")
    endif()
endif()
if(UNIX AND NOT APPLE)
    find_package(lcms2 REQUIRED)
endif()

# OpenColorIO
find_package(OpenColorIO CONFIG REQUIRED)
if (TARGET OpenColorIO::OpenColorIO)
    add_library(OpenColorIO ALIAS OpenColorIO::OpenColorIO)
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
   
# OpenImageIO
find_package(OpenImageIO CONFIG REQUIRED)

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
    find_package(libraw CONFIG REQUIRED)
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

# Lensfun
set(PUERHLAB_LENSFUN_ROOT
    "${CMAKE_SOURCE_DIR}/pu-erh_lab/third_party/lensfun/install"
    CACHE PATH "Lensfun install prefix (must contain include/, lib/, and bin/)")

if (WIN32)
    # vcpkg-style package if available.
    find_package(lensfun CONFIG QUIET)

    # Fallback for local source build installs that only provide headers/libs.
    if (NOT lensfun_FOUND)
        unset(LENSFUN_INCLUDE_DIR CACHE)
        unset(LENSFUN_IMPLIB CACHE)
        unset(LENSFUN_DLL CACHE)

        find_path(LENSFUN_INCLUDE_DIR
            NAMES lensfun/lensfun.h
            PATHS "${PUERHLAB_LENSFUN_ROOT}/include"
            NO_DEFAULT_PATH
            NO_CMAKE_FIND_ROOT_PATH
        )
        find_library(LENSFUN_IMPLIB
            NAMES lensfun
            PATHS "${PUERHLAB_LENSFUN_ROOT}/lib"
            NO_DEFAULT_PATH
            NO_CMAKE_FIND_ROOT_PATH
        )
        find_file(LENSFUN_DLL
            NAMES lensfun.dll
            PATHS "${PUERHLAB_LENSFUN_ROOT}/bin"
            NO_DEFAULT_PATH
            NO_CMAKE_FIND_ROOT_PATH
        )

        if (NOT LENSFUN_INCLUDE_DIR OR NOT LENSFUN_IMPLIB)
            message(FATAL_ERROR
                "Lensfun was not found. Install via vcpkg or set PUERHLAB_LENSFUN_ROOT "
                "to a valid Lensfun install prefix (include/lib/bin).")
        endif()

        add_library(lensfun::lensfun SHARED IMPORTED)
        set_target_properties(lensfun::lensfun PROPERTIES
            IMPORTED_IMPLIB "${LENSFUN_IMPLIB}"
            INTERFACE_INCLUDE_DIRECTORIES "${LENSFUN_INCLUDE_DIR}"
        )
        if (LENSFUN_DLL)
            set_target_properties(lensfun::lensfun PROPERTIES
                IMPORTED_LOCATION "${LENSFUN_DLL}"
            )
        endif()
    endif()

    if (TARGET lensfun AND NOT TARGET lensfun::lensfun)
        add_library(lensfun::lensfun ALIAS lensfun)
    endif()
    if (TARGET lensfun::lensfun AND NOT TARGET lensfun)
        add_library(lensfun INTERFACE)
        target_link_libraries(lensfun INTERFACE lensfun::lensfun)
    endif()
else()
    find_package(lensfun QUIET)
    if (NOT lensfun_FOUND AND NOT TARGET lensfun AND NOT TARGET lensfun::lensfun)
        message(FATAL_ERROR "Lensfun not found. Install via Homebrew (lensfun) or vcpkg.")
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
set(TEST_RES_DIR "${CMAKE_SOURCE_DIR}/pu-erh_lab/tests/resources" CACHE PATH "Path to test resources")
file(TO_CMAKE_PATH "${TEST_RES_DIR}" TEST_RES_DIR)
add_compile_definitions(
    TEST_DB_PATH=\"${TEST_RES_DIR}/temp_folder/test.db\"
    TEST_META_PATH=\"${TEST_RES_DIR}/temp_folder/meta.json\"
    TEST_IMG_PATH=\"${TEST_RES_DIR}/sample_images\"
    TEST_PROFILER_OUTPUT_PATH=\"${TEST_RES_DIR}/temp_folder/test_profile.prof\"
    CONFIG_PATH=\"${CMAKE_SOURCE_DIR}/pu-erh_lab/src/config/\"
)

include(GNUInstallDirs)

add_subdirectory(${CMAKE_SOURCE_DIR}/pu-erh_lab/src)
add_subdirectory(${CMAKE_SOURCE_DIR}/pu-erh_lab/tests)



file(GLOB_RECURSE ALL_SOURCE_FILES
    "${CMAKE_SOURCE_DIR}/pu-erh_lab/src/*.c"
    "${CMAKE_SOURCE_DIR}/pu-erh_lab/src/*.cpp"
    "${CMAKE_SOURCE_DIR}/pu-erh_lab/src/*.cu"
    "${CMAKE_SOURCE_DIR}/pu-erh_lab/src/*.h"
    "${CMAKE_SOURCE_DIR}/pu-erh_lab/src/*.hpp"
    "${CMAKE_SOURCE_DIR}/pu-erh_lab/src/*.cuh")

# Files compiled by clang-tidy; keep to TUs so compile_commands entries are used.
file(GLOB_RECURSE TIDY_SOURCE_FILES
    "${CMAKE_SOURCE_DIR}/pu-erh_lab/src/*.c"
    "${CMAKE_SOURCE_DIR}/pu-erh_lab/src/*.cpp"
    "${CMAKE_SOURCE_DIR}/pu-erh_lab/src/*.cu"
    "${CMAKE_SOURCE_DIR}/pu-erh_lab/tests/*.cpp")

# Header filter: allow both '/' and '\\' so Windows paths match.
set(MY_HEADER_FILTER ".*pu-erh_lab[\\\\/](src|tests)[\\\\/].*")

add_custom_target(format
    COMMAND clang-format
    -i
    -style=file
    ${ALL_SOURCE_FILES}
    COMMENT "Running clang-format on source files"
)

add_custom_target(tidy
    COMMAND clang-tidy
    "\"-p=${CMAKE_BINARY_DIR}\""
    "\"--config-file=${CMAKE_SOURCE_DIR}/.clang-tidy\""
    "\"--header-filter=${MY_HEADER_FILTER}\""
    "\"--extra-arg=--driver-mode=cl\""
    "\"--extra-arg=-fms-compatibility\""
    "\"--extra-arg=-fms-extensions\""
    "-fix"
    "-fix-errors"
    ${TIDY_SOURCE_FILES}
    COMMENT "Running clang-tidy on source files"
)

set(CPACK_GENERATOR "ZIP")
set(CPACK_PACKAGE_NAME "pu-erh_lab")
set(CPACK_PACKAGE_VERSION "${PROJECT_VERSION}")
set(CPACK_PACKAGE_FILE_NAME "${CPACK_PACKAGE_NAME}-${CPACK_PACKAGE_VERSION}-${CMAKE_SYSTEM_NAME}-${CMAKE_SYSTEM_PROCESSOR}")
include(CPack)
