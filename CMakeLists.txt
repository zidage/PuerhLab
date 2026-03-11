cmake_minimum_required(VERSION 3.19)


#set(CMAKE_C_COMPILER "clang")
#set(CMAKE_CXX_COMPILER "clang++")
# Compiler flags.

project(pu-erh_lab VERSION 0.1.2 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
# set(BUILD_STATIC_LIBS ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(APPLE)
    set(PUERHLAB_ENABLE_CUDA_DEFAULT OFF)
    set(PUERHLAB_ENABLE_METAL_DEFAULT ON)
    set(PUERHLAB_ENABLE_OPENGL_EDITOR_DEFAULT OFF)
else()
    set(PUERHLAB_ENABLE_CUDA_DEFAULT ON)
    set(PUERHLAB_ENABLE_METAL_DEFAULT OFF)
    set(PUERHLAB_ENABLE_OPENGL_EDITOR_DEFAULT ON)
endif()

option(PUERHLAB_ENABLE_CUDA "Enable CUDA-accelerated pipeline backends when CUDA is available." ${PUERHLAB_ENABLE_CUDA_DEFAULT})
option(PUERHLAB_ENABLE_METAL "Enable Metal-accelerated pipeline backends when building on Apple platforms." ${PUERHLAB_ENABLE_METAL_DEFAULT})
option(PUERHLAB_ENABLE_OPENGL_EDITOR "Build the OpenGL-based editor UI when the required backend is available." ${PUERHLAB_ENABLE_OPENGL_EDITOR_DEFAULT})
option(PUERHLAB_BUILD_TESTS "Build pu-erh_lab tests and demos." ON)

function(puerhlab_append_existing_prefix prefix)
    if(EXISTS "${prefix}")
        list(APPEND CMAKE_PREFIX_PATH "${prefix}")
        set(CMAKE_PREFIX_PATH "${CMAKE_PREFIX_PATH}" PARENT_SCOPE)
    endif()
endfunction()

puerhlab_append_existing_prefix("/opt/homebrew")
puerhlab_append_existing_prefix("/opt/homebrew/opt/qt")
puerhlab_append_existing_prefix("/opt/homebrew/opt/opencv")
puerhlab_append_existing_prefix("/opt/homebrew/opt/opencolorio")
puerhlab_append_existing_prefix("/opt/homebrew/opt/openimageio")
puerhlab_append_existing_prefix("/opt/homebrew/opt/exiv2")
puerhlab_append_existing_prefix("/opt/homebrew/opt/glib")
puerhlab_append_existing_prefix("/opt/homebrew/opt/libraw")
puerhlab_append_existing_prefix("/opt/homebrew/opt/xxhash")
puerhlab_append_existing_prefix("/opt/homebrew/opt/duckdb")
puerhlab_append_existing_prefix("/opt/homebrew/opt/little-cms2")
puerhlab_append_existing_prefix("/opt/homebrew/opt/highway")
puerhlab_append_existing_prefix("/usr/local")
puerhlab_append_existing_prefix("/usr/local/opt/qt")
puerhlab_append_existing_prefix("/usr/local/opt/opencv")
puerhlab_append_existing_prefix("/usr/local/opt/opencolorio")
puerhlab_append_existing_prefix("/usr/local/opt/openimageio")
puerhlab_append_existing_prefix("/usr/local/opt/exiv2")
puerhlab_append_existing_prefix("/usr/local/opt/glib")
puerhlab_append_existing_prefix("/usr/local/opt/libraw")
puerhlab_append_existing_prefix("/usr/local/opt/xxhash")
puerhlab_append_existing_prefix("/usr/local/opt/duckdb")
puerhlab_append_existing_prefix("/usr/local/opt/little-cms2")
puerhlab_append_existing_prefix("/usr/local/opt/highway")

set(PUERHLAB_LIBRARY_HINTS "")
set(PUERHLAB_INCLUDE_HINTS "")
foreach(_prefix IN LISTS CMAKE_PREFIX_PATH)
    if(EXISTS "${_prefix}")
        list(APPEND PUERHLAB_LIBRARY_HINTS "${_prefix}/lib")
        list(APPEND PUERHLAB_INCLUDE_HINTS "${_prefix}/include")
    endif()
endforeach()
list(REMOVE_DUPLICATES PUERHLAB_LIBRARY_HINTS)
list(REMOVE_DUPLICATES PUERHLAB_INCLUDE_HINTS)



if (MSVC)
    set(CMAKE_CXX_FLAGS_DEBUG "/Zi /arch:AVX /bigobj /W4 /external:anglebrackets /external:W0 /D_DEBUG /EHsc /openmp /Zc:preprocessor")
    set(CMAKE_CXX_FLAGS_RELEASE "/O2 /Ob3 /arch:AVX2 /bigobj /W4 /external:anglebrackets /external:W0 /EHsc /openmp /GL /Zc:preprocessor")
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
    endif()
endif()

# Qt6
set(PUERHLAB_QT_COMPONENTS
    Core
    LinguistTools
    Widgets
    Quick
    Qml
    QuickControls2
    QuickDialogs2
    QuickEffects
)
if(PUERHLAB_BUILD_TESTS)
    list(APPEND PUERHLAB_QT_COMPONENTS Test)
endif()
find_package(Qt6 REQUIRED COMPONENTS ${PUERHLAB_QT_COMPONENTS})

set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTORCC ON)

# CUDA
set(PUERHLAB_CUDA_ENABLED OFF)
if(PUERHLAB_ENABLE_CUDA)
    find_package(CUDAToolkit QUIET)
endif()
if (PUERHLAB_ENABLE_CUDA AND CUDAToolkit_FOUND)
    enable_language(CUDA)
    add_definitions(-DHAVE_CUDA)
    set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)
    set(CMAKE_CUDA_ARCHITECTURES 60-real 61-real 75-real 86-real 86-virtual 89-real 90-real)
    include_directories(SYSTEM ${CUDAToolkit_INCLUDE_DIRS})
    add_compile_options($<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Release>>:-O3>)
    add_compile_options($<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Release>>:--use_fast_math>)
    add_compile_options($<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Release>>:--extra-device-vectorization>)
    add_compile_options($<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Debug>>:-G>)
    if(MSVC)
        list(APPEND CMAKE_CUDA_HOST_COMPILER_OPTIONS /W4 /external:anglebrackets /external:W0 /D_DEBUG /EHsc /arch:AVX2 /openmp)
    endif()
    set(PUERHLAB_CUDA_ENABLED ON)
elseif(PUERHLAB_ENABLE_CUDA)
    message(STATUS "CUDA toolkit not found; building without CUDA backends.")
endif()

# Metal
set(PUERHLAB_METAL_ENABLED OFF)
if(PUERHLAB_ENABLE_METAL AND APPLE)
    add_compile_definitions(HAVE_METAL)
    set(PUERHLAB_METAL_ENABLED ON)
elseif(PUERHLAB_ENABLE_METAL)
    message(STATUS "Metal support is only available on Apple platforms; building without Metal backends.")
endif()

# Dependencies
# GTest
if(PUERHLAB_BUILD_TESTS)
    include(FetchContent)
    FetchContent_Declare(
      googletest
      URL https://github.com/google/googletest/archive/refs/tags/v1.17.0.zip
    )
    # For Windows: Prevent overriding the parent project's compiler/linker settings
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
    enable_testing()
endif()

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
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs calib3d highgui)

# Eigen3
find_package(Eigen3 CONFIG REQUIRED)

set(PUERHLAB_REAL_WIDGET_EDITOR OFF)
set(PUERHLAB_REAL_OPENGL_EDITOR OFF)
set(PUERHLAB_RHI_WIDGET_EDITOR OFF)
if(PUERHLAB_ENABLE_OPENGL_EDITOR AND PUERHLAB_CUDA_ENABLED)
    find_package(Qt6 REQUIRED COMPONENTS OpenGL OpenGLWidgets)
    find_package(OpenGL REQUIRED)
    set(PUERHLAB_REAL_OPENGL_EDITOR ON)
    set(PUERHLAB_REAL_WIDGET_EDITOR ON)
    add_compile_definitions(PUERHLAB_HAS_LEGACY_GL_VIEWER)
elseif(PUERHLAB_ENABLE_OPENGL_EDITOR)
    message(STATUS "OpenGL editor requested, but no accelerated backend is available. Building editor stub only.")
endif()

if(APPLE AND PUERHLAB_METAL_ENABLED)
    find_package(Qt6 REQUIRED COMPONENTS ShaderTools GuiPrivate)
    set(PUERHLAB_RHI_WIDGET_EDITOR ON)
    set(PUERHLAB_REAL_WIDGET_EDITOR ON)
    add_compile_definitions(PUERHLAB_HAS_RHI_VIEWER)
endif()

# Profiler/hwy/lcms2
if (WIN32)
find_package(easy_profiler REQUIRED)
add_compile_definitions(HAS_EASY_PROFILER)
endif()
find_package(hwy CONFIG REQUIRED)

if (WIN32)
find_package(lcms2 CONFIG REQUIRED)
elseif(APPLE)
    find_library(LCMS2_LIB NAMES lcms2 PATHS ${PUERHLAB_LIBRARY_HINTS})
    find_path(LCMS2_INC NAMES lcms2.h PATHS ${PUERHLAB_INCLUDE_HINTS})
    if(LCMS2_LIB AND LCMS2_INC)
        add_library(lcms2::lcms2 SHARED IMPORTED)
        set_target_properties(lcms2::lcms2 PROPERTIES IMPORTED_LOCATION "${LCMS2_LIB}" INTERFACE_INCLUDE_DIRECTORIES "${LCMS2_INC}")
    else()
        message(FATAL_ERROR "lcms2 not found. Install via Homebrew (little-cms2) or set CMAKE_PREFIX_PATH to its prefix.")
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
    find_library(OCIO_LIB NAMES OpenColorIO PATHS ${PUERHLAB_LIBRARY_HINTS})
    find_path(OCIO_INC NAMES OpenColorIO/OpenColorIO.h PATHS ${PUERHLAB_INCLUDE_HINTS})
    if(OCIO_LIB AND OCIO_INC)
        add_library(OpenColorIO SHARED IMPORTED)
        set_target_properties(OpenColorIO PROPERTIES IMPORTED_LOCATION "${OCIO_LIB}" INTERFACE_INCLUDE_DIRECTORIES "${OCIO_INC}")
    else()
        message(FATAL_ERROR "OpenColorIO not found. Install via Homebrew (opencolorio) or set CMAKE_PREFIX_PATH to its prefix.")
    endif()
endif()
   
# OpenImageIO
find_package(OpenImageIO CONFIG QUIET)
if (NOT TARGET OpenImageIO::OpenImageIO)
    if (TARGET OpenImageIO)
        add_library(OpenImageIO::OpenImageIO ALIAS OpenImageIO)
    else()
        find_library(OPENIMAGEIO_LIB NAMES OpenImageIO PATHS ${PUERHLAB_LIBRARY_HINTS})
        find_path(OPENIMAGEIO_INC NAMES OpenImageIO/imageio.h PATHS ${PUERHLAB_INCLUDE_HINTS})
        if(OPENIMAGEIO_LIB AND OPENIMAGEIO_INC)
            add_library(OpenImageIO::OpenImageIO SHARED IMPORTED)
            set_target_properties(OpenImageIO::OpenImageIO PROPERTIES
                IMPORTED_LOCATION "${OPENIMAGEIO_LIB}"
                INTERFACE_INCLUDE_DIRECTORIES "${OPENIMAGEIO_INC}"
            )
        else()
            message(FATAL_ERROR "OpenImageIO not found. Install via Homebrew (openimageio) or set CMAKE_PREFIX_PATH to its prefix.")
        endif()
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
  find_library(EXIV2_LIB NAMES exiv2 PATHS ${PUERHLAB_LIBRARY_HINTS})
  find_path(EXIV2_INC NAMES exiv2/exiv2.hpp PATHS ${PUERHLAB_INCLUDE_HINTS})
  if(EXIV2_LIB AND EXIV2_INC)
    add_library(Exiv2 SHARED IMPORTED)
    set_target_properties(Exiv2 PROPERTIES IMPORTED_LOCATION "${EXIV2_LIB}" INTERFACE_INCLUDE_DIRECTORIES "${EXIV2_INC}")
  else()
    message(FATAL_ERROR "Exiv2 not found. Install via Homebrew (exiv2) or set CMAKE_PREFIX_PATH to its prefix.")
  endif()
endif()

# LibRaw
if (WIN32)
    find_package(libraw CONFIG REQUIRED)
else()
    find_path(LIBRAW_INC NAMES libraw/libraw.h PATHS ${PUERHLAB_INCLUDE_HINTS})
    find_library(LIBRAW_LIB NAMES raw_r raw libraw PATHS ${PUERHLAB_LIBRARY_HINTS})
    if(LIBRAW_INC AND LIBRAW_LIB)
        add_library(libraw::raw_r SHARED IMPORTED)
        set_target_properties(libraw::raw_r PROPERTIES
            IMPORTED_LOCATION "${LIBRAW_LIB}"
            INTERFACE_INCLUDE_DIRECTORIES "${LIBRAW_INC}"
        )
    else()
        message(FATAL_ERROR "LibRaw not found. Install via Homebrew (libraw) or set CMAKE_PREFIX_PATH to its prefix.")
    endif()
endif()

if(TARGET libraw::raw AND NOT TARGET libraw::raw_r)
    add_library(libraw::raw_r ALIAS libraw::raw)
elseif(TARGET libraw::raw_r AND NOT TARGET libraw::raw)
    add_library(libraw::raw ALIAS libraw::raw_r)
endif()

if(TARGET libraw::raw_r AND NOT TARGET LibRaw)
    add_library(LibRaw ALIAS libraw::raw_r)
endif()

# DuckDB API
if (WIN32)
    add_library(DuckDB SHARED IMPORTED)
    set_target_properties(DuckDB PROPERTIES
        IMPORTED_LOCATION "${CMAKE_SOURCE_DIR}/pu-erh_lab/third_party/libduckdb-windows/duckdb.dll"
        IMPORTED_IMPLIB "${CMAKE_SOURCE_DIR}/pu-erh_lab/third_party/libduckdb-windows/duckdb.lib"
        INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_SOURCE_DIR}/pu-erh_lab/third_party/libduckdb-windows")
else()        
  find_path(DUCKDB_INC NAMES duckdb.h PATHS ${PUERHLAB_INCLUDE_HINTS})
  find_library(DUCKDB_LIB NAMES duckdb PATHS ${PUERHLAB_LIBRARY_HINTS})
  if(DUCKDB_INC AND DUCKDB_LIB)
    add_library(DuckDB SHARED IMPORTED)
    set_target_properties(DuckDB PROPERTIES IMPORTED_LOCATION "${DUCKDB_LIB}" INTERFACE_INCLUDE_DIRECTORIES "${DUCKDB_INC}")
  else()
    message(FATAL_ERROR "DuckDB not found. Install via Homebrew (duckdb) or set CMAKE_PREFIX_PATH to its prefix.")
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

add_subdirectory(${CMAKE_SOURCE_DIR}/pu-erh_lab/src/third_party)
add_subdirectory(${CMAKE_SOURCE_DIR}/pu-erh_lab/src)
if(PUERHLAB_BUILD_TESTS)
    add_subdirectory(${CMAKE_SOURCE_DIR}/pu-erh_lab/tests)
endif()

set(PUERHLAB_MAIN_BIN puerhlab_main)
if(TARGET ${PUERHLAB_MAIN_BIN})
    install(TARGETS ${PUERHLAB_MAIN_BIN}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    )

    set(_config_dir "${CMAKE_SOURCE_DIR}/pu-erh_lab/src/config")
    install(DIRECTORY "${_config_dir}/fonts/" DESTINATION "${CMAKE_INSTALL_BINDIR}/fonts")
    install(DIRECTORY "${_config_dir}/LUTs/"  DESTINATION "${CMAKE_INSTALL_BINDIR}/LUTs")
    install(DIRECTORY "${_config_dir}/lens_calib/" DESTINATION "${CMAKE_INSTALL_BINDIR}/config/lens_calib")
    install(DIRECTORY "${_config_dir}/nikon_lens/" DESTINATION "${CMAKE_INSTALL_BINDIR}/config/nikon_lens")
endif()

if(WIN32 AND TARGET ${PUERHLAB_MAIN_BIN})
    set(CMAKE_INSTALL_SYSTEM_RUNTIME_DESTINATION "${CMAKE_INSTALL_BINDIR}")
    include(InstallRequiredSystemLibraries)

    set(PUERHLAB_RUNTIME_DEP_DIRS "" CACHE STRING "Extra directories to search for runtime DLL dependencies during install.")
    set(PUERHLAB_RUNTIME_EXES "$<TARGET_FILE:${PUERHLAB_MAIN_BIN}>")
    set(PUERHLAB_RUNTIME_SEARCH_DIRS "$<TARGET_FILE_DIR:${PUERHLAB_MAIN_BIN}>")

    set(_third_party_exiv2_bin "${CMAKE_SOURCE_DIR}/pu-erh_lab/third_party/exiv2_x64-windows/bin")
    set(_third_party_duckdb_bin "${CMAKE_SOURCE_DIR}/pu-erh_lab/third_party/libduckdb-windows")

    set(_vcpkg_bin_dir "")
    if(DEFINED VCPKG_INSTALLED_DIR AND DEFINED VCPKG_TARGET_TRIPLET)
        set(_vcpkg_bin_dir "${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/bin")
    endif()

    set(_qt6_bin_dir "")
    get_target_property(_qt6_core_loc Qt6::Core IMPORTED_LOCATION_RELEASE)
    if(NOT _qt6_core_loc)
        get_target_property(_qt6_core_loc Qt6::Core IMPORTED_LOCATION)
    endif()
    if(_qt6_core_loc)
        get_filename_component(_qt6_bin_dir "${_qt6_core_loc}" DIRECTORY)
    endif()

    set(_cuda_bin_dir "")
    if(PUERHLAB_CUDA_ENABLED AND CUDAToolkit_BIN_DIR)
        set(_cuda_bin_dir "${CUDAToolkit_BIN_DIR}")
    endif()

    install(CODE
        "message(STATUS \"[pu-erh_lab] Installing runtime DLL dependencies...\")
         set(_bindir \"\${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_BINDIR}\")
         file(MAKE_DIRECTORY \"\${_bindir}\")
         set(_search_dirs
             \"\${_bindir}\"
             ${PUERHLAB_RUNTIME_SEARCH_DIRS}
             \"${_third_party_exiv2_bin}\"
             \"${_third_party_duckdb_bin}\"
             \"${_vcpkg_bin_dir}\"
             \"${_qt6_bin_dir}\"
             \${PUERHLAB_RUNTIME_DEP_DIRS}
         )
         file(GET_RUNTIME_DEPENDENCIES
             EXECUTABLES ${PUERHLAB_RUNTIME_EXES}
             DIRECTORIES \${_search_dirs}
             RESOLVED_DEPENDENCIES_VAR _resolved
             UNRESOLVED_DEPENDENCIES_VAR _unresolved
             PRE_EXCLUDE_REGEXES
               \"api-ms-win-.*\"
               \"ext-ms-.*\"
             POST_EXCLUDE_REGEXES
               \".*[\\\\\\\\/][Ww][Ii][Nn][Dd][Oo][Ww][Ss][\\\\\\\\/].*\"
               \".*[\\\\\\\\/][Ss][Yy][Ss][Tt][Ee][Mm]32[\\\\\\\\/].*\"
               \".*[\\\\\\\\/][Ss][Yy][Ss][Ww][Oo][Ww]64[\\\\\\\\/].*\"
               \"azureattest.*\\\\.dll\"
               \"hvsifiletrust\\\\.dll\"
               \"pdmutilities\\\\.dll\"
               \"wpaxholder\\\\.dll\"
               \"opencv_cudaarithm.*\\\\.dll\"
               \"opencv_cudafilters.*\\\\.dll\"
               \"opencv_cudaimgproc.*\\\\.dll\"
               \"opencv_cudawarping.*\\\\.dll\"
               \"cublasLt64_.*\\\\.dll\"
         )
         list(REMOVE_DUPLICATES _resolved)
         foreach(_dep IN LISTS _resolved)
             file(INSTALL DESTINATION \"\${_bindir}\" TYPE SHARED_LIBRARY FILES \"\${_dep}\")
         endforeach()
         if(NOT \"${_cuda_bin_dir}\" STREQUAL \"\")
             file(GLOB _cuda_runtime_dlls \"${_cuda_bin_dir}/cudart64_*.dll\")
             foreach(_cuda_runtime_dll IN LISTS _cuda_runtime_dlls)
                 file(INSTALL DESTINATION \"\${_bindir}\" TYPE SHARED_LIBRARY FILES \"\${_cuda_runtime_dll}\")
             endforeach()
             if(NOT _cuda_runtime_dlls)
                 message(WARNING \"[pu-erh_lab] cudart64_*.dll not found in: ${_cuda_bin_dir}\")
             endif()
         endif()
         if(_unresolved)
             message(WARNING \"[pu-erh_lab] Unresolved runtime deps: \${_unresolved}\")
         endif()
        "
    )

    if(NOT COMMAND qt_generate_deploy_app_script)
        message(FATAL_ERROR "qt_generate_deploy_app_script is unavailable. Use a Qt 6 build with deployment tooling.")
    endif()

    option(PUERHLAB_DEPLOY_SOFTWARE_OPENGL "Bundle opengl32sw.dll for software OpenGL fallback." OFF)
    set(PUERHLAB_QT_DEPLOY_TOOL_OPTIONS
        --no-compiler-runtime
        --no-translations
        --no-system-d3d-compiler
        --no-system-dxc-compiler
        CACHE STRING
        "Semicolon-separated options forwarded to Qt deploy tool during install."
    )

    set(_deploy_tool_options ${PUERHLAB_QT_DEPLOY_TOOL_OPTIONS})
    if(NOT PUERHLAB_DEPLOY_SOFTWARE_OPENGL)
        list(APPEND _deploy_tool_options --no-opengl-sw)
    endif()
    list(APPEND _deploy_tool_options --qmldir "${CMAKE_SOURCE_DIR}/pu-erh_lab/src/ui/puerhlab_main/qml")

    qt_generate_deploy_app_script(
        TARGET ${PUERHLAB_MAIN_BIN}
        OUTPUT_SCRIPT _puerhlab_deploy_script
        NO_UNSUPPORTED_PLATFORM_ERROR
        DEPLOY_TOOL_OPTIONS ${_deploy_tool_options}
    )
    install(SCRIPT ${_puerhlab_deploy_script})
endif()



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
