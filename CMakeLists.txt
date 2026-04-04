cmake_minimum_required(VERSION 3.19)


#set(CMAKE_C_COMPILER "clang")
#set(CMAKE_CXX_COMPILER "clang++")
# Compiler flags.

project(pu-erh_lab VERSION 0.2.2 LANGUAGES C CXX)

set(BUILD_SHARED_LIBS OFF CACHE BOOL "Build pu-erh_lab libraries as static by default." FORCE)

if(APPLE)
    enable_language(OBJCXX)
endif()

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
option(PUERHLAB_MACOS_BUNDLE
       "Build puerhlab_main as a macOS .app bundle (intended for packaging/install)."
       OFF)
option(PUERHLAB_MACDEPLOYQT_STRIP
       "Allow macdeployqt to strip binaries during macOS app deployment."
       OFF)
set(PUERHLAB_MACOS_CODESIGN_IDENTITY "-"
    CACHE STRING
    "Code-sign identity for macOS bundle after deployment (use '-' for ad-hoc signing, empty to skip).")
set(PUERHLAB_QT_PREFIX ""
    CACHE PATH
    "Optional Qt installation prefix for macOS packaging builds (contains bin/, lib/cmake/Qt6, plugins/, and qml/).")

function(puerhlab_append_existing_prefix prefix)
    if(EXISTS "${prefix}")
        list(APPEND CMAKE_PREFIX_PATH "${prefix}")
        set(CMAKE_PREFIX_PATH "${CMAKE_PREFIX_PATH}" PARENT_SCOPE)
    endif()
endfunction()

set(_puerhlab_qt_prefix_hint "")
if(PUERHLAB_QT_PREFIX)
    file(TO_CMAKE_PATH "${PUERHLAB_QT_PREFIX}" _puerhlab_qt_prefix_hint)
    if(EXISTS "${_puerhlab_qt_prefix_hint}")
        list(PREPEND CMAKE_PREFIX_PATH "${_puerhlab_qt_prefix_hint}")
    else()
        message(WARNING "PUERHLAB_QT_PREFIX does not exist: ${_puerhlab_qt_prefix_hint}")
    endif()
endif()

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
puerhlab_append_existing_prefix("/opt/homebrew/opt/openblas")
puerhlab_append_existing_prefix("/opt/homebrew/opt/gcc")

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
puerhlab_append_existing_prefix("/usr/local/opt/openblas")
puerhlab_append_existing_prefix("/usr/local/opt/gcc")

set(PUERHLAB_LIBRARY_HINTS "")
set(PUERHLAB_INCLUDE_HINTS "")

foreach(_prefix IN LISTS CMAKE_PREFIX_PATH)
    if(EXISTS "${_prefix}")
        if(EXISTS "${_prefix}/lib")
            list(APPEND PUERHLAB_LIBRARY_HINTS "${_prefix}/lib")
        endif()
        if(EXISTS "${_prefix}/include")
            list(APPEND PUERHLAB_INCLUDE_HINTS "${_prefix}/include")
        endif()
    endif()
endforeach()

# Extra GCC runtime directories for libgfortran/libquadmath/libgcc_s.
# Homebrew gcc may place runtime dylibs in versioned subdirectories.
foreach(_gcc_prefix IN ITEMS "/opt/homebrew/opt/gcc" "/usr/local/opt/gcc")
    if(EXISTS "${_gcc_prefix}")
        if(EXISTS "${_gcc_prefix}/lib")
            list(APPEND PUERHLAB_LIBRARY_HINTS "${_gcc_prefix}/lib")
        endif()

        file(GLOB _puerhlab_gcc_runtime_dirs
            LIST_DIRECTORIES true
            "${_gcc_prefix}/lib/gcc/current"
            "${_gcc_prefix}/lib/gcc/*"
        )
        foreach(_dir IN LISTS _puerhlab_gcc_runtime_dirs)
            if(IS_DIRECTORY "${_dir}")
                list(APPEND PUERHLAB_LIBRARY_HINTS "${_dir}")
            endif()
        endforeach()
        unset(_puerhlab_gcc_runtime_dirs)
    endif()
endforeach()

list(REMOVE_DUPLICATES PUERHLAB_LIBRARY_HINTS)
list(REMOVE_DUPLICATES PUERHLAB_INCLUDE_HINTS)



if (MSVC)
    set(CMAKE_CXX_FLAGS_DEBUG "/Zi /arch:AVX /bigobj /W4 /external:anglebrackets /external:W0 /D_DEBUG /EHsc /openmp /Zc:preprocessor")
    set(CMAKE_CXX_FLAGS_RELEASE "/O2 /arch:AVX2 /bigobj /W4 /external:anglebrackets /external:W0 /EHsc /openmp /GL /Zc:preprocessor /MP")
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
        $<$<CONFIG:Debug>:-fno-omit-frame-pointer>
        $<$<CONFIG:Release>:-O3>
        $<$<CONFIG:Release>:-DNDEBUG>
    )

    if(APPLE)
        add_compile_definitions(__MACOSX_CORE__)
    endif()
endif()

# Qt6
set(PUERHLAB_QT_COMPONENTS
    Core
    LinguistTools
    Svg
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

if((WIN32 AND PUERHLAB_CUDA_ENABLED) OR (APPLE AND PUERHLAB_METAL_ENABLED))
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
    if(APPLE AND PUERHLAB_MACOS_BUNDLE)
        # Match the app bundle name to the QML module URI (PuerhLab.Main)
        # so that Qt's QML deploy puts files in the correct bundle.
        set_target_properties(${PUERHLAB_MAIN_BIN} PROPERTIES OUTPUT_NAME PuerhLab)
        set_property(TARGET ${PUERHLAB_MAIN_BIN} APPEND PROPERTY INSTALL_RPATH "@executable_path/../Frameworks")

        # macOS: install the .app bundle into the prefix root so that
        # CPack picks up  PuerhLab.app/  directly.
        install(TARGETS ${PUERHLAB_MAIN_BIN}
            BUNDLE  DESTINATION .
            RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        )

        # Copy asset folders into Contents/MacOS/ (next to the executable)
        set(_config_dir "${CMAKE_SOURCE_DIR}/pu-erh_lab/src/config")
        set(_bundle_macos_dest "PuerhLab.app/Contents/MacOS")
        set(_bundle_resources_dest "PuerhLab.app/Contents/Resources")
        set(_bundle_frameworks_dest "PuerhLab.app/Contents/Frameworks")
        install(DIRECTORY "${_config_dir}/fonts/" DESTINATION "${_bundle_macos_dest}/fonts")
        install(DIRECTORY "${_config_dir}/LUTs/"  DESTINATION "${_bundle_macos_dest}/LUTs")
        install(DIRECTORY "${_config_dir}/icc/" DESTINATION "${_bundle_macos_dest}/config/icc")
        install(DIRECTORY "${_config_dir}/lens_calib/" DESTINATION "${_bundle_macos_dest}/config/lens_calib")
        install(DIRECTORY "${_config_dir}/nikon_lens/" DESTINATION "${_bundle_macos_dest}/config/nikon_lens")
        if(PUERHLAB_METAL_ENABLED AND PUERHLAB_METAL_RUNTIME_LIBS)
            install(FILES ${PUERHLAB_METAL_RUNTIME_LIBS}
                DESTINATION "${_bundle_resources_dest}/metallib"
            )
        endif()
        if(DEFINED PUERHLAB_LENSFUN_LIBRARY_DIR AND NOT "${PUERHLAB_LENSFUN_LIBRARY_DIR}" STREQUAL "")
            set(_lensfun_soname_path "${PUERHLAB_LENSFUN_LIBRARY_DIR}/liblensfun.2.dylib")
            file(REAL_PATH "${_lensfun_soname_path}" _lensfun_real_path)
            if(EXISTS "${_lensfun_real_path}")
                install(FILES "${_lensfun_real_path}"
                    DESTINATION "${_bundle_frameworks_dest}"
                    OPTIONAL
                )
                get_filename_component(_lensfun_real_name "${_lensfun_real_path}" NAME)
                if(NOT "${_lensfun_real_name}" STREQUAL "liblensfun.2.dylib")
                    install(CODE
                        "set(_frameworks_dir \"\${CMAKE_INSTALL_PREFIX}/${_bundle_frameworks_dest}\")
                         file(MAKE_DIRECTORY \"\${_frameworks_dir}\")
                         file(REMOVE \"\${_frameworks_dir}/liblensfun.2.dylib\")
                         execute_process(
                             COMMAND \"${CMAKE_COMMAND}\" -E create_symlink \"${_lensfun_real_name}\" \"\${_frameworks_dir}/liblensfun.2.dylib\"
                             RESULT_VARIABLE _puerhlab_lensfun_symlink_result
                         )
                         if(NOT _puerhlab_lensfun_symlink_result EQUAL 0)
                             message(FATAL_ERROR \"Failed to create lensfun soname symlink in \${_frameworks_dir}\")
                         endif()
                        "
                    )
                endif()
            else()
                install(FILES "${_lensfun_soname_path}"
                    DESTINATION "${_bundle_frameworks_dest}"
                    OPTIONAL
                )
            endif()
        endif()

        # Ensure JPEG XL runtime deps are present for OpenImageIO/Homebrew builds.
        set(_jxl_cms_soname_path "")
        foreach(_lib_dir IN LISTS PUERHLAB_LIBRARY_HINTS)
            if(EXISTS "${_lib_dir}/libjxl_cms.0.11.dylib")
                set(_jxl_cms_soname_path "${_lib_dir}/libjxl_cms.0.11.dylib")
                break()
            endif()
        endforeach()
        if(_jxl_cms_soname_path)
            file(REAL_PATH "${_jxl_cms_soname_path}" _jxl_cms_real_path)
            if(EXISTS "${_jxl_cms_real_path}")
                install(FILES "${_jxl_cms_real_path}"
                    DESTINATION "${_bundle_frameworks_dest}"
                    OPTIONAL
                )
                get_filename_component(_jxl_cms_real_name "${_jxl_cms_real_path}" NAME)
                if(NOT "${_jxl_cms_real_name}" STREQUAL "libjxl_cms.0.11.dylib")
                    install(CODE
                        "set(_frameworks_dir \"\${CMAKE_INSTALL_PREFIX}/${_bundle_frameworks_dest}\")
                         file(MAKE_DIRECTORY \"\${_frameworks_dir}\")
                         file(REMOVE \"\${_frameworks_dir}/libjxl_cms.0.11.dylib\")
                         execute_process(
                             COMMAND \"${CMAKE_COMMAND}\" -E create_symlink \"${_jxl_cms_real_name}\" \"\${_frameworks_dir}/libjxl_cms.0.11.dylib\"
                             RESULT_VARIABLE _puerhlab_jxl_cms_symlink_result
                         )
                         if(NOT _puerhlab_jxl_cms_symlink_result EQUAL 0)
                             message(FATAL_ERROR \"Failed to create libjxl_cms soname symlink in \${_frameworks_dir}\")
                         endif()
                        "
                    )
                endif()
            else()
                install(FILES "${_jxl_cms_soname_path}"
                    DESTINATION "${_bundle_frameworks_dest}"
                    OPTIONAL
                )
            endif()
        else()
            message(WARNING "libjxl_cms.0.11.dylib was not found in PUERHLAB_LIBRARY_HINTS; app may fail to launch on systems without JPEG XL runtime installed.")
        endif()
    else()
        install(TARGETS ${PUERHLAB_MAIN_BIN}
            RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        )

        set(_config_dir "${CMAKE_SOURCE_DIR}/pu-erh_lab/src/config")
        install(DIRECTORY "${_config_dir}/fonts/" DESTINATION "${CMAKE_INSTALL_BINDIR}/fonts")
        install(DIRECTORY "${_config_dir}/LUTs/"  DESTINATION "${CMAKE_INSTALL_BINDIR}/LUTs")
        install(DIRECTORY "${_config_dir}/icc/" DESTINATION "${CMAKE_INSTALL_BINDIR}/config/icc")
        install(DIRECTORY "${_config_dir}/lens_calib/" DESTINATION "${CMAKE_INSTALL_BINDIR}/config/lens_calib")
        install(DIRECTORY "${_config_dir}/nikon_lens/" DESTINATION "${CMAKE_INSTALL_BINDIR}/config/nikon_lens")
        if(APPLE AND PUERHLAB_METAL_ENABLED AND PUERHLAB_METAL_RUNTIME_LIBS)
            install(FILES ${PUERHLAB_METAL_RUNTIME_LIBS}
                DESTINATION "${CMAKE_INSTALL_BINDIR}/metallib"
            )
        endif()
    endif()
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

    if(NOT COMMAND qt_generate_deploy_qml_app_script)
        message(FATAL_ERROR "qt_generate_deploy_qml_app_script is unavailable. Use Qt 6.3+ with deployment tooling.")
    endif()

    option(PUERHLAB_DEPLOY_SOFTWARE_OPENGL "Bundle opengl32sw.dll for software OpenGL fallback." OFF)

    set(_puerhlab_qt_deploy_tool_options_default
        --no-compiler-runtime
        --no-system-d3d-compiler
        --no-system-dxc-compiler
    )
    set(_puerhlab_qt_deploy_tool_options_legacy_default
        --no-compiler-runtime
        --no-translations
        --no-system-d3d-compiler
        --no-system-dxc-compiler
    )

    if(NOT DEFINED PUERHLAB_QT_DEPLOY_TOOL_OPTIONS OR
       "${PUERHLAB_QT_DEPLOY_TOOL_OPTIONS}" STREQUAL "${_puerhlab_qt_deploy_tool_options_legacy_default}")
        set(PUERHLAB_QT_DEPLOY_TOOL_OPTIONS
            ${_puerhlab_qt_deploy_tool_options_default}
            CACHE STRING
            "Semicolon-separated options forwarded to Qt deploy tool during install."
            FORCE
        )
    else()
        set(PUERHLAB_QT_DEPLOY_TOOL_OPTIONS
            "${PUERHLAB_QT_DEPLOY_TOOL_OPTIONS}"
            CACHE STRING
            "Semicolon-separated options forwarded to Qt deploy tool during install."
        )
    endif()

    set(_deploy_tool_options ${PUERHLAB_QT_DEPLOY_TOOL_OPTIONS})
    if(NOT PUERHLAB_DEPLOY_SOFTWARE_OPENGL)
        list(APPEND _deploy_tool_options --no-opengl-sw)
    endif()

    qt_generate_deploy_qml_app_script(
        TARGET ${PUERHLAB_MAIN_BIN}
        OUTPUT_SCRIPT _puerhlab_deploy_script
        NO_UNSUPPORTED_PLATFORM_ERROR
        DEPLOY_TOOL_OPTIONS ${_deploy_tool_options}
    )
    install(SCRIPT ${_puerhlab_deploy_script})
endif()

# ── macOS deployment ────────────────────────────────────────────────────────
# Bundles external dylibs + Qt frameworks/plugins/QML modules into
# PuerhLab.app so the packaged ZIP/DMG is self-contained.
#
# Build & package workflow:
#   cmake --preset macos_release
#   cmake --build --preset macos_release
#   cmake --install build/macos-release          # populates build/install/
#   cd build/macos-release && cpack              # creates the DMG + ZIP
#
# End-user can run the app from the terminal after mounting/extracting:
#   ./PuerhLab.app/Contents/MacOS/PuerhLab
# ────────────────────────────────────────────────────────────────────────────
if(APPLE AND PUERHLAB_MACOS_BUNDLE AND TARGET ${PUERHLAB_MAIN_BIN})

    # Use Qt's QML deploy API (macdeployqt underneath) and pass Homebrew/3rd-party
    # library search roots via -libpath.
    if(NOT COMMAND qt_generate_deploy_qml_app_script)
        message(FATAL_ERROR
            "qt_generate_deploy_qml_app_script is unavailable. "
            "Use Qt 6.3+ with deployment tooling.")
    endif()

    set(_puerhlab_macos_deploy_tool_options "")
    if(NOT PUERHLAB_MACDEPLOYQT_STRIP)
        list(APPEND _puerhlab_macos_deploy_tool_options -no-strip)
    endif()

    set(_puerhlab_macos_libpaths "")
    foreach(_lib_dir IN LISTS PUERHLAB_LIBRARY_HINTS)
        if(EXISTS "${_lib_dir}")
            list(APPEND _puerhlab_macos_libpaths "-libpath=${_lib_dir}")
        endif()
    endforeach()

    if(DEFINED PUERHLAB_LENSFUN_LIBRARY_DIR AND NOT "${PUERHLAB_LENSFUN_LIBRARY_DIR}" STREQUAL "")
        list(APPEND _puerhlab_macos_libpaths "-libpath=${PUERHLAB_LENSFUN_LIBRARY_DIR}")
    endif()

    list(REMOVE_DUPLICATES _puerhlab_macos_libpaths)
    list(APPEND _puerhlab_macos_deploy_tool_options ${_puerhlab_macos_libpaths})

    set(_puerhlab_macos_deploy_args NO_UNSUPPORTED_PLATFORM_ERROR)
    if(_puerhlab_macos_deploy_tool_options)
        list(APPEND _puerhlab_macos_deploy_args
            DEPLOY_TOOL_OPTIONS
            ${_puerhlab_macos_deploy_tool_options}
        )
    endif()

    qt_generate_deploy_qml_app_script(
        TARGET ${PUERHLAB_MAIN_BIN}
        OUTPUT_SCRIPT _puerhlab_macos_deploy_script
        ${_puerhlab_macos_deploy_args}
    )
    install(SCRIPT ${_puerhlab_macos_deploy_script})

    # Build a search-dir list for manual GCC/OpenBLAS runtime fixups.
    set(_puerhlab_runtime_dirs ${PUERHLAB_LIBRARY_HINTS})
    if(DEFINED PUERHLAB_LENSFUN_LIBRARY_DIR AND NOT "${PUERHLAB_LENSFUN_LIBRARY_DIR}" STREQUAL "")
        list(APPEND _puerhlab_runtime_dirs "${PUERHLAB_LENSFUN_LIBRARY_DIR}")
    endif()
    list(REMOVE_DUPLICATES _puerhlab_runtime_dirs)
    list(JOIN _puerhlab_runtime_dirs ";" _puerhlab_runtime_dirs_joined)

    # Qt/macdeployqt already does the main deployment pass.
    # Here we only patch the OpenBLAS -> libgfortran -> libquadmath/libgcc_s chain.
    install(CODE
        "
        set(_puerhlab_app_path \"\${CMAKE_INSTALL_PREFIX}/PuerhLab.app\")
        set(_puerhlab_fw_dir   \"\${_puerhlab_app_path}/Contents/Frameworks\")
        set(_puerhlab_search_dirs \"${_puerhlab_runtime_dirs_joined}\")

        if(NOT EXISTS \"\${_puerhlab_app_path}\")
            message(FATAL_ERROR \"macOS bundle not found: \${_puerhlab_app_path}\")
        endif()

        file(MAKE_DIRECTORY \"\${_puerhlab_fw_dir}\")

        function(_puerhlab_find_lib _out _name)
            set(_found \"\")
            foreach(_dir IN LISTS _puerhlab_search_dirs)
                if(EXISTS \"\${_dir}/\${_name}\")
                    set(_found \"\${_dir}/\${_name}\")
                    break()
                endif()
            endforeach()
            set(\${_out} \"\${_found}\" PARENT_SCOPE)
        endfunction()

        function(_puerhlab_copy_lib_if_missing _name)
            set(_dst \"\${_puerhlab_fw_dir}/\${_name}\")
            if(NOT EXISTS \"\${_dst}\")
                _puerhlab_find_lib(_src \"\${_name}\")
                if(\"\${_src}\" STREQUAL \"\")
                    message(FATAL_ERROR \"Required runtime library not found: \${_name}\")
                endif()

                execute_process(
                    COMMAND /bin/cp -fL \"\${_src}\" \"\${_dst}\"
                    RESULT_VARIABLE _copy_result
                )
                if(NOT _copy_result EQUAL 0)
                    message(FATAL_ERROR \"Failed to copy \${_src} -> \${_dst}\")
                endif()
            endif()

            execute_process(COMMAND /bin/chmod u+w \"\${_dst}\" ERROR_QUIET)
        endfunction()

        # Ensure the GCC runtime chain exists in the bundle.
        _puerhlab_copy_lib_if_missing(\"libgfortran.5.dylib\")
        _puerhlab_copy_lib_if_missing(\"libquadmath.0.dylib\")
        _puerhlab_copy_lib_if_missing(\"libgcc_s.1.1.dylib\")

        # Patch libgfortran to use sibling dylibs in Frameworks.
        execute_process(
            COMMAND /usr/bin/install_name_tool
                    -change @rpath/libquadmath.0.dylib @loader_path/libquadmath.0.dylib
                    \"\${_puerhlab_fw_dir}/libgfortran.5.dylib\"
            RESULT_VARIABLE _ignore_quadmath
            ERROR_QUIET
        )
        execute_process(
            COMMAND /usr/bin/install_name_tool
                    -change @rpath/libgcc_s.1.1.dylib @loader_path/libgcc_s.1.1.dylib
                    \"\${_puerhlab_fw_dir}/libgfortran.5.dylib\"
            RESULT_VARIABLE _ignore_gcc
            ERROR_QUIET
        )

        # Make libopenblas prefer the bundled libgfortran when applicable.
        if(EXISTS \"\${_puerhlab_fw_dir}/libopenblas.0.dylib\")
            execute_process(COMMAND /bin/chmod u+w \"\${_puerhlab_fw_dir}/libopenblas.0.dylib\" ERROR_QUIET)

            execute_process(
                COMMAND /usr/bin/install_name_tool
                        -change @rpath/libgfortran.5.dylib @loader_path/libgfortran.5.dylib
                        \"\${_puerhlab_fw_dir}/libopenblas.0.dylib\"
                RESULT_VARIABLE _ignore_openblas_rpath
                ERROR_QUIET
            )
            execute_process(
                COMMAND /usr/bin/install_name_tool
                        -change @executable_path/../Frameworks/libgfortran.5.dylib @loader_path/libgfortran.5.dylib
                        \"\${_puerhlab_fw_dir}/libopenblas.0.dylib\"
                RESULT_VARIABLE _ignore_openblas_exec
                ERROR_QUIET
            )
        endif()

        # Optional sanity check for the patched libgfortran.
        execute_process(
            COMMAND /usr/bin/otool -L \"\${_puerhlab_fw_dir}/libgfortran.5.dylib\"
            RESULT_VARIABLE _gfortran_otool_result
            OUTPUT_VARIABLE _gfortran_otool_output
            ERROR_VARIABLE  _gfortran_otool_error
        )
        if(NOT _gfortran_otool_result EQUAL 0)
            message(FATAL_ERROR
                \"otool failed for libgfortran.5.dylib: \${_gfortran_otool_error}\")
        endif()

        message(STATUS \"Patched GCC runtime chain for macOS bundle\")
        "
    )

    # Sign only after all bundle mutations are complete.
    if(NOT "${PUERHLAB_MACOS_CODESIGN_IDENTITY}" STREQUAL "")
        install(CODE
            "set(_puerhlab_app_path \"\${CMAKE_INSTALL_PREFIX}/PuerhLab.app\")
             if(EXISTS \"\${_puerhlab_app_path}\")
                 if(EXISTS \"/usr/bin/codesign\")
                     execute_process(
                         COMMAND /usr/bin/codesign
                                 --force
                                 --deep
                                 --sign \"${PUERHLAB_MACOS_CODESIGN_IDENTITY}\"
                                 --timestamp=none
                                 \"\${_puerhlab_app_path}\"
                         RESULT_VARIABLE _puerhlab_codesign_result
                     )
                     if(NOT _puerhlab_codesign_result EQUAL 0)
                         message(FATAL_ERROR
                             \"codesign failed for \${_puerhlab_app_path} (exit=\${_puerhlab_codesign_result})\")
                     endif()
                 else()
                     message(WARNING \"/usr/bin/codesign not found; skipping macOS bundle re-sign\")
                 endif()
             else()
                 message(WARNING \"macOS bundle not found for re-sign: \${_puerhlab_app_path}\")
             endif()
            "
        )
    endif()
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

set(CPACK_PACKAGE_NAME "pu-erh_lab")
set(CPACK_PACKAGE_VERSION "${PROJECT_VERSION}")
set(CPACK_PACKAGE_FILE_NAME "${CPACK_PACKAGE_NAME}-${CPACK_PACKAGE_VERSION}-${CMAKE_SYSTEM_NAME}-${CMAKE_SYSTEM_PROCESSOR}")
if(APPLE AND PUERHLAB_MACOS_BUNDLE)
    set(CPACK_GENERATOR "DragNDrop;ZIP")
    set(CPACK_DMG_VOLUME_NAME "PuerhLab")
    # Include the .app bundle at the top level of the ZIP so the user
    # extracts it and finds PuerhLab.app/ directly.
    set(CPACK_INCLUDE_TOPLEVEL_DIRECTORY OFF)
else()
    set(CPACK_GENERATOR "ZIP")
endif()
include(CPack)
