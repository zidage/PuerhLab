set(PUERHLAB_LENSFUN_SOURCE_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}/lensfun"
    CACHE PATH
    "Path to the bundled Lensfun source checkout.")
set(PUERHLAB_LENSFUN_BINARY_DIR
    "${CMAKE_BINARY_DIR}/third_party/lensfun"
    CACHE PATH
    "Out-of-source build directory for the bundled Lensfun build.")
set(PUERHLAB_LENSFUN_INSTALL_DIR
    "${PUERHLAB_LENSFUN_BINARY_DIR}/install"
    CACHE PATH
    "Install prefix for the bundled Lensfun build.")

file(MAKE_DIRECTORY
    "${PUERHLAB_LENSFUN_INSTALL_DIR}/bin"
    "${PUERHLAB_LENSFUN_INSTALL_DIR}/include"
    "${PUERHLAB_LENSFUN_INSTALL_DIR}/lib")

if(NOT EXISTS "${PUERHLAB_LENSFUN_SOURCE_DIR}/CMakeLists.txt")
    message(FATAL_ERROR
        "Bundled Lensfun source was not found at ${PUERHLAB_LENSFUN_SOURCE_DIR}. "
        "Run `git submodule update --init --recursive pu-erh_lab/src/third_party/lensfun` and re-configure.")
endif()

set(_puerhlab_lensfun_cmake_args
    "-DCMAKE_INSTALL_PREFIX:PATH=${PUERHLAB_LENSFUN_INSTALL_DIR}"
    "-DCMAKE_INSTALL_DATAROOTDIR:PATH=share"
    "-DCMAKE_MODULE_PATH:STRING=${CMAKE_CURRENT_SOURCE_DIR}/cmake"
    "-DBUILD_STATIC:BOOL=OFF"
    "-DBUILD_TESTS:BOOL=OFF"
    "-DBUILD_LENSTOOL:BOOL=OFF"
    "-DBUILD_DOC:BOOL=OFF"
    "-DINSTALL_PYTHON_MODULE:BOOL=OFF"
    "-DINSTALL_HELPER_SCRIPTS:BOOL=OFF"
)

if(CMAKE_BUILD_TYPE)
    list(APPEND _puerhlab_lensfun_cmake_args
        "-DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}")
endif()

if(CMAKE_PREFIX_PATH)
    set(_puerhlab_lensfun_prefix_path "${CMAKE_PREFIX_PATH}")
    string(REPLACE ";" "\\;" _puerhlab_lensfun_prefix_path "${_puerhlab_lensfun_prefix_path}")
    list(APPEND _puerhlab_lensfun_cmake_args
        "-DCMAKE_PREFIX_PATH:STRING=${_puerhlab_lensfun_prefix_path}")
endif()

foreach(_tool_var IN ITEMS CMAKE_TOOLCHAIN_FILE CMAKE_C_COMPILER CMAKE_CXX_COMPILER CMAKE_MAKE_PROGRAM)
    if(DEFINED ${_tool_var} AND NOT "${${_tool_var}}" STREQUAL "")
        list(APPEND _puerhlab_lensfun_cmake_args
            "-D${_tool_var}:FILEPATH=${${_tool_var}}")
    endif()
endforeach()

if(DEFINED CMAKE_OSX_ARCHITECTURES AND NOT "${CMAKE_OSX_ARCHITECTURES}" STREQUAL "")
    set(_puerhlab_lensfun_osx_arches "${CMAKE_OSX_ARCHITECTURES}")
    string(REPLACE ";" "\\;" _puerhlab_lensfun_osx_arches "${_puerhlab_lensfun_osx_arches}")
    list(APPEND _puerhlab_lensfun_cmake_args
        "-DCMAKE_OSX_ARCHITECTURES:STRING=${_puerhlab_lensfun_osx_arches}")
endif()

if(DEFINED CMAKE_OSX_DEPLOYMENT_TARGET AND NOT "${CMAKE_OSX_DEPLOYMENT_TARGET}" STREQUAL "")
    list(APPEND _puerhlab_lensfun_cmake_args
        "-DCMAKE_OSX_DEPLOYMENT_TARGET:STRING=${CMAKE_OSX_DEPLOYMENT_TARGET}")
endif()

if(WIN32)
    set(_puerhlab_lensfun_legacy_glib2_base
        "${CMAKE_SOURCE_DIR}/pu-erh_lab/third_party/glib-2.28.1")
    set(_puerhlab_lensfun_default_glib2_base
        "${_puerhlab_lensfun_legacy_glib2_base}")
    if(DEFINED VCPKG_INSTALLED_DIR AND DEFINED VCPKG_TARGET_TRIPLET)
        set(_puerhlab_lensfun_vcpkg_glib2_base
            "${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}")
        if(EXISTS "${_puerhlab_lensfun_vcpkg_glib2_base}/include/glib-2.0/glib.h")
            set(_puerhlab_lensfun_default_glib2_base
                "${_puerhlab_lensfun_vcpkg_glib2_base}")
        endif()
    endif()

    set(_puerhlab_lensfun_configured_glib2_base "${PUERHLAB_LENSFUN_GLIB2_BASE_DIR}")
    if(NOT DEFINED PUERHLAB_LENSFUN_GLIB2_BASE_DIR
       OR "${_puerhlab_lensfun_configured_glib2_base}" STREQUAL ""
       OR ("${_puerhlab_lensfun_configured_glib2_base}" STREQUAL "${_puerhlab_lensfun_legacy_glib2_base}"
           AND NOT EXISTS "${_puerhlab_lensfun_configured_glib2_base}"))
        set(_puerhlab_lensfun_configured_glib2_base
            "${_puerhlab_lensfun_default_glib2_base}")
        set(_puerhlab_lensfun_force_glib2_base FORCE)
    else()
        set(_puerhlab_lensfun_force_glib2_base)
    endif()

    set(PUERHLAB_LENSFUN_GLIB2_BASE_DIR
        "${_puerhlab_lensfun_configured_glib2_base}"
        CACHE PATH
        "Base path of the GLib2 package used by the bundled Lensfun build on Windows."
        ${_puerhlab_lensfun_force_glib2_base})
    if(EXISTS "${PUERHLAB_LENSFUN_GLIB2_BASE_DIR}")
        list(APPEND _puerhlab_lensfun_cmake_args
            "-DGLIB2_BASE_DIR:PATH=${PUERHLAB_LENSFUN_GLIB2_BASE_DIR}")
    else()
        message(STATUS
            "PUERHLAB_LENSFUN_GLIB2_BASE_DIR does not exist. "
            "Install vcpkg package `glib` or set it explicitly if the bundled Lensfun configure step cannot find GLib2 on Windows.")
    endif()
endif()

if(WIN32)
    set(PUERHLAB_LENSFUN_IMPORTED_LOCATION "${PUERHLAB_LENSFUN_INSTALL_DIR}/bin/lensfun.dll")
    set(PUERHLAB_LENSFUN_IMPORTED_IMPLIB "${PUERHLAB_LENSFUN_INSTALL_DIR}/lib/lensfun.lib")
elseif(APPLE)
    set(PUERHLAB_LENSFUN_IMPORTED_LOCATION "${PUERHLAB_LENSFUN_INSTALL_DIR}/lib/liblensfun.dylib")
else()
    set(PUERHLAB_LENSFUN_IMPORTED_LOCATION "${PUERHLAB_LENSFUN_INSTALL_DIR}/lib/liblensfun.so")
endif()

set(_puerhlab_lensfun_byproducts
    "${PUERHLAB_LENSFUN_IMPORTED_LOCATION}"
    "${PUERHLAB_LENSFUN_INSTALL_DIR}/include/lensfun/lensfun.h"
)
if(WIN32)
    list(APPEND _puerhlab_lensfun_byproducts "${PUERHLAB_LENSFUN_IMPORTED_IMPLIB}")
endif()

set(_puerhlab_lensfun_args_manifest "${CMAKE_CURRENT_BINARY_DIR}/puerhlab_lensfun_args.txt")
string(REPLACE ";" "\n" _puerhlab_lensfun_args_manifest_contents "${_puerhlab_lensfun_cmake_args}")
file(GENERATE OUTPUT "${_puerhlab_lensfun_args_manifest}"
    CONTENT "${_puerhlab_lensfun_args_manifest_contents}\n")

set(_puerhlab_lensfun_configure_stamp "${PUERHLAB_LENSFUN_BINARY_DIR}/.puerhlab-lensfun-configured")

add_custom_command(
    OUTPUT "${_puerhlab_lensfun_configure_stamp}"
    COMMAND ${CMAKE_COMMAND} -E make_directory "${PUERHLAB_LENSFUN_BINARY_DIR}"
    COMMAND ${CMAKE_COMMAND}
        -S "${PUERHLAB_LENSFUN_SOURCE_DIR}"
        -B "${PUERHLAB_LENSFUN_BINARY_DIR}"
        -G "${CMAKE_GENERATOR}"
        ${_puerhlab_lensfun_cmake_args}
    COMMAND ${CMAKE_COMMAND} -E touch "${_puerhlab_lensfun_configure_stamp}"
    DEPENDS
        "${PUERHLAB_LENSFUN_SOURCE_DIR}/CMakeLists.txt"
        "${_puerhlab_lensfun_args_manifest}"
    VERBATIM
)

add_custom_command(
    OUTPUT ${_puerhlab_lensfun_byproducts}
    COMMAND ${CMAKE_COMMAND} --build "${PUERHLAB_LENSFUN_BINARY_DIR}"
    COMMAND ${CMAKE_COMMAND} --install "${PUERHLAB_LENSFUN_BINARY_DIR}" --prefix "${PUERHLAB_LENSFUN_INSTALL_DIR}"
    DEPENDS "${_puerhlab_lensfun_configure_stamp}"
    VERBATIM
)

add_custom_target(puerhlab_lensfun_build DEPENDS ${_puerhlab_lensfun_byproducts})

add_library(puerhlab_lensfun SHARED IMPORTED GLOBAL)
set_target_properties(puerhlab_lensfun PROPERTIES
    IMPORTED_LOCATION "${PUERHLAB_LENSFUN_IMPORTED_LOCATION}"
    INTERFACE_INCLUDE_DIRECTORIES "${PUERHLAB_LENSFUN_INSTALL_DIR}/include"
)
if(WIN32)
    set_target_properties(puerhlab_lensfun PROPERTIES
        IMPORTED_IMPLIB "${PUERHLAB_LENSFUN_IMPORTED_IMPLIB}")
endif()
add_dependencies(puerhlab_lensfun puerhlab_lensfun_build)

if(NOT TARGET lensfun::lensfun)
    add_library(lensfun::lensfun ALIAS puerhlab_lensfun)
endif()

set(PUERHLAB_LENSFUN_INCLUDE_DIR "${PUERHLAB_LENSFUN_INSTALL_DIR}/include" CACHE INTERNAL
    "Header directory for the bundled Lensfun build.")
set(PUERHLAB_LENSFUN_LIBRARY_DIR "${PUERHLAB_LENSFUN_INSTALL_DIR}/lib" CACHE INTERNAL
    "Runtime library directory for the bundled Lensfun build.")

set(PUERHLAB_ULTRAHDR_SOURCE_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}/libultrahdr"
    CACHE PATH
    "Path to the bundled libultrahdr source checkout.")

if(NOT EXISTS "${PUERHLAB_ULTRAHDR_SOURCE_DIR}/CMakeLists.txt")
    message(FATAL_ERROR
        "Bundled libultrahdr source was not found at ${PUERHLAB_ULTRAHDR_SOURCE_DIR}. "
        "Ensure pu-erh_lab/src/third_party/libultrahdr is present and re-configure.")
endif()

set(UHDR_BUILD_EXAMPLES OFF CACHE BOOL "Disable libultrahdr examples" FORCE)
set(UHDR_BUILD_TESTS OFF CACHE BOOL "Disable libultrahdr tests" FORCE)
set(UHDR_BUILD_BENCHMARK OFF CACHE BOOL "Disable libultrahdr benchmarks" FORCE)
set(UHDR_BUILD_FUZZERS OFF CACHE BOOL "Disable libultrahdr fuzzers" FORCE)
set(UHDR_BUILD_DEPS OFF CACHE BOOL "Use system libjpeg when building libultrahdr" FORCE)
set(UHDR_BUILD_JAVA OFF CACHE BOOL "Disable libultrahdr JNI bindings" FORCE)
set(UHDR_BUILD_PACKAGING OFF CACHE BOOL "Disable libultrahdr packaging" FORCE)
set(UHDR_ENABLE_INSTALL OFF CACHE BOOL "Disable libultrahdr install target" FORCE)
set(UHDR_ENABLE_GLES OFF CACHE BOOL "Disable libultrahdr GLES path" FORCE)
set(UHDR_ENABLE_WERROR OFF CACHE BOOL "Disable libultrahdr -Werror" FORCE)
set(UHDR_WRITE_XMP ON CACHE BOOL "Enable XMP gainmap metadata packets" FORCE)
set(UHDR_WRITE_ISO ON CACHE BOOL "Enable ISO 21496-1 gainmap metadata packets" FORCE)

add_subdirectory(
    "${PUERHLAB_ULTRAHDR_SOURCE_DIR}"
    "${CMAKE_BINARY_DIR}/third_party/libultrahdr"
)

if(TARGET uhdr)
    add_library(puerhlab_ultrahdr INTERFACE)
    target_link_libraries(puerhlab_ultrahdr INTERFACE uhdr)
elseif(TARGET uhdr-static)
    add_library(puerhlab_ultrahdr INTERFACE)
    target_link_libraries(puerhlab_ultrahdr INTERFACE uhdr-static)
else()
    message(FATAL_ERROR "Bundled libultrahdr was configured, but no usable target was exported.")
endif()

if(NOT TARGET puerhlab::uhdr)
    add_library(puerhlab::uhdr ALIAS puerhlab_ultrahdr)
endif()

set(PUERHLAB_METALCPP_SOURCE_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}/metal-cpp"
    CACHE PATH
    "Path to the bundled metal-cpp source checkout.")
set(PUERHLAB_METALCPP_HELPER_INCLUDE_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
    CACHE INTERNAL
    "Header directory for project-local metal-cpp wrapper headers.")

if(NOT EXISTS "${PUERHLAB_METALCPP_SOURCE_DIR}/Metal/Metal.hpp")
    message(FATAL_ERROR
        "Bundled metal-cpp source was not found at ${PUERHLAB_METALCPP_SOURCE_DIR}. "
        "Run `git submodule update --init --recursive pu-erh_lab/src/third_party/metal-cpp` and re-configure.")
endif()

add_library(puerhlab_metal_cpp INTERFACE)
target_include_directories(puerhlab_metal_cpp
    INTERFACE
        "${PUERHLAB_METALCPP_HELPER_INCLUDE_DIR}"
)
target_include_directories(puerhlab_metal_cpp
    SYSTEM INTERFACE
        "${PUERHLAB_METALCPP_SOURCE_DIR}"
)
target_compile_features(puerhlab_metal_cpp INTERFACE cxx_std_20)

if(APPLE)
    find_library(PUERHLAB_FOUNDATION_FRAMEWORK Foundation REQUIRED)
    find_library(PUERHLAB_QUARTZCORE_FRAMEWORK QuartzCore REQUIRED)
    find_library(PUERHLAB_METAL_FRAMEWORK Metal REQUIRED)

    add_library(puerhlab_metal_cpp_runtime STATIC
        "${CMAKE_CURRENT_SOURCE_DIR}/puerhlab_metal_cpp_implementation.cpp"
    )
    set_target_properties(puerhlab_metal_cpp_runtime PROPERTIES
        AUTOMOC OFF
        AUTOUIC OFF
        AUTORCC OFF
    )
    target_link_libraries(puerhlab_metal_cpp_runtime
        PUBLIC
            "${PUERHLAB_FOUNDATION_FRAMEWORK}"
            "${PUERHLAB_QUARTZCORE_FRAMEWORK}"
            "${PUERHLAB_METAL_FRAMEWORK}"
    )
    target_include_directories(puerhlab_metal_cpp_runtime
        PRIVATE
            "${PUERHLAB_METALCPP_HELPER_INCLUDE_DIR}"
    )
    target_include_directories(puerhlab_metal_cpp_runtime
        SYSTEM PRIVATE
            "${PUERHLAB_METALCPP_SOURCE_DIR}"
    )

    target_link_libraries(puerhlab_metal_cpp
        INTERFACE
            puerhlab_metal_cpp_runtime
            "${PUERHLAB_FOUNDATION_FRAMEWORK}"
            "${PUERHLAB_QUARTZCORE_FRAMEWORK}"
            "${PUERHLAB_METAL_FRAMEWORK}"
    )
endif()

if(NOT TARGET puerhlab::metal_cpp)
    add_library(puerhlab::metal_cpp ALIAS puerhlab_metal_cpp)
endif()

set(PUERHLAB_METALCPP_INCLUDE_DIR "${PUERHLAB_METALCPP_SOURCE_DIR}" CACHE INTERNAL
    "Header directory for the bundled metal-cpp checkout.")
