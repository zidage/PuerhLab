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
    set(PUERHLAB_LENSFUN_GLIB2_BASE_DIR
        "${CMAKE_SOURCE_DIR}/pu-erh_lab/third_party/glib-2.28.1"
        CACHE PATH
        "Base path of the GLib2 package used by the bundled Lensfun build on Windows.")
    if(EXISTS "${PUERHLAB_LENSFUN_GLIB2_BASE_DIR}")
        list(APPEND _puerhlab_lensfun_cmake_args
            "-DGLIB2_BASE_DIR:PATH=${PUERHLAB_LENSFUN_GLIB2_BASE_DIR}")
    else()
        message(STATUS
            "PUERHLAB_LENSFUN_GLIB2_BASE_DIR does not exist. "
            "Set it explicitly if the bundled Lensfun configure step cannot find GLib2 on Windows.")
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
