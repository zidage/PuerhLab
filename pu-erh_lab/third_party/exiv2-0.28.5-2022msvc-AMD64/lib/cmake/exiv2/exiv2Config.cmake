
####### Expanded from @PACKAGE_INIT@ by configure_package_config_file() #######
####### Any changes to this file will be overwritten by the next CMake run ####
####### The input file was exiv2Config.cmake.in                            ########

get_filename_component(PACKAGE_PREFIX_DIR "${CMAKE_CURRENT_LIST_DIR}/../../../" ABSOLUTE)

macro(set_and_check _var _file)
  set(${_var} "${_file}")
  if(NOT EXISTS "${_file}")
    message(FATAL_ERROR "File or directory ${_file} referenced by variable ${_var} does not exist !")
  endif()
endmacro()

macro(check_required_components _NAME)
  foreach(comp ${${_NAME}_FIND_COMPONENTS})
    if(NOT ${_NAME}_${comp}_FOUND)
      if(${_NAME}_FIND_REQUIRED_${comp})
        set(${_NAME}_FOUND FALSE)
      endif()
    endif()
  endforeach()
endmacro()

####################################################################################

cmake_minimum_required(VERSION 3.12)
include(CMakeFindDependencyMacro)

if(NOT TRUE) # if(NOT BUILD_SHARED_LIBS)
  if(TRUE) # if(EXIV2_ENABLE_PNG)
    find_dependency(ZLIB REQUIRED)
  endif()

  if(OFF) # if(EXIV2_ENABLE_WEBREADY)
    if(OFF) # if(EXIV2_ENABLE_CURL)
      find_dependency(CURL REQUIRED)
    endif()
  endif()

  if(ON) # if(EXIV2_ENABLE_XMP)
    find_dependency(EXPAT REQUIRED)
  endif()

  if(FALSE) # if(EXIV2_ENABLE_NLS)
    find_dependency(Intl REQUIRED)
  endif()

  if(FALSE) # if(EXV_HAVE_LIBICONV)
    find_dependency(Iconv REQUIRED)
  endif()
endif()

include("${CMAKE_CURRENT_LIST_DIR}/exiv2Targets.cmake")

check_required_components(exiv2)

# compatibility with non-aliased users
add_library(exiv2lib ALIAS Exiv2::exiv2lib)
