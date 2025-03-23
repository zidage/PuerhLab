#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "Exiv2::exiv2lib" for configuration "Release"
set_property(TARGET Exiv2::exiv2lib APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(Exiv2::exiv2lib PROPERTIES
  IMPORTED_IMPLIB_RELEASE "${_IMPORT_PREFIX}/lib/exiv2.lib"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/exiv2.dll"
  )

list(APPEND _cmake_import_check_targets Exiv2::exiv2lib )
list(APPEND _cmake_import_check_files_for_Exiv2::exiv2lib "${_IMPORT_PREFIX}/lib/exiv2.lib" "${_IMPORT_PREFIX}/bin/exiv2.dll" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
