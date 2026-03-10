IF (NOT MSVC)
    FIND_PACKAGE(PkgConfig)
    PKG_SEARCH_MODULE(GLIB2 glib-2.0)
    IF(WIN32 AND NOT BUILD_STATIC)
        FIND_FILE(GLIB2_DLL
                NAMES glib-2.0-0.dll glib-2.dll glib-2-vs9.dll libglib-2.0-0.dll
                PATHS "${GLIB2_LIBRARY_DIRS}/../bin"
                NO_SYSTEM_ENVIRONMENT_PATH)
    ENDIF()
ENDIF()

IF (NOT GLIB2_FOUND OR NOT PKG_CONFIG_FOUND)
    SET(_GLIB2_GLIBCONFIG_PATHS
        /usr/local/lib
        /usr/lib
        /usr/lib64
        /opt/local/lib
        ${GLIB2_BASE_DIR}
        ${GLIB2_BASE_DIR}/lib
        ${GLIB2_BASE_DIR}/debug/lib
        ${GLIB2_BASE_DIR}/include
        ${CMAKE_PREFIX_PATH}
        ${CMAKE_LIBRARY_PATH}
    )

    SET(_GLIB2_INCLUDE_PATHS
        /usr/local/include
        /usr/include
        /opt/local/include
        ${GLIB2_BASE_DIR}
        ${GLIB2_BASE_DIR}/include
        ${CMAKE_PREFIX_PATH}
    )

    SET(_GLIB2_LIBRARY_PATHS
        /usr/local/lib
        /usr/lib
        /usr/lib64
        /opt/local/lib
        ${GLIB2_BASE_DIR}
        ${GLIB2_BASE_DIR}/lib
        ${GLIB2_BASE_DIR}/debug/lib
        ${CMAKE_PREFIX_PATH}
    )

    FIND_PATH(GLIB2_GLIB2CONFIG_INCLUDE_PATH
        NAMES glibconfig.h
        PATHS ${_GLIB2_GLIBCONFIG_PATHS}
        PATH_SUFFIXES
          glib-2.0/include
          lib/glib-2.0/include
          debug/lib/glib-2.0/include
          include/glib-2.0/include
    )


    FIND_PATH(GLIB2_INCLUDE_DIRS
        NAMES glib.h
        PATHS ${_GLIB2_INCLUDE_PATHS}
        PATH_SUFFIXES
            gtk-2.0
            glib-2.0
            glib20
            include/glib-2.0
    )

    FIND_LIBRARY(GLIB2_LIBRARIES
        NAMES  glib-2.0 glib20 glib
        PATHS ${_GLIB2_LIBRARY_PATHS}
        PATH_SUFFIXES
            lib
            debug/lib
    )

    IF(GLIB2_GLIB2CONFIG_INCLUDE_PATH AND GLIB2_INCLUDE_DIRS AND GLIB2_LIBRARIES)
        SET( GLIB2_INCLUDE_DIRS  ${GLIB2_GLIB2CONFIG_INCLUDE_PATH} ${GLIB2_INCLUDE_DIRS} )
        SET( GLIB2_LIBRARIES ${GLIB2_LIBRARIES} )
        SET( GLIB2_FOUND 1 )
    ELSE()
        SET( GLIB2_INCLUDE_DIRS )
        SET( GLIB2_LIBRARIES )
        SET( GLIB2_FOUND 0)
    ENDIF()

    IF(WIN32 AND NOT BUILD_STATIC)
        FIND_FILE(GLIB2_DLL
                NAMES glib-2.0-0.dll glib-2.dll glib-2-vs9.dll libglib-2.0-0.dll
                PATHS
                    ${GLIB2_BASE_DIR}
                    ${CMAKE_PREFIX_PATH}
                PATH_SUFFIXES
                    bin
                    debug/bin
                NO_SYSTEM_ENVIRONMENT_PATH)
    ENDIF()
ENDIF ()

#INCLUDE( FindPackageHandleStandardArgs )
#FIND_PACKAGE_HANDLE_STANDARD_ARGS( GLIB2 DEFAULT_MSG GLIB2_LIBRARIES GLIB2_GLIB2CONFIG_INCLUDE_PATH GLIB2_GLIB2_INCLUDE_PATH )

IF (NOT GLIB2_FOUND AND GLIB2_FIND_REQUIRED)
        MESSAGE(FATAL_ERROR "Could not find glib2")
ENDIF()
