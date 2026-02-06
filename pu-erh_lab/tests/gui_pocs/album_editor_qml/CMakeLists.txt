find_package(Qt6 REQUIRED COMPONENTS Quick Qml QuickControls2 QuickDialogs2 Widgets)

add_executable(AlbumEditorQmlDemo
    main.cpp
    AlbumBackend.cpp
    AlbumBackend.h
    EditorDialog.cpp
    EditorDialog.h
    FilterRuleModel.cpp
    FilterRuleModel.h
)

target_include_directories(AlbumEditorQmlDemo
    PRIVATE
        ${CMAKE_SOURCE_DIR}/pu-erh_lab/src/include
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(AlbumEditorQmlDemo
    PRIVATE
        Qt6::Quick
        Qt6::Qml
        Qt6::QuickControls2
        Qt6::QuickDialogs2
        Qt6::Widgets
        ProjectService
        ImportService
        ExportService
        PipelineMgmtService
        EditHistoryMgmtService
        ThumbnailService
        SleeveFilterService
        FSService
        EditPipeline
        EditViewer
)

if(WIN32)
    target_link_libraries(AlbumEditorQmlDemo PRIVATE easy_profiler)
endif()

qt_add_qml_module(AlbumEditorQmlDemo
    URI PuerhLab.AlbumEditorQml
    VERSION 1.0
    QML_FILES
        qml/Main.qml
)

target_compile_definitions(AlbumEditorQmlDemo PRIVATE PUERHLAB_SOURCE_DIR="${CMAKE_SOURCE_DIR}")

set(PUERHLAB_DEMO_FONT_SRC "${CMAKE_SOURCE_DIR}/pu-erh_lab/src/config/fonts/main_IBM.ttf")
add_custom_command(TARGET AlbumEditorQmlDemo POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_FILE_DIR:AlbumEditorQmlDemo>/fonts"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${PUERHLAB_DEMO_FONT_SRC}" "$<TARGET_FILE_DIR:AlbumEditorQmlDemo>/fonts/main_IBM.ttf"
    VERBATIM
)
