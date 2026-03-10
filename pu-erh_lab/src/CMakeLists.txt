# ==============================================================================
# Helper Macro to reduce boilerplate
# ==============================================================================
# Usage: def_library(Name SOURCES src1 src2 PUBLIC_DEPS lib1 PRIVATE_DEPS lib2)
macro(def_library target_name)
    cmake_parse_arguments(ARG "" "" "SOURCES;PUBLIC_DEPS;PRIVATE_DEPS" ${ARGN})
    
    add_library(${target_name} ${ARG_SOURCES})
    target_include_directories(${target_name} PUBLIC include)
    
    if(ARG_PUBLIC_DEPS)
        target_link_libraries(${target_name} PUBLIC ${ARG_PUBLIC_DEPS})
    endif()
    
    if(ARG_PRIVATE_DEPS)
        target_link_libraries(${target_name} PRIVATE ${ARG_PRIVATE_DEPS})
    endif()
endmacro()

# ==============================================================================
# 1. Utilities
# ==============================================================================

set(PUERHLAB_OPENCV_CORE_DEPS
    opencv_core
)
set(PUERHLAB_OPENCV_IMAGE_DEPS
    opencv_core
    opencv_imgproc
    opencv_imgcodecs
    opencv_calib3d
    opencv_highgui
)

def_library(ThreadPool SOURCES concurrency/thread_pool.cpp)

def_library(TimeProvider SOURCES utils/clock/time_provider.cpp)

def_library(StrConv 
    SOURCES utils/string/converter.cpp 
    PUBLIC_DEPS utfcpp
)

if(PUERHLAB_METAL_ENABLED)
    set(METAL_UTILS_SRC "${CMAKE_CURRENT_SOURCE_DIR}/metal/metal_utils/metal_convert.metal")
    set(METAL_UTILS_AIR "${CMAKE_CURRENT_BINARY_DIR}/metal/metal_utils/metal_convert.air")
    set(METAL_UTILS_METALLIB "${CMAKE_CURRENT_BINARY_DIR}/metal/metal_utils/metal_convert.metallib")
    set(METAL_GEOMETRY_UTILS_SRC "${CMAKE_CURRENT_SOURCE_DIR}/metal/metal_utils/geometry_utils.metal")
    set(METAL_GEOMETRY_UTILS_AIR "${CMAKE_CURRENT_BINARY_DIR}/metal/metal_utils/geometry_utils.air")
    set(METAL_GEOMETRY_UTILS_METALLIB "${CMAKE_CURRENT_BINARY_DIR}/metal/metal_utils/geometry_utils.metallib")
    set(METAL_LENS_CALIB_SRC "${CMAKE_CURRENT_SOURCE_DIR}/edit/operators/geometry/metal_shader/lens_calib.metal")
    set(METAL_LENS_CALIB_AIR "${CMAKE_CURRENT_BINARY_DIR}/edit/operators/geometry/metal_shader/lens_calib.air")
    set(METAL_LENS_CALIB_METALLIB "${CMAKE_CURRENT_BINARY_DIR}/edit/operators/geometry/metal_shader/lens_calib.metallib")
    set(METAL_TO_LINEAR_REF_SRC "${CMAKE_CURRENT_SOURCE_DIR}/decoders/processor/operators/gpu/metal_shader/to_linear_ref.metal")
    set(METAL_TO_LINEAR_REF_AIR "${CMAKE_CURRENT_BINARY_DIR}/decoders/processor/operators/gpu/metal_shader/to_linear_ref.air")
    set(METAL_TO_LINEAR_REF_METALLIB "${CMAKE_CURRENT_BINARY_DIR}/decoders/processor/operators/gpu/metal_shader/to_linear_ref.metallib")
    set(METAL_DEBAYER_RCD_SRC "${CMAKE_CURRENT_SOURCE_DIR}/decoders/processor/operators/gpu/metal_shader/debayer_rcd.metal")
    set(METAL_DEBAYER_RCD_AIR "${CMAKE_CURRENT_BINARY_DIR}/decoders/processor/operators/gpu/metal_shader/debayer_rcd.air")
    set(METAL_DEBAYER_RCD_METALLIB "${CMAKE_CURRENT_BINARY_DIR}/decoders/processor/operators/gpu/metal_shader/debayer_rcd.metallib")
    set(METAL_CVT_REF_SPACE_SRC "${CMAKE_CURRENT_SOURCE_DIR}/decoders/processor/operators/gpu/metal_shader/cvt_ref_space.metal")
    set(METAL_CVT_REF_SPACE_AIR "${CMAKE_CURRENT_BINARY_DIR}/decoders/processor/operators/gpu/metal_shader/cvt_ref_space.air")
    set(METAL_CVT_REF_SPACE_METALLIB "${CMAKE_CURRENT_BINARY_DIR}/decoders/processor/operators/gpu/metal_shader/cvt_ref_space.metallib")
    set(METAL_HIGHLIGHT_RECONSTRUCT_SRC "${CMAKE_CURRENT_SOURCE_DIR}/decoders/processor/operators/gpu/metal_shader/highlight_reconstruct.metal")
    set(METAL_HIGHLIGHT_RECONSTRUCT_AIR "${CMAKE_CURRENT_BINARY_DIR}/decoders/processor/operators/gpu/metal_shader/highlight_reconstruct.air")
    set(METAL_HIGHLIGHT_RECONSTRUCT_METALLIB "${CMAKE_CURRENT_BINARY_DIR}/decoders/processor/operators/gpu/metal_shader/highlight_reconstruct.metallib")
    find_program(PUERHLAB_XCRUN_EXECUTABLE xcrun REQUIRED)
    execute_process(
        COMMAND ${PUERHLAB_XCRUN_EXECUTABLE} --sdk macosx --find metal
        RESULT_VARIABLE PUERHLAB_METAL_COMPILER_STATUS
        OUTPUT_VARIABLE PUERHLAB_METAL_EXECUTABLE
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_VARIABLE PUERHLAB_METAL_COMPILER_ERROR
    )
    if(NOT PUERHLAB_METAL_COMPILER_STATUS EQUAL 0)
        string(STRIP "${PUERHLAB_METAL_COMPILER_ERROR}" PUERHLAB_METAL_COMPILER_ERROR)
        message(FATAL_ERROR "Metal compiler is not available via xcrun. ${PUERHLAB_METAL_COMPILER_ERROR}")
    endif()
    execute_process(
        COMMAND ${PUERHLAB_XCRUN_EXECUTABLE} --sdk macosx --find metallib
        RESULT_VARIABLE PUERHLAB_METALLIB_STATUS
        OUTPUT_VARIABLE PUERHLAB_METALLIB_EXECUTABLE
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_VARIABLE PUERHLAB_METALLIB_ERROR
    )
    if(NOT PUERHLAB_METALLIB_STATUS EQUAL 0)
        string(STRIP "${PUERHLAB_METALLIB_ERROR}" PUERHLAB_METALLIB_ERROR)
        message(FATAL_ERROR "Metal library linker is not available via xcrun. ${PUERHLAB_METALLIB_ERROR}")
    endif()

    add_custom_command(
        OUTPUT ${METAL_UTILS_AIR}
        COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/metal/metal_utils"
        COMMAND ${PUERHLAB_METAL_EXECUTABLE} -c "${METAL_UTILS_SRC}" -o "${METAL_UTILS_AIR}"
        DEPENDS ${METAL_UTILS_SRC}
        COMMENT "Compiling Metal utility shaders"
        VERBATIM
    )
    add_custom_command(
        OUTPUT ${METAL_UTILS_METALLIB}
        COMMAND ${PUERHLAB_METALLIB_EXECUTABLE} "${METAL_UTILS_AIR}" -o "${METAL_UTILS_METALLIB}"
        DEPENDS ${METAL_UTILS_AIR}
        COMMENT "Linking Metal utility library"
        VERBATIM
    )
    add_custom_target(MetalUtilsShaders DEPENDS ${METAL_UTILS_METALLIB})

    add_custom_command(
        OUTPUT ${METAL_GEOMETRY_UTILS_AIR}
        COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/metal/metal_utils"
        COMMAND ${PUERHLAB_METAL_EXECUTABLE} -c "${METAL_GEOMETRY_UTILS_SRC}" -o "${METAL_GEOMETRY_UTILS_AIR}"
        DEPENDS ${METAL_GEOMETRY_UTILS_SRC}
        COMMENT "Compiling Metal geometry utility shaders"
        VERBATIM
    )
    add_custom_command(
        OUTPUT ${METAL_GEOMETRY_UTILS_METALLIB}
        COMMAND ${PUERHLAB_METALLIB_EXECUTABLE} "${METAL_GEOMETRY_UTILS_AIR}" -o "${METAL_GEOMETRY_UTILS_METALLIB}"
        DEPENDS ${METAL_GEOMETRY_UTILS_AIR}
        COMMENT "Linking Metal geometry utility library"
        VERBATIM
    )
    add_custom_target(MetalGeometryUtilsShaders DEPENDS ${METAL_GEOMETRY_UTILS_METALLIB})

    add_custom_command(
        OUTPUT ${METAL_LENS_CALIB_AIR}
        COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/edit/operators/geometry/metal_shader"
        COMMAND ${PUERHLAB_METAL_EXECUTABLE} -c "${METAL_LENS_CALIB_SRC}" -o "${METAL_LENS_CALIB_AIR}"
        DEPENDS ${METAL_LENS_CALIB_SRC}
        COMMENT "Compiling Metal lens calibration shader"
        VERBATIM
    )
    add_custom_command(
        OUTPUT ${METAL_LENS_CALIB_METALLIB}
        COMMAND ${PUERHLAB_METALLIB_EXECUTABLE} "${METAL_LENS_CALIB_AIR}" -o "${METAL_LENS_CALIB_METALLIB}"
        DEPENDS ${METAL_LENS_CALIB_AIR}
        COMMENT "Linking Metal lens calibration library"
        VERBATIM
    )
    add_custom_target(LensCalibOpMetalShaders DEPENDS ${METAL_LENS_CALIB_METALLIB})

    add_custom_command(
        OUTPUT ${METAL_TO_LINEAR_REF_AIR}
        COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/decoders/processor/operators/gpu/metal_shader"
        COMMAND ${PUERHLAB_METAL_EXECUTABLE} -c "${METAL_TO_LINEAR_REF_SRC}" -o "${METAL_TO_LINEAR_REF_AIR}"
        DEPENDS ${METAL_TO_LINEAR_REF_SRC}
        COMMENT "Compiling Metal ToLinearRef shader"
        VERBATIM
    )
    add_custom_command(
        OUTPUT ${METAL_TO_LINEAR_REF_METALLIB}
        COMMAND ${PUERHLAB_METALLIB_EXECUTABLE} "${METAL_TO_LINEAR_REF_AIR}" -o "${METAL_TO_LINEAR_REF_METALLIB}"
        DEPENDS ${METAL_TO_LINEAR_REF_AIR}
        COMMENT "Linking Metal ToLinearRef library"
        VERBATIM
    )
    add_custom_command(
        OUTPUT ${METAL_DEBAYER_RCD_AIR}
        COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/decoders/processor/operators/gpu/metal_shader"
        COMMAND ${PUERHLAB_METAL_EXECUTABLE} -c "${METAL_DEBAYER_RCD_SRC}" -o "${METAL_DEBAYER_RCD_AIR}"
        DEPENDS ${METAL_DEBAYER_RCD_SRC}
        COMMENT "Compiling Metal Debayer RCD shader"
        VERBATIM
    )
    add_custom_command(
        OUTPUT ${METAL_DEBAYER_RCD_METALLIB}
        COMMAND ${PUERHLAB_METALLIB_EXECUTABLE} "${METAL_DEBAYER_RCD_AIR}" -o "${METAL_DEBAYER_RCD_METALLIB}"
        DEPENDS ${METAL_DEBAYER_RCD_AIR}
        COMMENT "Linking Metal Debayer RCD library"
        VERBATIM
    )
    add_custom_command(
        OUTPUT ${METAL_HIGHLIGHT_RECONSTRUCT_AIR}
        COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/decoders/processor/operators/gpu/metal_shader"
        COMMAND ${PUERHLAB_METAL_EXECUTABLE} -c "${METAL_HIGHLIGHT_RECONSTRUCT_SRC}" -o "${METAL_HIGHLIGHT_RECONSTRUCT_AIR}"
        DEPENDS ${METAL_HIGHLIGHT_RECONSTRUCT_SRC}
        COMMENT "Compiling Metal HighlightReconstruct shader"
        VERBATIM
    )
    add_custom_command(
        OUTPUT ${METAL_CVT_REF_SPACE_AIR}
        COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/decoders/processor/operators/gpu/metal_shader"
        COMMAND ${PUERHLAB_METAL_EXECUTABLE} -c "${METAL_CVT_REF_SPACE_SRC}" -o "${METAL_CVT_REF_SPACE_AIR}"
        DEPENDS ${METAL_CVT_REF_SPACE_SRC}
        COMMENT "Compiling Metal ConvertToRefSpace shader"
        VERBATIM
    )
    add_custom_command(
        OUTPUT ${METAL_HIGHLIGHT_RECONSTRUCT_METALLIB}
        COMMAND ${PUERHLAB_METALLIB_EXECUTABLE} "${METAL_HIGHLIGHT_RECONSTRUCT_AIR}" -o "${METAL_HIGHLIGHT_RECONSTRUCT_METALLIB}"
        DEPENDS ${METAL_HIGHLIGHT_RECONSTRUCT_AIR}
        COMMENT "Linking Metal HighlightReconstruct library"
        VERBATIM
    )
    add_custom_command(
        OUTPUT ${METAL_CVT_REF_SPACE_METALLIB}
        COMMAND ${PUERHLAB_METALLIB_EXECUTABLE} "${METAL_CVT_REF_SPACE_AIR}" -o "${METAL_CVT_REF_SPACE_METALLIB}"
        DEPENDS ${METAL_CVT_REF_SPACE_AIR}
        COMMENT "Linking Metal ConvertToRefSpace library"
        VERBATIM
    )
    add_custom_target(RawProcessorOpMetalShaders DEPENDS
        ${METAL_TO_LINEAR_REF_METALLIB}
        ${METAL_DEBAYER_RCD_METALLIB}
        ${METAL_HIGHLIGHT_RECONSTRUCT_METALLIB}
        ${METAL_CVT_REF_SPACE_METALLIB}
    )

    def_library(MetalContext SOURCES metal/metal_context.cpp)
    target_link_libraries(MetalContext PUBLIC puerhlab::metal_cpp)
    def_library(MetalUtils
        SOURCES metal/metal_utils/metal_convert_utils.cpp
        PUBLIC_DEPS MetalContext puerhlab::metal_cpp ${PUERHLAB_OPENCV_CORE_DEPS}
    )
    add_dependencies(MetalUtils MetalUtilsShaders)
    target_compile_definitions(MetalUtils PRIVATE PUERHLAB_METAL_UTILS_METALLIB_PATH="${METAL_UTILS_METALLIB}")
    def_library(MetalImage SOURCES image/metal_image.cpp PUBLIC_DEPS MetalContext MetalUtils puerhlab::metal_cpp ${PUERHLAB_OPENCV_CORE_DEPS})
    def_library(MetalGeometryUtils
        SOURCES metal/metal_utils/geometry_utils.cpp
        PUBLIC_DEPS MetalImage puerhlab::metal_cpp ${PUERHLAB_OPENCV_CORE_DEPS}
    )
    add_dependencies(MetalGeometryUtils MetalGeometryUtilsShaders)
    target_compile_definitions(MetalGeometryUtils PRIVATE PUERHLAB_METAL_GEOMETRY_UTILS_METALLIB_PATH="${METAL_GEOMETRY_UTILS_METALLIB}")
endif()

# ==============================================================================
# 2. Core Image Handling 
# ==============================================================================

# ImageBuffer (Platform specific deps handling)
set(IMAGE_BUFFER_SRCS image/image_buffer.cpp)
add_library(ImageBuffer ${IMAGE_BUFFER_SRCS})
target_include_directories(ImageBuffer PUBLIC include)

# Common dependencies
set(IMAGE_BUFFER_DEPS Exiv2 ${PUERHLAB_OPENCV_CORE_DEPS} libraw::raw_r TimeProvider JSON OpenMP::OpenMP_CXX StrConv)

# Platform specific dependencies
if (WIN32)
    target_link_libraries(ImageBuffer PUBLIC ${IMAGE_BUFFER_DEPS} easy_profiler xxHash::xxhash)
elseif(PUERHLAB_METAL_ENABLED)
    target_link_libraries(ImageBuffer PUBLIC ${IMAGE_BUFFER_DEPS} xxHash MetalImage)
else()
    target_link_libraries(ImageBuffer PUBLIC ${IMAGE_BUFFER_DEPS} xxHash)
endif()

# Image Base
def_library(Image 
    SOURCES image/image.cpp 
    image/metadata_extractor.cpp
    PUBLIC_DEPS ImageBuffer
)

# Image Pool
def_library(ImagePool 
    SOURCES storage/image_pool/image_pool_manager.cpp
    PRIVATE_DEPS Image
)

# ==============================================================================
# 3. Processors & Operators
# ==============================================================================

# --- RawProcessorOp Configuration (CPU/GPU Split) ---
set(RAW_OP_CPU_SRCS
    decoders/processor/operators/cpu/color_space_conv.cpp
    decoders/processor/operators/cpu/debayer_ahd.cpp
    decoders/processor/operators/cpu/debayer_amaze.cpp
    decoders/processor/operators/cpu/debayer_rcd.cpp
    decoders/processor/operators/cpu/highlight_reconstruct.cpp
    decoders/processor/operators/cpu/white_balance.cpp
    decoders/processor/operators/cpu/raw_proc_utils.cpp
)

if(PUERHLAB_CUDA_ENABLED)
    set(RAW_OP_GPU_SRCS
        decoders/processor/operators/gpu/cuda_color_space_conv.cu
        decoders/processor/operators/gpu/cuda_debayer_ahd.cu
        decoders/processor/operators/gpu/cuda_debayer_rcd.cu
        decoders/processor/operators/gpu/cuda_highlight_reconstruct.cu
        decoders/processor/operators/gpu/cuda_image_ops.cu
        decoders/processor/operators/gpu/cuda_rotate.cu
        decoders/processor/operators/gpu/cuda_white_balance.cu
    )
elseif(PUERHLAB_METAL_ENABLED)
    set(RAW_OP_GPU_SRCS
        decoders/processor/operators/gpu/metal_cvt_ref_space.cpp
        decoders/processor/operators/gpu/metal_debayer_rcd.cpp
        decoders/processor/operators/gpu/metal_highlight_reconstruct.cpp
        decoders/processor/operators/gpu/metal_to_linear_ref.cpp
    )
endif()

# RawProcessorOp Target
def_library(RawProcessorOp
    SOURCES ${RAW_OP_CPU_SRCS} ${RAW_OP_GPU_SRCS}
    PUBLIC_DEPS ImageBuffer libraw::raw_r Exiv2 hwy::hwy Eigen3::Eigen
)

if(PUERHLAB_CUDA_ENABLED)
    set_target_properties(RawProcessorOp PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    target_compile_definitions(RawProcessorOp PUBLIC HAVE_CUDA)
endif()
if(PUERHLAB_METAL_ENABLED)
    add_dependencies(RawProcessorOp RawProcessorOpMetalShaders)
    target_compile_definitions(RawProcessorOp PRIVATE
        PUERHLAB_METAL_TO_LINEAR_REF_METALLIB_PATH="${METAL_TO_LINEAR_REF_METALLIB}"
        PUERHLAB_METAL_DEBAYER_RCD_METALLIB_PATH="${METAL_DEBAYER_RCD_METALLIB}"
        PUERHLAB_METAL_HIGHLIGHT_RECONSTRUCT_METALLIB_PATH="${METAL_HIGHLIGHT_RECONSTRUCT_METALLIB}"
        PUERHLAB_METAL_CVT_REF_SPACE_METALLIB_PATH="${METAL_CVT_REF_SPACE_METALLIB}"
    )
endif()

# --- Edit Operators ---
set(OPERATORS_SRCS
    edit/operators/operator_factory.cpp
    edit/operators/basic/exposure_op.cpp
    edit/operators/basic/exposure_op_vec.cpp
    edit/operators/basic/contrast_op.cpp
    edit/operators/basic/black_op.cpp
    edit/operators/basic/color_temp_op.cpp
    edit/operators/basic/highlight_op.cpp
    edit/operators/basic/shadow_op.cpp
    edit/operators/basic/white_op.cpp
    edit/operators/color/tint_op.cpp
    edit/operators/color/conversion/Oklab_cvt.cpp
    edit/operators/color/saturation_op.cpp
    edit/operators/color/vibrance_op.cpp
    edit/operators/color/HLS_op.cpp
    edit/operators/detail/clarity_op.cpp
    edit/operators/detail/sharpen_op.cpp
    edit/operators/wheel/color_wheel_op.cpp
    edit/operators/curve/curve_op.cpp
    edit/operators/cst/cst_op.cpp
    edit/operators/cst/lmt_op.cpp
    edit/operators/cst/aces_odt_cpu.cpp
    edit/operators/cst/open_drt_cpu.cpp
    edit/operators/cst/odt_op.cpp
    edit/operators/raw/raw_decode_op.cpp
    edit/operators/geometry/resize_op.cpp
    edit/operators/geometry/crop_rotate_op.cpp
    edit/operators/geometry/lens_calib_op.cpp
    edit/operators/operator_registeration.cpp
)

if(PUERHLAB_CUDA_ENABLED)
    list(APPEND OPERATORS_SRCS
        edit/operators/geometry/cuda_geometry_ops.cu
        edit/operators/geometry/cuda_lens_calib_ops.cu
    )
endif()
if(PUERHLAB_METAL_ENABLED)
    list(APPEND OPERATORS_SRCS
        edit/operators/geometry/metal_lens_calib.cpp
    )
endif()

def_library(Operators
    SOURCES ${OPERATORS_SRCS}
    PUBLIC_DEPS ImageBuffer JSON OpenColorIO::OpenColorIO hwy::hwy lensfun::lensfun
)
if(PUERHLAB_METAL_ENABLED)
    add_dependencies(Operators LensCalibOpMetalShaders)
    target_link_libraries(Operators PRIVATE MetalGeometryUtils)
    target_compile_definitions(Operators PRIVATE
        PUERHLAB_METAL_LENS_CALIB_METALLIB_PATH="${METAL_LENS_CALIB_METALLIB}"
    )
endif()
if(DEFINED PUERHLAB_LENSFUN_INCLUDE_DIR AND NOT "${PUERHLAB_LENSFUN_INCLUDE_DIR}" STREQUAL "")
    target_include_directories(Operators SYSTEM BEFORE PUBLIC "${PUERHLAB_LENSFUN_INCLUDE_DIR}")
endif()

# --- Edit Pipeline ---
set(PIPELINE_SRCS
    edit/pipeline/pipeline_cpu.cpp
    edit/pipeline/pipeline_stage.cpp
    edit/pipeline/pipeline_gpu_wrapper.cpp
)

if(PUERHLAB_CUDA_ENABLED)
    set(GPU_PIPELINE_SRCS
        edit/pipeline/pipeline_gpu_impl.cu
    )
endif()

def_library(EditPipeline
    SOURCES ${PIPELINE_SRCS} ${GPU_PIPELINE_SRCS}
    PUBLIC_DEPS Operators
)

if(PUERHLAB_CUDA_ENABLED)
    set_target_properties(EditPipeline PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    target_compile_definitions(EditPipeline PUBLIC HAVE_CUDA)
    target_include_directories(EditPipeline PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
    target_link_libraries(EditPipeline PRIVATE CUDA::cudart)
endif()

set(RENDERER_SRCS renderer/pipeline_scheduler.cpp)
def_library(PipelineScheduler
    SOURCES ${RENDERER_SRCS}
    PUBLIC_DEPS EditPipeline ThreadPool)

# --- Raw Processor Wrapper ---
def_library(RawProcessor
    SOURCES decoders/processor/raw_processor.cpp
    PUBLIC_DEPS Image libraw::raw_r Exiv2 RawProcessorOp
    PRIVATE_DEPS opencv_imgproc
)

# ==============================================================================
# 4. Decoding & IO
# ==============================================================================

set(DECODER_SRCS
    decoders/decoder_scheduler.cpp
    decoders/raw_decoder.cpp
    decoders/thumbnail_decoder.cpp
    decoders/metadata_decoder.cpp
)

def_library(ImageDecoder
    SOURCES ${DECODER_SRCS}
    PUBLIC_DEPS Image ThreadPool libraw::raw_r RawProcessor
    PRIVATE_DEPS ${PUERHLAB_OPENCV_IMAGE_DEPS}
)

def_library(ImageLoader 
    SOURCES io/image/image_loader.cpp
    PUBLIC_DEPS ImageDecoder
)

# ==============================================================================
# 5. Business Logic & State
# ==============================================================================

set(HISTORY_SRCS 
    edit/history/edit_history.cpp 
    edit/history/version.cpp 
    edit/history/edit_transaction.cpp
)

def_library(EditHistory
    SOURCES ${HISTORY_SRCS}
    PUBLIC_DEPS Image TimeProvider JSON EditPipeline
)

set(FILTER_SRCS
    sleeve/sleeve_filter/filter_sql.cpp
    sleeve/sleeve_filter/filter_combo.cpp
)

def_library(SleeveFilter
    SOURCES ${FILTER_SRCS}
    PRIVATE_DEPS Image EditHistory JSON
)

set(ELEMENT_SRCS
    sleeve/sleeve_element/sleeve_element_factory.cpp
    sleeve/sleeve_element/sleeve_element.cpp
    sleeve/sleeve_element/sleeve_file.cpp
    sleeve/sleeve_element/sleeve_folder.cpp
)

def_library(SleeveElement
    SOURCES ${ELEMENT_SRCS}
    PUBLIC_DEPS Image ImageLoader SleeveFilter EditHistory ImagePool
)

# ==============================================================================
# 6. Storage & Database
# ==============================================================================

# ORM Layer
def_library(DuckORM
    SOURCES storage/mapper/duckorm/duckdb_orm.cpp storage/mapper/duckorm/duckdb_types.cpp
    PUBLIC_DEPS DuckDB
)

# Mappers
set(MAPPER_SRCS
    storage/mapper/sleeve/element/file_mapper.cpp
    storage/mapper/sleeve/element/folder_mapper.cpp
    storage/mapper/sleeve/element/element_mapper.cpp
    storage/mapper/sleeve/element/element_id_mapper.cpp
    storage/mapper/sleeve/base/base_mapper.cpp
    storage/mapper/image/image_mapper.cpp
    storage/mapper/sleeve/filter/filter_mapper.cpp
    storage/mapper/sleeve/edit_history/history_mapper.cpp
    storage/mapper/pipeline/pipeline_mapper.cpp
)

def_library(SleeveMapper
    SOURCES ${MAPPER_SRCS}
    PUBLIC_DEPS DuckDB DuckORM Image SleeveElement ImagePool EditPipeline PipelineScheduler
)

# Services
set(SERVICE_SRCS
    storage/service/image/image_service.cpp
    storage/service/sleeve/base/base_service.cpp
    storage/service/sleeve/base/root_service.cpp
    storage/service/sleeve/element/element_service.cpp
    storage/service/sleeve/element/element_id_service.cpp
    storage/service/sleeve/element/file_service.cpp
    storage/service/sleeve/element/folder_service.cpp
    storage/service/sleeve/edit_history/history_service.cpp
    storage/service/pipeline/pipeline_service.cpp
)

def_library(StorageService
    SOURCES ${SERVICE_SRCS}
    PUBLIC_DEPS DuckDB DuckORM SleeveMapper StrConv PipelineScheduler
)

# Controllers
set(CONTROLLER_SRCS
    storage/controller/controller_types.cpp
    storage/controller/db_controller.cpp
    storage/controller/image/image_controller.cpp
    storage/controller/sleeve/element_controller.cpp
)

def_library(SleeveController
    SOURCES ${CONTROLLER_SRCS}
    PUBLIC_DEPS DuckDB StorageService StrConv
)

# ==============================================================================
# 7. Filesystem & View Manager
# ==============================================================================

set(FS_SRCS
    sleeve/sleeve_base.cpp
    sleeve/path_resolver.cpp
    sleeve/storage_service.cpp
    sleeve/dentry_cache_manager.cpp
    sleeve/sleeve_filesystem.cpp
)

def_library(SleeveFS
    SOURCES ${FS_SRCS}
    PUBLIC_DEPS SleeveElement SleeveController
)

# def_library(SleeveView
#     SOURCES sleeve/sleeve_view.cpp
#     PUBLIC_DEPS Image SleeveFS ImagePool PipelineScheduler
# )

def_library(SleeveManager
    SOURCES sleeve/sleeve_manager.cpp
    PUBLIC_DEPS SleeveFS PipelineScheduler
)

# ==============================================================================
# 8. I/O  
# ==============================================================================

def_library(ImageWriter
    SOURCES io/image/image_writer.cpp
    PUBLIC_DEPS OpenImageIO::OpenImageIO ImageBuffer
    PRIVATE_DEPS ${PUERHLAB_OPENCV_IMAGE_DEPS}
)

# ==============================================================================
# 9. Services
# ==============================================================================
def_library(FSService
    SOURCES app/sleeve_service.cpp
    PUBLIC_DEPS SleeveFS EditHistory
)



def_library(ImagePoolService
    SOURCES app/image_pool_service.cpp
    PUBLIC_DEPS ImagePool Image StorageService
)

def_library(ProjectService
    SOURCES app/project_service.cpp
    PUBLIC_DEPS FSService StorageService ImagePoolService StrConv JSON
)

def_library(PipelineMgmtService
    SOURCES app/pipeline_service.cpp
    PUBLIC_DEPS ProjectService EditPipeline
    )

def_library(EditHistoryMgmtService
    SOURCES app/history_mgmt_service.cpp
    PUBLIC_DEPS ProjectService EditHistory
)

def_library(ImportService
    SOURCES app/import_service.cpp
    PUBLIC_DEPS SleeveManager ImageWriter ImagePoolService
)

def_library(ThumbnailService
    SOURCES app/thumbnail_service.cpp
    PUBLIC_DEPS EditPipeline ThreadPool StrConv ImagePoolService PipelineMgmtService
)

def_library(ExportService
    SOURCES app/export_service.cpp
    PUBLIC_DEPS ImageWriter ImageLoader ImagePoolService PipelineMgmtService PipelineScheduler ThreadPool
)

def_library(SleeveFilterService
    SOURCES app/sleeve_filter_service.cpp
    PUBLIC_DEPS SleeveElement StorageService SleeveFilter StrConv
)

# ==============================================================================
# 9. UI Components
# ==============================================================================
if(PUERHLAB_REAL_OPENGL_EDITOR)
    def_library(EditViewer
        SOURCES ui/edit_viewer/edit_viewer.cpp include/ui/edit_viewer/edit_viewer.hpp
        PUBLIC_DEPS EditPipeline OpenGL::GL Qt6::OpenGLWidgets
    )

    if(PUERHLAB_CUDA_ENABLED)
        set_target_properties(EditViewer PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
        target_compile_definitions(EditViewer PUBLIC HAVE_CUDA)
    endif()
endif()

# ── AlbumBackendLib: reusable static library for album_backend ──────────────
add_library(AlbumBackendLib STATIC
    ui/puerhlab_main/album_backend/album_backend.cpp
    ui/puerhlab_main/album_backend/path_utils.cpp
    ui/puerhlab_main/album_backend/packed_project.cpp
    ui/puerhlab_main/album_backend/project_handler.cpp
    ui/puerhlab_main/album_backend/thumbnail_manager.cpp
    ui/puerhlab_main/album_backend/folder_controller.cpp
    ui/puerhlab_main/album_backend/image_controller.cpp
    ui/puerhlab_main/album_backend/stats_engine.cpp
    ui/puerhlab_main/album_backend/import_export.cpp
    ui/puerhlab_main/album_backend/editor_controller.cpp
    include/ui/puerhlab_main/album_backend/album_backend.hpp
    include/ui/puerhlab_main/album_backend/album_types.hpp
    include/ui/puerhlab_main/album_backend/path_utils.hpp
    include/ui/puerhlab_main/album_backend/packed_project.hpp
    include/ui/puerhlab_main/album_backend/project_handler.hpp
    include/ui/puerhlab_main/album_backend/thumbnail_manager.hpp
    include/ui/puerhlab_main/album_backend/folder_controller.hpp
    include/ui/puerhlab_main/album_backend/image_controller.hpp
    include/ui/puerhlab_main/album_backend/stats_engine.hpp
    include/ui/puerhlab_main/album_backend/import_export.hpp
    include/ui/puerhlab_main/album_backend/editor_controller.hpp
)

target_include_directories(AlbumBackendLib
    PUBLIC
        ${CMAKE_SOURCE_DIR}/pu-erh_lab/src/include
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/ui/puerhlab_main
        ${CMAKE_CURRENT_SOURCE_DIR}/ui/puerhlab_main/album_backend
)

target_link_libraries(AlbumBackendLib
    PUBLIC
        Qt6::Quick
        Qt6::Qml
        Qt6::QuickControls2
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
)

if(TARGET EditViewer)
    target_link_libraries(AlbumBackendLib PUBLIC EditViewer)
endif()

if(WIN32)
    target_link_libraries(AlbumBackendLib PUBLIC easy_profiler)
endif()

set(PUERHLAB_EDITOR_DIALOG_SRCS
    ui/puerhlab_main/editor_dialog/editor_dialog_stub.cpp
    include/ui/puerhlab_main/editor_dialog/editor_dialog.hpp
)

if(PUERHLAB_REAL_OPENGL_EDITOR)
    set(PUERHLAB_EDITOR_DIALOG_SRCS
        ui/puerhlab_main/editor_dialog/editor_dialog.cpp
        ui/puerhlab_main/editor_dialog/dialog.cpp
        ui/puerhlab_main/editor_dialog/state.cpp
        ui/puerhlab_main/editor_dialog/controllers/image_controller.cpp
        ui/puerhlab_main/editor_dialog/controllers/pipeline_controller.cpp
        ui/puerhlab_main/editor_dialog/controllers/history_controller.cpp
        ui/puerhlab_main/editor_dialog/controllers/render_controller.cpp
        ui/puerhlab_main/editor_dialog/modules/histogram.cpp
        ui/puerhlab_main/editor_dialog/modules/color_temp.cpp
        ui/puerhlab_main/editor_dialog/modules/color_wheel.cpp
        ui/puerhlab_main/editor_dialog/modules/curve.cpp
        ui/puerhlab_main/editor_dialog/modules/lens_calib.cpp
        ui/puerhlab_main/editor_dialog/modules/geometry.cpp
        ui/puerhlab_main/editor_dialog/modules/hls.cpp
        ui/puerhlab_main/editor_dialog/modules/pipeline_io.cpp
        ui/puerhlab_main/editor_dialog/modules/versioning.cpp
        ui/puerhlab_main/editor_dialog/widgets/spinner.cpp
        ui/puerhlab_main/editor_dialog/widgets/histogram_widget.cpp
        ui/puerhlab_main/editor_dialog/widgets/trackball.cpp
        ui/puerhlab_main/editor_dialog/widgets/tone_curve_widget.cpp
        ui/puerhlab_main/editor_dialog/widgets/history_cards.cpp
        include/ui/puerhlab_main/editor_dialog/editor_dialog.hpp
        ui/puerhlab_main/editor_dialog/dialog.hpp
        ui/puerhlab_main/editor_dialog/state.hpp
        ui/puerhlab_main/editor_dialog/controllers/image_controller.hpp
        ui/puerhlab_main/editor_dialog/controllers/pipeline_controller.hpp
        ui/puerhlab_main/editor_dialog/controllers/history_controller.hpp
        ui/puerhlab_main/editor_dialog/controllers/render_controller.hpp
        ui/puerhlab_main/editor_dialog/modules/histogram.hpp
        ui/puerhlab_main/editor_dialog/modules/color_temp.hpp
        ui/puerhlab_main/editor_dialog/modules/color_wheel.hpp
        ui/puerhlab_main/editor_dialog/modules/curve.hpp
        ui/puerhlab_main/editor_dialog/modules/lens_calib.hpp
        ui/puerhlab_main/editor_dialog/modules/geometry.hpp
        ui/puerhlab_main/editor_dialog/modules/versioning.hpp
        ui/puerhlab_main/editor_dialog/widgets/spinner.hpp
        ui/puerhlab_main/editor_dialog/widgets/histogram_widget.hpp
        ui/puerhlab_main/editor_dialog/widgets/trackball.hpp
        ui/puerhlab_main/editor_dialog/widgets/tone_curve_widget.hpp
        ui/puerhlab_main/editor_dialog/widgets/history_cards.hpp
    )
endif()

# ── puerhlab_main executable ────────────────────────────────────────────────
add_executable(puerhlab_main
    ui/puerhlab_main/main.cpp
    ui/puerhlab_main/app_theme.cpp
    config/resource.qrc
    include/ui/puerhlab_main/app_theme.hpp
    ${PUERHLAB_EDITOR_DIALOG_SRCS}
)

target_include_directories(puerhlab_main
    PRIVATE
        ${CMAKE_SOURCE_DIR}/pu-erh_lab/src/include
        ${CMAKE_CURRENT_SOURCE_DIR}/ui/puerhlab_main
        ${CMAKE_CURRENT_SOURCE_DIR}/ui/puerhlab_main/album_backend
        ${CMAKE_CURRENT_SOURCE_DIR}/ui/puerhlab_main/editor_dialog
)

target_link_libraries(puerhlab_main
    PRIVATE
        AlbumBackendLib
        Qt6::Quick
        Qt6::Qml
        Qt6::QuickControls2
        Qt6::QuickDialogs2
        Qt6::Widgets
        Qt6::QuickEffects
)

qt_add_qml_module(puerhlab_main
    URI PuerhLab.Main
    VERSION 1.0
    QML_FILES
        ui/puerhlab_main/qml/Main.qml
        ui/puerhlab_main/qml/AlbumExportDialog.qml
        ui/puerhlab_main/qml/ImportProgressRing.qml
        ui/puerhlab_main/qml/InspectorPanel.qml
        ui/puerhlab_main/qml/StatsCard.qml
        ui/puerhlab_main/qml/ImageContextMenu.qml
        ui/puerhlab_main/qml/ThumbnailGridView.qml
        ui/puerhlab_main/qml/ThumbnailListView.qml
)

target_compile_definitions(puerhlab_main PRIVATE PUERHLAB_SOURCE_DIR="${CMAKE_SOURCE_DIR}")

if(DEFINED PUERHLAB_LENSFUN_LIBRARY_DIR AND NOT "${PUERHLAB_LENSFUN_LIBRARY_DIR}" STREQUAL "")
    set_property(TARGET puerhlab_main APPEND PROPERTY BUILD_RPATH "${PUERHLAB_LENSFUN_LIBRARY_DIR}")
endif()

set(PUERHLAB_MAIN_FONT_SRC "${CMAKE_SOURCE_DIR}/pu-erh_lab/src/config/fonts/main_Inter.ttf")
add_custom_command(TARGET puerhlab_main POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_FILE_DIR:puerhlab_main>/fonts"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${PUERHLAB_MAIN_FONT_SRC}" "$<TARGET_FILE_DIR:puerhlab_main>/fonts/main_Inter.ttf"
    VERBATIM
)

if(WIN32 AND TARGET puerhlab_lensfun)
    add_custom_command(TARGET puerhlab_main POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:puerhlab_lensfun>"
            "$<TARGET_FILE_DIR:puerhlab_main>/lensfun.dll"
        VERBATIM
    )
endif()
