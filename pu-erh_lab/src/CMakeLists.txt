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
if(PUERHLAB_METAL_ENABLED)
    target_link_libraries(ThreadPool PUBLIC puerhlab::metal_cpp)
endif()

def_library(TimeProvider SOURCES utils/clock/time_provider.cpp)

def_library(UiLocalization
    SOURCES
        include/ui/puerhlab_main/i18n.hpp
        include/ui/puerhlab_main/language_manager.hpp
        ui/puerhlab_main/i18n.cpp
        ui/puerhlab_main/language_manager.cpp
    PUBLIC_DEPS
        Qt6::Core
        Qt6::Qml
)

def_library(StrConv 
    SOURCES utils/string/converter.cpp 
    PUBLIC_DEPS utfcpp
)

if(PUERHLAB_METAL_ENABLED)
    if(POLICY CMP0116)
        cmake_policy(SET CMP0116 NEW)
    endif()

    set(METAL_UTILS_SRC "${CMAKE_CURRENT_SOURCE_DIR}/metal/metal_utils/metal_convert.metal")
    set(METAL_UTILS_AIR "${CMAKE_CURRENT_BINARY_DIR}/metal/metal_utils/metal_convert.air")
    set(METAL_UTILS_METALLIB "${CMAKE_CURRENT_BINARY_DIR}/metal/metal_utils/metal_convert.metallib")
    set(METAL_GEOMETRY_UTILS_SRC "${CMAKE_CURRENT_SOURCE_DIR}/metal/metal_utils/geometry_utils.metal")
    set(METAL_GEOMETRY_UTILS_AIR "${CMAKE_CURRENT_BINARY_DIR}/metal/metal_utils/geometry_utils.air")
    set(METAL_GEOMETRY_UTILS_METALLIB "${CMAKE_CURRENT_BINARY_DIR}/metal/metal_utils/geometry_utils.metallib")
    set(METAL_LENS_CALIB_SRC "${CMAKE_CURRENT_SOURCE_DIR}/edit/operators/geometry/metal_shader/lens_calib.metal")
    set(METAL_LENS_CALIB_AIR "${CMAKE_CURRENT_BINARY_DIR}/edit/operators/geometry/metal_shader/lens_calib.air")
    set(METAL_LENS_CALIB_METALLIB "${CMAKE_CURRENT_BINARY_DIR}/edit/operators/geometry/metal_shader/lens_calib.metallib")
    set(METAL_FUSED_PIPELINE_SRC "${CMAKE_CURRENT_SOURCE_DIR}/edit/pipeline/metal_shader/fused_pipeline.metal")
    set(METAL_FUSED_PIPELINE_AIR "${CMAKE_CURRENT_BINARY_DIR}/edit/pipeline/metal_shader/fused_pipeline.air")
    set(METAL_FUSED_PIPELINE_METALLIB "${CMAKE_CURRENT_BINARY_DIR}/edit/pipeline/metal_shader/fused_pipeline.metallib")
    set(METAL_FUSED_PIPELINE_DEPS
        "${CMAKE_CURRENT_SOURCE_DIR}/edit/operators/GPU_kernels/metal_shader/common.metal"
        "${CMAKE_CURRENT_SOURCE_DIR}/edit/operators/GPU_kernels/metal_shader/basic.metal"
        "${CMAKE_CURRENT_SOURCE_DIR}/edit/operators/GPU_kernels/metal_shader/color.metal"
        "${CMAKE_CURRENT_SOURCE_DIR}/edit/operators/GPU_kernels/metal_shader/cst.metal"
        "${CMAKE_CURRENT_SOURCE_DIR}/edit/operators/GPU_kernels/metal_shader/detail.metal"
    )
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
    set(METAL_XTRANS_INTERPOLATE_SRC "${CMAKE_CURRENT_SOURCE_DIR}/decoders/processor/operators/gpu/metal_shader/xtrans_interpolate.metal")
    set(METAL_XTRANS_INTERPOLATE_AIR "${CMAKE_CURRENT_BINARY_DIR}/decoders/processor/operators/gpu/metal_shader/xtrans_interpolate.air")
    set(METAL_XTRANS_INTERPOLATE_METALLIB "${CMAKE_CURRENT_BINARY_DIR}/decoders/processor/operators/gpu/metal_shader/xtrans_interpolate.metallib")
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

    set(PUERHLAB_METAL_COMMON_COMPILE_FLAGS
        -MMD
        -fno-color-diagnostics
    )
    if(DEFINED CMAKE_OSX_DEPLOYMENT_TARGET AND NOT "${CMAKE_OSX_DEPLOYMENT_TARGET}" STREQUAL "")
        set(PUERHLAB_METAL_SHADER_DEPLOYMENT_TARGET_DEFAULT "${CMAKE_OSX_DEPLOYMENT_TARGET}")
    else()
        set(PUERHLAB_METAL_SHADER_DEPLOYMENT_TARGET_DEFAULT "13.0")
    endif()
    set(PUERHLAB_METAL_SHADER_DEPLOYMENT_TARGET
        "${PUERHLAB_METAL_SHADER_DEPLOYMENT_TARGET_DEFAULT}"
        CACHE STRING
        "macOS deployment target used when compiling Metal shaders."
    )
    set(PUERHLAB_METAL_PLATFORM_COMPILE_FLAGS)
    if(NOT "${PUERHLAB_METAL_SHADER_DEPLOYMENT_TARGET}" STREQUAL "")
        list(APPEND PUERHLAB_METAL_PLATFORM_COMPILE_FLAGS
            "-mmacosx-version-min=${PUERHLAB_METAL_SHADER_DEPLOYMENT_TARGET}"
        )
    endif()
    set(PUERHLAB_METAL_MODULE_CACHE_DIR "${CMAKE_CURRENT_BINARY_DIR}/metal/module-cache")
    set(PUERHLAB_METAL_CONFIG_COMPILE_FLAGS
        $<$<CONFIG:Debug>:-gline-tables-only>
        $<$<CONFIG:RelWithDebInfo>:-O2>
        $<$<CONFIG:RelWithDebInfo>:-gline-tables-only>
        $<$<OR:$<CONFIG:Release>,$<CONFIG:MinSizeRel>>:-O3>
    )

    function(puerhlab_add_metal_library shader_target source_file air_file metallib_file compile_comment
             link_comment)
        set(options)
        set(oneValueArgs)
        set(multiValueArgs DEPENDS)
        cmake_parse_arguments(ARG "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

        get_filename_component(_air_dir "${air_file}" DIRECTORY)
        set(_depfile "${air_file}.d")

        add_custom_command(
            OUTPUT "${air_file}"
            COMMAND ${CMAKE_COMMAND} -E make_directory "${_air_dir}"
            COMMAND ${CMAKE_COMMAND} -E make_directory "${PUERHLAB_METAL_MODULE_CACHE_DIR}"
            COMMAND ${PUERHLAB_METAL_EXECUTABLE}
                    ${PUERHLAB_METAL_COMMON_COMPILE_FLAGS}
                    ${PUERHLAB_METAL_PLATFORM_COMPILE_FLAGS}
                    ${PUERHLAB_METAL_CONFIG_COMPILE_FLAGS}
                    -fmodules-cache-path="${PUERHLAB_METAL_MODULE_CACHE_DIR}"
                    -MF "${_depfile}"
                    -c "${source_file}"
                    -o "${air_file}"
            DEPENDS "${source_file}" ${ARG_DEPENDS}
            DEPFILE "${_depfile}"
            BYPRODUCTS "${_depfile}"
            COMMENT "${compile_comment}"
            VERBATIM
            COMMAND_EXPAND_LISTS
        )
        add_custom_command(
            OUTPUT "${metallib_file}"
            COMMAND ${PUERHLAB_METALLIB_EXECUTABLE} "${air_file}" -o "${metallib_file}"
            DEPENDS "${air_file}"
            COMMENT "${link_comment}"
            VERBATIM
        )
        add_custom_target(${shader_target} DEPENDS "${metallib_file}")
    endfunction()

    puerhlab_add_metal_library(
        MetalUtilsShaders
        "${METAL_UTILS_SRC}"
        "${METAL_UTILS_AIR}"
        "${METAL_UTILS_METALLIB}"
        "Compiling Metal utility shaders"
        "Linking Metal utility library"
    )

    puerhlab_add_metal_library(
        MetalGeometryUtilsShaders
        "${METAL_GEOMETRY_UTILS_SRC}"
        "${METAL_GEOMETRY_UTILS_AIR}"
        "${METAL_GEOMETRY_UTILS_METALLIB}"
        "Compiling Metal geometry utility shaders"
        "Linking Metal geometry utility library"
    )

    puerhlab_add_metal_library(
        LensCalibOpMetalShaders
        "${METAL_LENS_CALIB_SRC}"
        "${METAL_LENS_CALIB_AIR}"
        "${METAL_LENS_CALIB_METALLIB}"
        "Compiling Metal lens calibration shader"
        "Linking Metal lens calibration library"
    )

    puerhlab_add_metal_library(
        EditPipelineMetalShaders
        "${METAL_FUSED_PIPELINE_SRC}"
        "${METAL_FUSED_PIPELINE_AIR}"
        "${METAL_FUSED_PIPELINE_METALLIB}"
        "Compiling Metal fused pipeline shader"
        "Linking Metal fused pipeline library"
        DEPENDS ${METAL_FUSED_PIPELINE_DEPS}
    )

    puerhlab_add_metal_library(
        MetalToLinearRefShaders
        "${METAL_TO_LINEAR_REF_SRC}"
        "${METAL_TO_LINEAR_REF_AIR}"
        "${METAL_TO_LINEAR_REF_METALLIB}"
        "Compiling Metal ToLinearRef shader"
        "Linking Metal ToLinearRef library"
    )

    puerhlab_add_metal_library(
        MetalDebayerRcdShaders
        "${METAL_DEBAYER_RCD_SRC}"
        "${METAL_DEBAYER_RCD_AIR}"
        "${METAL_DEBAYER_RCD_METALLIB}"
        "Compiling Metal Debayer RCD shader"
        "Linking Metal Debayer RCD library"
    )

    puerhlab_add_metal_library(
        MetalHighlightReconstructShaders
        "${METAL_HIGHLIGHT_RECONSTRUCT_SRC}"
        "${METAL_HIGHLIGHT_RECONSTRUCT_AIR}"
        "${METAL_HIGHLIGHT_RECONSTRUCT_METALLIB}"
        "Compiling Metal HighlightReconstruct shader"
        "Linking Metal HighlightReconstruct library"
    )

    puerhlab_add_metal_library(
        MetalXTransInterpolateShaders
        "${METAL_XTRANS_INTERPOLATE_SRC}"
        "${METAL_XTRANS_INTERPOLATE_AIR}"
        "${METAL_XTRANS_INTERPOLATE_METALLIB}"
        "Compiling Metal X-Trans interpolate shader"
        "Linking Metal X-Trans interpolate library"
    )

    puerhlab_add_metal_library(
        MetalCvtRefSpaceShaders
        "${METAL_CVT_REF_SPACE_SRC}"
        "${METAL_CVT_REF_SPACE_AIR}"
        "${METAL_CVT_REF_SPACE_METALLIB}"
        "Compiling Metal ConvertToRefSpace shader"
        "Linking Metal ConvertToRefSpace library"
    )

    add_custom_target(RawProcessorOpMetalShaders DEPENDS
        "${METAL_TO_LINEAR_REF_METALLIB}"
        "${METAL_DEBAYER_RCD_METALLIB}"
        "${METAL_HIGHLIGHT_RECONSTRUCT_METALLIB}"
        "${METAL_XTRANS_INTERPOLATE_METALLIB}"
        "${METAL_CVT_REF_SPACE_METALLIB}"
    )
    set(PUERHLAB_METAL_RUNTIME_LIBS
        "${METAL_UTILS_METALLIB}"
        "${METAL_GEOMETRY_UTILS_METALLIB}"
        "${METAL_LENS_CALIB_METALLIB}"
        "${METAL_FUSED_PIPELINE_METALLIB}"
        "${METAL_TO_LINEAR_REF_METALLIB}"
        "${METAL_DEBAYER_RCD_METALLIB}"
        "${METAL_HIGHLIGHT_RECONSTRUCT_METALLIB}"
        "${METAL_XTRANS_INTERPOLATE_METALLIB}"
        "${METAL_CVT_REF_SPACE_METALLIB}"
    )
    set(PUERHLAB_METAL_RUNTIME_LIBS "${PUERHLAB_METAL_RUNTIME_LIBS}" CACHE INTERNAL
        "Metal shader libraries required at runtime."
    )

    def_library(MetalContext SOURCES metal/metal_context.cpp)
    target_link_libraries(MetalContext PUBLIC puerhlab::metal_cpp)
    def_library(MetalImage SOURCES image/metal_image.cpp PUBLIC_DEPS MetalContext puerhlab::metal_cpp ${PUERHLAB_OPENCV_CORE_DEPS})
    def_library(MetalUtils
        SOURCES metal/metal_utils/metal_convert_utils.cpp
        PUBLIC_DEPS MetalContext MetalImage puerhlab::metal_cpp ${PUERHLAB_OPENCV_CORE_DEPS}
    )
    add_dependencies(MetalUtils MetalUtilsShaders)
    target_compile_definitions(MetalUtils PRIVATE PUERHLAB_METAL_UTILS_METALLIB_PATH="${METAL_UTILS_METALLIB}")
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
set(IMAGE_BUFFER_DEPS Exiv2 ${PUERHLAB_OPENCV_CORE_DEPS} libraw::raw_r TimeProvider JSON StrConv)

# Platform specific dependencies
if (WIN32)
    target_link_libraries(ImageBuffer PUBLIC ${IMAGE_BUFFER_DEPS} easy_profiler xxHash::xxhash)
elseif(PUERHLAB_METAL_ENABLED)
    target_link_libraries(ImageBuffer PUBLIC ${IMAGE_BUFFER_DEPS} xxHash MetalImage MetalUtils)
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
        decoders/processor/operators/gpu/metal_xtrans_interpolate.cpp
    )
endif()

# RawProcessorOp Target
def_library(RawProcessorOp
    SOURCES ${RAW_OP_CPU_SRCS} ${RAW_OP_GPU_SRCS}
    PUBLIC_DEPS ImageBuffer libraw::raw_r Exiv2 hwy::hwy Eigen3::Eigen
)

if(PUERHLAB_CUDA_ENABLED)
    set_target_properties(RawProcessorOp PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    set_target_properties(RawProcessorOp PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
    target_compile_definitions(RawProcessorOp PUBLIC HAVE_CUDA)
endif()
if(PUERHLAB_METAL_ENABLED)
    add_dependencies(RawProcessorOp RawProcessorOpMetalShaders)
    target_compile_definitions(RawProcessorOp PRIVATE
        PUERHLAB_METAL_TO_LINEAR_REF_METALLIB_PATH="${METAL_TO_LINEAR_REF_METALLIB}"
        PUERHLAB_METAL_DEBAYER_RCD_METALLIB_PATH="${METAL_DEBAYER_RCD_METALLIB}"
        PUERHLAB_METAL_HIGHLIGHT_RECONSTRUCT_METALLIB_PATH="${METAL_HIGHLIGHT_RECONSTRUCT_METALLIB}"
        PUERHLAB_METAL_XTRANS_INTERPOLATE_METALLIB_PATH="${METAL_XTRANS_INTERPOLATE_METALLIB}"
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
    PRIVATE_DEPS ${PUERHLAB_OPENCV_IMAGE_DEPS} RawProcessor
)
if(PUERHLAB_CUDA_ENABLED)
    set_target_properties(Operators PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
endif()
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
    edit/scope/scope_analyzer.cpp
    edit/scope/final_display_frame_tap.cpp
)

if(PUERHLAB_CUDA_ENABLED)
    set(GPU_PIPELINE_SRCS
        edit/pipeline/pipeline_gpu_impl.cu
        edit/scope/cuda_scope_analyzer.cu
    )
endif()
if(PUERHLAB_METAL_ENABLED)
    list(APPEND GPU_PIPELINE_SRCS
        edit/pipeline/pipeline_metal_impl.cpp
    )
endif()

def_library(EditPipeline
    SOURCES ${PIPELINE_SRCS} ${GPU_PIPELINE_SRCS}
    PUBLIC_DEPS Operators
    PRIVATE_DEPS ThreadPool opencv_imgproc
)

if(PUERHLAB_CUDA_ENABLED)
    set_target_properties(EditPipeline PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    set_target_properties(EditPipeline PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
    target_compile_definitions(EditPipeline PUBLIC HAVE_CUDA)
    target_include_directories(EditPipeline PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
    target_link_libraries(EditPipeline PRIVATE CUDA::cudart)
endif()
if(PUERHLAB_METAL_ENABLED)
    add_dependencies(EditPipeline EditPipelineMetalShaders)
    target_compile_definitions(EditPipeline PUBLIC HAVE_METAL)
    target_compile_definitions(EditPipeline PRIVATE
        PUERHLAB_METAL_FUSED_PIPELINE_METALLIB_PATH="${METAL_FUSED_PIPELINE_METALLIB}"
    )
    target_link_libraries(EditPipeline PRIVATE MetalImage MetalContext)
endif()

set(RENDERER_SRCS renderer/pipeline_scheduler.cpp)
def_library(PipelineScheduler
    SOURCES ${RENDERER_SRCS}
    PUBLIC_DEPS EditPipeline ThreadPool ImageLoader)

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
    sleeve/storage_service.cpp
    storage/controller/controller_types.cpp
    storage/controller/db_controller.cpp
    storage/controller/image/image_controller.cpp
    storage/controller/sleeve/element_controller.cpp
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

# ==============================================================================
# 7. Filesystem & View Manager
# ==============================================================================

set(FS_SRCS
    sleeve/sleeve_base.cpp
    sleeve/path_resolver.cpp
    sleeve/dentry_cache_manager.cpp
    sleeve/sleeve_filesystem.cpp
)

def_library(SleeveFS
    SOURCES ${FS_SRCS}
    PUBLIC_DEPS SleeveElement StorageService
)

def_library(SleeveManager
    SOURCES sleeve/sleeve_manager.cpp
    PUBLIC_DEPS SleeveFS PipelineScheduler
)

# ==============================================================================
# 8. I/O  
# ==============================================================================

if(PUERHLAB_ULTRAHDR_AVAILABLE)
    def_library(UltraHdrWriter
        SOURCES io/image/ultra_hdr_writer.cpp
        PUBLIC_DEPS OpenImageIO::OpenImageIO ImageBuffer Exiv2 puerhlab::uhdr
        PRIVATE_DEPS opencv_core
    )

    if(APPLE)
        target_link_libraries(UltraHdrWriter PRIVATE "-framework CoreGraphics" "-framework CoreFoundation")
    endif()
endif()

def_library(ImageWriter
    SOURCES io/image/image_writer.cpp
    PUBLIC_DEPS OpenImageIO::OpenImageIO ImageBuffer
    PRIVATE_DEPS ${PUERHLAB_OPENCV_IMAGE_DEPS}
)
if(PUERHLAB_ULTRAHDR_AVAILABLE)
    target_link_libraries(ImageWriter PUBLIC UltraHdrWriter)
    target_compile_definitions(ImageWriter PUBLIC PUERHLAB_HAS_ULTRAHDR=1)
endif()

if(APPLE)
    target_link_libraries(ImageWriter PRIVATE "-framework CoreGraphics" "-framework CoreFoundation")
endif()

# ==============================================================================
# 9. Services
# ==============================================================================
def_library(FSService
    SOURCES app/sleeve_service.cpp
    PUBLIC_DEPS SleeveFS EditHistory
)

def_library(AlbumBrowseService
    SOURCES app/album_browse_service.cpp
    PUBLIC_DEPS FSService SleeveElement
)


def_library(ImagePoolService
    SOURCES app/image_pool_service.cpp
    PUBLIC_DEPS ImagePool Image StorageService
)

def_library(ProjectService
    SOURCES app/project_service.cpp
    PUBLIC_DEPS FSService StorageService ImagePoolService AlbumBrowseService
                StrConv JSON
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
    PUBLIC_DEPS SleeveManager ImageWriter ImagePoolService FSService
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
if(PUERHLAB_REAL_WIDGET_EDITOR)
    set(EDIT_VIEWER_COMMON_SRCS
        ui/edit_viewer/edit_viewer.cpp
        ui/edit_viewer/edit_viewer_overlay_geometry.cpp
        ui/edit_viewer/viewport_mapper.cpp
        ui/edit_viewer/crop_geometry.cpp
        ui/edit_viewer/view_transform_controller.cpp
        ui/edit_viewer/crop_interaction_controller.cpp
        include/ui/edit_viewer/edit_viewer.hpp
        include/ui/edit_viewer/edit_viewer_overlay_geometry.hpp
        include/ui/edit_viewer/edit_viewer_surface.hpp
        include/ui/edit_viewer/color_manager.hpp
        include/ui/edit_viewer/viewer_state.hpp
        include/ui/edit_viewer/viewport_mapper.hpp
        include/ui/edit_viewer/crop_geometry.hpp
        include/ui/edit_viewer/view_transform_controller.hpp
        include/ui/edit_viewer/crop_interaction_controller.hpp
    )

    set(EDIT_VIEWER_BACKEND_SRCS)
    set(EDIT_VIEWER_PUBLIC_DEPS EditPipeline Qt6::Widgets)

    if(PUERHLAB_REAL_OPENGL_EDITOR)
        list(APPEND EDIT_VIEWER_BACKEND_SRCS
            ui/edit_viewer/frame_mailbox.cpp
            ui/edit_viewer/gl_edit_viewer_surface.cpp
            ui/edit_viewer/opengl_viewer_renderer.cpp
            include/ui/edit_viewer/frame_mailbox.hpp
            include/ui/edit_viewer/gl_edit_viewer_surface.hpp
        )
        list(APPEND EDIT_VIEWER_PUBLIC_DEPS OpenGL::GL Qt6::OpenGLWidgets)
    endif()

    if(PUERHLAB_RHI_WIDGET_EDITOR)
        list(APPEND EDIT_VIEWER_BACKEND_SRCS
            ui/edit_viewer/color_manager.cpp
            ui/edit_viewer/rhi_edit_viewer_surface.cpp
            include/ui/edit_viewer/rhi_edit_viewer_surface.hpp
        )
        if(APPLE)
            list(REMOVE_ITEM EDIT_VIEWER_BACKEND_SRCS ui/edit_viewer/color_manager.cpp)
            list(APPEND EDIT_VIEWER_BACKEND_SRCS ui/edit_viewer/color_manager.mm)
            set_source_files_properties(ui/edit_viewer/color_manager.mm PROPERTIES LANGUAGE OBJCXX)
        endif()
        list(APPEND EDIT_VIEWER_PUBLIC_DEPS Qt6::GuiPrivate Qt6::ShaderTools)
    endif()

    def_library(EditViewer
        SOURCES ${EDIT_VIEWER_COMMON_SRCS} ${EDIT_VIEWER_BACKEND_SRCS}
        PUBLIC_DEPS ${EDIT_VIEWER_PUBLIC_DEPS}
    )

    if(PUERHLAB_CUDA_ENABLED)
        set_target_properties(EditViewer PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
        target_compile_definitions(EditViewer PUBLIC HAVE_CUDA)
    endif()

    if(PUERHLAB_RHI_WIDGET_EDITOR AND APPLE)
        target_link_libraries(EditViewer PRIVATE
            "-framework AppKit"
            "-framework QuartzCore"
            "-framework CoreGraphics"
        )
    endif()
    if(PUERHLAB_RHI_WIDGET_EDITOR AND WIN32)
        target_link_libraries(EditViewer PRIVATE d3d11 dxgi)
    endif()

    if(PUERHLAB_RHI_WIDGET_EDITOR)
        qt_add_shaders(EditViewer "editviewer_rhi_shaders"
            PREFIX
                "/shaders/edit_viewer"
            BASE
                "ui/edit_viewer/shaders"
            FILES
                ui/edit_viewer/shaders/rhi_image.vert
                ui/edit_viewer/shaders/rhi_image.frag
        )
    endif()
endif()

# ── AlbumBackendLib: reusable static library for album_backend ──────────────
add_library(AlbumBackendLib STATIC
    ui/puerhlab_main/album_backend/album_backend.cpp
    ui/puerhlab_main/album_backend/path_utils.cpp
    app/project_package_backend.cpp
    ui/puerhlab_main/album_backend/project_handler.cpp
    app/project_package_service.cpp
    ui/puerhlab_main/album_backend/thumbnail_manager.cpp
    ui/puerhlab_main/album_backend/folder_controller.cpp
    ui/puerhlab_main/album_backend/image_controller.cpp
    ui/puerhlab_main/album_backend/stats_engine.cpp
    ui/puerhlab_main/album_backend/import_export.cpp
    ui/puerhlab_main/album_backend/editor_controller.cpp
    include/ui/puerhlab_main/album_backend/album_backend.hpp
    include/ui/puerhlab_main/album_backend/album_types.hpp
    include/ui/puerhlab_main/album_backend/path_utils.hpp
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
        UiLocalization
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

if(PUERHLAB_REAL_WIDGET_EDITOR)
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
        ui/puerhlab_main/editor_dialog/scope/scope_panel.cpp
        ui/puerhlab_main/editor_dialog/scope/scope_plot_widgets.cpp
        ui/puerhlab_main/editor_dialog/scope/scope_renderer.cpp
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
        ui/puerhlab_main/editor_dialog/scope/scope_panel.hpp
        ui/puerhlab_main/editor_dialog/scope/scope_plot_widgets.hpp
        ui/puerhlab_main/editor_dialog/scope/scope_renderer.hpp
        ui/puerhlab_main/editor_dialog/widgets/spinner.hpp
        ui/puerhlab_main/editor_dialog/widgets/histogram_widget.hpp
        ui/puerhlab_main/editor_dialog/widgets/trackball.hpp
        ui/puerhlab_main/editor_dialog/widgets/tone_curve_widget.hpp
        ui/puerhlab_main/editor_dialog/widgets/history_cards.hpp
    )
endif()

# ── puerhlab_main executable ────────────────────────────────────────────────
set(PUERHLAB_MAIN_SOURCES
    ui/puerhlab_main/main.cpp
    ui/puerhlab_main/app_theme.cpp
    config/resource.qrc
    include/ui/puerhlab_main/app_theme.hpp
    include/ui/puerhlab_main/i18n.hpp
    include/ui/puerhlab_main/language_manager.hpp
    ${PUERHLAB_EDITOR_DIALOG_SRCS}
)
if(APPLE AND PUERHLAB_MACOS_BUNDLE)
    add_executable(puerhlab_main MACOSX_BUNDLE ${PUERHLAB_MAIN_SOURCES})
else()
    add_executable(puerhlab_main ${PUERHLAB_MAIN_SOURCES})
endif()

if(APPLE)
    set(PUERHLAB_MAIN_ICON_PNG "${CMAKE_CURRENT_SOURCE_DIR}/config/ICON/unnamed.png")
    set(PUERHLAB_MAIN_ICONSET_DIR "${CMAKE_CURRENT_BINARY_DIR}/puerhlab_main.iconset")
    set(PUERHLAB_MAIN_ICON_ICNS "${CMAKE_CURRENT_BINARY_DIR}/puerhlab_main.icns")

    find_program(PUERHLAB_SIPS_EXECUTABLE sips REQUIRED)
    find_program(PUERHLAB_ICONUTIL_EXECUTABLE iconutil REQUIRED)

    add_custom_command(
        OUTPUT "${PUERHLAB_MAIN_ICON_ICNS}"
        COMMAND ${CMAKE_COMMAND} -E rm -rf "${PUERHLAB_MAIN_ICONSET_DIR}"
        COMMAND ${CMAKE_COMMAND} -E make_directory "${PUERHLAB_MAIN_ICONSET_DIR}"
        COMMAND ${PUERHLAB_SIPS_EXECUTABLE} -z 16 16 "${PUERHLAB_MAIN_ICON_PNG}" --out "${PUERHLAB_MAIN_ICONSET_DIR}/icon_16x16.png"
        COMMAND ${PUERHLAB_SIPS_EXECUTABLE} -z 32 32 "${PUERHLAB_MAIN_ICON_PNG}" --out "${PUERHLAB_MAIN_ICONSET_DIR}/icon_16x16@2x.png"
        COMMAND ${PUERHLAB_SIPS_EXECUTABLE} -z 32 32 "${PUERHLAB_MAIN_ICON_PNG}" --out "${PUERHLAB_MAIN_ICONSET_DIR}/icon_32x32.png"
        COMMAND ${PUERHLAB_SIPS_EXECUTABLE} -z 64 64 "${PUERHLAB_MAIN_ICON_PNG}" --out "${PUERHLAB_MAIN_ICONSET_DIR}/icon_32x32@2x.png"
        COMMAND ${PUERHLAB_SIPS_EXECUTABLE} -z 128 128 "${PUERHLAB_MAIN_ICON_PNG}" --out "${PUERHLAB_MAIN_ICONSET_DIR}/icon_128x128.png"
        COMMAND ${PUERHLAB_SIPS_EXECUTABLE} -z 256 256 "${PUERHLAB_MAIN_ICON_PNG}" --out "${PUERHLAB_MAIN_ICONSET_DIR}/icon_128x128@2x.png"
        COMMAND ${PUERHLAB_SIPS_EXECUTABLE} -z 256 256 "${PUERHLAB_MAIN_ICON_PNG}" --out "${PUERHLAB_MAIN_ICONSET_DIR}/icon_256x256.png"
        COMMAND ${PUERHLAB_SIPS_EXECUTABLE} -z 512 512 "${PUERHLAB_MAIN_ICON_PNG}" --out "${PUERHLAB_MAIN_ICONSET_DIR}/icon_256x256@2x.png"
        COMMAND ${PUERHLAB_SIPS_EXECUTABLE} -z 512 512 "${PUERHLAB_MAIN_ICON_PNG}" --out "${PUERHLAB_MAIN_ICONSET_DIR}/icon_512x512.png"
        COMMAND ${PUERHLAB_SIPS_EXECUTABLE} -z 1024 1024 "${PUERHLAB_MAIN_ICON_PNG}" --out "${PUERHLAB_MAIN_ICONSET_DIR}/icon_512x512@2x.png"
        COMMAND ${PUERHLAB_ICONUTIL_EXECUTABLE} -c icns "${PUERHLAB_MAIN_ICONSET_DIR}" -o "${PUERHLAB_MAIN_ICON_ICNS}"
        DEPENDS "${PUERHLAB_MAIN_ICON_PNG}"
        COMMENT "Generating macOS app icon from ${PUERHLAB_MAIN_ICON_PNG}"
        VERBATIM
    )

    add_custom_target(PuerhLabMainIcon DEPENDS "${PUERHLAB_MAIN_ICON_ICNS}")
    add_dependencies(puerhlab_main PuerhLabMainIcon)
    set_source_files_properties("${PUERHLAB_MAIN_ICON_ICNS}" PROPERTIES MACOSX_PACKAGE_LOCATION "Resources")
    target_sources(puerhlab_main PRIVATE "${PUERHLAB_MAIN_ICON_ICNS}")
    set_target_properties(puerhlab_main PROPERTIES MACOSX_BUNDLE_ICON_FILE "puerhlab_main.icns")
endif()

if(WIN32)
    set(PUERHLAB_MAIN_ICON_RC "${CMAKE_CURRENT_SOURCE_DIR}/config/ICON/puerhlab_main.rc")
    if(EXISTS "${PUERHLAB_MAIN_ICON_RC}")
        target_sources(puerhlab_main PRIVATE "${PUERHLAB_MAIN_ICON_RC}")
    else()
        message(WARNING "Windows app icon resource not found: ${PUERHLAB_MAIN_ICON_RC}")
    endif()
endif()

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
        UiLocalization
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
set_target_properties(puerhlab_main PROPERTIES
    QT_QMLCACHEGEN_ARGUMENTS "--only-bytecode"
)
if(APPLE)
    set_target_properties(puerhlab_main PROPERTIES
        MACOSX_BUNDLE ${PUERHLAB_MACOS_BUNDLE}
    )
endif()

set(PUERHLAB_TRANSLATION_TS_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/puerhlab_main/i18n/puerhlab_main_en.ts
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/puerhlab_main/i18n/puerhlab_main_zh_CN.ts
)

qt_add_translations(TARGETS puerhlab_main
    SOURCE_TARGETS
        puerhlab_main
        AlbumBackendLib
        UiLocalization
    TS_FILES
        ${PUERHLAB_TRANSLATION_TS_FILES}
    RESOURCE_PREFIX
        "/i18n"
    QM_OUTPUT_DIRECTORY
        "${CMAKE_CURRENT_BINARY_DIR}"
)

if(COMMAND qt_finalize_target)
    qt_finalize_target(puerhlab_main)
elseif(COMMAND qt_finalize_executable)
    qt_finalize_executable(puerhlab_main)
endif()

target_compile_definitions(puerhlab_main PRIVATE PUERHLAB_SOURCE_DIR="${CMAKE_SOURCE_DIR}")

if(DEFINED PUERHLAB_LENSFUN_LIBRARY_DIR AND NOT "${PUERHLAB_LENSFUN_LIBRARY_DIR}" STREQUAL "")
    set_property(TARGET puerhlab_main APPEND PROPERTY BUILD_RPATH "${PUERHLAB_LENSFUN_LIBRARY_DIR}")
endif()

set(PUERHLAB_MAIN_FONT_SRC "${CMAKE_SOURCE_DIR}/pu-erh_lab/src/config/fonts/main_Inter.ttf")
set(PUERHLAB_MAIN_ZH_FONT_SRC "${CMAKE_SOURCE_DIR}/pu-erh_lab/src/config/fonts/main_NotoSans_zh.ttf")
if(APPLE AND PUERHLAB_MACOS_BUNDLE)
    # Inside .app bundle, resources go next to the executable in Contents/MacOS/
    add_custom_command(TARGET puerhlab_main POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_BUNDLE_CONTENT_DIR:puerhlab_main>/MacOS/fonts"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "${PUERHLAB_MAIN_FONT_SRC}" "$<TARGET_BUNDLE_CONTENT_DIR:puerhlab_main>/MacOS/fonts/main_Inter.ttf"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "${PUERHLAB_MAIN_ZH_FONT_SRC}" "$<TARGET_BUNDLE_CONTENT_DIR:puerhlab_main>/MacOS/fonts/main_NotoSans_zh.ttf"
        VERBATIM
    )
else()
    add_custom_command(TARGET puerhlab_main POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_FILE_DIR:puerhlab_main>/fonts"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "${PUERHLAB_MAIN_FONT_SRC}" "$<TARGET_FILE_DIR:puerhlab_main>/fonts/main_Inter.ttf"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "${PUERHLAB_MAIN_ZH_FONT_SRC}" "$<TARGET_FILE_DIR:puerhlab_main>/fonts/main_NotoSans_zh.ttf"
        VERBATIM
    )
endif()

if(APPLE AND PUERHLAB_METAL_ENABLED AND PUERHLAB_METAL_RUNTIME_LIBS)
    if(PUERHLAB_MACOS_BUNDLE)
        set(_puerhlab_metallib_runtime_dir "$<TARGET_BUNDLE_CONTENT_DIR:puerhlab_main>/Resources/metallib")
    else()
        set(_puerhlab_metallib_runtime_dir "$<TARGET_FILE_DIR:puerhlab_main>/metallib")
    endif()

    add_custom_command(TARGET puerhlab_main POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory "${_puerhlab_metallib_runtime_dir}"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${PUERHLAB_METAL_RUNTIME_LIBS}
            "${_puerhlab_metallib_runtime_dir}"
        COMMAND_EXPAND_LISTS
        VERBATIM
    )
endif()

if(WIN32 AND TARGET puerhlab_lensfun)
    add_custom_command(TARGET puerhlab_main POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:puerhlab_lensfun>"
            "$<TARGET_FILE_DIR:puerhlab_main>/lensfun.dll"
        VERBATIM
    )
endif()
