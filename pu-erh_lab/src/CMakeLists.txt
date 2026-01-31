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

def_library(ThreadPool SOURCES concurrency/thread_pool.cpp)

def_library(TimeProvider SOURCES utils/clock/time_provider.cpp)

def_library(StrConv 
    SOURCES utils/string/converter.cpp 
    PUBLIC_DEPS utfcpp
)

# ==============================================================================
# 2. Core Image Handling 
# ==============================================================================

# ImageBuffer (Platform specific deps handling)
set(IMAGE_BUFFER_SRCS image/image_buffer.cpp)
add_library(ImageBuffer ${IMAGE_BUFFER_SRCS})
target_include_directories(ImageBuffer PUBLIC include)

# Common dependencies
set(IMAGE_BUFFER_DEPS Exiv2 ${OpenCV_LIBS} libraw::raw_r TimeProvider JSON OpenMP::OpenMP_CXX StrConv)

# Platform specific dependencies
if (WIN32)
    target_link_libraries(ImageBuffer PUBLIC ${IMAGE_BUFFER_DEPS} easy_profiler xxHash::xxhash)
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

if(CUDAToolkit_FOUND)
    set(RAW_OP_GPU_SRCS
        decoders/processor/operators/gpu/cuda_color_space_conv.cu
        decoders/processor/operators/gpu/cuda_debayer_ahd.cu
        decoders/processor/operators/gpu/cuda_highlight_reconstruct.cu
        decoders/processor/operators/gpu/cuda_white_balance.cu
    )
endif()

# RawProcessorOp Target
def_library(RawProcessorOp
    SOURCES ${RAW_OP_CPU_SRCS} ${RAW_OP_GPU_SRCS}
    PUBLIC_DEPS ImageBuffer libraw::raw_r Exiv2 hwy::hwy Eigen3::Eigen
)

if(CUDAToolkit_FOUND)
    set_target_properties(RawProcessorOp PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    target_compile_definitions(RawProcessorOp PUBLIC HAVE_CUDA)
endif()

# --- Edit Operators ---
set(OPERATORS_SRCS
    edit/operators/operator_factory.cpp
    edit/operators/basic/exposure_op.cpp
    edit/operators/basic/exposure_op_vec.cpp
    edit/operators/basic/contrast_op.cpp
    edit/operators/basic/black_op.cpp
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
    edit/operators/cst/odt_op.cpp
    edit/operators/raw/raw_decode_op.cpp
    edit/operators/geometry/resize_op.cpp
    edit/operators/operator_registeration.cpp
)

def_library(Operators
    SOURCES ${OPERATORS_SRCS}
    PUBLIC_DEPS ImageBuffer JSON OpenColorIO::OpenColorIO hwy::hwy glad::glad
)

# --- Edit Pipeline ---
set(PIPELINE_SRCS
    edit/pipeline/pipeline_cpu.cpp
    edit/pipeline/pipeline_stage.cpp
    
)

if(CUDAToolkit_FOUND)
    set(GPU_PIPELINE_SRCS
        edit/pipeline/pipeline_gpu_impl.cu
    )
endif()

def_library(EditPipeline
    SOURCES ${PIPELINE_SRCS} ${GPU_PIPELINE_SRCS}
    PUBLIC_DEPS Operators glad::glad OpenGL::GL
)

if(CUDAToolkit_FOUND)
    set_target_properties(EditPipeline PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    target_compile_definitions(EditPipeline PUBLIC HAVE_CUDA)
endif()

set(RENDERER_SRCS renderer/pipeline_scheduler.cpp)
def_library(PipelineScheduler
    SOURCES ${RENDERER_SRCS}
    PUBLIC_DEPS EditPipeline ThreadPool)

# --- Raw Processor Wrapper ---
def_library(RawProcessor
    SOURCES decoders/processor/raw_processor.cpp
    PUBLIC_DEPS Image libraw::raw_r Exiv2 RawProcessorOp
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
    PUBLIC_DEPS Image ThreadPool ${OpenCV_LIBS} libraw::raw_r RawProcessor
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
)

# ==============================================================================
# 9. Services
# ==============================================================================
def_library(FSService
    SOURCES app/sleeve_service.cpp
    PUBLIC_DEPS SleeveFS EditHistory ImageWriter 
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

def_library(ImportService
    SOURCES app/import_service.cpp
    PUBLIC_DEPS SleeveManager ImageWriter ImagePoolService
)

def_library(ThumbnailService
    SOURCES app/thumbnail_service.cpp
    PUBLIC_DEPS EditPipeline ThreadPool StrConv ImagePoolService PipelineMgmtService
)

def_library(SleeveFilterService
    SOURCES app/sleeve_filter_service.cpp
    PUBLIC_DEPS SleeveElement StorageService SleeveFilter StrConv
)

# ==============================================================================
# 9. UI Components
# ==============================================================================
def_library(EditViewer
    SOURCES ui/edit_viewer/edit_viewer.cpp include/ui/edit_viewer/edit_viewer.hpp
    PUBLIC_DEPS EditPipeline glad::glad OpenGL::GL Qt6::OpenGLWidgets
)

if(CUDAToolkit_FOUND)
    set_target_properties(EditViewer PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    target_compile_definitions(EditViewer PUBLIC HAVE_CUDA)
endif()

