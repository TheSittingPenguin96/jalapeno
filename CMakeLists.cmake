cmake_minimum_required(VERSION 3.18)
project(jalapeno VERSION 0.1.0 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find CUDA
find_package(CUDAToolkit REQUIRED)

# Find Python
find_package(Python 3.8 REQUIRED COMPONENTS Development)

# Library options
option(JALAPENO_BUILD_TESTS "Build tests" ON)
option(JALAPENO_BUILD_EXAMPLES "Build examples" ON)
option(JALAPENO_USE_NCCL "Use NCCL for multi-GPU" OFF)
option(JALAPENO_ENABLE_TRACING "Enable performance tracing" ON)

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${Python_INCLUDE_DIRS}
    ${CUDAToolkit_INCLUDE_DIRS}
)

# Core library
add_library(jalapeno_core SHARED
    src/core/tensor.cpp
    src/core/memory_manager.cpp
    src/core/paging_policy.cpp
    src/core/async_executor.cpp
    src/core/hardware_profiler.cpp
    src/core/layer_streamer.cpp
    src/core/prefetch_predictor.cpp
    src/core/execution_planner.cpp
)

# CUDA components
add_library(jalapeno_cuda STATIC
    src/cuda/kernels.cu
    src/cuda/memory_ops.cu
    src/cuda/quantization.cu
)

target_link_libraries(jalapeno_core
    PRIVATE
    jalapeno_cuda
    ${CUDAToolkit_LIBRARIES}
    ${Python_LIBRARIES}
    pthread
    dl
    rt
)

# Python extension
pybind11_add_module(jalapeno_python
    python/pybind_module.cpp
    python/pybind_tensor.cpp
    python/pybind_runtime.cpp
    python/pybind_layer_streamer.cpp
)

target_link_libraries(jalapeno_python
    PRIVATE
    jalapeno_core
)

# Install
install(TARGETS jalapeno_core jalapeno_python
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)

install(DIRECTORY include/jalapeno
    DESTINATION include
)

# Tests
if(JALAPENO_BUILD_TESTS)
    add_executable(test_jalapeno
        tests/test_tensor.cpp
        tests/test_memory_manager.cpp
    )
    target_link_libraries(test_jalapeno jalapeno_core)
    
    add_test(NAME tensor_tests COMMAND test_jalapeno)
endif()

# Examples
if(JALAPENO_BUILD_EXAMPLES)
    add_executable(example_layer_streaming
        examples/layer_streaming.cpp
    )
    target_link_libraries(example_layer_streaming jalapeno_core)
endif()
