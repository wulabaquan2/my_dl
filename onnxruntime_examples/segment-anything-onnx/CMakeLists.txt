# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
cmake_minimum_required(VERSION 3.26)
project(ONNXRUNTIME_ROOTDIR)
find_package(OpenCV)

add_executable(model-explorer model-explorer.cpp)
target_link_libraries(model-explorer PRIVATE ${OpenCV_LIBS})
option(ONNX_GPU "enable gpu"  OFF)
if(ONNX_GPU)
#cuda12不支持onnxruntime-1.14.1,改为onnxruntime-1.21.0(仅用于测试GPU时间)
    add_definitions(-DUSE_CUDA)
    find_package(CUDA REQUIRED)
    target_link_libraries(model-explorer PRIVATE ${CUDA_LIBRARIES})
    include_directories(${CUDA_INCLUDE_DIRS})
    set(ONNXRUNTIME_ROOTDIR "D:/onnxruntime-win-x64-gpu-1.21.0")
    target_link_libraries(model-explorer PRIVATE "${ONNXRUNTIME_ROOTDIR}/lib/onnxruntime.lib")
    target_link_libraries(model-explorer PRIVATE "${ONNXRUNTIME_ROOTDIR}/lib/onnxruntime_providers_cuda.lib")
    target_link_libraries(model-explorer PRIVATE "${ONNXRUNTIME_ROOTDIR}/lib/onnxruntime_providers_shared.lib")
    target_link_libraries(model-explorer PRIVATE "${ONNXRUNTIME_ROOTDIR}/lib/onnxruntime_providers_tensorrt.lib")
    
else()
    set(ONNXRUNTIME_ROOTDIR "D:/onnxruntime-win-x64-1.14.1")
    target_link_libraries(model-explorer PRIVATE "${ONNXRUNTIME_ROOTDIR}/lib/onnxruntime.lib")
    target_link_libraries(model-explorer PRIVATE "${ONNXRUNTIME_ROOTDIR}/lib/onnxruntime_providers_shared.lib")
endif()
target_include_directories(model-explorer PRIVATE "${ONNXRUNTIME_ROOTDIR}/include")

#TODO: fix the build error
#add_executable(batch-model-explorer batch-model-explorer.cpp)
#target_link_libraries(batch-model-explorer PRIVATE onnxruntime)
set_target_properties(model-explorer PROPERTIES
    VS_DEBUGGER_ENVIRONMENT "PATH=%PATH%;${ONNXRUNTIME_ROOTDIR}/lib;D:/opencv4100/opencv/build/x64/vc16/bin"
    VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
)

