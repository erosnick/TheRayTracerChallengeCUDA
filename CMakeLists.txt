cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

project(TheRayTracerChallengeCUDA LANGUAGES C CXX CUDA)

find_package(CUDA REQUIRED)

# Set CUDA flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-extended-lambda")

set(CMAKE_CXX_STANDARD 17)

# 用反斜杠转义宏值中的空格避免被截断
add_compile_definitions(HOST_DEVICE=__host__\ __device__)

# Add source files
file(GLOB CUDA_SOURCES "Src/*.cu")
file(GLOB CUDA_INCLUDES "Src/*.cuh")
file(GLOB CPP_SOURCES "Src/*.cpp")
file(GLOB CPP_INCLUDES "Src/*.h")
file(GLOB GLAD_SOURCES "ThirdParty/GLAD/src/glad.c")

# Add executable target
add_executable(TheRayTracerChallengeCUDA ${CUDA_SOURCES} ${CUDA_INCLUDES} ${CPP_SOURCES} ${CPP_INCLUDES} ${GLAD_SOURCES})

# Set target properties
set_target_properties(TheRayTracerChallengeCUDA
    PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "61"
)
target_include_directories(TheRayTracerChallengeCUDA PRIVATE ${CUDA_INCLUDE_DIRS})
target_include_directories(TheRayTracerChallengeCUDA PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/ThirdParty/CUDA/Common)
target_include_directories(TheRayTracerChallengeCUDA PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/ThirdParty/GLAD/include)
target_include_directories(TheRayTracerChallengeCUDA PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/ThirdParty/GLFW/include)
target_include_directories(TheRayTracerChallengeCUDA PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/ThirdParty/stb)
target_link_directories(TheRayTracerChallengeCUDA PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/ThirdParty/GLFW/lib)
target_link_libraries(TheRayTracerChallengeCUDA PRIVATE ${CUDA_LIBRARIES})
target_link_libraries(TheRayTracerChallengeCUDA PRIVATE "glfw3.lib")
