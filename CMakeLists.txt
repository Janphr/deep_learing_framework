cmake_minimum_required(VERSION 3.10)
project(framework)

set(CMAKE_CXX_STANDARD 14)

find_package(Eigen3 REQUIRED)
find_package(OpenCV)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/build)

include_directories(
        include
        ${EIGEN3_INCLUDE_DIRS}
        ${OpenCV_INCLUDE_DIRS}
)

list(APPEND SOURCEFILES
        src/Tensor.cpp
        src/InputLayer.cpp
        src/FullyConnectedLayer.cpp
        src/SoftmaxLayer.cpp
        src/ActivationLayer.cpp
        src/SigmoidLayer.cpp
        src/CrossEntropyLayer.cpp
        src/MeanSquaredErrorLayer.cpp
        src/SGDTrainer.cpp
        src/Network.cpp
        src/Conv2DLayer.cpp
        src/MaxPooling.cpp
        )

file(GLOB_RECURSE all_include_files RELATIVE "${CMAKE_SOURCE_DIR}" *.h *.hpp)

add_custom_target(framework_INCLUDES DEPENDS ${all_include_files} SOURCES ${all_include_files} )

add_library(framework SHARED ${SOURCEFILES})

target_link_libraries(framework ${LINK_LIBS} ${OpenCV_LIBS})

add_executable(dl_framework app/conv_mnist.cpp)

target_link_libraries(dl_framework framework)
