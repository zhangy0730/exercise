cmake_minimum_required(VERSION 2.8)
project(BA)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS "-std=c++14 -O3")

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)


# Ceres
find_package(Ceres REQUIRED)
include_directories(${CERES_INCLUDE_DIRS})

# Eigen
include_directories("/usr/local/include/eigen3")

add_executable(bundle_adjustment main.cpp)
target_link_libraries(bundle_adjustment ${OpenCV_LIBS} ${CERES_LIBRARIES})