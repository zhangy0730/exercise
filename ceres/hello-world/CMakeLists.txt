cmake_minimum_required(VERSION 2.8)
project(hello_world)


set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS "-std=c++14 -O3")

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)


# Ceres
find_package(Ceres REQUIRED)
include_directories(${CERES_INCLUDE_DIRS})

# Eigen
include_directories("/usr/local/include/eigen3")


# set(PROJECT_COMMON_LIB
#     ${G2O_CORE_LIBRARY}
#     ${G2O_STUFF_LIBRARY}
#     ${G2O_SOLVER_CSPARSE}
#     ${G2O_TYPES_SBA}
#     ${G2O_SOLVER_CSPARSE_EXTENSION}
#     ${CSPARSE_LIBRARY}
#     ${GLOG_LIBRARIES}
#     ${OpenCV_LIBRARIES}
#     ${Pangolin_LIBRARIES}
#     ${YAML_CPP_LIBRARIES})

add_executable(hello_world main.cpp)
target_link_libraries(hello_world ${OpenCV_LIBS} ${CERES_LIBRARIES})

add_executable(powell powell.cpp)
target_link_libraries(powell ${OpenCV_LIBS} ${CERES_LIBRARIES})

add_executable(curve_fitting curve_fitting.cpp)
target_link_libraries(curve_fitting ${OpenCV_LIBS} ${CERES_LIBRARIES})
