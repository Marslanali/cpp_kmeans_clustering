cmake_minimum_required(VERSION 3.15)
project(kmeans)

set(CMAKE_CXX_STANDARD 11)

add_executable(kmeans main.cpp)

find_package(Armadillo 5.4 REQUIRED)
include_directories(include ${ARMADILLO_INCLUDE_DIRS} )

find_library(ARMADILLO_LIBRARIES armadillo)
target_link_libraries(kmeans ${ARMADILLO_LIBRARIES})

find_package(PythonLibs 2.7)
target_include_directories(kmeans PRIVATE ${PYTHON_INCLUDE_DIRS})
target_link_libraries(kmeans ${PYTHON_LIBRARIES})