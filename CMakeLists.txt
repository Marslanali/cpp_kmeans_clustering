cmake_minimum_required(VERSION 3.5.1)
project(kmeans)

############################################################
# Find OS, Linux Kernel, Processor, Compiler, CMake Version
############################################################
message(STATUS "Operating System :" ${CMAKE_SYSTEM_NAME})
message(STATUS "Linux Kernel Version: " ${CMAKE_SYSTEM})
message(STATUS "System Processor: " ${CMAKE_SYSTEM_PROCESSOR})
message(STATUS "Compiler Version: " ${CMAKE_CXX_COMPILER})
message(STATUS "CMake Version :" ${CMAKE_SYSTEM_VERSION})

############################################################
# Set C++11
############################################################
set(CMAKE_CXX_STANDARD 11)

############################################################
# Find Armadillo, GTest package
############################################################
find_package(PythonLibs 2.7)
find_package(GTest REQUIRED)
find_package(Armadillo 4.04 REQUIRED)


############################################################
# Target Include Directories
############################################################
include_directories(include ${GTEST_INCLUDE_DIRS} ${PYTHON_INCLUDE_DIRS}
                            ${ARMADILLO_INCLUDE_DIRS} ${MLPACK_INCLUDE_DIRS})

############################################################
#  Create an executable for kmeans
############################################################
add_executable(test-keamn
        ${PROJECT_SOURCE_DIR}/test/test-kmeans-clustering.cpp
        )

