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
############################################################
# Create an executable for kmeans Unit Test
############################################################
SET( google-unit-test-kmean-clustering StaticTest )
add_executable(google-unit-test-kmean-clustering
        ${PROJECT_SOURCE_DIR}/test/unit-test-kmeans-clustering.cpp
        )
############################################################
# Target Link Libaries
############################################################
# link the new random_forest_library target with the hello_binary target
target_link_libraries(test-youtube-random-forest-model
        PRIVATE
        random_forest_library
        youtube_fun_library
        ${PYTHON_LIBRARIES}  ${ARMADILLO_LIBRARIES} libmlpack.so
        )

target_link_libraries(google-unit-test-youtube-random-forest-model
        PRIVATE
        random_forest_library
        youtube_fun_library
        PUBLIC
        ${GTEST_LIBRARIES} ${PYTHON_LIBRARIES}  ${ARMADILLO_LIBRARIES} libmlpack.so pthread
        )

