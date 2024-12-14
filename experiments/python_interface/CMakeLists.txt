cmake_minimum_required(VERSION 3.14)
project(test_RSVD LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_WARNINGS_FLAGS "-Wall -Wpedantic -Wextra")

#pybind11
set(pybind11_DIR /Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages/pybind11/share/cmake/pybind11)
find_package(pybind11 REQUIRED)

#Eigen
find_package (Eigen3 3.4 REQUIRED NO_MODULE)

#setting the flags
set(CMAKE_CXX_FLAGS "-O3 -std=c++20 ${OpenMP_CXX_FLAGS} ${CMAKE_CXX_FLAGS} ${CMAKE_WARNINGS_FLAGS} ")
set(CMAKE_SHARED_LIBRARY_CXX_FLAGS "${OpenMP_CXX_FLAGS} ${CMAKE_SHARED_LIBRARY_CXX_FLAGS}")

include_directories(../..)
include_directories(${json_SOURCE_DIR}/include/)

pybind11_add_module(randSVD randSVD_pymodule.cpp)
target_link_libraries(randSVD PRIVATE Eigen3::Eigen)

