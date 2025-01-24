set(CMAKE_CXX_STANDARD 20)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_WARNINGS_FLAGS "-Wall -Wpedantic -Wextra")

include_directories(../../)
include_directories(../../fdaPDE/)
include_directories(../utils/)

find_package (Eigen3 3.4 REQUIRED NO_MODULE)

# json (for testing)
include(FetchContent)
FetchContent_Declare(json
        GIT_REPOSITORY https://github.com/nlohmann/json.git
        GIT_TAG v3.7.3)

FetchContent_GetProperties(json)
if(NOT json_POPULATED)
    FetchContent_Populate(json)
    add_subdirectory(${json_SOURCE_DIR} ${json_BINARY_DIR} EXCLUDE_FROM_ALL)
endif()

if(CMAKE_BUILD_TYPE MATCHES Debug)
    add_definitions(-DDEBUG)
    set(CMAKE_CXX_FLAGS "-std=c++20 ${CMAKE_CXX_FLAGS} ${CMAKE_WARNINGS_FLAGS} -I${CMAKE_CURRENT_SOURCE_DIR}/ -I${CMAKE_CURRENT_SOURCE_DIR}/fdaPDE-core/ -I${json_SOURCE_DIR}/include/")
else()
    set(CMAKE_CXX_FLAGS "-std=c++20 -O3 ${CMAKE_CXX_FLAGS} ${CMAKE_WARNINGS_FLAGS} -I${CMAKE_CURRENT_SOURCE_DIR}/ -I${CMAKE_CURRENT_SOURCE_DIR}/fdaPDE-core/ -I${json_SOURCE_DIR}/include/")
endif()


add_executable(rpchol main.cpp)
target_link_libraries (rpchol Eigen3::Eigen)