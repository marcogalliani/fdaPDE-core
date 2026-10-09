# Tests of the execution module (from upstream codex/execution-foundation) and of the assertion macros, as separate
# executables: each execution test configures the runtime singleton of its own process.
#   cmake -S tests -B build/tests && cmake --build build/tests --parallel 4 && ctest --test-dir build/tests --output-on-failure
# On macOS with Homebrew GCC add -DCMAKE_OSX_SYSROOT="" (the SDK sysroot clashes with GCC's fixed headers).
cmake_minimum_required(VERSION 3.20)
project(fdapde_core_tests LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(POLICY CMP0135)
  cmake_policy(SET CMP0135 NEW)
endif()
include(FetchContent)
FetchContent_Declare(googletest
  URL https://github.com/google/googletest/archive/f8d7d77c06936315286eb55f8de22cd23c188571.zip)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)
find_package(Threads REQUIRED)

enable_testing()
include(GoogleTest)

function(fdapde_configure_target target)
  target_include_directories(${target} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/..")
  target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic -Werror)
  target_link_libraries(${target} PRIVATE GTest::gtest_main Threads::Threads)
endfunction()

add_executable(fdapde_assert_compat_test utility/assert_compat.cpp)
fdapde_configure_target(fdapde_assert_compat_test)
gtest_discover_tests(fdapde_assert_compat_test PROPERTIES TIMEOUT 30)

add_executable(fdapde_execution_test
  execution/runtime.cpp execution/task_graph.cpp execution/parallel_algorithms.cpp execution/queues.cpp)
add_executable(fdapde_execution_configuration_test execution/configuration.cpp)
add_executable(fdapde_execution_queue_capacity_test execution/queue_capacity.cpp)
foreach(target fdapde_execution_test fdapde_execution_configuration_test fdapde_execution_queue_capacity_test)
  fdapde_configure_target(${target})
  gtest_discover_tests(${target} PROPERTIES TIMEOUT 120)
endforeach()
