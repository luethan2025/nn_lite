cmake_minimum_required(VERSION 3.10)
project(nn_lite
        DESCRIPTION "Neural networks for Arduino hardware"
        LANGUAGES CXX)

if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  message(STATUS "You're using ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
  message(STATUS "You're using ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Include header files
include_directories(include)
include_directories(util/include)

# Compiling executables
add_executable(linear_test
  include/linear.h
  src/linear.cpp
  util/include/isclose.h
  util/src/isclose.cpp
  util/include/linear_test_util.h
  util/src/linear_test_util.cpp
  util/include/print_util.h
  util/src/print_util.cpp
  tests/linear_test.cpp)
