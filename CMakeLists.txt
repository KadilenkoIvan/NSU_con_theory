cmake_minimum_required(VERSION 3.0.0)
project(sin_sum VERSION 0.1.0 LANGUAGES C CXX)

include(CTest)
enable_testing()

add_executable(sin_sum main.cpp)

if(USE_FLOAT)
    add_definitions(-DFLOAT)
endif()

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)