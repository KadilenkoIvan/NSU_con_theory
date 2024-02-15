cmake_minimum_required(VERSION 3.0.0)
project(sin_sum VERSION 0.1.0 LANGUAGES C CXX)

include(CTest)
enable_testing()

add_executable(sinsum main.cpp)

if(USE_FLOAT)
	target_compile_definitions(sinsum PRIVATE -DFLOAT=1)
else()
	target_compile_definitions(sinsum PRIVATE -DFLOAT=0)
endif()

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
