cmake_minimum_required(VERSION 3.0.0)
project(task2 VERSION 0.1.0 LANGUAGES C)

find_package(OpenMP)

include(CTest)
enable_testing()

add_executable(task2 main.c)

target_link_libraries(task2 PRIVATE m)

target_link_libraries(task2 PRIVATE "${OpenMP_C_FLAGS}")
target_compile_options(task2 PRIVATE "${OpenMP_C_FLAGS}")

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)