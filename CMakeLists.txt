cmake_minimum_required(VERSION 3.15)
project(ll-cuda-cpu)

set(CMAKE_CXX_STANDARD 17)

find_package(ZLIB REQUIRED)
find_package(Curses REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")

# cuda
option(ENABLE_CUDA "Enable CUDA support" ON)
if(ENABLE_CUDA)
    find_package(CUDA REQUIRED)
    include_directories(${CUDA_INCLUDE_DIRS})
    list(APPEND CUDA_LIBRARIES ${CUDA_CUDA_LIBRARY})
    add_definitions("-DHAVE_CUDA")
    set(CUST_DEFINITIONS "-DHAVE_CUDA")
else()
    set(CUDA_LIBRARIES "")
endif()

# llvm
set(llvm_config_cmd "llvm-config")
set(llvm_config_inc_arg "--includedir")
set(llvm_config_ld_arg "--ldflags")
set(llvm_config_lib_arg "--libs")
execute_process(COMMAND ${llvm_config_cmd} ${llvm_config_inc_arg}
        OUTPUT_VARIABLE LLVM_INC_FLAGS)
execute_process(COMMAND ${llvm_config_cmd} ${llvm_config_lib_arg}
        OUTPUT_VARIABLE LLVM_LIB_FLAGS)
execute_process(COMMAND ${llvm_config_cmd} ${llvm_config_ld_arg}
        OUTPUT_VARIABLE LLVM_LD_FLAGS)
include_directories(${LLVM_INC_FLAGS} "/usr/local/cuda/include/")
string(REPLACE "\n" " " LLVM_LINKER_FLAGS "${LLVM_LIB_FLAGS} ${LLVM_LD_FLAGS}")
string(STRIP "${LLVM_LINKER_FLAGS}" LLVM_LINKER_FLAGS)
set(llvm_clangpp_cmd clang++)

add_custom_command(
        DEPENDS cpu_runtime_functions.cpp
        OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/cpu_runtime_functions.bc
        COMMAND ${llvm_clangpp_cmd}
        ARGS -std=c++14 -O3 -c -emit-llvm
        ${CLANG_SDK_INC}
        ${CLANG_CRT_INC}
        ${CUST_DEFINITIONS}
        ${CMAKE_CURRENT_SOURCE_DIR}/cpu_runtime_functions.cpp
)

add_custom_target(
        cpu_runtime
        DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/cpu_runtime_functions.bc
)

if(ENABLE_CUDA)
    add_library(runtime cpu_runtime_functions.cpp cpu_runtime_functions.bc ${CMAKE_CURRENT_BINARY_DIR}/cuda_runtime_functions.a)
    add_dependencies(runtime cpu_runtime cuda_runtime)

    add_custom_command(
            DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/cuda_runtime_functions.o
            OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/cuda_runtime_functions.a
            COMMAND nvcc
            ARGS
            -Xcompiler -fPIC
            -D__STDC_LIMIT_MACROS
            -D__STDC_CONSTANT_MACROS
            -D_FORCE_INLINES
            ${CUST_DEFINITIONS}
            -arch sm_30
            -lib ${CMAKE_CURRENT_BINARY_DIR}/cuda_runtime_functions.o
            ${NVCC_BUILD_TYPE_ARGS}
            -o ${CMAKE_CURRENT_BINARY_DIR}/cuda_runtime_functions.a
    )

    add_custom_command(
            DEPENDS cuda_runtime_functions.cu
            OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/cuda_runtime_functions.o
            COMMAND nvcc
            ARGS
            -Xcompiler -fPIC
            -D__STDC_LIMIT_MACROS
            -D__STDC_CONSTANT_MACROS
            -D_FORCE_INLINES
            -Xcompiler -Wno-return-type-c-linkage --expt-relaxed-constexpr
            ${CUST_DEFINITIONS}
            --device-link
            -arch sm_30
            -std=c++14
            ${NVCC_BUILD_TYPE_ARGS}
            -c ${CMAKE_CURRENT_SOURCE_DIR}/cuda_runtime_functions.cu
    )

    add_custom_target(
            cuda_runtime
            DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/cuda_runtime_functions.a
    )
else()
    add_library(runtime cpu_runtime_functions.cpp cpu_runtime_functions.bc)
    add_dependencies(runtime cpu_runtime)
endif()

target_link_libraries(
        runtime
        ${LLVM_LINKER_FLAGS}
)

if(ENABLE_CUDA)
    add_executable(ll-cuda-cpu main.cpp cuda_backend.cpp cpu_backend.cpp)
    target_link_libraries(ll-cuda-cpu ${CUDA_LIBRARIES} ${LLVM_LINKER_FLAGS} ${CMAKE_DL_LIBS} ${CURSES_LIBRARIES} ${ZLIB_LIBRARIES} ${LOCALE_LINK_FLAG} runtime)
else()
    add_executable(ll-cuda-cpu main.cpp cpu_backend.cpp)
    target_link_libraries(ll-cuda-cpu ${LLVM_LINKER_FLAGS} ${CMAKE_DL_LIBS} ${CURSES_LIBRARIES} ${ZLIB_LIBRARIES} ${LOCALE_LINK_FLAG} runtime)
endif()
