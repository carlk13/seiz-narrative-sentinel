cmake_minimum_required(VERSION 3.18)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    project(seiz_cpp_core LANGUAGES CXX)
endif()

if(NOT DEFINED Python3_EXECUTABLE)
    get_filename_component(_REPO_ROOT "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE)
    if(WIN32 AND EXISTS "${_REPO_ROOT}/.venv/Scripts/python.exe")
        set(Python3_EXECUTABLE "${_REPO_ROOT}/.venv/Scripts/python.exe")
    elseif(EXISTS "${_REPO_ROOT}/.venv/bin/python")
        set(Python3_EXECUTABLE "${_REPO_ROOT}/.venv/bin/python")
    endif()
endif()

if(DEFINED Python3_EXECUTABLE AND EXISTS "${Python3_EXECUTABLE}")
    if(NOT DEFINED Python3_INCLUDE_DIR)
        execute_process(
            COMMAND "${Python3_EXECUTABLE}" -c "import sysconfig; print(sysconfig.get_path('include'))"
            OUTPUT_VARIABLE _PY_INC_RAW
            OUTPUT_STRIP_TRAILING_WHITESPACE
            RESULT_VARIABLE _PY_INC_RES
        )
        if(_PY_INC_RES EQUAL 0 AND NOT _PY_INC_RAW STREQUAL "")
            file(TO_CMAKE_PATH "${_PY_INC_RAW}" _PY_INC)
            if(EXISTS "${_PY_INC}/Python.h")
                set(Python3_INCLUDE_DIR "${_PY_INC}")
            endif()
        endif()
    endif()
    if(WIN32 AND NOT DEFINED Python3_LIBRARY)
        execute_process(
            COMMAND "${Python3_EXECUTABLE}" -c "import os, sys; ver=f'{sys.version_info.major}{sys.version_info.minor}'; p=os.path.join(sys.base_prefix, 'libs', f'python{ver}.lib'); print(p if os.path.exists(p) else '')"
            OUTPUT_VARIABLE _PY_LIB_RAW
            OUTPUT_STRIP_TRAILING_WHITESPACE
            RESULT_VARIABLE _PY_LIB_RES
        )
        if(_PY_LIB_RES EQUAL 0 AND NOT _PY_LIB_RAW STREQUAL "")
            file(TO_CMAKE_PATH "${_PY_LIB_RAW}" _PY_LIB)
            if(EXISTS "${_PY_LIB}")
                set(Python3_LIBRARY "${_PY_LIB}")
            endif()
        endif()
    endif()
endif()

find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)

if(NOT DEFINED pybind11_DIR)
    execute_process(
        COMMAND "${Python3_EXECUTABLE}" -m pybind11 --cmakedir
        OUTPUT_VARIABLE _PYBIND11_CMAKE_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE _PYBIND11_RESULT
    )
    if(_PYBIND11_RESULT EQUAL 0 AND EXISTS "${_PYBIND11_CMAKE_DIR}")
        set(pybind11_DIR "${_PYBIND11_CMAKE_DIR}")
    endif()
endif()

find_package(pybind11 CONFIG REQUIRED)

find_package(Eigen3 3.4 QUIET NO_MODULE)
if(NOT TARGET Eigen3::Eigen)
    include(FetchContent)
    FetchContent_Declare(
        eigen3
        URL https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz
        SOURCE_SUBDIR _skip_eigen_tests
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    )
    FetchContent_MakeAvailable(eigen3)
    if(NOT TARGET Eigen3::Eigen)
        add_library(Eigen3_Eigen INTERFACE)
        target_include_directories(Eigen3_Eigen SYSTEM INTERFACE "${eigen3_SOURCE_DIR}")
        add_library(Eigen3::Eigen ALIAS Eigen3_Eigen)
    endif()
endif()

pybind11_add_module(seiz_cpp MODULE "${CMAKE_CURRENT_SOURCE_DIR}/model.cc")

target_link_libraries(seiz_cpp PRIVATE Eigen3::Eigen)

target_compile_options(seiz_cpp PRIVATE
    $<$<CXX_COMPILER_ID:MSVC>:/O2 /arch:AVX2 /W4>
    $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-O3 -Wall -Wextra>
    $<$<AND:$<NOT:$<CXX_COMPILER_ID:MSVC>>,$<OR:$<STREQUAL:${CMAKE_SYSTEM_PROCESSOR},x86_64>,$<STREQUAL:${CMAKE_SYSTEM_PROCESSOR},AMD64>>>:-mavx2 -mfma>
)

set_target_properties(seiz_cpp PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    LIBRARY_OUTPUT_DIRECTORY_RELEASE "${CMAKE_CURRENT_SOURCE_DIR}"
    LIBRARY_OUTPUT_DIRECTORY_DEBUG "${CMAKE_CURRENT_SOURCE_DIR}"
    LIBRARY_OUTPUT_DIRECTORY_RELWITHDEBINFO "${CMAKE_CURRENT_SOURCE_DIR}"
    LIBRARY_OUTPUT_DIRECTORY_MINSIZEREL "${CMAKE_CURRENT_SOURCE_DIR}"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_CURRENT_SOURCE_DIR}"
    RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_CURRENT_SOURCE_DIR}"
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
)

if(WIN32)
    add_custom_command(TARGET seiz_cpp POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:seiz_cpp>"
            "${CMAKE_CURRENT_SOURCE_DIR}/seiz_cpp.so"
        COMMENT "Creating seiz_cpp.so copy alongside Windows .pyd extension"
    )
endif()
