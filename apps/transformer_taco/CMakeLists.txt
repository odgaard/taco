cmake_minimum_required(VERSION 3.13)

# find_package(Torch REQUIRED)

if(POLICY CMP0048)
  cmake_policy(SET CMP0048 NEW)
endif()
project(Transformer)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS} -g -lstdc++fs")


# include_directories("src/transformer")
# include(CTest)
include_directories(${PROJECT_SOURCE_DIR}/src/transformer/gtest/)
include_directories(${TACO_TEST_DIR} ${TACO_SRC_DIR})
file(GLOB SOURCE_CODE ${PROJECT_SOURCE_DIR}/src/transformer/*.cpp)
file(GLOB GTEST_SOURCE ${PROJECT_SOURCE_DIR}/src/transformer/gtest/gtest-all.cc)
file(GLOB GTEST_HEADER ${PROJECT_SOURCE_DIR}/src/transformer/gtest/gtest.h)
add_library(gtest ${GTEST_HEADER} ${GTEST_SOURCE})
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-missing-declarations -DTACO_TEST_DIR=\"${TACO_TEST_DIR}\"")

file(GLOB TEST_HEADERS *.h)
file(GLOB TEST_SOURCES *.cpp)
# set(CMAKE_FIND_LIBRARY_SUFFIXES ".a")
# set(TORCH_LIBRARY, "${TORCH_INSTALL_PREFIX}/lib/libtorch.so")

add_executable(${PROJECT_NAME} ${SOURCE_CODE})
# add_executable(${PROJECT_NAME} transformer_test.cpp)

# target_link_libraries(${PROJECT_NAME} gtest)

# Add xtensor to project
target_include_directories(${PROJECT_NAME} PUBLIC ${xtensor_INCLUDE_DIRS} ${GTEST_HEADER})
# target_link_libraries(${PROJECT_NAME} PUBLIC xtensor xtensor-blas)

# find_dependency(BLAS REQUIRED)
# find_dependency(LAPACK REQUIRED)

# include("${CMAKE_CURRENT_LIST_DIR}/xtensor-blasTargets.cmake")
  # get_target_property(xtensor-blas_INCLUDE_DIRS xtensor-blas INTERFACE_INCLUDE_DIRECTORIES)

# set_target_properties(xtensor-blas PROPERTIES 
    # INTERFACE_LINK_LIBRARIES "${BLAS_LIBRARIES};${LAPACK_LIBRARIES}"
# )
# set_target_properties(xtensor-blas PROPERTIES
        # INTERFACE_COMPILE_FEATURES cxx_std_14
# )

# To let the app be a standalone project 
if (NOT TACO_INCLUDE_DIR)
  if (NOT DEFINED ENV{TACO_INCLUDE_DIR} OR NOT DEFINED ENV{TACO_LIBRARY_DIR})
    message(FATAL_ERROR "Set the environment variables TACO_INCLUDE_DIR and TACO_LIBRARY_DIR")
  endif ()
  set(TACO_INCLUDE_DIR $ENV{TACO_INCLUDE_DIR})
  set(TACO_LIBRARY_DIR $ENV{TACO_LIBRARY_DIR})
  find_library(taco taco ${TACO_LIBRARY_DIR})
  target_link_libraries(${PROJECT_NAME} LINK_PUBLIC ${taco} gtest pthread) # "${TORCH_LIBRARIES}")
else()
  set_target_properties("${PROJECT_NAME}" PROPERTIES OUTPUT_NAME "taco-${PROJECT_NAME}")
  target_link_libraries(${PROJECT_NAME} LINK_PUBLIC taco gtest pthread) # "${TORCH_LIBRARIES}")
endif ()


# Include taco headers
include_directories(${TACO_INCLUDE_DIR})

if(${CMAKE_VERSION} VERSION_LESS "3.9.0")
  add_test(NAME taco-test COMMAND taco-test)
else()
  cmake_policy(SET CMP0054 NEW)
  cmake_policy(SET CMP0057 NEW)
  include(GoogleTest)
  gtest_add_tests(TARGET taco-test)
endif()

add_test(transformer_test ${PROJECT_NAME})