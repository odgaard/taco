cmake_minimum_required(VERSION 3.13)

if(POLICY CMP0048)
  cmake_policy(SET CMP0048 NEW)
endif()
project(Transformer)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -lstdc++fs")

# include_directories("src/transformer")
# include(CTest)
file(GLOB SOURCE_CODE ${PROJECT_SOURCE_DIR}/src/transformer/*.cpp)
add_executable(${PROJECT_NAME} ${SOURCE_CODE})
# add_executable(${PROJECT_NAME} transformer_test.cpp)


# To let the app be a standalone project 
if (NOT TACO_INCLUDE_DIR)
  if (NOT DEFINED ENV{TACO_INCLUDE_DIR} OR NOT DEFINED ENV{TACO_LIBRARY_DIR})
    message(FATAL_ERROR "Set the environment variables TACO_INCLUDE_DIR and TACO_LIBRARY_DIR")
  endif ()
  set(TACO_INCLUDE_DIR $ENV{TACO_INCLUDE_DIR})
  set(TACO_LIBRARY_DIR $ENV{TACO_LIBRARY_DIR})
  find_library(taco taco ${TACO_LIBRARY_DIR})
  target_link_libraries(${PROJECT_NAME} LINK_PUBLIC ${taco})
else()
  set_target_properties("${PROJECT_NAME}" PROPERTIES OUTPUT_NAME "taco-${PROJECT_NAME}")
  target_link_libraries(${PROJECT_NAME} LINK_PUBLIC taco)
endif ()

# Include taco headers
include_directories(${TACO_INCLUDE_DIR})

# add_test(transformer_test ${PROJECT_NAME})