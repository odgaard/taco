# Minimum version required
cmake_minimum_required(VERSION 3.8)

# Project name and language configurations
project(taco_dse C CXX)

# Set policies and flags
if(POLICY CMP0048)
  cmake_policy(SET CMP0048 NEW)
endif()

# Set C++ flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -g -lstdc++fs")

# Include taco headers
include_directories(${TACO_INCLUDE_DIR})
include_directories(${CMAKE_BINARY_DIR})

# Find necessary packages
find_package(Protobuf CONFIG REQUIRED)
find_package(Threads REQUIRED)
find_package(gRPC CONFIG REQUIRED)

option(protobuf_MODULE_COMPATIBLE TRUE)
message(STATUS "Using protobuf ${Protobuf_VERSION}")
message(STATUS "Using gRPC ${gRPC_VERSION}")

set(_PROTOBUF_LIBPROTOBUF protobuf::libprotobuf)
set(_REFLECTION gRPC::grpc++_reflection)
set(_GRPC_GRPCPP gRPC::grpc++)

# Protobuf configurations
find_program(_PROTOBUF_PROTOC protoc)
find_program(_GRPC_CPP_PLUGIN_EXECUTABLE grpc_cpp_plugin)

# Proto file
get_filename_component(cs_proto "config_service.proto" ABSOLUTE)
get_filename_component(cs_proto_path "${cs_proto}" PATH)

# Generated sources
set(cs_proto_srcs config_service.pb.cc)
set(cs_proto_hdrs config_service.pb.h)
set(cs_grpc_srcs config_service.grpc.pb.cc)
set(cs_grpc_hdrs config_service.grpc.pb.h)

add_custom_command(
      OUTPUT "${cs_proto_srcs}" "${cs_proto_hdrs}" "${cs_grpc_srcs}" "${cs_grpc_hdrs}"
      COMMAND ${_PROTOBUF_PROTOC}
      ARGS --grpc_out "${CMAKE_CURRENT_BINARY_DIR}"
        --cpp_out "${CMAKE_CURRENT_BINARY_DIR}"
        -I "${cs_proto_path}"
        --plugin=protoc-gen-grpc="${_GRPC_CPP_PLUGIN_EXECUTABLE}"
        "${cs_proto}"
      DEPENDS "${cs_proto}")

# Include generated *.pb.h files
include_directories("${CMAKE_CURRENT_BINARY_DIR}")

# cs_grpc_proto library configurations
add_library(cs_grpc_proto
  ${cs_grpc_srcs}
  ${cs_grpc_hdrs}
  ${cs_proto_srcs}
  ${cs_proto_hdrs}
)
target_link_libraries(cs_grpc_proto
  ${_REFLECTION}
  ${_GRPC_GRPCPP}
  ${_PROTOBUF_LIBPROTOBUF})

# Configuration for taco_dse project
file(GLOB SOURCE_CODE ${PROJECT_SOURCE_DIR}/*.cpp ${PROJECT_SOURCE_DIR}/*.h)

# Check TACO include and library directories
if (NOT TACO_INCLUDE_DIR)
  if (NOT DEFINED ENV{TACO_INCLUDE_DIR} OR NOT DEFINED ENV{TACO_LIBRARY_DIR})
    message(FATAL_ERROR "Set the environment variables TACO_INCLUDE_DIR and TACO_LIBRARY_DIR")
  endif ()
  set(TACO_INCLUDE_DIR $ENV{TACO_INCLUDE_DIR})
  set(TACO_LIBRARY_DIR $ENV{TACO_LIBRARY_DIR})
  find_library(taco taco ${TACO_LIBRARY_DIR})
  add_executable(${PROJECT_NAME} ${SOURCE_CODE})
  target_link_libraries(${PROJECT_NAME} LINK_PUBLIC ${taco} stdc++fs cs_grpc_proto)
else()
  add_executable(${PROJECT_NAME} ${SOURCE_CODE})
  set_target_properties("${PROJECT_NAME}" PROPERTIES OUTPUT_NAME "taco-${PROJECT_NAME}")
  target_link_libraries(${PROJECT_NAME} LINK_PUBLIC taco stdc++fs cs_grpc_proto)
endif ()
