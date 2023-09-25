#!/bin/sh
cd grpc
mkdir -p cmake/build
cd cmake/build && \
cmake -DgRPC_INSTALL=ON \
    -DgRPC_BUILD_TESTS=OFF \
    ../.. && \
make -j 4 && \
make install && \
cd ../..
