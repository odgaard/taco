# Use a smaller base image
FROM python:3.9-slim-buster

ENV HYPERMAPPER_HOME=/app/hypermapper_dev \
    SUITESPARSE_PATH=/app/data/suitesparse \
    FROST_PATH=/app/data/FROSTT \
    DEBIAN_FRONTEND=noninteractive

# Set the working directory
WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY hypermapper_dev/requirements.txt .

# Install dependencies
RUN apt-get update && apt-get install -y \
  build-essential \
  cmake \
  git \
  curl \
  unzip \
  libomp-dev \
  python3-dev \
  zlib1g-dev \
  libssl-dev \
  libopenmpi-dev && \
  pip install -r requirements.txt && \
  apt-get clean && \
  rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Copy current directory files to docker
COPY . .

# Create build directory, build the project, and clean up
RUN mkdir build && \
    cd build && \
    cmake -DCMAKE_BUILD_TYPE=Release -DOPENMP=ON .. && \
    make -j8 && \
    mv ../cpp_taco_* . && \
    cd ..

# Here we assume that "cpp_taco_*" files are meant to stay in "/app/build". 
# If that's not the case, please adjust the path accordingly.

