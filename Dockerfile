FROM ubuntu:24.04

RUN apt-get update
RUN apt-get install -y build-essential cmake git
RUN apt-get install -y libgdal-dev libopencv-dev libeigen3-dev nlohmann-json3-dev

WORKDIR /dem2mosaic
COPY . ./

RUN mkdir build && cd build && cmake .. && make -j
