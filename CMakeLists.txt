cmake_minimum_required(VERSION 3.21)
project(MatrixMultiplicationSIMD)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "-march=native")

add_executable(MatrixMultiplicationSIMD main.cpp Utils.h)
