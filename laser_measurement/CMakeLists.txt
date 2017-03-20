cmake_minimum_required(VERSION 2.8)
add_executable(laser_mesurement main.cpp kalman_filter.cpp tracking.cpp)
include_directories(./Eigen)
