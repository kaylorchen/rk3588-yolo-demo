cmake_minimum_required(VERSION 3.22)
project(yolov8)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
find_package(spdlog REQUIRED)

add_executable(yolov8_demo main.cpp videofile.cpp image_process.cpp)
target_link_libraries(yolov8_demo spdlog::spdlog systemd ${OpenCV_LIBS})