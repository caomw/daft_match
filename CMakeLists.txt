cmake_minimum_required(VERSION 2.8.3)
project(daft_match)

find_package(catkin REQUIRED COMPONENTS daft)

find_package(OpenCV REQUIRED)
include_directories (${OpenCV_INCLUDE_DIR})

catkin_package(
  CATKIN_DEPENDS daft
)

## Specify additional locations of header files
include_directories(include ${catkin_INCLUDE_DIRS} ${Boost_INCLUDE_DIRS})

## Declare a cpp executable
add_executable(daft_match daft_match.cpp)

## Add dependencies to the executable
add_dependencies(daft_match ${PROJECT_NAME})

## Specify libraries to link a library or executable target against
target_link_libraries(daft_match
  ${catkin_LIBRARIES}
)

