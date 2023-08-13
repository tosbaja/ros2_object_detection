cmake_minimum_required(VERSION 3.5)
project(opencv_package)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(OpenCV REQUIRED)
find_package(image_transport REQUIRED)

include_directories(
  include
  ${rclcpp_INCLUDE_DIRS}
  ${sensor_msgs_INCLUDE_DIRS}
  ${cv_bridge_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${image_transport_INCLUDE_DIRS}
)

add_executable(opencv_node src/opencv_node.cpp)

ament_target_dependencies(opencv_node
  rclcpp
  sensor_msgs
  cv_bridge
  OpenCV
  image_transport
)

target_link_libraries(opencv_node
  ${rclcpp_LIBRARIES}
  ${sensor_msgs_LIBRARIES}
  ${cv_bridge_LIBRARIES}
  ${OpenCV_LIBRARIES}
  ${image_transport_LIBRARIES}
)

install(TARGETS
  opencv_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY
  launch
  config
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
