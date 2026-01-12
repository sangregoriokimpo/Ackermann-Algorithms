cmake_minimum_required(VERSION 3.8)
project(autonomous_driver)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# ---------------- Dependencies ----------------
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(ackermann_msgs REQUIRED)

find_package(pcl_conversions REQUIRED)
find_package(PCL REQUIRED)

find_package(tf2_ros REQUIRED)
find_package(tf2_sensor_msgs REQUIRED)

include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${PCL_INCLUDE_DIRS}
)

# ---------------- Executables ----------------
add_executable(autonomous_driver src/autonomous_driver.cpp)
ament_target_dependencies(autonomous_driver
  rclcpp sensor_msgs geometry_msgs pcl_conversions
)
target_link_libraries(autonomous_driver ${PCL_LIBRARIES})

add_executable(rgbd_driver src/rgbd_driver.cpp)
ament_target_dependencies(rgbd_driver
  rclcpp sensor_msgs geometry_msgs
)

add_executable(fusion_driver src/fusion_driver.cpp)
ament_target_dependencies(fusion_driver
  rclcpp sensor_msgs geometry_msgs visualization_msgs nav_msgs pcl_conversions
  tf2_ros tf2_sensor_msgs
)
target_link_libraries(fusion_driver ${PCL_LIBRARIES})

add_executable(2d_driver src/2d_driver.cpp)
ament_target_dependencies(2d_driver
  rclcpp sensor_msgs geometry_msgs
)

add_executable(fusion_gap_driver src/fusion_gap_driver.cpp)
ament_target_dependencies(fusion_gap_driver
  rclcpp sensor_msgs geometry_msgs visualization_msgs nav_msgs pcl_conversions
  tf2_ros tf2_sensor_msgs
)
target_link_libraries(fusion_gap_driver ${PCL_LIBRARIES})

add_executable(fusion_gap_driverE0 src/fusion_gap_driverE0.cpp)
ament_target_dependencies(fusion_gap_driverE0
  rclcpp sensor_msgs geometry_msgs visualization_msgs nav_msgs pcl_conversions
  tf2_ros tf2_sensor_msgs
)
target_link_libraries(fusion_gap_driverE0 ${PCL_LIBRARIES})

# ---------------- Install ----------------
install(TARGETS
  autonomous_driver
  rgbd_driver
  fusion_driver
  fusion_gap_driver
  fusion_gap_driverE0
  2d_driver
  DESTINATION lib/${PROJECT_NAME}
)

# ---------------- Tests ----------------
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
