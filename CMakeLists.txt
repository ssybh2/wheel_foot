cmake_minimum_required(VERSION 3.8)
project(wheel_leg_description)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
# uncomment the following section in order to fill in
# further dependencies manually.
# find_package(<dependency> REQUIRED)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()


install(PROGRAMS
  scripts/fourbar_joint_publisher.py
  scripts/fourbar_keyboard_cmd.py
  DESTINATION lib/${PROJECT_NAME}
)

install(
  PROGRAMS
    scripts/balance_state_monitor.py
  DESTINATION lib/${PROJECT_NAME}
)

install(
  PROGRAMS
    scripts/lqr_balance_controller.py
  DESTINATION lib/${PROJECT_NAME}
)

install(
  PROGRAMS
    scripts/leg_height_hold.py
  DESTINATION lib/${PROJECT_NAME}
)

install(
  PROGRAMS
    scripts/leg_effort_symmetric_hold.py
    scripts/leg_roll_calibrator.py
    scripts/leg_height_calibrator.py
  DESTINATION lib/${PROJECT_NAME}
)

# BEGIN CONTROLLED_DISTURBANCE_PLUGIN
find_package(rclcpp REQUIRED)
find_package(std_srvs REQUIRED)
find_package(gazebo_ros REQUIRED)
find_package(gazebo REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${GAZEBO_CXX_FLAGS}")

add_library(controlled_disturbance_plugin SHARED
  src/controlled_disturbance_plugin.cpp
)

target_compile_features(controlled_disturbance_plugin PUBLIC cxx_std_17)

target_include_directories(controlled_disturbance_plugin PRIVATE
  ${GAZEBO_INCLUDE_DIRS}
)

target_link_directories(controlled_disturbance_plugin PRIVATE
  ${GAZEBO_LIBRARY_DIRS}
)

target_link_libraries(controlled_disturbance_plugin
  ${GAZEBO_LIBRARIES}
)

ament_target_dependencies(controlled_disturbance_plugin
  rclcpp
  std_srvs
  gazebo_ros
)

install(TARGETS controlled_disturbance_plugin
  LIBRARY DESTINATION lib
)
# END CONTROLLED_DISTURBANCE_PLUGIN


install(
  PROGRAMS
    scripts/wheel_keyboard_teleop.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
install(
  DIRECTORY urdf meshes launch rviz worlds config sdf
  DESTINATION share/${PROJECT_NAME}
)


install(
  DIRECTORY sdf
  DESTINATION share/${PROJECT_NAME}
)


install(
  PROGRAMS
    scripts/simple_balance_controller.py
  DESTINATION lib/${PROJECT_NAME}
)
