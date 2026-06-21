cmake_minimum_required(VERSION 3.8)
project(wheel_leg_description)

find_package(ament_cmake REQUIRED)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()


install(
  PROGRAMS
    scripts/five_bar_kinematics.py
    scripts/five_bar_kinematics_node.py
  DESTINATION lib/${PROJECT_NAME}
)
install(
  DIRECTORY urdf meshes launch rviz worlds sdf
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
