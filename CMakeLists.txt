cmake_minimum_required(VERSION 2.8.3)
project(ros_deeplabv3)

find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  manipulation_context_slam_msgs
  roscpp
  rospy
  sensor_msgs
  std_msgs
)


catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES ros_deeplabv3
   CATKIN_DEPENDS cv_bridge manipulation_context_slam_msgs roscpp rospy sensor_msgs std_msgs
#  DEPENDS system_lib
)


include_directories(
# include
  ${catkin_INCLUDE_DIRS}
)

catkin_install_python(PROGRAMS src/deeplabv3.py
                      DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})
