cmake_minimum_required(VERSION 2.8)
project(slamv1)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -pthread")

find_package(OpenCV 3)
if(NOT OpenCV_FOUND)
   find_package(OpenCV 2.4.3 QUIET)
   if(NOT OpenCV_FOUND)
      message(FATAL_ERROR "OpenCV > 2.4.3 not found.")
   endif()
endif()

find_package(Sophus REQUIRED)
find_package(glog REQUIRED)
include_directories(${Sophus_INCLUDE_DIRS})

set(VTK_DIR /home/e1ysium/Desktop/VTK-6.3.0/build)

# pcl 
find_package(PCL 1.8 REQUIRED)
set( PCL_DIR "/usr/include/pcl-1.8/pcl")
include_directories(${PCL_INCLUDE_DIRS})
add_definitions(${PCL_DEFINITIONS})

MESSAGE("OPENCV VERSION:")
MESSAGE(${OpenCV_VERSION})
MESSAGE("PCL INCLUDE FOLDER:")
MESSAGE(${PCL_INCLUDE_DIRS})
MESSAGE("PCL FOLDER:")
MESSAGE(${PCL_DIR})
message(STATUS "VTK library status:")
set(${VTK_INCLUDE_DIRS} "/usr/local/include/vtk-6.3")
set(${VTK_LIBRARIES} "/home/e1ysium/Desktop/VTK-6.3.0/build/lib/")
message(STATUS "VTK Version: ${VTK_VERSION}")
message(STATUS "library:${VTK_LIBRARIES}")
message(STATUS "include path:${VTK_INCLUDE_DIRS}")

find_package(Eigen3 REQUIRED)
find_package(Pangolin REQUIRED)

# Sophus
find_package(Sophus REQUIRED)
include_directories(${Sophus_INCLUDE_DIRS})

set( DBoW3_LIBS "/usr/local/lib/libDBoW3.so.0.0" )
set( DBoW3_INCLUDE_DIRS "/usr/local/include" )

include_directories(
${OpenCV_LIBS}
${EIGEN3_LIBS}
${PCL_INCLUDE_DIRS}
${PROJECT_SOURCE_DIR}/include
${EIGEN3_INCLUDE_DIR}
${Pangolin_INCLUDE_DIRS}
${G2O_INCLUDE_DIRS}
${Sophus_INCLUDE_DIRS}
/usr/local/include/ceres
)

find_package(VTK REQUIRED)
vtk_module_config(VTK
  vtkCommonCore
  vtkFiltersSources
  vtkFiltersSources
  vtkInteractionStyle
  vtkRendering${VTK_RENDERING_BACKEND}
)
include(${VTK_USE_FILE})

aux_source_directory(./include/ src_path)

add_library(${PROJECT_NAME}
include/mapping/pointclouds_map.h
include/mapping/pointclouds_map.cpp
include/mapping/posture.h
include/mapping/posture.cpp
include/datapath/datapath.h
include/datapath/datapath.cpp
include/define/flags.h
include/define/feature.h
include/define/frame.h
include/define/mappoint.h
include/define/map.h
include/algorithm/front_module.h
include/algorithm/front_module.cpp
include/algorithm/back_module.h
include/algorithm/back_module.cpp
include/viewer/viewer.h
include/viewer/viewer.cpp
include/loop_closure/dbow.h
include/loop_closure/dbow.cpp
include/loop_closure/loop_closing.h
include/loop_closure/loop_closing.cpp
)

add_executable(slam_main_program main.cpp)

target_link_libraries(slam_main_program PRIVATE 
${CMAKE_THREAD_LIBS_INIT} 
${OpenCV_LIBS}
${EIGEN3_LIBS}
${Pangolin_LIBRARIES}
${src_path}
${PROJECT_NAME}
g2o_core g2o_stuff g2o_types_sba g2o_solver_csparse g2o_csparse_extension
${Sophus_LIBRARIES}
fmt
glog
${DBoW3_LIBS}
${PCL_LIBRARIES}
${VTK_LIBRARIES}
)
