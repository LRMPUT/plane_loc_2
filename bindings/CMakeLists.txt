cmake_minimum_required(VERSION 2.8.3)
project(plane_loc_py)

## Compile as C++11, supported in ROS Kinetic and newer
#add_compile_options(-std=c++14)
set(CMAKE_CXX_STANDARD 14)

LIST(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../cmake_modules)

## System dependencies are found with CMake's conventions

# Boost after all other packages to get correct components libs for linking
find_package(Boost COMPONENTS system filesystem serialization REQUIRED)

# OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS} )

# PCL
find_package(PCL REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

# Eigen
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

# CGAL and its components
find_package( CGAL QUIET COMPONENTS  )

if ( NOT CGAL_FOUND )

    message(STATUS "This project requires the CGAL library, and will not be compiled.")
    return()

endif()

# include helper file
include( ${CGAL_USE_FILE} )

find_package(libnabo REQUIRED)

include_directories(
        ../include/plane_loc
        ../include
)

add_subdirectory(pybind11)
pybind11_add_module(plane_loc_py
        planeLocPy.cpp
        ../src/plane_loc/ConcaveHull.cpp
        ../src/plane_loc/Map.cpp
        ../src/plane_loc/Matching.cpp
        ../src/plane_loc/Misc.cpp
        ../src/plane_loc/ObjInstanceView.cpp
        ../src/plane_loc/ObjInstance.cpp
        ../src/plane_loc/PlaneEstimator.cpp
        ../src/plane_loc/UnionFind.cpp
        ../src/PoseMisc.cpp
        )

target_link_libraries(plane_loc_py
        PRIVATE
        ${OpenCV_LIBS}
        ${PCL_LIBRARIES}
        libnabo::nabo
        Boost::boost Boost::system Boost::filesystem Boost::serialization
        )