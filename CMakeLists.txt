cmake_minimum_required(VERSION 3.10)
project(GeoRefHut)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE RELEASE)

# Add the executable
add_executable(GeoRefHut src/GeoRefHut.cpp dependencies/tinyfiledialogs/tinyfiledialogs.cpp)

# Specify library directories
set(E57Format_DIR "${CMAKE_SOURCE_DIR}/dependencies/libE57Format")
set(g2o_DIR "${CMAKE_SOURCE_DIR}/dependencies/g2o")
set(Pangolin_DIR "${CMAKE_SOURCE_DIR}/dependencies/Pangolin/build")

# Find required packages
find_package(Pangolin REQUIRED PATHS ${Pangolin_DIR})
find_package(small_gicp REQUIRED PATHS "${CMAKE_SOURCE_DIR}/dependencies/small_gicp")
find_package(E57Format REQUIRED PATHS ${E57Format_DIR})
find_package(GLU REQUIRED)
find_package(g2o REQUIRED PATHS ${g2o_DIR})

include_directories(src)
include_directories(${CMAKE_SOURCE_DIR}/dependencies/tinyfiledialogs)
include_directories(${CMAKE_SOURCE_DIR}/dependencies/json/single_include)
include_directories(${CMAKE_SOURCE_DIR}/dependencies/small_gicp/include)
include_directories(${CMAKE_SOURCE_DIR}/dependencies/cilantro/include)
include_directories(${g2o_DIR})  # Includes the path to g2o/config.h
include_directories(/usr/local/include)
include_directories(${CMAKE_SOURCE_DIR}/dependencies/Pangolin/components/)

# Link libraries to the target
target_link_libraries(GeoRefHut
    E57Format
    pango_core
    pango_display
    pango_opengl
    pango_plot
    pango_video
    GLU
    small_gicp
    g2o::stuff
    g2o::core
    g2o::types_slam3d
    g2o::solver_cholmod
    g2o::solver_csparse
)
