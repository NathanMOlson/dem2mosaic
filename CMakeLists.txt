cmake_minimum_required(VERSION 3.1)

include(ExternalProject)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED yes)

FIND_PACKAGE(OpenMP)
if(OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

FIND_PACKAGE(Eigen3 REQUIRED)
FIND_PACKAGE(OpenCV REQUIRED)

include_directories(SYSTEM ${OpenCV_INCLUDE_DIRS})
include_directories(SYSTEM ${EIGEN3_INCLUDE_DIRS})

externalproject_add(ext_mapmap
    PREFIX          ext_mapmap
    GIT_REPOSITORY  https://github.com/dthuerck/mapmap_cpu.git
    GIT_TAG         fa526e0
    UPDATE_COMMAND  ""
    SOURCE_DIR      ${CMAKE_SOURCE_DIR}/elibs/mapmap
    CONFIGURE_COMMAND ""
    BUILD_COMMAND   ""
    INSTALL_COMMAND ""
)

include_directories(SYSTEM
    ${CMAKE_SOURCE_DIR}/elibs/mapmap/
    ${CMAKE_SOURCE_DIR}/elibs/mapmap/mapmap
    ${CMAKE_SOURCE_DIR}/elibs/mapmap/ext/dset
)

add_executable(dem2mosaic dem2mosaic.cpp quadmesh.cpp image_view.cpp undistorter.cpp save_geotiff.cpp proj_info.cpp)
target_compile_options(dem2mosaic PUBLIC -Wall -Wextra -Werror)
add_dependencies(dem2mosaic ext_mapmap)
set_property(TARGET dem2mosaic PROPERTY INTERPROCEDURAL_OPTIMIZATION True)
include_directories(${TBB_INCLUDE_DIRS})
include_directories(CMAKE_CURRENT_SOURCE_DIR)
target_link_libraries(dem2mosaic tbb opencv_core opencv_imgproc opencv_imgcodecs opencv_calib3d gdal tiff geotiff)
install(TARGETS dem2mosaic RUNTIME DESTINATION bin)
