cmake_minimum_required(VERSION 2.8)

# Set a default build type if none was specified
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "Setting build type to 'RelWithDebInfo' as none was specified.")
    set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "Choose the type of build." FORCE)
    # Set the possible values of build type for cmake-gui
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release"
      "MinSizeRel" "RelWithDebInfo")
endif()

project(serial-borg-moea)

enable_language(C)

set(CMAKE_C_STANDARD 99)

set(CFLAGS "-O3")
set(LIBS "-lm -lnsl -lsocket -lresolv")
#set(LDFLAGS "-Wl,-R,\.")

set(CMAKE_C_FLAGS_DEBUG "-DMY_DEBUG -O0 -g2")
set(CMAKE_C_FLAGS_MINSIZEREL "${CFLAGS}")
set(CMAKE_C_FLAGS_RELWITHDEBINFO "-O0 -g2")
set(CMAKE_C_FLAGS_RELEASE "${CFLAGS}")

set(CMAKE_EXE_LINKER_FLAGS "${LDFLAGS}")

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY bin)
set(CMAKE_SHARED_LIBRARY_PREFIX "")

add_executable(dtlz2_serial dtlz2_serial.c)
add_executable(dtlz2_advanced dtlz2_advanced.c)
add_executable(borg frontend.c borg.c mt19937ar.c)

add_library(libborg SHARED frontend.c borg.c mt19937ar.c)

target_link_libraries(borg m)
target_link_libraries(libborg m)
target_link_libraries(dtlz2_serial m libborg)
target_link_libraries(dtlz2_advanced m libborg)

# MPI
find_package(MPI)
message(">MPI inc =  ${MPI_INCLUDE_PATH}  ")
message(">MPI lib =  ${MPI_LIBRARIES}  ")
include_directories(${MPI_INCLUDE_PATH})

add_definitions(-DOMPI_SKIP_MPICXX)
    
add_executable(dtlz2_ms dtlz2_ms.c)
add_library(libborgms SHARED frontend.c borgms.c mt19937ar.c)
    
target_link_libraries(libborgms m ${MPI_LIBRARIES})
target_link_libraries(dtlz2_ms libborgms m)

