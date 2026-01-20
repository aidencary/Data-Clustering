cmake_minimum_required(VERSION 3.15)
project(DataClustering)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add executable
add_executable(DataClustering 
    main.cpp
    Kmeans.h
    Point.h
)

# Copy data files to build directory
file(GLOB DATA_FILES "*.txt")
foreach(DATA_FILE ${DATA_FILES})
    get_filename_component(FILE_NAME ${DATA_FILE} NAME)
    configure_file(${DATA_FILE} ${CMAKE_BINARY_DIR}/${FILE_NAME} COPYONLY)
endforeach()
