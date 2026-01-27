cmake_minimum_required(VERSION 3.14)
project(batch_loader VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Optimization flags
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG")

# Find zstd
find_package(PkgConfig REQUIRED)
pkg_check_modules(ZSTD REQUIRED libzstd)

# Find threads
find_package(Threads REQUIRED)

# Create shared library
add_library(batch_loader SHARED
    batch_loader.cpp
)

target_include_directories(batch_loader
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
    PRIVATE
        ${ZSTD_INCLUDE_DIRS}
)

target_link_libraries(batch_loader
    PRIVATE
        ${ZSTD_LIBRARIES}
        Threads::Threads
)

# Set library properties
set_target_properties(batch_loader PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION 1
    PUBLIC_HEADER batch_loader.h
)

# Install rules
install(TARGETS batch_loader
    LIBRARY DESTINATION lib
    PUBLIC_HEADER DESTINATION include
)

# Print configuration summary
message(STATUS "")
message(STATUS "=== Batch Loader Build Configuration ===")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ compiler: ${CMAKE_CXX_COMPILER}")
message(STATUS "C++ flags: ${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_RELEASE}")
message(STATUS "ZSTD include: ${ZSTD_INCLUDE_DIRS}")
message(STATUS "ZSTD libraries: ${ZSTD_LIBRARIES}")
message(STATUS "")
