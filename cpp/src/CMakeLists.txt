set(HEADER_LIST "${CMAKE_SOURCE_DIR}/include/collision_detection/impulse_based.h")
# Make an automatic library - will be static or dynamic based on user setting
add_library(detection_library collision_detection.cpp ${HEADER_LIST})

# We need this directory, and users of our library will need it too
target_include_directories(detection_library PUBLIC ../include)

# All users of this library will need at least C++11
target_compile_features(detection_library PUBLIC cxx_std_11)
