add_demo(2_mpm88 ${CMAKE_CURRENT_SOURCE_DIR}/app.cpp)
generate_aot_files(2_mpm88 "assets/mpm88.py" "vulkan")
generate_aot_files(2_mpm88 "assets/mpm88.py" "x64")
generate_aot_files(2_mpm88 "assets/mpm88.py" "cuda")
