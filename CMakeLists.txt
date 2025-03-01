cmake_minimum_required(VERSION 3.13)

project(TaichiAotDemo LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_BUILD_TYPE RelWithDebInfo)

option(TI_WITH_CPU "Compile with CPU backend" OFF)
option(TI_WITH_CUDA "Compile with CUDA backend" OFF)

# 1. Configure Environments and Third-party Libraries
include("cmake/env.cmake")

set(TAICHI_C_API_INSTALL_DIR $ENV{TAICHI_C_API_INSTALL_DIR})

configure_environment()
configure_third_party()

###############################################################
# If you're looking for minimal code to play with Taichi AOT, #
# then the tutorials are all what you need.                   #
###############################################################
# 2. Compile for tutorials - tutorials have standalone build system
add_subdirectory("0_tutorial_cgraph")
add_subdirectory("0_tutorial_kernel")
# Note: Tutorial readers dont have to bother with anything below this line


##############################################################
# The following codes aim at compiling for AOT Demos, which  #
# demonstrates how to integrate Taichi C-API into a          #
# customized renderer framework                              #
##############################################################
# 3. Compile for renderer framework
add_subdirectory(framework) # defined TaichiAotDemoFramework_INCLUDE_DIRECTORIES, RENDER_FRAMEWORK_TARGET 

# 4. Compile for each demo
include("cmake/demo.cmake")

set(HEADLESS_DEMO_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/headless)
set(GLFW_DEMO_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/glfw)

add_subdirectory(1_hello_world)
add_subdirectory(1_hello_world_with_interop)
add_subdirectory(2_mpm88)
add_subdirectory(3_implicit_fem)
add_subdirectory(4_texture_fractal)
add_subdirectory(5_sph)
add_subdirectory(6_taichi_sparse)
add_subdirectory(7_comet)

# 5. Install runtime libraries
install_shared_libraries(${GLFW_DEMO_OUTPUT_DIRECTORY}, "glfw_dummy")
install_shared_libraries(${HEADLESS_DEMO_OUTPUT_DIRECTORY}, "headless_dummy")
