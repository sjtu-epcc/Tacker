# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /workspace/tacker/runtime/nnf/nnf_tf_freezer/inception3-64/cuda_codegen

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /workspace/tacker/runtime/nnf/nnf_tf_freezer/inception3-64/cuda_codegen

# Include any dependencies generated for this target.
include CMakeFiles/main_test.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/main_test.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/main_test.dir/flags.make

CMakeFiles/main_test.dir/main_test.cpp.o: CMakeFiles/main_test.dir/flags.make
CMakeFiles/main_test.dir/main_test.cpp.o: main_test.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/workspace/tacker/runtime/nnf/nnf_tf_freezer/inception3-64/cuda_codegen/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/main_test.dir/main_test.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/main_test.dir/main_test.cpp.o -c /workspace/tacker/runtime/nnf/nnf_tf_freezer/inception3-64/cuda_codegen/main_test.cpp

CMakeFiles/main_test.dir/main_test.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/main_test.dir/main_test.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /workspace/tacker/runtime/nnf/nnf_tf_freezer/inception3-64/cuda_codegen/main_test.cpp > CMakeFiles/main_test.dir/main_test.cpp.i

CMakeFiles/main_test.dir/main_test.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/main_test.dir/main_test.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /workspace/tacker/runtime/nnf/nnf_tf_freezer/inception3-64/cuda_codegen/main_test.cpp -o CMakeFiles/main_test.dir/main_test.cpp.s

# Object files for target main_test
main_test_OBJECTS = \
"CMakeFiles/main_test.dir/main_test.cpp.o"

# External object files for target main_test
main_test_EXTERNAL_OBJECTS =

main_test: CMakeFiles/main_test.dir/main_test.cpp.o
main_test: CMakeFiles/main_test.dir/build.make
main_test: /usr/local/cuda/lib64/libcudart_static.a
main_test: /usr/lib/x86_64-linux-gnu/librt.so
main_test: libnnfusion_naive_rt.so
main_test: /usr/lib/x86_64-linux-gnu/libcuda.so
main_test: /usr/local/cuda/lib64/libcudart.so
main_test: /usr/local/cuda/lib64/libcudart_static.a
main_test: /usr/lib/x86_64-linux-gnu/librt.so
main_test: /usr/local/cuda/lib64/libcublas.so
main_test: CMakeFiles/main_test.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/workspace/tacker/runtime/nnf/nnf_tf_freezer/inception3-64/cuda_codegen/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable main_test"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/main_test.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/main_test.dir/build: main_test

.PHONY : CMakeFiles/main_test.dir/build

CMakeFiles/main_test.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/main_test.dir/cmake_clean.cmake
.PHONY : CMakeFiles/main_test.dir/clean

CMakeFiles/main_test.dir/depend:
	cd /workspace/tacker/runtime/nnf/nnf_tf_freezer/inception3-64/cuda_codegen && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /workspace/tacker/runtime/nnf/nnf_tf_freezer/inception3-64/cuda_codegen /workspace/tacker/runtime/nnf/nnf_tf_freezer/inception3-64/cuda_codegen /workspace/tacker/runtime/nnf/nnf_tf_freezer/inception3-64/cuda_codegen /workspace/tacker/runtime/nnf/nnf_tf_freezer/inception3-64/cuda_codegen /workspace/tacker/runtime/nnf/nnf_tf_freezer/inception3-64/cuda_codegen/CMakeFiles/main_test.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/main_test.dir/depend

