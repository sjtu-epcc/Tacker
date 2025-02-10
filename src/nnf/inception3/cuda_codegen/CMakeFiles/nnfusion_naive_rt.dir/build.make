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
include CMakeFiles/nnfusion_naive_rt.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/nnfusion_naive_rt.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/nnfusion_naive_rt.dir/flags.make

CMakeFiles/nnfusion_naive_rt.dir/nnfusion_naive_rt_generated_nnfusion_rt.cu.o: CMakeFiles/nnfusion_naive_rt.dir/nnfusion_naive_rt_generated_nnfusion_rt.cu.o.depend
CMakeFiles/nnfusion_naive_rt.dir/nnfusion_naive_rt_generated_nnfusion_rt.cu.o: CMakeFiles/nnfusion_naive_rt.dir/nnfusion_naive_rt_generated_nnfusion_rt.cu.o.Release.cmake
CMakeFiles/nnfusion_naive_rt.dir/nnfusion_naive_rt_generated_nnfusion_rt.cu.o: nnfusion_rt.cu
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/workspace/tacker/runtime/nnf/nnf_tf_freezer/inception3-64/cuda_codegen/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building NVCC (Device) object CMakeFiles/nnfusion_naive_rt.dir/nnfusion_naive_rt_generated_nnfusion_rt.cu.o"
	cd /workspace/tacker/runtime/nnf/nnf_tf_freezer/inception3-64/cuda_codegen/CMakeFiles/nnfusion_naive_rt.dir && /usr/bin/cmake -E make_directory /workspace/tacker/runtime/nnf/nnf_tf_freezer/inception3-64/cuda_codegen/CMakeFiles/nnfusion_naive_rt.dir//.
	cd /workspace/tacker/runtime/nnf/nnf_tf_freezer/inception3-64/cuda_codegen/CMakeFiles/nnfusion_naive_rt.dir && /usr/bin/cmake -D verbose:BOOL=$(VERBOSE) -D build_configuration:STRING=Release -D generated_file:STRING=/workspace/tacker/runtime/nnf/nnf_tf_freezer/inception3-64/cuda_codegen/CMakeFiles/nnfusion_naive_rt.dir//./nnfusion_naive_rt_generated_nnfusion_rt.cu.o -D generated_cubin_file:STRING=/workspace/tacker/runtime/nnf/nnf_tf_freezer/inception3-64/cuda_codegen/CMakeFiles/nnfusion_naive_rt.dir//./nnfusion_naive_rt_generated_nnfusion_rt.cu.o.cubin.txt -P /workspace/tacker/runtime/nnf/nnf_tf_freezer/inception3-64/cuda_codegen/CMakeFiles/nnfusion_naive_rt.dir//nnfusion_naive_rt_generated_nnfusion_rt.cu.o.Release.cmake

# Object files for target nnfusion_naive_rt
nnfusion_naive_rt_OBJECTS =

# External object files for target nnfusion_naive_rt
nnfusion_naive_rt_EXTERNAL_OBJECTS = \
"/workspace/tacker/runtime/nnf/nnf_tf_freezer/inception3-64/cuda_codegen/CMakeFiles/nnfusion_naive_rt.dir/nnfusion_naive_rt_generated_nnfusion_rt.cu.o"

libnnfusion_naive_rt.so: CMakeFiles/nnfusion_naive_rt.dir/nnfusion_naive_rt_generated_nnfusion_rt.cu.o
libnnfusion_naive_rt.so: CMakeFiles/nnfusion_naive_rt.dir/build.make
libnnfusion_naive_rt.so: /usr/local/cuda/lib64/libcudart_static.a
libnnfusion_naive_rt.so: /usr/lib/x86_64-linux-gnu/librt.so
libnnfusion_naive_rt.so: /usr/lib/x86_64-linux-gnu/libcuda.so
libnnfusion_naive_rt.so: /usr/local/cuda/lib64/libcudart.so
libnnfusion_naive_rt.so: /usr/local/cuda/lib64/libcudart_static.a
libnnfusion_naive_rt.so: /usr/lib/x86_64-linux-gnu/librt.so
libnnfusion_naive_rt.so: /usr/local/cuda/lib64/libcublas.so
libnnfusion_naive_rt.so: /usr/lib/x86_64-linux-gnu/libcuda.so
libnnfusion_naive_rt.so: /usr/local/cuda/lib64/libcudart.so
libnnfusion_naive_rt.so: /usr/local/cuda/lib64/libcublas.so
libnnfusion_naive_rt.so: CMakeFiles/nnfusion_naive_rt.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/workspace/tacker/runtime/nnf/nnf_tf_freezer/inception3-64/cuda_codegen/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX shared library libnnfusion_naive_rt.so"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/nnfusion_naive_rt.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/nnfusion_naive_rt.dir/build: libnnfusion_naive_rt.so

.PHONY : CMakeFiles/nnfusion_naive_rt.dir/build

CMakeFiles/nnfusion_naive_rt.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/nnfusion_naive_rt.dir/cmake_clean.cmake
.PHONY : CMakeFiles/nnfusion_naive_rt.dir/clean

CMakeFiles/nnfusion_naive_rt.dir/depend: CMakeFiles/nnfusion_naive_rt.dir/nnfusion_naive_rt_generated_nnfusion_rt.cu.o
	cd /workspace/tacker/runtime/nnf/nnf_tf_freezer/inception3-64/cuda_codegen && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /workspace/tacker/runtime/nnf/nnf_tf_freezer/inception3-64/cuda_codegen /workspace/tacker/runtime/nnf/nnf_tf_freezer/inception3-64/cuda_codegen /workspace/tacker/runtime/nnf/nnf_tf_freezer/inception3-64/cuda_codegen /workspace/tacker/runtime/nnf/nnf_tf_freezer/inception3-64/cuda_codegen /workspace/tacker/runtime/nnf/nnf_tf_freezer/inception3-64/cuda_codegen/CMakeFiles/nnfusion_naive_rt.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/nnfusion_naive_rt.dir/depend

