# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.5

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
CMAKE_SOURCE_DIR = /home/joshua/TrackObj

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/joshua/TrackObj

# Include any dependencies generated for this target.
include CMakeFiles/test_kf.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/test_kf.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/test_kf.dir/flags.make

CMakeFiles/test_kf.dir/test.cpp.o: CMakeFiles/test_kf.dir/flags.make
CMakeFiles/test_kf.dir/test.cpp.o: test.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/joshua/TrackObj/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/test_kf.dir/test.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/test_kf.dir/test.cpp.o -c /home/joshua/TrackObj/test.cpp

CMakeFiles/test_kf.dir/test.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/test_kf.dir/test.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/joshua/TrackObj/test.cpp > CMakeFiles/test_kf.dir/test.cpp.i

CMakeFiles/test_kf.dir/test.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/test_kf.dir/test.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/joshua/TrackObj/test.cpp -o CMakeFiles/test_kf.dir/test.cpp.s

CMakeFiles/test_kf.dir/test.cpp.o.requires:

.PHONY : CMakeFiles/test_kf.dir/test.cpp.o.requires

CMakeFiles/test_kf.dir/test.cpp.o.provides: CMakeFiles/test_kf.dir/test.cpp.o.requires
	$(MAKE) -f CMakeFiles/test_kf.dir/build.make CMakeFiles/test_kf.dir/test.cpp.o.provides.build
.PHONY : CMakeFiles/test_kf.dir/test.cpp.o.provides

CMakeFiles/test_kf.dir/test.cpp.o.provides.build: CMakeFiles/test_kf.dir/test.cpp.o


# Object files for target test_kf
test_kf_OBJECTS = \
"CMakeFiles/test_kf.dir/test.cpp.o"

# External object files for target test_kf
test_kf_EXTERNAL_OBJECTS =

test_kf: CMakeFiles/test_kf.dir/test.cpp.o
test_kf: CMakeFiles/test_kf.dir/build.make
test_kf: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_stitching3.so.3.3.1
test_kf: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_superres3.so.3.3.1
test_kf: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_videostab3.so.3.3.1
test_kf: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_aruco3.so.3.3.1
test_kf: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_bgsegm3.so.3.3.1
test_kf: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_bioinspired3.so.3.3.1
test_kf: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_ccalib3.so.3.3.1
test_kf: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_cvv3.so.3.3.1
test_kf: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_dpm3.so.3.3.1
test_kf: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_face3.so.3.3.1
test_kf: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_fuzzy3.so.3.3.1
test_kf: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_hdf3.so.3.3.1
test_kf: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_img_hash3.so.3.3.1
test_kf: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_line_descriptor3.so.3.3.1
test_kf: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_optflow3.so.3.3.1
test_kf: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_reg3.so.3.3.1
test_kf: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_rgbd3.so.3.3.1
test_kf: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_saliency3.so.3.3.1
test_kf: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_stereo3.so.3.3.1
test_kf: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_structured_light3.so.3.3.1
test_kf: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_surface_matching3.so.3.3.1
test_kf: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_tracking3.so.3.3.1
test_kf: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_xfeatures2d3.so.3.3.1
test_kf: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_ximgproc3.so.3.3.1
test_kf: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_xobjdetect3.so.3.3.1
test_kf: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_xphoto3.so.3.3.1
test_kf: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_shape3.so.3.3.1
test_kf: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_photo3.so.3.3.1
test_kf: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_datasets3.so.3.3.1
test_kf: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_plot3.so.3.3.1
test_kf: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_text3.so.3.3.1
test_kf: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_dnn3.so.3.3.1
test_kf: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_ml3.so.3.3.1
test_kf: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_video3.so.3.3.1
test_kf: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_calib3d3.so.3.3.1
test_kf: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_features2d3.so.3.3.1
test_kf: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_highgui3.so.3.3.1
test_kf: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_videoio3.so.3.3.1
test_kf: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_viz3.so.3.3.1
test_kf: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_phase_unwrapping3.so.3.3.1
test_kf: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_flann3.so.3.3.1
test_kf: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_imgcodecs3.so.3.3.1
test_kf: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_objdetect3.so.3.3.1
test_kf: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_imgproc3.so.3.3.1
test_kf: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_core3.so.3.3.1
test_kf: CMakeFiles/test_kf.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/joshua/TrackObj/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable test_kf"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/test_kf.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/test_kf.dir/build: test_kf

.PHONY : CMakeFiles/test_kf.dir/build

CMakeFiles/test_kf.dir/requires: CMakeFiles/test_kf.dir/test.cpp.o.requires

.PHONY : CMakeFiles/test_kf.dir/requires

CMakeFiles/test_kf.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/test_kf.dir/cmake_clean.cmake
.PHONY : CMakeFiles/test_kf.dir/clean

CMakeFiles/test_kf.dir/depend:
	cd /home/joshua/TrackObj && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/joshua/TrackObj /home/joshua/TrackObj /home/joshua/TrackObj /home/joshua/TrackObj /home/joshua/TrackObj/CMakeFiles/test_kf.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/test_kf.dir/depend

