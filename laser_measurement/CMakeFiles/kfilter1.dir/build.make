# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 2.8

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
CMAKE_SOURCE_DIR = /root/carnd_term2_notebook/kfilter

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /root/carnd_term2_notebook/kfilter

# Include any dependencies generated for this target.
include CMakeFiles/kfilter1.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/kfilter1.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/kfilter1.dir/flags.make

CMakeFiles/kfilter1.dir/practice-1.cpp.o: CMakeFiles/kfilter1.dir/flags.make
CMakeFiles/kfilter1.dir/practice-1.cpp.o: practice-1.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /root/carnd_term2_notebook/kfilter/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/kfilter1.dir/practice-1.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/kfilter1.dir/practice-1.cpp.o -c /root/carnd_term2_notebook/kfilter/practice-1.cpp

CMakeFiles/kfilter1.dir/practice-1.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/kfilter1.dir/practice-1.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /root/carnd_term2_notebook/kfilter/practice-1.cpp > CMakeFiles/kfilter1.dir/practice-1.cpp.i

CMakeFiles/kfilter1.dir/practice-1.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/kfilter1.dir/practice-1.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /root/carnd_term2_notebook/kfilter/practice-1.cpp -o CMakeFiles/kfilter1.dir/practice-1.cpp.s

CMakeFiles/kfilter1.dir/practice-1.cpp.o.requires:
.PHONY : CMakeFiles/kfilter1.dir/practice-1.cpp.o.requires

CMakeFiles/kfilter1.dir/practice-1.cpp.o.provides: CMakeFiles/kfilter1.dir/practice-1.cpp.o.requires
	$(MAKE) -f CMakeFiles/kfilter1.dir/build.make CMakeFiles/kfilter1.dir/practice-1.cpp.o.provides.build
.PHONY : CMakeFiles/kfilter1.dir/practice-1.cpp.o.provides

CMakeFiles/kfilter1.dir/practice-1.cpp.o.provides.build: CMakeFiles/kfilter1.dir/practice-1.cpp.o

# Object files for target kfilter1
kfilter1_OBJECTS = \
"CMakeFiles/kfilter1.dir/practice-1.cpp.o"

# External object files for target kfilter1
kfilter1_EXTERNAL_OBJECTS =

kfilter1: CMakeFiles/kfilter1.dir/practice-1.cpp.o
kfilter1: CMakeFiles/kfilter1.dir/build.make
kfilter1: CMakeFiles/kfilter1.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX executable kfilter1"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/kfilter1.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/kfilter1.dir/build: kfilter1
.PHONY : CMakeFiles/kfilter1.dir/build

CMakeFiles/kfilter1.dir/requires: CMakeFiles/kfilter1.dir/practice-1.cpp.o.requires
.PHONY : CMakeFiles/kfilter1.dir/requires

CMakeFiles/kfilter1.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/kfilter1.dir/cmake_clean.cmake
.PHONY : CMakeFiles/kfilter1.dir/clean

CMakeFiles/kfilter1.dir/depend:
	cd /root/carnd_term2_notebook/kfilter && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /root/carnd_term2_notebook/kfilter /root/carnd_term2_notebook/kfilter /root/carnd_term2_notebook/kfilter /root/carnd_term2_notebook/kfilter /root/carnd_term2_notebook/kfilter/CMakeFiles/kfilter1.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/kfilter1.dir/depend
