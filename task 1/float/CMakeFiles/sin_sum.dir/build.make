# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
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
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/ivank/parrallel/1_task

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/ivank/parrallel/1_task/float

# Include any dependencies generated for this target.
include CMakeFiles/sin_sum.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/sin_sum.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/sin_sum.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/sin_sum.dir/flags.make

CMakeFiles/sin_sum.dir/main.cpp.o: CMakeFiles/sin_sum.dir/flags.make
CMakeFiles/sin_sum.dir/main.cpp.o: ../main.cpp
CMakeFiles/sin_sum.dir/main.cpp.o: CMakeFiles/sin_sum.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/ivank/parrallel/1_task/float/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/sin_sum.dir/main.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/sin_sum.dir/main.cpp.o -MF CMakeFiles/sin_sum.dir/main.cpp.o.d -o CMakeFiles/sin_sum.dir/main.cpp.o -c /home/ivank/parrallel/1_task/main.cpp

CMakeFiles/sin_sum.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/sin_sum.dir/main.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/ivank/parrallel/1_task/main.cpp > CMakeFiles/sin_sum.dir/main.cpp.i

CMakeFiles/sin_sum.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/sin_sum.dir/main.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/ivank/parrallel/1_task/main.cpp -o CMakeFiles/sin_sum.dir/main.cpp.s

# Object files for target sin_sum
sin_sum_OBJECTS = \
"CMakeFiles/sin_sum.dir/main.cpp.o"

# External object files for target sin_sum
sin_sum_EXTERNAL_OBJECTS =

sin_sum: CMakeFiles/sin_sum.dir/main.cpp.o
sin_sum: CMakeFiles/sin_sum.dir/build.make
sin_sum: CMakeFiles/sin_sum.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/ivank/parrallel/1_task/float/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable sin_sum"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/sin_sum.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/sin_sum.dir/build: sin_sum
.PHONY : CMakeFiles/sin_sum.dir/build

CMakeFiles/sin_sum.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/sin_sum.dir/cmake_clean.cmake
.PHONY : CMakeFiles/sin_sum.dir/clean

CMakeFiles/sin_sum.dir/depend:
	cd /home/ivank/parrallel/1_task/float && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/ivank/parrallel/1_task /home/ivank/parrallel/1_task /home/ivank/parrallel/1_task/float /home/ivank/parrallel/1_task/float /home/ivank/parrallel/1_task/float/CMakeFiles/sin_sum.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/sin_sum.dir/depend
