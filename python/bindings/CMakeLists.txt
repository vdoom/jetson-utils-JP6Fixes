# clear CMakeCache of Python version
unset(PYTHONINTERP_FOUND CACHE)
unset(PYTHON_EXECUTABLE CACHE)
unset(PYTHON_VERSION_STRING CACHE)
unset(PYTHON_VERSION_MAJOR CACHE)
unset(PYTHON_VERSION_MINOR CACHE)
unset(PYTHON_VERSION_PATCH CACHE)

unset(PYTHON_INCLUDE_PATH CACHE)
unset(PYTHON_INCLUDE_DIRS CACHE)
unset(PYTHON_INCLUDE_DIR CACHE)
unset(PYTHON_LIBRARY CACHE)
unset(PYTHON_LIBRARIES CACHE)
unset(PYTHON_DEBUG_LIBRARIES CACHE)
unset(PYTHON_MODULE_PREFIX CACHE)
unset(PYTHON_MODULE_EXTENSION CACHE)

unset(PYTHONLIBS_FOUND CACHE)
unset(PYTHONLIBS_VERSION_STRING CACHE)

# locate requested python version
message("-- detecting Python ${PYTHON_BINDING_VERSION}...")

# FindPythonInterp/FindPythonLibs are deprecated and removed by CMake 4 (which
# ships with JetPack 7), so prefer FindPython3 and keep the old modules as a
# fallback for the CMake versions that predate it (< 3.12)
find_package(Python3 ${PYTHON_BINDING_VERSION} QUIET COMPONENTS Interpreter Development NumPy)

if(Python3_Interpreter_FOUND AND Python3_Development_FOUND)
	set(PYTHON_VERSION_MAJOR ${Python3_VERSION_MAJOR})
	set(PYTHON_VERSION_MINOR ${Python3_VERSION_MINOR})
	set(PYTHONLIBS_VERSION_STRING ${Python3_VERSION})
	set(PYTHON_INCLUDE_DIRS ${Python3_INCLUDE_DIRS})
	set(PYTHON_LIBRARIES Python3::Python)

	if(Python3_NumPy_FOUND)
		set(NUMPY_FOUND ON)
		set(NUMPY_VERSION ${Python3_NumPy_VERSION})
		set(NUMPY_INCLUDE_DIR ${Python3_NumPy_INCLUDE_DIRS})
	else()
		set(NUMPY_FOUND OFF)
	endif()
else()
	find_package(PythonInterp ${PYTHON_BINDING_VERSION} QUIET)
	find_package(PythonLibs ${PYTHON_BINDING_VERSION} QUIET)

	if(NOT ${PYTHONLIBS_FOUND})
		message("-- Python ${PYTHON_BINDING_VERSION} wasn't found")
		return()
	endif()

	# attempt to find numpy
	set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}")
	message("-- CMake module path:  ${CMAKE_MODULE_PATH}")
	find_package(NumPy QUIET)
endif()

message("-- found Python version:  ${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR} (${PYTHONLIBS_VERSION_STRING})")
message("-- found Python include:  ${PYTHON_INCLUDE_DIRS}")
message("-- found Python library:  ${PYTHON_LIBRARIES}") 

include_directories(${PYTHON_INCLUDE_DIRS})

if(${NUMPY_FOUND})
	message("-- found NumPy version:  ${NUMPY_VERSION}")
	message("-- found NumPy include:  ${NUMPY_INCLUDE_DIR}")

	add_definitions(-DHAS_NUMPY)
	include_directories(${NUMPY_INCLUDE_DIR})
	link_directories(${NUMPY_INCLUDE_DIR}/../lib)
else()
	message("-- NumPy not found")
endif()

# build the bindings
file(GLOB pythonUtilsSources *.cpp)

cuda_add_library(jetson-utils-python-${PYTHON_VERSION_MAJOR}${PYTHON_VERSION_MINOR} SHARED ${pythonUtilsSources})
target_link_libraries(jetson-utils-python-${PYTHON_VERSION_MAJOR}${PYTHON_VERSION_MINOR} jetson-utils ${PYTHON_LIBRARIES})

if(${NUMPY_FOUND})
	if(DEFINED NUMPY_VERSION AND NUMPY_VERSION VERSION_GREATER_EQUAL "2.0")
		message("-- NumPy version ${NUMPY_VERSION} detected (>= 2.0) - skipping npymath linkage")
	else()
		# link npymath by its full path - it's a static lib that lives inside the
		# numpy package, and a bare '-lnpymath' doesn't carry the -L search path
		# over to the targets that link this one (i.e. the jetson-inference bindings)
		find_library(NPYMATH_LIB npymath HINTS "${NUMPY_INCLUDE_DIR}/../lib" "${NUMPY_INCLUDE_DIR}/../../_core/lib")

		if(NPYMATH_LIB)
			message("-- NumPy version ${NUMPY_VERSION} detected (< 2.0) - linking npymath:  ${NPYMATH_LIB}")
			target_link_libraries(jetson-utils-python-${PYTHON_VERSION_MAJOR}${PYTHON_VERSION_MINOR} ${NPYMATH_LIB})
		else()
			message("-- NumPy version ${NUMPY_VERSION} detected (< 2.0) but npymath wasn't found - skipping npymath linkage")
		endif()
	endif()
endif()



set_target_properties(jetson-utils-python-${PYTHON_VERSION_MAJOR}${PYTHON_VERSION_MINOR}
				  PROPERTIES
					PREFIX ""
					OUTPUT_NAME "jetson_utils_python"
					LIBRARY_OUTPUT_DIRECTORY "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/python/${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}")

# on x86, install under /opt/conda/lib/pythonX.X/site-packages
# otherwise, install under /usr/lib/pythonX.X/dist-packages
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64")
	set(PYTHON_BINDING_INSTALL_DIR /opt/conda/lib/python${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}/site-packages)
else()
	set(PYTHON_BINDING_INSTALL_DIR /usr/lib/python${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}/dist-packages)
endif()

install(TARGETS jetson-utils-python-${PYTHON_VERSION_MAJOR}${PYTHON_VERSION_MINOR} DESTINATION ${PYTHON_BINDING_INSTALL_DIR})
install(DIRECTORY ../jetson_utils DESTINATION ${PYTHON_BINDING_INSTALL_DIR})
