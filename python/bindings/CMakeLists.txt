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

message("-- detecting Python ${PYTHON_BINDING_VERSION}...")

if(DEFINED ENV{VIRTUAL_ENV})
  set(Python3_FIND_VIRTUALENV FIRST)
endif()

# Interpreter + NumPy only — we intentionally do not link libpython into the
# extension module; Py* symbols are resolved at dlopen time from the host
# interpreter. This lets the build succeed against altinstalled Python
# distributions that ship only a static libpython.
find_package(Python3 ${PYTHON_BINDING_VERSION} QUIET COMPONENTS Interpreter NumPy)

if(NOT Python3_Interpreter_FOUND)
  find_package(PythonInterp ${PYTHON_BINDING_VERSION} QUIET)
  if(NOT PYTHONINTERP_FOUND)
    message("-- Python ${PYTHON_BINDING_VERSION} interpreter wasn't found")
    return()
  endif()
  set(Python3_EXECUTABLE ${PYTHON_EXECUTABLE})
  set(Python3_VERSION_MAJOR ${PYTHON_VERSION_MAJOR})
  set(Python3_VERSION_MINOR ${PYTHON_VERSION_MINOR})
endif()

set(PYTHON_VERSION_MAJOR ${Python3_VERSION_MAJOR})
set(PYTHON_VERSION_MINOR ${Python3_VERSION_MINOR})

execute_process(
  COMMAND ${Python3_EXECUTABLE} -c
    "import sysconfig; print(sysconfig.get_paths()['include'])"
  OUTPUT_VARIABLE PYTHON_INCLUDE_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE)

message("-- found Python version:  ${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}")
message("-- found Python interp:   ${Python3_EXECUTABLE}")
message("-- found Python include:  ${PYTHON_INCLUDE_DIR}")

# NumPy
set(NUMPY_FOUND OFF)
if(Python3_NumPy_FOUND)
  set(NUMPY_FOUND ON)
  set(NUMPY_VERSION ${Python3_NumPy_VERSION})
  set(NUMPY_INCLUDE_DIR ${Python3_NumPy_INCLUDE_DIRS})
else()
  execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import numpy; print(numpy.get_include())"
    OUTPUT_VARIABLE NUMPY_INCLUDE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE _NUMPY_RC)
  if(_NUMPY_RC EQUAL 0 AND NUMPY_INCLUDE_DIR)
    set(NUMPY_FOUND ON)
    execute_process(
      COMMAND ${Python3_EXECUTABLE} -c "import numpy; print(numpy.__version__)"
      OUTPUT_VARIABLE NUMPY_VERSION
      OUTPUT_STRIP_TRAILING_WHITESPACE)
  endif()
endif()

if(NUMPY_FOUND)
  message("-- found NumPy version:   ${NUMPY_VERSION}")
  message("-- found NumPy include:   ${NUMPY_INCLUDE_DIR}")
  add_definitions(-DHAS_NUMPY)
else()
  message("-- NumPy not found")
endif()

# build the bindings
file(GLOB pythonUtilsSources *.cpp)

set(_PYMOD jetson-utils-python-${PYTHON_VERSION_MAJOR}${PYTHON_VERSION_MINOR})
set(CUDA_LINK_LIBRARIES_KEYWORD PRIVATE)
cuda_add_library(${_PYMOD} SHARED ${pythonUtilsSources})

# Headers only — do NOT link libpython.
target_include_directories(${_PYMOD} PRIVATE ${PYTHON_INCLUDE_DIR})
target_link_libraries(${_PYMOD} PRIVATE jetson-utils)

if(NUMPY_FOUND)
  target_include_directories(${_PYMOD} PRIVATE ${NUMPY_INCLUDE_DIR})

  if(DEFINED NUMPY_VERSION AND NUMPY_VERSION VERSION_GREATER_EQUAL "2.0")
    message("-- NumPy version ${NUMPY_VERSION} detected (>= 2.0) - skipping npymath linkage")
  else()
    find_library(NPYMATH_LIB NAMES npymath)
    if(NPYMATH_LIB)
      target_link_libraries(${_PYMOD} PRIVATE ${NPYMATH_LIB})
      message(STATUS "NumPy ${NUMPY_VERSION} (< 2.0): linking npymath -> ${NPYMATH_LIB}")
    else()
      message(WARNING "NumPy ${NUMPY_VERSION} (< 2.0) but 'npymath' not found: continuing without it")
    endif()
  endif()
endif()

set_target_properties(${_PYMOD}
  PROPERTIES
    PREFIX ""
    OUTPUT_NAME "jetson_utils_python"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/python/${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}")

# Install paths: honor -DPYTHON_BINDING_INSTALL_DIR=..., then active venv,
# otherwise historical defaults.
if(NOT DEFINED PYTHON_BINDING_INSTALL_DIR OR PYTHON_BINDING_INSTALL_DIR STREQUAL "")
  if(DEFINED ENV{VIRTUAL_ENV})
    set(PYTHON_BINDING_INSTALL_DIR "$ENV{VIRTUAL_ENV}/lib/python${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}/site-packages")
    message("-- installing Python bindings into active venv: ${PYTHON_BINDING_INSTALL_DIR}")
  elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64")
    set(PYTHON_BINDING_INSTALL_DIR /opt/conda/lib/python${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}/site-packages)
  else()
    set(PYTHON_BINDING_INSTALL_DIR /usr/lib/python${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}/dist-packages)
  endif()
endif()

install(TARGETS ${_PYMOD} DESTINATION ${PYTHON_BINDING_INSTALL_DIR})
install(DIRECTORY ../jetson_utils DESTINATION ${PYTHON_BINDING_INSTALL_DIR})
