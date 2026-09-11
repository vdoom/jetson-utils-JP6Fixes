
cmake_minimum_required(VERSION 3.5)

# determine if jetson-utils is being built as a submodule inside another repo,
# or if it's being build standalone (if the later, we need to do some configuration)
get_directory_property(hasParent PARENT_DIRECTORY)

if(hasParent)
	message("-- jetson-utils:  building as submodule, ${hasParent}")
else()
	message("-- jetson-utils:  building as standalone")
	
	# standalone project
	project(jetson-utils)
  
	# -std=gnu++11
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wno-write-strings")	
  
	# setup CUDA
	set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cuda")
	find_package(CUDA)
	message("-- CUDA version: ${CUDA_VERSION}")

	set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -O3)

# Jetson SM architectures:  53=Nano/TX1, 62=TX2, 72=Xavier, 87=Orin, 110=Thor
# newer CUDA toolkits drop support for older SM's (CUDA 13 / JetPack 7 removed
# 53/62/72), so the list gets filtered against what the installed nvcc actually
# accepts - that keeps JetPack 4/5/6 generating the same code as before, while
# letting JetPack 7 build for the architectures that are still supported.
	set(CUDA_ARCH_LIST "53;62;72;87;110" CACHE STRING "CUDA architectures (SM versions) to generate code for")

	if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
		execute_process(COMMAND ${CUDA_NVCC_EXECUTABLE} --list-gpu-arch
					OUTPUT_VARIABLE NVCC_ARCH_OUTPUT
					RESULT_VARIABLE NVCC_ARCH_RESULT
					ERROR_QUIET)

		string(REPLACE "\n" ";" NVCC_ARCH_SUPPORTED "${NVCC_ARCH_OUTPUT}")
		set(CUDA_ARCH_ENABLED "")

		foreach(SM ${CUDA_ARCH_LIST})
			if(NVCC_ARCH_RESULT EQUAL 0)
				# nvcc reports the architectures that it supports (CUDA 11.0 and newer)
				list(FIND NVCC_ARCH_SUPPORTED "compute_${SM}" SM_INDEX)
			else()
				# --list-gpu-arch is unavailable (CUDA 10 and older), fall back to version checks
				set(SM_INDEX -1)

				if(SM LESS 72 OR (SM EQUAL 72 AND CUDA_VERSION_MAJOR GREATER 9))
					set(SM_INDEX 0)
				endif()
			endif()

			if(NOT SM_INDEX EQUAL -1)
				set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -gencode arch=compute_${SM},code=sm_${SM})
				list(APPEND CUDA_ARCH_ENABLED SM_${SM})
			endif()
		endforeach()

		if(CUDA_ARCH_ENABLED)
			string(REPLACE ";" " " CUDA_ARCH_ENABLED "${CUDA_ARCH_ENABLED}")
			message("-- CUDA ${CUDA_VERSION} detected (${CMAKE_SYSTEM_PROCESSOR}), enabling ${CUDA_ARCH_ENABLED}")
		else()
			message(WARNING "CUDA ${CUDA_VERSION} doesn't support any of the SM architectures in CUDA_ARCH_LIST (${CUDA_ARCH_LIST}), falling back to the nvcc defaults")
		endif()
	endif()
	
	# setup project output paths
	set(PROJECT_OUTPUT_DIR  ${PROJECT_BINARY_DIR}/${CMAKE_SYSTEM_PROCESSOR})
	set(PROJECT_INCLUDE_DIR ${PROJECT_OUTPUT_DIR}/include)

	file(MAKE_DIRECTORY ${PROJECT_INCLUDE_DIR})
	file(MAKE_DIRECTORY ${PROJECT_OUTPUT_DIR}/bin)

	message("-- system arch:  ${CMAKE_SYSTEM_PROCESSOR}")
	message("-- output path:  ${PROJECT_OUTPUT_DIR}")

	set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_OUTPUT_DIR}/bin)
	set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_OUTPUT_DIR}/lib)
	set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_OUTPUT_DIR}/lib)
	
	# detect distro version
	find_program(LSB_RELEASE_EXEC lsb_release)

	execute_process(COMMAND "${LSB_RELEASE_EXEC}" --short --id OUTPUT_VARIABLE LSB_RELEASE_ID OUTPUT_STRIP_TRAILING_WHITESPACE)
	execute_process(COMMAND "${LSB_RELEASE_EXEC}" --short --release OUTPUT_VARIABLE LSB_RELEASE_VERSION OUTPUT_STRIP_TRAILING_WHITESPACE)
	execute_process(COMMAND "${LSB_RELEASE_EXEC}" --short --codename OUTPUT_VARIABLE LSB_RELEASE_CODENAME OUTPUT_STRIP_TRAILING_WHITESPACE)

	message("-- distro ID:       ${LSB_RELEASE_ID}")
	message("-- distro version:  ${LSB_RELEASE_VERSION}")
	message("-- distro codename: ${LSB_RELEASE_CODENAME}")

	# enable includes like <jetson_utils/xyz.h>
	include_directories(${PROJECT_INCLUDE_DIR})

endif()

# option for enabling/disabling NVMM memory in multimedia stack
find_library(NVBUF_UTILS NAMES nvbuf_utils PATHS /usr/lib/${CMAKE_SYSTEM_PROCESSOR}-linux-gnu/tegra)
message("-- nvbuf_utils:  ${NVBUF_UTILS}")

if(NVBUF_UTILS)
	set(ENABLE_NVMM_DEFAULT ON)
else()
	set(ENABLE_NVMM_DEFAULT OFF)
endif()

option(ENABLE_NVMM "Enable use of NVMM zero-copy memory in video and camera streaming" ${ENABLE_NVMM_DEFAULT})
message("-- NVMM zero-copy memory:  ENABLE_NVMM=${ENABLE_NVMM}")

if(ENABLE_NVMM)
	add_definitions(-DENABLE_NVMM)
endif()

# additional paths for includes and libraries
include_directories(${PROJECT_INCLUDE_DIR}/jetson-utils)

include_directories(
	/usr/include/gstreamer-1.0 
	/usr/include/libxml2  /usr/include/libsoup-2.4 
	/usr/include/glib-2.0 /usr/include/json-glib-1.0
	/usr/lib/${CMAKE_SYSTEM_PROCESSOR}-linux-gnu/gstreamer-1.0/include 
	/usr/lib/${CMAKE_SYSTEM_PROCESSOR}-linux-gnu/glib-2.0/include/
)

if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
	include_directories(/usr/src/jetson_multimedia_api/include)
	link_directories(/usr/lib/aarch64-linux-gnu/tegra)
endif()

# build library
file(GLOB jetsonUtilsSrc cpp/*/*.cpp cuda/*.cpp cuda/*.cu)
file(GLOB jetsonUtilsInc cpp/*/*.h cpp/*/*.hpp cpp/*/*.inl cuda/*.h cuda/*.cuh)

cuda_add_library(jetson-utils SHARED ${jetsonUtilsSrc})

target_link_libraries( jetson-utils 
	GL GLU GLEW ${CUDA_nppicc_LIBRARY}
	gstreamer-1.0 gstapp-1.0 gstpbutils-1.0 
	gstwebrtc-1.0 gstsdp-1.0 gstrtspserver-1.0 
	json-glib-1.0 soup-2.4
)	

if(NVBUF_UTILS)
	target_link_libraries(jetson-utils nvbuf_utils)
endif()

# transfer all headers to the include directory 
file(MAKE_DIRECTORY ${PROJECT_INCLUDE_DIR}/jetson-utils)

foreach(include ${jetsonUtilsInc})
	message("-- Copying ${include}")
	configure_file(${include} ${PROJECT_INCLUDE_DIR}/jetson-utils COPYONLY)
endforeach()
	
# install headers
foreach(include ${jetsonUtilsInc})
	install(FILES "${include}" DESTINATION include/jetson-utils)
endforeach()

# install the shared library
install(TARGETS jetson-utils DESTINATION lib EXPORT jetson-utilsConfig)

# install the cmake project, for importing
install(EXPORT jetson-utilsConfig DESTINATION share/jetson-utils/cmake)

# build python bindings
add_subdirectory(python)

# build sample apps
add_subdirectory(cpp/video/video-viewer)

#add_subdirectory(camera/camera-viewer)
#add_subdirectory(display/gl-display-test)
#add_subdirectory(network/webrtc-server)
#add_subdirectory(network/rtsp-server)

