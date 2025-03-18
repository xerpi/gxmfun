cmake_minimum_required(VERSION 3.10)

if(NOT DEFINED CMAKE_TOOLCHAIN_FILE)
  if(DEFINED ENV{VITASDK})
    set(CMAKE_TOOLCHAIN_FILE "$ENV{VITASDK}/share/vita.toolchain.cmake" CACHE PATH "toolchain file")
  else()
    message(FATAL_ERROR "Please define VITASDK to point to your SDK path!")
  endif()
endif()

project(gxmfun)

include("${VITASDK}/share/vita.cmake" REQUIRED)

set(VITA_APP_NAME "GXM Fun")
set(VITA_TITLEID "GXMFUN000")
set(VITA_VERSION  "01.00")

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(VITA_MKSFOEX_FLAGS "${VITA_MKSFOEX_FLAGS} -d PARENTAL_LEVEL=1")

include_directories(
	include
)

link_directories(
	${CMAKE_BINARY_DIR}
)

set(SOURCES
	source/main.c
	source/camera.c
	source/math_utils.c
)

set(VERTEX_SHADERS
	shader/clear_v.cg
	shader/color_v.cg
	shader/cube_v.cg
	shader/disable_color_buffer_v.cg
)

set(FRAGMENT_SHADERS
	shader/clear_f.cg
	shader/color_f.cg
	shader/cube_f.cg
	shader/disable_color_buffer_f.cg
)

foreach(shader ${VERTEX_SHADERS})
	get_filename_component(shader_we ${shader} NAME_WE)
	add_custom_command(OUTPUT "${shader_we}.gxp"
		COMMAND $ENV{PSP2CGC} -profile sce_vp_psp2 "${PROJECT_SOURCE_DIR}/${shader}"
			-o "${CMAKE_BINARY_DIR}/${shader_we}.gxp"
		DEPENDS ${shader}
		COMMENT "Compiling ${shader} to ${shader_we}.gxp"
	)
	list(APPEND SHADER_GXPS "${shader_we}.gxp")
endforeach()

foreach(shader ${FRAGMENT_SHADERS})
	get_filename_component(shader_we ${shader} NAME_WE)
	add_custom_command(OUTPUT "${shader_we}.gxp"
		COMMAND $ENV{PSP2CGC} -profile sce_fp_psp2 "${PROJECT_SOURCE_DIR}/${shader}"
			-o "${CMAKE_BINARY_DIR}/${shader_we}.gxp"
		DEPENDS ${shader}
		COMMENT "Compiling ${shader} to ${shader_we}.gxp"
	)
	list(APPEND SHADER_GXPS "${shader_we}.gxp")
endforeach()

foreach(gxp ${SHADER_GXPS})
	get_filename_component(gxp_we ${gxp} NAME_WE)
	add_custom_command(OUTPUT "${gxp_we}_gxp.o"
		COMMAND arm-vita-eabi-objcopy --input binary --output elf32-littlearm
			--binary-architecture arm "${gxp}"
			"${CMAKE_BINARY_DIR}/${gxp_we}_gxp.o"
		WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
		DEPENDS ${gxp}
		COMMENT "Objcopying ${gxp} to ${gxp_we}_gxp.o"
	)
	list(APPEND SHADER_OBJECTS "${gxp_we}_gxp.o")
endforeach()

add_executable(${PROJECT_NAME}
	${SOURCES}
	${SHADER_OBJECTS}
)

target_link_libraries(${PROJECT_NAME}
	-lm
	SceDisplay_stub
	SceGxm_stub
	SceCtrl_stub
	SceSysmodule_stub
)

vita_create_self(${PROJECT_NAME}.self ${PROJECT_NAME})

vita_create_vpk(${PROJECT_NAME}.vpk ${VITA_TITLEID} ${PROJECT_NAME}.self
	VERSION ${VITA_VERSION}
	NAME ${VITA_APP_NAME}
)

add_custom_target(vpksend
	COMMAND curl -T ${PROJECT_NAME}.vpk ftp://$(PSVITAIP):1337/ux0:/
	DEPENDS ${PROJECT_NAME}.vpk
)

add_custom_target(send
	COMMAND curl -T ${PROJECT_NAME}.self ftp://$(PSVITAIP):1337/ux0:/app/${TITLE_ID}/eboot.bin
	DEPENDS ${PROJECT_NAME}.self
)
