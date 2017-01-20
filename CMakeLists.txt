cmake_minimum_required(VERSION 2.8)

set(CMAKE_SYSTEM_NAME "Generic")
set(CMAKE_C_COMPILER "arm-vita-eabi-gcc")
set(CMAKE_CXX_COMPILER "arm-vita-eabi-g++")

project(gxmfun)
set(TITLE_ID "GXMFUN000")
set(TITLE_NAME "GXM Fun")

set(CMAKE_C_FLAGS "-Wl,-q -Wall -O2")
set(CMAKE_CXX_FLAGS "${CMAKE_C_FLAGS} -std=c++11 -fno-rtti -fno-exceptions")

include_directories(
	include
)

link_directories(
	${CMAKE_BINARY_DIR}
)

set(SOURCES
	source/main.c
	source/netlog.c
	source/math_utils.c
)

set(VERTEX_SHADERS
	shader/clear_v.cg
	shader/color_v.cg
	shader/cube_v.cg
)

set(FRAGMENT_SHADERS
	shader/clear_f.cg
	shader/color_f.cg
	shader/cube_f.cg
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

add_executable(${PROJECT_NAME}.elf
	${SOURCES}
	${SHADER_OBJECTS}
)

target_link_libraries(${PROJECT_NAME}.elf
	-lm
	SceDisplay_stub
	SceGxm_stub
	SceCtrl_stub
	SceSysmodule_stub
)

add_custom_command(OUTPUT ${PROJECT_NAME}.velf
	COMMAND vita-elf-create ${PROJECT_NAME}.elf ${PROJECT_NAME}.velf
	DEPENDS ${PROJECT_NAME}.elf
)

add_custom_command(OUTPUT ${PROJECT_NAME}.self
	COMMAND vita-make-fself -c -s ${PROJECT_NAME}.velf ${PROJECT_NAME}.self
	DEPENDS ${PROJECT_NAME}.velf
)

add_custom_command(OUTPUT ${PROJECT_NAME}.vpk
	COMMAND vita-mksfoex -s TITLE_ID=${TITLE_ID} "${TITLE_NAME}" param.sfo
	COMMAND vita-pack-vpk -s param.sfo -b ${PROJECT_NAME}.self ${PROJECT_NAME}.vpk
	DEPENDS ${PROJECT_NAME}.self
)

add_custom_target(vpksend
	COMMAND curl -T ${PROJECT_NAME}.vpk ftp://$(PSVITAIP):1337/ux0:/
	DEPENDS ${PROJECT_NAME}.vpk
)

add_custom_target(send
	COMMAND curl -T ${PROJECT_NAME}.self ftp://$(PSVITAIP):1337/ux0:/app/${TITLE_ID}/eboot.bin
	DEPENDS ${PROJECT_NAME}.self
)
