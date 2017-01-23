#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <psp2/gxm.h>
#include <psp2/display.h>
#include <psp2/ctrl.h>
#include <psp2/kernel/sysmem.h>
#include "math_utils.h"
#include "camera.h"
#include "netlog.h"

#define ALIGN(x, a) (((x) + ((a) - 1)) & ~((a) - 1))
#define abs(x) (((x) < 0) ? -(x) : (x))

#define ANALOG_THRESHOLD 20

#define DISPLAY_WIDTH 960
#define DISPLAY_HEIGHT 544
#define DISPLAY_STRIDE 1024
#define DISPLAY_BUFFER_COUNT 2
#define DISPLAY_COLOR_FORMAT SCE_GXM_COLOR_FORMAT_A8B8G8R8
#define DISPLAY_PIXEL_FORMAT SCE_DISPLAY_PIXELFORMAT_A8B8G8R8
#define MAX_PENDING_SWAPS (DISPLAY_BUFFER_COUNT - 1)

struct clear_vertex {
	vector2f position;
};

struct position_vertex {
	vector3f position;
};

struct color_vertex {
	vector3f position;
	vector4f color;
};

struct mesh_vertex {
	vector3f position;
	vector3f normal;
	vector4f color;
};

struct phong_material {
	vector3f ambient;
	vector3f diffuse;
	vector3f specular;
	float shininess;
};

struct light {
	vector3f position;
	vector3f color;
};

struct portal {
	float width;
	float height;
	struct {
		matrix4x4 model_matrix;
	} end1, end2;
};


struct scene_state {
	/* Cube parameters */
	float trans_x;
	float trans_y;
	float trans_z;
	float rot_y;
	float rot_x;

	/* Light parameters */
	float light_distance;
	float light_x_rot;
	float light_y_rot;

	struct light light;

	struct portal portal;
};

struct phong_material_gxm_params {
	const SceGxmProgramParameter *ambient;
	const SceGxmProgramParameter *diffuse;
	const SceGxmProgramParameter *specular;
	const SceGxmProgramParameter *shininess;
};

struct light_gxm_params {
	const SceGxmProgramParameter *position;
	const SceGxmProgramParameter *color;
};

struct display_queue_callback_data {
	void *addr;
};

extern unsigned char _binary_disable_color_buffer_v_gxp_start;
extern unsigned char _binary_disable_color_buffer_f_gxp_start;
extern unsigned char _binary_clear_v_gxp_start;
extern unsigned char _binary_clear_f_gxp_start;
extern unsigned char _binary_cube_v_gxp_start;
extern unsigned char _binary_cube_f_gxp_start;

static const SceGxmProgram *const gxm_program_disable_color_buffer_v = (SceGxmProgram *)&_binary_disable_color_buffer_v_gxp_start;
static const SceGxmProgram *const gxm_program_disable_color_buffer_f = (SceGxmProgram *)&_binary_disable_color_buffer_f_gxp_start;
static const SceGxmProgram *const gxm_program_clear_v = (SceGxmProgram *)&_binary_clear_v_gxp_start;
static const SceGxmProgram *const gxm_program_clear_f = (SceGxmProgram *)&_binary_clear_f_gxp_start;
static const SceGxmProgram *const gxm_program_cube_v = (SceGxmProgram *)&_binary_cube_v_gxp_start;
static const SceGxmProgram *const gxm_program_cube_f = (SceGxmProgram *)&_binary_cube_f_gxp_start;

static SceGxmContext *gxm_context;
static SceUID vdm_ring_buffer_uid;
static void *vdm_ring_buffer_addr;
static SceUID vertex_ring_buffer_uid;
static void *vertex_ring_buffer_addr;
static SceUID fragment_ring_buffer_uid;
static void *fragment_ring_buffer_addr;
static SceUID fragment_usse_ring_buffer_uid;
static void *fragment_usse_ring_buffer_addr;
static SceGxmRenderTarget *gxm_render_target;
static SceGxmColorSurface gxm_color_surfaces[DISPLAY_BUFFER_COUNT];
static SceUID gxm_color_surfaces_uid[DISPLAY_BUFFER_COUNT];
static void *gxm_color_surfaces_addr[DISPLAY_BUFFER_COUNT];
static SceGxmSyncObject *gxm_sync_objects[DISPLAY_BUFFER_COUNT];
static unsigned int gxm_front_buffer_index;
static unsigned int gxm_back_buffer_index;
static SceUID gxm_depth_stencil_surface_uid;
static void *gxm_depth_stencil_surface_addr;
static SceGxmDepthStencilSurface gxm_depth_stencil_surface;
static SceGxmShaderPatcher *gxm_shader_patcher;
static SceUID gxm_shader_patcher_buffer_uid;
static void *gxm_shader_patcher_buffer_addr;
static SceUID gxm_shader_patcher_vertex_usse_uid;
static void *gxm_shader_patcher_vertex_usse_addr;
static SceUID gxm_shader_patcher_fragment_usse_uid;
static void *gxm_shader_patcher_fragment_usse_addr;

static SceGxmShaderPatcherId gxm_disable_color_buffer_vertex_program_id;
static SceGxmShaderPatcherId gxm_disable_color_buffer_fragment_program_id;
static const SceGxmProgramParameter *gxm_disable_color_buffer_vertex_program_position_param;
static const SceGxmProgramParameter *gxm_disable_color_buffer_vertex_program_u_mvp_matrix_param;
static SceGxmVertexProgram *gxm_disable_color_buffer_vertex_program_patched;
static SceGxmFragmentProgram *gxm_disable_color_buffer_fragment_program_patched;

static SceGxmShaderPatcherId gxm_clear_vertex_program_id;
static SceGxmShaderPatcherId gxm_clear_fragment_program_id;
static const SceGxmProgramParameter *gxm_clear_vertex_program_position_param;
static const SceGxmProgramParameter *gxm_clear_fragment_program_u_clear_color_param;
static SceGxmVertexProgram *gxm_clear_vertex_program_patched;
static SceGxmFragmentProgram *gxm_clear_fragment_program_patched;

static SceGxmShaderPatcherId gxm_cube_vertex_program_id;
static SceGxmShaderPatcherId gxm_cube_fragment_program_id;
static const SceGxmProgramParameter *gxm_cube_vertex_program_position_param;
static const SceGxmProgramParameter *gxm_cube_vertex_program_normal_param;
static const SceGxmProgramParameter *gxm_cube_vertex_program_color_param;
static const SceGxmProgramParameter *gxm_cube_vertex_program_u_mvp_matrix_param;
static const SceGxmProgramParameter *gxm_cube_fragment_program_u_modelview_matrix_param;
static const SceGxmProgramParameter *gxm_cube_fragment_program_u_normal_matrix_param;
static struct phong_material_gxm_params gxm_cube_fragment_program_phong_material_params;
static struct light_gxm_params gxm_cube_fragment_program_light_params;
static SceGxmVertexProgram *gxm_cube_vertex_program_patched;
static SceGxmFragmentProgram *gxm_cube_fragment_program_patched;

static struct mesh_vertex *cube_mesh_data;
static unsigned short *cube_indices_data;
static struct mesh_vertex *floor_mesh_data;
static unsigned short *floor_indices_data;
static struct mesh_vertex *portal_frame_mesh_data;
static unsigned short *portal_frame_indices_data;

static void set_vertex_default_uniform_data(const SceGxmProgramParameter *param,
	unsigned int component_count, const void *data);
static void set_fragment_default_uniform_data(const SceGxmProgramParameter *param,
	unsigned int component_count, const void *data);
static void set_cube_fragment_material_uniform_params(const struct phong_material *material,
	const struct phong_material_gxm_params *params);
static void set_cube_matrices_uniform_params(matrix4x4 mvp_matrix,
	matrix4x4 modelview_matrix, matrix3x3 normal_matrix);
static void set_cube_fragment_light_uniform_params(const struct light *light,
	const struct light_gxm_params *params);

static void update_camera(struct camera *camera, SceCtrlData *pad);
static void draw_scene(const struct scene_state *state, matrix4x4 projection_matrix, matrix4x4 view_matrix);
static void update_scene(struct scene_state *state, const struct camera *camera, SceCtrlData *pad);

static void draw_cube(const struct scene_state *state, const matrix4x4 projection_matrix,
	const matrix4x4 view_matrix, const matrix4x4 model_matrix, const struct phong_material *material);

static void *gpu_alloc_map(SceKernelMemBlockType type, SceGxmMemoryAttribFlags gpu_attrib, size_t size, SceUID *uid);
static void gpu_unmap_free(SceUID uid);
static void *gpu_vertex_usse_alloc_map(size_t size, SceUID *uid, unsigned int *usse_offset);
static void gpu_vertex_usse_unmap_free(SceUID uid);
static void *gpu_fragment_usse_alloc_map(size_t size, SceUID *uid, unsigned int *usse_offset);
static void gpu_fragment_usse_unmap_free(SceUID uid);
static void *shader_patcher_host_alloc_cb(void *user_data, unsigned int size);
static void shader_patcher_host_free_cb(void *user_data, void *mem);
static void display_queue_callback(const void *callbackData);

int main(int argc, char *argv[])
{
	int i;

	netlog_init();
	netlog("GXM fun by xerpi\n");

	sceCtrlSetSamplingMode(SCE_CTRL_MODE_ANALOG);

	SceGxmInitializeParams gxm_init_params;
	memset(&gxm_init_params, 0, sizeof(gxm_init_params));
	gxm_init_params.flags = 0;
	gxm_init_params.displayQueueMaxPendingCount = MAX_PENDING_SWAPS;
	gxm_init_params.displayQueueCallback = display_queue_callback;
	gxm_init_params.displayQueueCallbackDataSize = sizeof(struct display_queue_callback_data);
	gxm_init_params.parameterBufferSize = SCE_GXM_DEFAULT_PARAMETER_BUFFER_SIZE;

	sceGxmInitialize(&gxm_init_params);

	vdm_ring_buffer_addr = gpu_alloc_map(SCE_KERNEL_MEMBLOCK_TYPE_USER_CDRAM_RW,
		SCE_GXM_MEMORY_ATTRIB_READ, SCE_GXM_DEFAULT_VDM_RING_BUFFER_SIZE,
		&vdm_ring_buffer_uid);

	vertex_ring_buffer_addr = gpu_alloc_map(SCE_KERNEL_MEMBLOCK_TYPE_USER_CDRAM_RW,
		SCE_GXM_MEMORY_ATTRIB_READ, SCE_GXM_DEFAULT_VERTEX_RING_BUFFER_SIZE,
		&vertex_ring_buffer_uid);

	fragment_ring_buffer_addr = gpu_alloc_map(SCE_KERNEL_MEMBLOCK_TYPE_USER_CDRAM_RW,
		SCE_GXM_MEMORY_ATTRIB_READ, SCE_GXM_DEFAULT_FRAGMENT_RING_BUFFER_SIZE,
		&fragment_ring_buffer_uid);

	unsigned int fragment_usse_offset;
	fragment_usse_ring_buffer_addr = gpu_fragment_usse_alloc_map(
		SCE_GXM_DEFAULT_FRAGMENT_USSE_RING_BUFFER_SIZE,
		&fragment_ring_buffer_uid, &fragment_usse_offset);

	SceGxmContextParams gxm_context_params;
	memset(&gxm_context_params, 0, sizeof(gxm_context_params));
	gxm_context_params.hostMem = malloc(SCE_GXM_MINIMUM_CONTEXT_HOST_MEM_SIZE);
	gxm_context_params.hostMemSize = SCE_GXM_MINIMUM_CONTEXT_HOST_MEM_SIZE;
	gxm_context_params.vdmRingBufferMem = vdm_ring_buffer_addr;
	gxm_context_params.vdmRingBufferMemSize = SCE_GXM_DEFAULT_VDM_RING_BUFFER_SIZE;
	gxm_context_params.vertexRingBufferMem = vertex_ring_buffer_addr;
	gxm_context_params.vertexRingBufferMemSize = SCE_GXM_DEFAULT_VERTEX_RING_BUFFER_SIZE;
	gxm_context_params.fragmentRingBufferMem = fragment_ring_buffer_addr;
	gxm_context_params.fragmentRingBufferMemSize = SCE_GXM_DEFAULT_FRAGMENT_RING_BUFFER_SIZE;
	gxm_context_params.fragmentUsseRingBufferMem = fragment_usse_ring_buffer_addr;
	gxm_context_params.fragmentUsseRingBufferMemSize = SCE_GXM_DEFAULT_FRAGMENT_USSE_RING_BUFFER_SIZE;
	gxm_context_params.fragmentUsseRingBufferOffset = fragment_usse_offset;

	sceGxmCreateContext(&gxm_context_params, &gxm_context);

	SceGxmRenderTargetParams render_target_params;
	memset(&render_target_params, 0, sizeof(render_target_params));
	render_target_params.flags = 0;
	render_target_params.width = DISPLAY_WIDTH;
	render_target_params.height = DISPLAY_HEIGHT;
	render_target_params.scenesPerFrame = 1;
	render_target_params.multisampleMode = SCE_GXM_MULTISAMPLE_NONE;
	render_target_params.multisampleLocations = 0;
	render_target_params.driverMemBlock = -1;

	sceGxmCreateRenderTarget(&render_target_params, &gxm_render_target);

	for (i = 0; i < DISPLAY_BUFFER_COUNT; i++) {
		gxm_color_surfaces_addr[i] = gpu_alloc_map(SCE_KERNEL_MEMBLOCK_TYPE_USER_CDRAM_RW,
			SCE_GXM_MEMORY_ATTRIB_READ | SCE_GXM_MEMORY_ATTRIB_WRITE,
			ALIGN(4 * DISPLAY_STRIDE * DISPLAY_HEIGHT, 1 * 1024 * 1024),
			&gxm_color_surfaces_uid[i]);

		memset(gxm_color_surfaces_addr[i], 0, DISPLAY_STRIDE * DISPLAY_HEIGHT);

		sceGxmColorSurfaceInit(&gxm_color_surfaces[i],
			DISPLAY_COLOR_FORMAT,
			SCE_GXM_COLOR_SURFACE_LINEAR,
			SCE_GXM_COLOR_SURFACE_SCALE_NONE,
			SCE_GXM_OUTPUT_REGISTER_SIZE_32BIT,
			DISPLAY_WIDTH,
			DISPLAY_HEIGHT,
			DISPLAY_STRIDE,
			gxm_color_surfaces_addr[i]);

		sceGxmSyncObjectCreate(&gxm_sync_objects[i]);
	}

	unsigned int depth_stencil_width = ALIGN(DISPLAY_WIDTH, SCE_GXM_TILE_SIZEX);
	unsigned int depth_stencil_height = ALIGN(DISPLAY_HEIGHT, SCE_GXM_TILE_SIZEY);
	unsigned int depth_stencil_samples = depth_stencil_width * depth_stencil_height;

	gxm_depth_stencil_surface_addr = gpu_alloc_map(SCE_KERNEL_MEMBLOCK_TYPE_USER_CDRAM_RW,
		SCE_GXM_MEMORY_ATTRIB_READ | SCE_GXM_MEMORY_ATTRIB_WRITE,
		4 * depth_stencil_samples, &gxm_depth_stencil_surface_uid);

	sceGxmDepthStencilSurfaceInit(&gxm_depth_stencil_surface,
		SCE_GXM_DEPTH_STENCIL_FORMAT_S8D24,
		SCE_GXM_DEPTH_STENCIL_SURFACE_TILED,
		depth_stencil_width,
		gxm_depth_stencil_surface_addr,
		NULL);

	static const unsigned int shader_patcher_buffer_size = 64 * 1024;
	static const unsigned int shader_patcher_vertex_usse_size = 64 * 1024;
	static const unsigned int shader_patcher_fragment_usse_size = 64 * 1024;

	gxm_shader_patcher_buffer_addr = gpu_alloc_map(SCE_KERNEL_MEMBLOCK_TYPE_USER_CDRAM_RW,
		SCE_GXM_MEMORY_ATTRIB_READ | SCE_GXM_MEMORY_ATTRIB_READ,
		shader_patcher_buffer_size, &gxm_shader_patcher_buffer_uid);

	unsigned int shader_patcher_vertex_usse_offset;
	gxm_shader_patcher_vertex_usse_addr = gpu_vertex_usse_alloc_map(
		shader_patcher_vertex_usse_size, &gxm_shader_patcher_vertex_usse_uid,
		&shader_patcher_vertex_usse_offset);

	unsigned int shader_patcher_fragment_usse_offset;
	gxm_shader_patcher_fragment_usse_addr = gpu_fragment_usse_alloc_map(
		shader_patcher_fragment_usse_size, &gxm_shader_patcher_fragment_usse_uid,
		&shader_patcher_fragment_usse_offset);

	SceGxmShaderPatcherParams shader_patcher_params;
	memset(&shader_patcher_params, 0, sizeof(shader_patcher_params));
	shader_patcher_params.userData = NULL;
	shader_patcher_params.hostAllocCallback = shader_patcher_host_alloc_cb;
	shader_patcher_params.hostFreeCallback = shader_patcher_host_free_cb;
	shader_patcher_params.bufferAllocCallback = NULL;
	shader_patcher_params.bufferFreeCallback = NULL;
	shader_patcher_params.bufferMem = gxm_shader_patcher_buffer_addr;
	shader_patcher_params.bufferMemSize = shader_patcher_buffer_size;
	shader_patcher_params.vertexUsseAllocCallback = NULL;
	shader_patcher_params.vertexUsseFreeCallback = NULL;
	shader_patcher_params.vertexUsseMem = gxm_shader_patcher_vertex_usse_addr;
	shader_patcher_params.vertexUsseMemSize = shader_patcher_vertex_usse_size;
	shader_patcher_params.vertexUsseOffset = shader_patcher_vertex_usse_offset;
	shader_patcher_params.fragmentUsseAllocCallback = NULL;
	shader_patcher_params.fragmentUsseFreeCallback = NULL;
	shader_patcher_params.fragmentUsseMem = gxm_shader_patcher_fragment_usse_addr;
	shader_patcher_params.fragmentUsseMemSize = shader_patcher_fragment_usse_size;
	shader_patcher_params.fragmentUsseOffset = shader_patcher_fragment_usse_offset;

	sceGxmShaderPatcherCreate(&shader_patcher_params, &gxm_shader_patcher);

	sceGxmShaderPatcherRegisterProgram(gxm_shader_patcher, gxm_program_disable_color_buffer_v,
		&gxm_disable_color_buffer_vertex_program_id);
	sceGxmShaderPatcherRegisterProgram(gxm_shader_patcher, gxm_program_disable_color_buffer_f,
		&gxm_disable_color_buffer_fragment_program_id);

	const SceGxmProgram *disable_color_buffer_vertex_program =
		sceGxmShaderPatcherGetProgramFromId(gxm_disable_color_buffer_vertex_program_id);
	const SceGxmProgram *disable_color_buffer_fragment_program =
		sceGxmShaderPatcherGetProgramFromId(gxm_disable_color_buffer_fragment_program_id);

	gxm_disable_color_buffer_vertex_program_position_param = sceGxmProgramFindParameterByName(
		disable_color_buffer_vertex_program, "position");

	gxm_disable_color_buffer_vertex_program_u_mvp_matrix_param = sceGxmProgramFindParameterByName(
		disable_color_buffer_vertex_program, "u_mvp_matrix");

	SceGxmVertexAttribute disable_color_buffer_attributes;
	SceGxmVertexStream disable_color_buffer_stream;
	disable_color_buffer_attributes.streamIndex = 0;
	disable_color_buffer_attributes.offset = 0;
	disable_color_buffer_attributes.format = SCE_GXM_ATTRIBUTE_FORMAT_F32;
	disable_color_buffer_attributes.componentCount = 3;
	disable_color_buffer_attributes.regIndex = sceGxmProgramParameterGetResourceIndex(
		gxm_disable_color_buffer_vertex_program_position_param);
	disable_color_buffer_stream.stride = sizeof(struct position_vertex);
	disable_color_buffer_stream.indexSource = SCE_GXM_INDEX_SOURCE_INDEX_16BIT;

	sceGxmShaderPatcherCreateVertexProgram(gxm_shader_patcher,
		gxm_disable_color_buffer_vertex_program_id, &disable_color_buffer_attributes,
		1, &disable_color_buffer_stream, 1, &gxm_disable_color_buffer_vertex_program_patched);

	SceGxmBlendInfo disable_color_buffer_blend_info;
	memset(&disable_color_buffer_blend_info, 0, sizeof(disable_color_buffer_blend_info));
	disable_color_buffer_blend_info.colorMask = SCE_GXM_COLOR_MASK_NONE;
	disable_color_buffer_blend_info.colorFunc = SCE_GXM_BLEND_FUNC_NONE;
	disable_color_buffer_blend_info.alphaFunc = SCE_GXM_BLEND_FUNC_NONE;
	disable_color_buffer_blend_info.colorSrc = SCE_GXM_BLEND_FACTOR_ZERO;
	disable_color_buffer_blend_info.colorDst = SCE_GXM_BLEND_FACTOR_ZERO;
	disable_color_buffer_blend_info.alphaSrc = SCE_GXM_BLEND_FACTOR_ZERO;
	disable_color_buffer_blend_info.alphaDst = SCE_GXM_BLEND_FACTOR_ZERO;

	sceGxmShaderPatcherCreateFragmentProgram(gxm_shader_patcher,
		gxm_disable_color_buffer_fragment_program_id,
		SCE_GXM_OUTPUT_REGISTER_FORMAT_UCHAR4,
		SCE_GXM_MULTISAMPLE_NONE,
		&disable_color_buffer_blend_info,
		disable_color_buffer_fragment_program,
		&gxm_disable_color_buffer_fragment_program_patched);

	sceGxmShaderPatcherRegisterProgram(gxm_shader_patcher, gxm_program_clear_v,
		&gxm_clear_vertex_program_id);
	sceGxmShaderPatcherRegisterProgram(gxm_shader_patcher, gxm_program_clear_f,
		&gxm_clear_fragment_program_id);

	const SceGxmProgram *clear_vertex_program =
		sceGxmShaderPatcherGetProgramFromId(gxm_clear_vertex_program_id);
	const SceGxmProgram *clear_fragment_program =
		sceGxmShaderPatcherGetProgramFromId(gxm_clear_fragment_program_id);

	gxm_clear_vertex_program_position_param = sceGxmProgramFindParameterByName(
		clear_vertex_program, "position");

	gxm_clear_fragment_program_u_clear_color_param = sceGxmProgramFindParameterByName(
		clear_fragment_program, "u_clear_color");

	SceGxmVertexAttribute clear_vertex_attribute;
	SceGxmVertexStream clear_vertex_stream;
	clear_vertex_attribute.streamIndex = 0;
	clear_vertex_attribute.offset = 0;
	clear_vertex_attribute.format = SCE_GXM_ATTRIBUTE_FORMAT_F32;
	clear_vertex_attribute.componentCount = 2;
	clear_vertex_attribute.regIndex = sceGxmProgramParameterGetResourceIndex(
		gxm_clear_vertex_program_position_param);
	clear_vertex_stream.stride = sizeof(struct clear_vertex);
	clear_vertex_stream.indexSource = SCE_GXM_INDEX_SOURCE_INDEX_16BIT;

	sceGxmShaderPatcherCreateVertexProgram(gxm_shader_patcher,
		gxm_clear_vertex_program_id, &clear_vertex_attribute,
		1, &clear_vertex_stream, 1, &gxm_clear_vertex_program_patched);

	sceGxmShaderPatcherCreateFragmentProgram(gxm_shader_patcher,
		gxm_clear_fragment_program_id, SCE_GXM_OUTPUT_REGISTER_FORMAT_UCHAR4,
		SCE_GXM_MULTISAMPLE_NONE, NULL, clear_fragment_program,
		&gxm_clear_fragment_program_patched);

	SceUID clear_vertices_uid;
	struct clear_vertex *const clear_vertices_data = gpu_alloc_map(
		SCE_KERNEL_MEMBLOCK_TYPE_USER_RW_UNCACHE, SCE_GXM_MEMORY_ATTRIB_READ,
		4 * sizeof(struct clear_vertex), &clear_vertices_uid);

	SceUID clear_indices_uid;
	unsigned short *const clear_indices_data = gpu_alloc_map(
		SCE_KERNEL_MEMBLOCK_TYPE_USER_RW_UNCACHE, SCE_GXM_MEMORY_ATTRIB_READ,
		4 * sizeof(unsigned short), &clear_indices_uid);

	clear_vertices_data[0].position = (vector2f){-1.0f, -1.0f};
	clear_vertices_data[1].position = (vector2f){ 1.0f, -1.0f};
	clear_vertices_data[2].position = (vector2f){-1.0f,  1.0f};
	clear_vertices_data[3].position = (vector2f){ 1.0f,  1.0f};

	clear_indices_data[0] = 0;
	clear_indices_data[1] = 1;
	clear_indices_data[2] = 2;
	clear_indices_data[3] = 3;

	sceGxmShaderPatcherRegisterProgram(gxm_shader_patcher, gxm_program_cube_v,
		&gxm_cube_vertex_program_id);
	sceGxmShaderPatcherRegisterProgram(gxm_shader_patcher, gxm_program_cube_f,
		&gxm_cube_fragment_program_id);

	const SceGxmProgram *cube_vertex_program =
		sceGxmShaderPatcherGetProgramFromId(gxm_cube_vertex_program_id);
	const SceGxmProgram *cube_fragment_program =
		sceGxmShaderPatcherGetProgramFromId(gxm_cube_fragment_program_id);

	gxm_cube_vertex_program_position_param = sceGxmProgramFindParameterByName(
		cube_vertex_program, "position");
	gxm_cube_vertex_program_normal_param = sceGxmProgramFindParameterByName(
		cube_vertex_program, "normal");
	gxm_cube_vertex_program_color_param = sceGxmProgramFindParameterByName(
		cube_vertex_program, "color");
	gxm_cube_vertex_program_u_mvp_matrix_param = sceGxmProgramFindParameterByName(
		cube_vertex_program, "u_mvp_matrix");

	gxm_cube_fragment_program_u_modelview_matrix_param = sceGxmProgramFindParameterByName(
		cube_fragment_program, "u_modelview_matrix");
	gxm_cube_fragment_program_u_normal_matrix_param = sceGxmProgramFindParameterByName(
		cube_fragment_program, "u_normal_matrix");

	gxm_cube_fragment_program_phong_material_params.ambient = sceGxmProgramFindParameterByName(
		cube_fragment_program, "u_material.ambient");
	gxm_cube_fragment_program_phong_material_params.diffuse = sceGxmProgramFindParameterByName(
		cube_fragment_program, "u_material.diffuse");
	gxm_cube_fragment_program_phong_material_params.specular = sceGxmProgramFindParameterByName(
		cube_fragment_program, "u_material.specular");
	gxm_cube_fragment_program_phong_material_params.shininess = sceGxmProgramFindParameterByName(
		cube_fragment_program, "u_material.shininess");

	gxm_cube_fragment_program_light_params.position = sceGxmProgramFindParameterByName(
		cube_fragment_program, "u_light.position");
	gxm_cube_fragment_program_light_params.color = sceGxmProgramFindParameterByName(
		cube_fragment_program, "u_light.color");

	SceGxmVertexAttribute cube_vertex_attributes[3];
	SceGxmVertexStream cube_vertex_stream;
	cube_vertex_attributes[0].streamIndex = 0;
	cube_vertex_attributes[0].offset = 0;
	cube_vertex_attributes[0].format = SCE_GXM_ATTRIBUTE_FORMAT_F32;
	cube_vertex_attributes[0].componentCount = 3;
	cube_vertex_attributes[0].regIndex = sceGxmProgramParameterGetResourceIndex(
		gxm_cube_vertex_program_position_param);
	cube_vertex_attributes[1].streamIndex = 0;
	cube_vertex_attributes[1].offset = sizeof(vector3f);
	cube_vertex_attributes[1].format = SCE_GXM_ATTRIBUTE_FORMAT_F32;
	cube_vertex_attributes[1].componentCount = 3;
	cube_vertex_attributes[1].regIndex = sceGxmProgramParameterGetResourceIndex(
		gxm_cube_vertex_program_normal_param);
	cube_vertex_attributes[2].streamIndex = 0;
	cube_vertex_attributes[2].offset = 2 * sizeof(vector3f);
	cube_vertex_attributes[2].format = SCE_GXM_ATTRIBUTE_FORMAT_F32;
	cube_vertex_attributes[2].componentCount = 4;
	cube_vertex_attributes[2].regIndex = sceGxmProgramParameterGetResourceIndex(
		gxm_cube_vertex_program_color_param);
	cube_vertex_stream.stride = sizeof(struct mesh_vertex);
	cube_vertex_stream.indexSource = SCE_GXM_INDEX_SOURCE_INDEX_16BIT;

	sceGxmShaderPatcherCreateVertexProgram(gxm_shader_patcher,
		gxm_cube_vertex_program_id, cube_vertex_attributes,
		3, &cube_vertex_stream, 1, &gxm_cube_vertex_program_patched);

	sceGxmShaderPatcherCreateFragmentProgram(gxm_shader_patcher,
		gxm_cube_fragment_program_id, SCE_GXM_OUTPUT_REGISTER_FORMAT_UCHAR4,
		SCE_GXM_MULTISAMPLE_NONE, NULL, cube_fragment_program,
		&gxm_cube_fragment_program_patched);

	SceUID cube_mesh_uid;
	cube_mesh_data = gpu_alloc_map(
		SCE_KERNEL_MEMBLOCK_TYPE_USER_RW_UNCACHE, SCE_GXM_MEMORY_ATTRIB_READ,
		36 * sizeof(struct mesh_vertex), &cube_mesh_uid);

	SceUID cube_indices_uid;
	cube_indices_data = gpu_alloc_map(
		SCE_KERNEL_MEMBLOCK_TYPE_USER_RW_UNCACHE, SCE_GXM_MEMORY_ATTRIB_READ,
		36 * sizeof(unsigned short), &cube_indices_uid);

	#define CUBE_SIZE 1.0f
	#define CUBE_HALF_SIZE (CUBE_SIZE / 2.0f)

	static const vector3f cube_vertices[] = {
		{.x = -CUBE_HALF_SIZE, .y = +CUBE_HALF_SIZE, .z = +CUBE_HALF_SIZE},
		{.x = -CUBE_HALF_SIZE, .y = -CUBE_HALF_SIZE, .z = +CUBE_HALF_SIZE},
		{.x = +CUBE_HALF_SIZE, .y = +CUBE_HALF_SIZE, .z = +CUBE_HALF_SIZE},
		{.x = +CUBE_HALF_SIZE, .y = -CUBE_HALF_SIZE, .z = +CUBE_HALF_SIZE},
		{.x = +CUBE_HALF_SIZE, .y = +CUBE_HALF_SIZE, .z = -CUBE_HALF_SIZE},
		{.x = +CUBE_HALF_SIZE, .y = -CUBE_HALF_SIZE, .z = -CUBE_HALF_SIZE},
		{.x = -CUBE_HALF_SIZE, .y = +CUBE_HALF_SIZE, .z = -CUBE_HALF_SIZE},
		{.x = -CUBE_HALF_SIZE, .y = -CUBE_HALF_SIZE, .z = -CUBE_HALF_SIZE}
	};

	static const vector3f cube_face_normals[] = {
		{.x =  0.0f, .y =  0.0f, .z =  1.0f}, /* Front */
		{.x =  1.0f, .y =  0.0f, .z =  0.0f}, /* Right */
		{.x =  0.0f, .y =  0.0f, .z = -1.0f}, /* Back */
		{.x = -1.0f, .y =  0.0f, .z =  0.0f}, /* Left */
		{.x =  0.0f, .y =  1.0f, .z =  0.0f}, /* Top */
		{.x =  0.0f, .y = -1.0f, .z =  0.0f}, /* Bottom */
	};

	static const vector4f cube_colors[] = {
		{.r = 1.0f, .g = 0.0f, .b = 0.0f, .a = 1.0f},
		{.r = 0.0f, .g = 1.0f, .b = 0.0f, .a = 1.0f},
		{.r = 0.0f, .g = 0.0f, .b = 1.0f, .a = 1.0f},
		{.r = 1.0f, .g = 1.0f, .b = 0.0f, .a = 1.0f},
		{.r = 0.0f, .g = 1.0f, .b = 1.0f, .a = 1.0f},
		{.r = 1.0f, .g = 0.0f, .b = 1.0f, .a = 1.0f},
	};

	cube_mesh_data[0] = (struct mesh_vertex){cube_vertices[0], cube_face_normals[0], cube_colors[0]};
	cube_mesh_data[1] = (struct mesh_vertex){cube_vertices[1], cube_face_normals[0], cube_colors[0]};
	cube_mesh_data[2] = (struct mesh_vertex){cube_vertices[2], cube_face_normals[0], cube_colors[0]};
	cube_mesh_data[3] = (struct mesh_vertex){cube_vertices[2], cube_face_normals[0], cube_colors[0]};
	cube_mesh_data[4] = (struct mesh_vertex){cube_vertices[1], cube_face_normals[0], cube_colors[0]};
	cube_mesh_data[5] = (struct mesh_vertex){cube_vertices[3], cube_face_normals[0], cube_colors[0]};

	cube_mesh_data[6] = (struct mesh_vertex){cube_vertices[2], cube_face_normals[1], cube_colors[1]};
	cube_mesh_data[7] = (struct mesh_vertex){cube_vertices[3], cube_face_normals[1], cube_colors[1]};
	cube_mesh_data[8] = (struct mesh_vertex){cube_vertices[4], cube_face_normals[1], cube_colors[1]};
	cube_mesh_data[9] = (struct mesh_vertex){cube_vertices[4], cube_face_normals[1], cube_colors[1]};
	cube_mesh_data[10] = (struct mesh_vertex){cube_vertices[3], cube_face_normals[1], cube_colors[1]};
	cube_mesh_data[11] = (struct mesh_vertex){cube_vertices[5], cube_face_normals[1], cube_colors[1]};

	cube_mesh_data[12] = (struct mesh_vertex){cube_vertices[4], cube_face_normals[2], cube_colors[2]};
	cube_mesh_data[13] = (struct mesh_vertex){cube_vertices[5], cube_face_normals[2], cube_colors[2]};
	cube_mesh_data[14] = (struct mesh_vertex){cube_vertices[6], cube_face_normals[2], cube_colors[2]};
	cube_mesh_data[15] = (struct mesh_vertex){cube_vertices[6], cube_face_normals[2], cube_colors[2]};
	cube_mesh_data[16] = (struct mesh_vertex){cube_vertices[5], cube_face_normals[2], cube_colors[2]};
	cube_mesh_data[17] = (struct mesh_vertex){cube_vertices[7], cube_face_normals[2], cube_colors[2]};

	cube_mesh_data[18] = (struct mesh_vertex){cube_vertices[6], cube_face_normals[3], cube_colors[3]};
	cube_mesh_data[19] = (struct mesh_vertex){cube_vertices[7], cube_face_normals[3], cube_colors[3]};
	cube_mesh_data[20] = (struct mesh_vertex){cube_vertices[0], cube_face_normals[3], cube_colors[3]};
	cube_mesh_data[21] = (struct mesh_vertex){cube_vertices[0], cube_face_normals[3], cube_colors[3]};
	cube_mesh_data[22] = (struct mesh_vertex){cube_vertices[7], cube_face_normals[3], cube_colors[3]};
	cube_mesh_data[23] = (struct mesh_vertex){cube_vertices[1], cube_face_normals[3], cube_colors[3]};

	cube_mesh_data[24] = (struct mesh_vertex){cube_vertices[6], cube_face_normals[4], cube_colors[4]};
	cube_mesh_data[25] = (struct mesh_vertex){cube_vertices[0], cube_face_normals[4], cube_colors[4]};
	cube_mesh_data[26] = (struct mesh_vertex){cube_vertices[4], cube_face_normals[4], cube_colors[4]};
	cube_mesh_data[27] = (struct mesh_vertex){cube_vertices[4], cube_face_normals[4], cube_colors[4]};
	cube_mesh_data[28] = (struct mesh_vertex){cube_vertices[0], cube_face_normals[4], cube_colors[4]};
	cube_mesh_data[29] = (struct mesh_vertex){cube_vertices[2], cube_face_normals[4], cube_colors[4]};

	cube_mesh_data[30] = (struct mesh_vertex){cube_vertices[1], cube_face_normals[5], cube_colors[5]};
	cube_mesh_data[31] = (struct mesh_vertex){cube_vertices[7], cube_face_normals[5], cube_colors[5]};
	cube_mesh_data[32] = (struct mesh_vertex){cube_vertices[3], cube_face_normals[5], cube_colors[5]};
	cube_mesh_data[33] = (struct mesh_vertex){cube_vertices[3], cube_face_normals[5], cube_colors[5]};
	cube_mesh_data[34] = (struct mesh_vertex){cube_vertices[7], cube_face_normals[5], cube_colors[5]};
	cube_mesh_data[35] = (struct mesh_vertex){cube_vertices[5], cube_face_normals[5], cube_colors[5]};

	for (i = 0; i < 36; i++)
		cube_indices_data[i] = i;

	SceUID floor_mesh_uid;
	floor_mesh_data = gpu_alloc_map(
		SCE_KERNEL_MEMBLOCK_TYPE_USER_RW_UNCACHE, SCE_GXM_MEMORY_ATTRIB_READ,
		4 * sizeof(struct mesh_vertex), &floor_mesh_uid);

	SceUID floor_indices_uid;
	floor_indices_data = gpu_alloc_map(
		SCE_KERNEL_MEMBLOCK_TYPE_USER_RW_UNCACHE, SCE_GXM_MEMORY_ATTRIB_READ,
		4 * sizeof(unsigned short), &floor_indices_uid);

	#define FLOOR_SIZE 20.0f
	#define FLOOR_HALF_SIZE (FLOOR_SIZE / 2.0f)

	static const vector3f floor_vertices[] = {
		{.x = -FLOOR_HALF_SIZE, .y = 0.0f, .z = -FLOOR_HALF_SIZE},
		{.x = -FLOOR_HALF_SIZE, .y = 0.0f, .z = +FLOOR_HALF_SIZE},
		{.x = +FLOOR_HALF_SIZE, .y = 0.0f, .z = -FLOOR_HALF_SIZE},
		{.x = +FLOOR_HALF_SIZE, .y = 0.0f, .z = +FLOOR_HALF_SIZE}
	};

	static const vector4f floor_color = {
		.r = 0.5f, .g = 0.5f, .b = 0.5f, .a = 1.0f
	};

	static const vector3f floor_normal = {
		.x = 0.0f, .y = 1.0f, .z = 0.0f
	};

	for (i = 0; i < 4; i++) {
		floor_mesh_data[i] = (struct mesh_vertex){floor_vertices[i], floor_normal, floor_color};
		floor_indices_data[i] = i;
	}

	SceUID portal_mesh_uid;
	struct position_vertex *const portal_mesh_data = gpu_alloc_map(
		SCE_KERNEL_MEMBLOCK_TYPE_USER_RW_UNCACHE, SCE_GXM_MEMORY_ATTRIB_READ,
		4 * sizeof(struct position_vertex), &portal_mesh_uid);

	SceUID portal_indices_uid;
	unsigned short *const portal_indices_data = gpu_alloc_map(
		SCE_KERNEL_MEMBLOCK_TYPE_USER_RW_UNCACHE, SCE_GXM_MEMORY_ATTRIB_READ,
		4 * sizeof(unsigned short), &portal_indices_uid);

	#define PORTAL_SIZE 4.0f
	#define PORTAL_HALF_SIZE (PORTAL_SIZE / 2.0f)

	static const vector3f portal_vertices[] = {
		{.x = -PORTAL_HALF_SIZE, .y = +PORTAL_HALF_SIZE, .z = 0.0f},
		{.x = -PORTAL_HALF_SIZE, .y = -PORTAL_HALF_SIZE, .z = 0.0f},
		{.x = +PORTAL_HALF_SIZE, .y = +PORTAL_HALF_SIZE, .z = 0.0f},
		{.x = +PORTAL_HALF_SIZE, .y = -PORTAL_HALF_SIZE, .z = 0.0f}
	};

	static const vector3f portal_normal = {
		.x = 0.0f, .y = 0.0f, .z = -1.0f
	};

	for (i = 0; i < 4; i++) {
		portal_mesh_data[i] = (struct position_vertex){portal_vertices[i]};
		portal_indices_data[i] = i;
	}

	SceUID portal_frame_mesh_uid;
	portal_frame_mesh_data = gpu_alloc_map(
		SCE_KERNEL_MEMBLOCK_TYPE_USER_RW_UNCACHE, SCE_GXM_MEMORY_ATTRIB_READ,
		12 * sizeof(struct mesh_vertex), &portal_frame_mesh_uid);

	SceUID portal_frame_indices_uid;
	portal_frame_indices_data = gpu_alloc_map(
		SCE_KERNEL_MEMBLOCK_TYPE_USER_RW_UNCACHE, SCE_GXM_MEMORY_ATTRIB_READ,
		8 * 3 * sizeof(unsigned short), &portal_frame_indices_uid);

	#define PORTAL_FRAME_SIZE 0.2f

	static const vector4f portal_frame_color = {
		.r = 0.3f, .g = 0.0f, .b = 1.0f, .a = 1.0f
	};

	static const vector2f portal_frame_position_offsets[12] = {
		{.x = -PORTAL_HALF_SIZE - PORTAL_FRAME_SIZE, .y = +PORTAL_HALF_SIZE + PORTAL_FRAME_SIZE},
		{.x = -PORTAL_HALF_SIZE - PORTAL_FRAME_SIZE, .y = +PORTAL_HALF_SIZE},
		{.x = +PORTAL_HALF_SIZE + PORTAL_FRAME_SIZE, .y = +PORTAL_HALF_SIZE + PORTAL_FRAME_SIZE},
		{.x = +PORTAL_HALF_SIZE + PORTAL_FRAME_SIZE, .y = +PORTAL_HALF_SIZE},
		{.x = -PORTAL_HALF_SIZE - PORTAL_FRAME_SIZE, .y = -PORTAL_HALF_SIZE},
		{.x = -PORTAL_HALF_SIZE, .y = +PORTAL_HALF_SIZE},
		{.x = -PORTAL_HALF_SIZE, .y = -PORTAL_HALF_SIZE},
		{.x = +PORTAL_HALF_SIZE, .y = +PORTAL_HALF_SIZE},
		{.x = +PORTAL_HALF_SIZE, .y = -PORTAL_HALF_SIZE},
		{.x = +PORTAL_HALF_SIZE + PORTAL_FRAME_SIZE, .y = -PORTAL_HALF_SIZE},
		{.x = -PORTAL_HALF_SIZE - PORTAL_FRAME_SIZE, .y = -PORTAL_HALF_SIZE - PORTAL_FRAME_SIZE},
		{.x = +PORTAL_HALF_SIZE + PORTAL_FRAME_SIZE, .y = -PORTAL_HALF_SIZE - PORTAL_FRAME_SIZE},
	};

	for (i = 0; i < 12; i++) {
		portal_frame_mesh_data[i] = (struct mesh_vertex){
			.position = {.x = portal_frame_position_offsets[i].x, .y = portal_frame_position_offsets[i].y, .z = 0.0f},
			.normal = portal_normal,
			.color = portal_frame_color
		};
	}

	portal_frame_indices_data[0] = 0;
	portal_frame_indices_data[1] = 1;
	portal_frame_indices_data[2] = 2;
	portal_frame_indices_data[3] = 2;
	portal_frame_indices_data[4] = 1;
	portal_frame_indices_data[5] = 3;
	portal_frame_indices_data[6] = 1;
	portal_frame_indices_data[7] = 4;
	portal_frame_indices_data[8] = 5;
	portal_frame_indices_data[9] = 5;
	portal_frame_indices_data[10] = 4;
	portal_frame_indices_data[11] = 6;
	portal_frame_indices_data[12] = 7;
	portal_frame_indices_data[13] = 8;
	portal_frame_indices_data[14] = 3;
	portal_frame_indices_data[15] = 3;
	portal_frame_indices_data[16] = 8;
	portal_frame_indices_data[17] = 9;
	portal_frame_indices_data[18] = 4;
	portal_frame_indices_data[19] = 10;
	portal_frame_indices_data[20] = 9;
	portal_frame_indices_data[21] = 9;
	portal_frame_indices_data[22] = 10;
	portal_frame_indices_data[23] = 11;


	gxm_front_buffer_index = 0;
	gxm_back_buffer_index = 0;

	matrix4x4 projection_matrix;
	matrix4x4_init_perspective(projection_matrix, 90.0f,
		DISPLAY_WIDTH / (float)DISPLAY_HEIGHT, 0.01f, 100.0f);

	SceCtrlData pad;
	memset(&pad, 0, sizeof(pad));

	static vector3f camera_initial_pos = {
		.x = 0.0f, .y = 1.5f, .z = 4.0f
	};

	static vector3f camera_initial_rot = {
		.x = 0.0f, .y = 0.0f, .z = 0.0f
	};

	struct camera camera;
	camera_init(&camera, &camera_initial_pos, &camera_initial_rot);

	struct scene_state scene_state;
	scene_state.trans_x = 0.0f;
	scene_state.trans_y = CUBE_SIZE + 0.1f;
	scene_state.trans_z = 0.0f;
	scene_state.rot_y = 0.0f; //DEG_TO_RAD(45.0f);
	scene_state.rot_x = 0.0f; //DEG_TO_RAD(45.0f);

	scene_state.light_distance = 8.0f;
	scene_state.light_x_rot = DEG_TO_RAD(20.0f);
	scene_state.light_y_rot = 0.0f;

	scene_state.portal.width = PORTAL_SIZE;
	scene_state.portal.height = PORTAL_SIZE;
	matrix4x4_identity(scene_state.portal.end1.model_matrix);
	matrix4x4_translate(scene_state.portal.end1.model_matrix, 0.0f,
		PORTAL_HALF_SIZE + PORTAL_FRAME_SIZE, 0.0f);
	matrix4x4_rotate_y(scene_state.portal.end1.model_matrix, M_PI);

	matrix4x4_identity(scene_state.portal.end2.model_matrix);
	matrix4x4_translate(scene_state.portal.end2.model_matrix, 1.5f,
		PORTAL_HALF_SIZE + PORTAL_FRAME_SIZE, 0.0f);
	matrix4x4_rotate_y(scene_state.portal.end2.model_matrix, -M_PI / 2.0f);

	static int run = 1;
	while (run) {
		sceCtrlPeekBufferPositive(0, &pad, 1);
		if (pad.buttons & SCE_CTRL_START)
			run = 0;

		update_camera(&camera, &pad);
		update_scene(&scene_state, &camera, &pad);

		sceGxmBeginScene(gxm_context,
			0,
			gxm_render_target,
			NULL,
			NULL,
			gxm_sync_objects[gxm_back_buffer_index],
			&gxm_color_surfaces[gxm_back_buffer_index],
			&gxm_depth_stencil_surface);

		{ /* Clear the color and the depth/stencil buffers */
			sceGxmSetVertexProgram(gxm_context, gxm_clear_vertex_program_patched);
			sceGxmSetFragmentProgram(gxm_context, gxm_clear_fragment_program_patched);

			static const float clear_color[4] = {
				1.0f, 1.0f, 1.0f, 1.0f
			};

			set_fragment_default_uniform_data(gxm_clear_fragment_program_u_clear_color_param,
				sizeof(clear_color) / sizeof(float), clear_color);

			sceGxmSetFrontStencilFunc(gxm_context,
				SCE_GXM_STENCIL_FUNC_ALWAYS,
				SCE_GXM_STENCIL_OP_ZERO,
				SCE_GXM_STENCIL_OP_ZERO,
				SCE_GXM_STENCIL_OP_ZERO,
				0, 0xFF);

			sceGxmSetVertexStream(gxm_context, 0, clear_vertices_data);
			sceGxmDraw(gxm_context, SCE_GXM_PRIMITIVE_TRIANGLE_STRIP,
				SCE_GXM_INDEX_FORMAT_U16, clear_indices_data, 4);
		}

		/*
		 * Step 1: Disable drawing to the color buffer and the depth buffer,
		 *         but enable writing to the stencil buffer.
		 * Step 2: Set the stencil operation to GL_INCR on sfail, meaning that
		 *         the stencil value will be incremented when the stencil test fails.
		 * Step 3: Set the stencil function to GL_NEVER, which makes sure that
		 *         the stencil test always fails on every pixel drawn.
		 */
		sceGxmSetFrontDepthWriteEnable(gxm_context,
			SCE_GXM_DEPTH_WRITE_DISABLED);
		sceGxmSetFrontStencilFunc(gxm_context,
			SCE_GXM_STENCIL_FUNC_NEVER,
			SCE_GXM_STENCIL_OP_INCR,
			SCE_GXM_STENCIL_OP_KEEP,
			SCE_GXM_STENCIL_OP_KEEP,
			0, 0xFF);

		/*
		 * Step 4: Draw the portal's frame. At this point the stencil buffer is filled
		 *         with zero's on the outside of the portal's frame and one's on the inside.
		 */
		{
			sceGxmSetVertexProgram(gxm_context, gxm_disable_color_buffer_vertex_program_patched);
			sceGxmSetFragmentProgram(gxm_context, gxm_disable_color_buffer_fragment_program_patched);

			matrix4x4 portal_mvp_matrix;
			matrix4x4 portal_modelview_matrix;

			matrix4x4_multiply(portal_modelview_matrix,
				camera.view_matrix, scene_state.portal.end1.model_matrix);
			matrix4x4_multiply(portal_mvp_matrix,
				projection_matrix, portal_modelview_matrix);

			set_vertex_default_uniform_data(gxm_disable_color_buffer_vertex_program_u_mvp_matrix_param,
				sizeof(portal_mvp_matrix) / sizeof(float), portal_mvp_matrix);

			sceGxmSetVertexStream(gxm_context, 0, portal_mesh_data);
			sceGxmDraw(gxm_context, SCE_GXM_PRIMITIVE_TRIANGLE_STRIP,
				SCE_GXM_INDEX_FORMAT_U16, portal_indices_data, 4);
		}

		/*
		 * Step 5: Generate the virtual camera's view matrix using the view
		 *         frustum clipping method.
		 * Step 6: Disable writing to the stencil buffer, but enable drawing to
		 *         the color buffer and the depth buffer.
		 * Step 7: Set the stencil function to GL_LEQUAL with reference value 1.
		 *         This will only draw where the stencil value is greater than
		 *         or equal to 1, which is inside the portal frame.
		 * Step 8: Draw the scene using the virtual camera from step 5. This will
		 *         only draw inside of the portal's frame because of the stencil test.
		 */
		sceGxmSetFrontDepthWriteEnable(gxm_context,
			SCE_GXM_DEPTH_WRITE_ENABLED);
		sceGxmSetFrontStencilRef(gxm_context, 1);
		sceGxmSetFrontStencilFunc(gxm_context,
			SCE_GXM_STENCIL_FUNC_LESS_EQUAL,
			SCE_GXM_STENCIL_OP_KEEP,
			SCE_GXM_STENCIL_OP_KEEP,
			SCE_GXM_STENCIL_OP_KEEP,
			0xFF, 0);
		{
			/*
			 * Render the scene from the other portal's end view.
			 *     P' = M2 * Mreflect * M1^-1 * P
			 * If V is the camera's view matrix:
			 *     V' = M2 * M1^-1 * V
			 */
			matrix4x4 m1, m2;
			matrix4x4 portal_end2_view_matrix;

			matrix4x4_invert(m1, scene_state.portal.end1.model_matrix);
			matrix4x4_multiply(m2, camera.view_matrix, m1);

			matrix4x4_multiply(portal_end2_view_matrix,
				m2, scene_state.portal.end2.model_matrix);

			draw_scene(&scene_state, projection_matrix, portal_end2_view_matrix);
		}

		/*
		 * Step 9: Disable the stencil test, disable drawing to the color
		 *         buffer, and enable drawing to the depth buffer.
		 * Step 10: Clear the depth buffer.
		 */
		sceGxmSetFrontStencilFunc(gxm_context,
			SCE_GXM_STENCIL_FUNC_ALWAYS,
			SCE_GXM_STENCIL_OP_KEEP,
			SCE_GXM_STENCIL_OP_KEEP,
			SCE_GXM_STENCIL_OP_KEEP,
			0, 0);

		{
			sceGxmSetVertexProgram(gxm_context, gxm_disable_color_buffer_vertex_program_patched);
			sceGxmSetFragmentProgram(gxm_context, gxm_disable_color_buffer_fragment_program_patched);

			sceGxmSetVertexStream(gxm_context, 0, clear_vertices_data);
			sceGxmDraw(gxm_context, SCE_GXM_PRIMITIVE_TRIANGLE_STRIP,
				SCE_GXM_INDEX_FORMAT_U16, clear_indices_data, 4);
		}

		/*
		 * Step 10: Draw the portal frame once again, this time
		 *          to the depth buffer which was just cleared.
		 */
		{
			sceGxmSetVertexProgram(gxm_context, gxm_disable_color_buffer_vertex_program_patched);
			sceGxmSetFragmentProgram(gxm_context, gxm_disable_color_buffer_fragment_program_patched);

			matrix4x4 portal_mvp_matrix;
			matrix4x4 portal_modelview_matrix;

			matrix4x4_multiply(portal_modelview_matrix,
				camera.view_matrix, scene_state.portal.end1.model_matrix);
			matrix4x4_multiply(portal_mvp_matrix,
				projection_matrix, portal_modelview_matrix);

			set_vertex_default_uniform_data(gxm_disable_color_buffer_vertex_program_u_mvp_matrix_param,
				sizeof(portal_mvp_matrix) / sizeof(float), portal_mvp_matrix);

			sceGxmSetVertexStream(gxm_context, 0, portal_mesh_data);
			sceGxmDraw(gxm_context, SCE_GXM_PRIMITIVE_TRIANGLE_STRIP,
				SCE_GXM_INDEX_FORMAT_U16, portal_indices_data, 4);
		}

		/*
		 * Step 11: Enable the color buffer again.
		 * step 12: Draw the whole scene with the regular camera.
		 */

		draw_scene(&scene_state, projection_matrix, camera.view_matrix);

		sceGxmEndScene(gxm_context, NULL, NULL);

		sceGxmPadHeartbeat(&gxm_color_surfaces[gxm_back_buffer_index],
			gxm_sync_objects[gxm_back_buffer_index]);

		struct display_queue_callback_data queue_cb_data;
		queue_cb_data.addr = gxm_color_surfaces_addr[gxm_back_buffer_index];

		sceGxmDisplayQueueAddEntry(gxm_sync_objects[gxm_front_buffer_index],
			gxm_sync_objects[gxm_back_buffer_index], &queue_cb_data);

		gxm_front_buffer_index = gxm_back_buffer_index;
		gxm_back_buffer_index = (gxm_back_buffer_index + 1) % DISPLAY_BUFFER_COUNT;
	}

	sceGxmDisplayQueueFinish();
	sceGxmFinish(gxm_context);

	gpu_unmap_free(clear_vertices_uid);
	gpu_unmap_free(clear_indices_uid);

	gpu_unmap_free(cube_mesh_uid);
	gpu_unmap_free(cube_indices_uid);

	gpu_unmap_free(floor_mesh_uid);
	gpu_unmap_free(floor_indices_uid);

	gpu_unmap_free(portal_mesh_uid);
	gpu_unmap_free(portal_indices_uid);

	gpu_unmap_free(portal_frame_mesh_uid);
	gpu_unmap_free(portal_frame_indices_uid);

	sceGxmShaderPatcherReleaseVertexProgram(gxm_shader_patcher,
		gxm_disable_color_buffer_vertex_program_patched);
	sceGxmShaderPatcherReleaseFragmentProgram(gxm_shader_patcher,
		gxm_disable_color_buffer_fragment_program_patched);

	sceGxmShaderPatcherReleaseVertexProgram(gxm_shader_patcher,
		gxm_clear_vertex_program_patched);
	sceGxmShaderPatcherReleaseFragmentProgram(gxm_shader_patcher,
		gxm_clear_fragment_program_patched);

	sceGxmShaderPatcherReleaseVertexProgram(gxm_shader_patcher,
		gxm_cube_vertex_program_patched);
	sceGxmShaderPatcherReleaseFragmentProgram(gxm_shader_patcher,
		gxm_cube_fragment_program_patched);

	sceGxmShaderPatcherUnregisterProgram(gxm_shader_patcher,
		gxm_clear_vertex_program_id);
	sceGxmShaderPatcherUnregisterProgram(gxm_shader_patcher,
		gxm_clear_fragment_program_id);

	sceGxmShaderPatcherUnregisterProgram(gxm_shader_patcher,
		gxm_cube_vertex_program_id);
	sceGxmShaderPatcherUnregisterProgram(gxm_shader_patcher,
		gxm_cube_fragment_program_id);

	sceGxmShaderPatcherDestroy(gxm_shader_patcher);

	gpu_unmap_free(gxm_shader_patcher_buffer_uid);
	gpu_vertex_usse_unmap_free(gxm_shader_patcher_vertex_usse_uid);
	gpu_fragment_usse_unmap_free(gxm_shader_patcher_fragment_usse_uid);

	gpu_unmap_free(gxm_depth_stencil_surface_uid);

	for (i = 0; i < DISPLAY_BUFFER_COUNT; i++) {
		gpu_unmap_free(gxm_color_surfaces_uid[i]);
		sceGxmSyncObjectDestroy(gxm_sync_objects[i]);
	}

	sceGxmDestroyRenderTarget(gxm_render_target);

	gpu_unmap_free(vdm_ring_buffer_uid);
	gpu_unmap_free(vertex_ring_buffer_uid);
	gpu_unmap_free(fragment_ring_buffer_uid);
	gpu_fragment_usse_unmap_free(fragment_usse_ring_buffer_uid);

	sceGxmDestroyContext(gxm_context);

	sceGxmTerminate();

	netlog_fini();

	return 0;
}

static void update_scene(struct scene_state *state, const struct camera *camera, SceCtrlData *pad)
{
	if (pad->buttons & SCE_CTRL_UP)
		state->trans_z -= 0.025f;
	else if (pad->buttons & SCE_CTRL_DOWN)
		state->trans_z += 0.025f;

	if (pad->buttons & SCE_CTRL_RIGHT)
		state->trans_x += 0.025f;
	else if (pad->buttons & SCE_CTRL_LEFT)
		state->trans_x -= 0.025f;

	if (pad->buttons & SCE_CTRL_SQUARE)
		state->rot_y += 0.025f;
	else if (pad->buttons & SCE_CTRL_CIRCLE)
		state->rot_y -= 0.025f;

	if (pad->buttons & SCE_CTRL_CROSS)
		state->rot_x += 0.025f;
	else if (pad->buttons & SCE_CTRL_TRIANGLE)
		state->rot_x -= 0.025f;

	vector3f light_position;
	light_position.x = state->light_distance * cosf(state->light_x_rot) * cosf(state->light_y_rot);
	light_position.y = state->light_distance * sinf(state->light_x_rot);
	light_position.z = state->light_distance * cosf(state->light_x_rot) * sinf(state->light_y_rot);

	state->light_y_rot += 0.1f;
	vector3f_matrix4x4_mult(&state->light.position,
		camera->view_matrix, &light_position, 1.0f);

	state->light.color = (vector3f){.r = 1.0f, .g = 1.0f, .b = 1.0f};

	/*matrix4x4_identity(state->portal.end2.model_matrix);
	matrix4x4_rotate_y(state->portal.end2.model_matrix, state->rot_y);
	matrix4x4_rotate_x(state->portal.end2.model_matrix, state->rot_x);
	matrix4x4_translate(state->portal.end2.model_matrix,
		0.0f + state->trans_x, 4.0f + state->trans_y, -10.0f + state->trans_z);*/
}

static void draw_scene(const struct scene_state *state, matrix4x4 projection_matrix, matrix4x4 view_matrix)
{
	sceGxmSetVertexProgram(gxm_context, gxm_cube_vertex_program_patched);
	sceGxmSetFragmentProgram(gxm_context, gxm_cube_fragment_program_patched);

	{ /* Draw the portal frame */
		static const struct phong_material portal_frame_material = {
			.ambient = {.r = 0.2f, .g = 0.2f, .b = 0.2f},
			.diffuse = {.r = 0.6f, .g = 0.6f, .b = 0.6f},
			.specular = {.r = 0.6f, .g = 0.6f, .b = 0.6f},
			.shininess = 40.0f
		};

		matrix4x4 portal_frame_mvp_matrix;
		matrix4x4 portal_frame_modelview_matrix;
		matrix3x3 portal_frame_normal_matrix;

		matrix4x4_multiply(portal_frame_modelview_matrix, view_matrix, state->portal.end1.model_matrix);
		matrix4x4_multiply(portal_frame_mvp_matrix, projection_matrix, portal_frame_modelview_matrix);
		matrix3x3_normal_matrix(portal_frame_normal_matrix, portal_frame_modelview_matrix);

		set_cube_fragment_light_uniform_params(&state->light,
			&gxm_cube_fragment_program_light_params);
		set_cube_fragment_material_uniform_params(&portal_frame_material,
			&gxm_cube_fragment_program_phong_material_params);
		set_cube_matrices_uniform_params(portal_frame_mvp_matrix,
			portal_frame_modelview_matrix, portal_frame_normal_matrix);

		sceGxmSetVertexStream(gxm_context, 0, portal_frame_mesh_data);
		sceGxmDraw(gxm_context, SCE_GXM_PRIMITIVE_TRIANGLES,
			SCE_GXM_INDEX_FORMAT_U16, portal_frame_indices_data, 24);
	}

	{ /* Draw the cube */
		static const struct phong_material cube1_material = {
			.ambient = {.r = 0.1f, .g = 0.1f, .b = 0.1f},
			.diffuse = {.r = 0.8f, .g = 0.8f, .b = 0.8f},
			.specular = {.r = 0.6f, .g = 0.6f, .b = 0.6f},
			.shininess = 80.0f
		};

		matrix4x4 cube1_model_matrix;
		matrix4x4_init_translation(cube1_model_matrix,
			5.0f, CUBE_SIZE + 0.1f, 0.0f);

		draw_cube(state, projection_matrix, view_matrix,
			cube1_model_matrix, &cube1_material);

		static const struct phong_material cube2_material = {
			.ambient = {.r = 0.1f, .g = 0.1f, .b = 0.1f},
			.diffuse = {.r = 0.8f, .g = 0.8f, .b = 0.8f},
			.specular = {.r = 0.6f, .g = 0.6f, .b = 0.6f},
			.shininess = 80.0f
		};

		matrix4x4 cube2_model_matrix;
		matrix4x4_init_translation(cube2_model_matrix, 0.0f, 2.0f, 1.5f);

		draw_cube(state, projection_matrix, view_matrix,
			cube2_model_matrix, &cube2_material);
	}

	{ /* Draw the floor */
		static const struct phong_material floor_material = {
			.ambient = {.r = 0.1f, .g = 0.1f, .b = 0.1f},
			.diffuse = {.r = 0.8f, .g = 0.8f, .b = 0.8f},
			.specular = {.r = 0.7f, .g = 0.7f, .b = 0.7f},
			.shininess = 20.0f
		};

		matrix4x4 floor_mvp_matrix;
		matrix4x4 floor_modelview_matrix;
		matrix3x3 floor_normal_matrix;
		matrix4x4 floor_model_matrix;

		matrix4x4_identity(floor_model_matrix);
		matrix4x4_multiply(floor_modelview_matrix, view_matrix, floor_model_matrix);
		matrix4x4_multiply(floor_mvp_matrix, projection_matrix, floor_modelview_matrix);
		matrix3x3_normal_matrix(floor_normal_matrix, floor_modelview_matrix);

		set_cube_fragment_light_uniform_params(&state->light,
			&gxm_cube_fragment_program_light_params);
		set_cube_fragment_material_uniform_params(&floor_material,
			&gxm_cube_fragment_program_phong_material_params);
		set_cube_matrices_uniform_params(floor_mvp_matrix,
			floor_modelview_matrix, floor_normal_matrix);

		sceGxmSetVertexStream(gxm_context, 0, floor_mesh_data);
		sceGxmDraw(gxm_context, SCE_GXM_PRIMITIVE_TRIANGLE_STRIP,
			SCE_GXM_INDEX_FORMAT_U16, floor_indices_data, 4);
	}
}

static void update_camera(struct camera *camera, SceCtrlData *pad)
{
	vector3f camera_direction;
	vector3f camera_right;

	camera_get_direction_vector(camera, &camera_direction);
	camera_get_right_vector(camera, &camera_right);

	float forward = 0.0f;
	float lateral = 0.0f;

	signed char lx = (signed char)pad->lx - 128;
	if (abs(lx) > ANALOG_THRESHOLD)
		lateral = lx / 1536.0f;

	signed char ly = (signed char)pad->ly - 128;
	if (abs(ly) > ANALOG_THRESHOLD)
		forward = ly / 1536.0f;

	signed char rx = (signed char)pad->rx - 128;
	if (abs(rx) > ANALOG_THRESHOLD)
		camera->rotation.y -= rx / 1536.0f;

	signed char ry = (signed char)pad->ry - 128;
	if (abs(ry) > ANALOG_THRESHOLD)
		camera->rotation.x -= ry / 1536.0f;

	if (pad->buttons & SCE_CTRL_RTRIGGER)
		camera->position.y += 0.1f;
	else if (pad->buttons & SCE_CTRL_LTRIGGER)
		camera->position.y -= 0.1f;

	vector3f_add_mult(&camera->position, &camera_direction, forward);
	vector3f_add_mult(&camera->position, &camera_right, lateral);

	camera_update_view_matrix(camera);
}

static void draw_cube(const struct scene_state *state, const matrix4x4 projection_matrix,
	const matrix4x4 view_matrix, const matrix4x4 model_matrix, const struct phong_material *material)
{
	matrix4x4 mvp_matrix;
	matrix4x4 modelview_matrix;
	matrix3x3 normal_matrix;

	matrix4x4_multiply(modelview_matrix, view_matrix, model_matrix);
	matrix4x4_multiply(mvp_matrix, projection_matrix, modelview_matrix);
	matrix3x3_normal_matrix(normal_matrix, modelview_matrix);

	set_cube_fragment_light_uniform_params(&state->light,
		&gxm_cube_fragment_program_light_params);
	set_cube_fragment_material_uniform_params(material,
		&gxm_cube_fragment_program_phong_material_params);
	set_cube_matrices_uniform_params(mvp_matrix,
		modelview_matrix, normal_matrix);

	sceGxmSetVertexStream(gxm_context, 0, cube_mesh_data);
	sceGxmDraw(gxm_context, SCE_GXM_PRIMITIVE_TRIANGLES,
		SCE_GXM_INDEX_FORMAT_U16, cube_indices_data, 36);
}

static void set_cube_fragment_light_uniform_params(const struct light *light,
	const struct light_gxm_params *params)
{
	set_fragment_default_uniform_data(params->position,
		sizeof(light->position) / sizeof(float), &light->position);
	set_fragment_default_uniform_data(params->color,
		sizeof(light->color) / sizeof(float), &light->color);
}

static void set_cube_matrices_uniform_params(matrix4x4 mvp_matrix,
	matrix4x4 modelview_matrix, matrix3x3 normal_matrix)
{
	set_vertex_default_uniform_data(gxm_cube_vertex_program_u_mvp_matrix_param,
		sizeof(matrix4x4) / sizeof(float), mvp_matrix);
	set_fragment_default_uniform_data(gxm_cube_fragment_program_u_modelview_matrix_param,
		sizeof(matrix4x4) / sizeof(float), modelview_matrix);
	set_fragment_default_uniform_data(gxm_cube_fragment_program_u_normal_matrix_param,
		sizeof(matrix3x3) / sizeof(float), normal_matrix);
}

static void set_cube_fragment_material_uniform_params(const struct phong_material *material,
	const struct phong_material_gxm_params *params)
{
	set_fragment_default_uniform_data(params->ambient,
		sizeof(material->ambient) / sizeof(float), &material->ambient);
	set_fragment_default_uniform_data(params->diffuse,
		sizeof(material->diffuse) / sizeof(float), &material->diffuse);
	set_fragment_default_uniform_data(params->specular,
		sizeof(material->specular) / sizeof(float), &material->specular);
	set_fragment_default_uniform_data(params->shininess,
		sizeof(material->shininess) / sizeof(float), &material->shininess);
}

void set_vertex_default_uniform_data(const SceGxmProgramParameter *param,
	unsigned int component_count, const void *data)
{
	void *uniform_buffer;
	sceGxmReserveVertexDefaultUniformBuffer(gxm_context, &uniform_buffer);
	sceGxmSetUniformDataF(uniform_buffer, param, 0, component_count, data);
}

void set_fragment_default_uniform_data(const SceGxmProgramParameter *param,
	unsigned int component_count, const void *data)
{
	void *uniform_buffer;
	sceGxmReserveFragmentDefaultUniformBuffer(gxm_context, &uniform_buffer);
	sceGxmSetUniformDataF(uniform_buffer, param, 0, component_count, data);
}

void *gpu_alloc_map(SceKernelMemBlockType type, SceGxmMemoryAttribFlags gpu_attrib, size_t size, SceUID *uid)
{
	SceUID memuid;
	void *addr;

	if (type == SCE_KERNEL_MEMBLOCK_TYPE_USER_CDRAM_RW)
		size = ALIGN(size, 256 * 1024);
	else
		size = ALIGN(size, 4 * 1024);

	memuid = sceKernelAllocMemBlock("gpumem", type, size, NULL);
	if (memuid < 0)
		return NULL;

	if (sceKernelGetMemBlockBase(memuid, &addr) < 0)
		return NULL;

	if (sceGxmMapMemory(addr, size, gpu_attrib) < 0) {
		sceKernelFreeMemBlock(memuid);
		return NULL;
	}

	if (uid)
		*uid = memuid;

	return addr;
}

void gpu_unmap_free(SceUID uid)
{
	void *addr;

	if (sceKernelGetMemBlockBase(uid, &addr) < 0)
		return;

	sceGxmUnmapMemory(addr);

	sceKernelFreeMemBlock(uid);
}

void *gpu_vertex_usse_alloc_map(size_t size, SceUID *uid, unsigned int *usse_offset)
{
	SceUID memuid;
	void *addr;

	size = ALIGN(size, 4 * 1024);

	memuid = sceKernelAllocMemBlock("gpu_vertex_usse",
		SCE_KERNEL_MEMBLOCK_TYPE_USER_RW_UNCACHE, size, NULL);
	if (memuid < 0)
		return NULL;

	if (sceKernelGetMemBlockBase(memuid, &addr) < 0)
		return NULL;

	if (sceGxmMapVertexUsseMemory(addr, size, usse_offset) < 0)
		return NULL;

	return addr;
}

void gpu_vertex_usse_unmap_free(SceUID uid)
{
	void *addr;

	if (sceKernelGetMemBlockBase(uid, &addr) < 0)
		return;

	sceGxmUnmapVertexUsseMemory(addr);

	sceKernelFreeMemBlock(uid);
}

void *gpu_fragment_usse_alloc_map(size_t size, SceUID *uid, unsigned int *usse_offset)
{
	SceUID memuid;
	void *addr;

	size = ALIGN(size, 4 * 1024);

	memuid = sceKernelAllocMemBlock("gpu_fragment_usse",
		SCE_KERNEL_MEMBLOCK_TYPE_USER_RW_UNCACHE, size, NULL);
	if (memuid < 0)
		return NULL;

	if (sceKernelGetMemBlockBase(memuid, &addr) < 0)
		return NULL;

	if (sceGxmMapFragmentUsseMemory(addr, size, usse_offset) < 0)
		return NULL;

	return addr;
}

void gpu_fragment_usse_unmap_free(SceUID uid)
{
	void *addr;

	if (sceKernelGetMemBlockBase(uid, &addr) < 0)
		return;

	sceGxmUnmapFragmentUsseMemory(addr);

	sceKernelFreeMemBlock(uid);
}

void *shader_patcher_host_alloc_cb(void *user_data, unsigned int size)
{
	return malloc(size);
}

void shader_patcher_host_free_cb(void *user_data, void *mem)
{
	return free(mem);
}

void display_queue_callback(const void *callbackData)
{
	SceDisplayFrameBuf display_fb;
	const struct display_queue_callback_data *cb_data = callbackData;

	memset(&display_fb, 0, sizeof(display_fb));
	display_fb.size = sizeof(display_fb);
	display_fb.base = cb_data->addr;
	display_fb.pitch = DISPLAY_STRIDE;
	display_fb.pixelformat = DISPLAY_PIXEL_FORMAT;
	display_fb.width = DISPLAY_WIDTH;
	display_fb.height = DISPLAY_HEIGHT;

	sceDisplaySetFrameBuf(&display_fb, SCE_DISPLAY_SETBUF_NEXTFRAME);

	sceDisplayWaitVblankStart();
}
