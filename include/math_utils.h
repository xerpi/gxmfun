#ifndef MATH_UTILS_H
#define MATH_UTILS_H

#include <math.h>

#define DEG_TO_RAD(x) ((x) * M_PI / 180.0)
#define RAD_TO_RAD(x) ((x) * 180.0 / M_PI)

typedef struct {
	float x, y;
} vector2f;

typedef struct {
	float x, y, z;
} vector3f;

typedef struct {
	float x, y, z, w;
} vector4f;

typedef float matrix3x3[3][3];
typedef float matrix4x4[4][4];

void matrix3x3_from_matrix4x4(const matrix4x4 src, matrix3x3 dst);

void matrix4x4_identity(matrix4x4 m);
void matrix4x4_copy(matrix4x4 dst, const matrix4x4 src);

void matrix4x4_multiply(matrix4x4 dst, const matrix4x4 src1, const matrix4x4 src2);

void matrix4x4_init_rotation_x(matrix4x4 m, float rad);
void matrix4x4_init_rotation_y(matrix4x4 m, float rad);
void matrix4x4_init_rotation_z(matrix4x4 m, float rad);

void matrix4x4_rotate_x(matrix4x4 m, float rad);
void matrix4x4_rotate_y(matrix4x4 m, float rad);
void matrix4x4_rotate_z(matrix4x4 m, float rad);

void matrix4x4_init_translation(matrix4x4 m, float x, float y, float z);
void matrix4x4_translate(matrix4x4 m, float x, float y, float z);

void matrix4x4_init_scaling(matrix4x4 m, float scale_x, float scale_y, float scale_z);
void matrix4x4_scale(matrix4x4 m, float scale_x, float scale_y, float scale_z);

void matrix4x4_transpose(const matrix4x4 m, matrix4x4 out);
int matrix4x4_invert(const matrix4x4 m, matrix4x4 inv_out);

void matrix4x4_init_orthographic(matrix4x4 m, float left, float right, float bottom, float top, float near, float far);
void matrix4x4_init_frustum(matrix4x4 m, float left, float right, float bottom, float top, float near, float far);
void matrix4x4_init_perspective(matrix4x4 m, float fov, float aspect, float near, float far);

/* Graphics related */

void matrix3x3_normal_matrix(const matrix4x4 m, matrix3x3 out);

#endif
