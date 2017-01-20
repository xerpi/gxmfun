#ifndef MATH_UTILS_H
#define MATH_UTILS_H

#include <math.h>

#define DEG_TO_RAD(x) ((x) * M_PI / 180.0)
#define RAD_TO_RAD(x) ((x) * 180.0 / M_PI)

typedef float matrix4x4[4][4];

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

void matrix4x4_init_orthographic(matrix4x4 m, float left, float right, float bottom, float top, float near, float far);
void matrix4x4_init_frustum(matrix4x4 m, float left, float right, float bottom, float top, float near, float far);
void matrix4x4_init_perspective(matrix4x4 m, float fov, float aspect, float near, float far);

#endif
