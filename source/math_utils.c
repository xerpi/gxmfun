#include <string.h>
#include <math.h>
#include "math_utils.h"

void vector3f_init(vector3f *v, float x, float y, float z)
{
	v->x = x;
	v->y = y;
	v->z = z;
}

void vector3f_copy(vector3f *dst, const vector3f *src)
{
	dst->x = src->x;
	dst->y = src->y;
	dst->z = src->z;
}

void vector3f_add(vector3f *v1, const vector3f *v2)
{
	v1->x += v2->x;
	v1->y += v2->y;
	v1->z += v2->z;
}

void vector3f_scalar_mult(vector3f *v, float a)
{
	v->x *= a;
	v->y *= a;
	v->z *= a;
}

void vector3f_add_mult(vector3f *v, const vector3f *u, float a)
{
	v->x += u->x * a;
	v->y += u->y * a;
	v->z += u->z * a;
}

void vector3f_opposite(vector3f *v1, const vector3f *v0)
{
	v1->x = -v0->x;
	v1->y = -v0->y;
	v1->z = -v0->z;
}

float vector3f_dot_product(const vector3f *v1, const vector3f *v2)
{
	return v1->x * v2->x + v1->y * v2->y + v1->z * v2->z;
}

void vector3f_cross_product(vector3f *w, const vector3f *u, const vector3f *v)
{
	w->x = u->y * v->z - u->z * v->y;
	w->y = u->z * v->x - u->x * v->z;
	w->z = u->x * v->y - u->y * v->x;
}

void vector3f_matrix4x4_mult(vector3f *u, const matrix4x4 m, const vector3f *v)
{
	u->x = m[0][0] * v->x + m[1][0] * v->y + m[2][0] * v->z;
	u->y = m[0][1] * v->x + m[1][1] * v->y + m[2][1] * v->z;
	u->z = m[0][2] * v->x + m[1][2] * v->y + m[2][2] * v->z;
}

void matrix3x3_from_matrix4x4(const matrix4x4 src, matrix3x3 dst)
{
	int i, j;

	for (i = 0; i < 3; i++) {
		for (j = 0; j < 3; j++)
			dst[i][j] = src[i][j];
	}
}

void matrix4x4_identity(matrix4x4 m)
{
	m[0][1] = m[0][2] = m[0][3] = 0.0f;
	m[1][0] = m[1][2] = m[1][3] = 0.0f;
	m[2][0] = m[2][1] = m[2][3] = 0.0f;
	m[3][0] = m[3][1] = m[3][2] = 0.0f;
	m[0][0] = m[1][1] = m[2][2] = m[3][3] = 1.0f;
}

void matrix4x4_copy(matrix4x4 dst, const matrix4x4 src)
{
	memcpy(dst, src, sizeof(matrix4x4));
}

void matrix4x4_multiply(matrix4x4 dst, const matrix4x4 src1, const matrix4x4 src2)
{
	int i, j, k;

	for (i = 0; i < 4; i++) {
		for (j = 0; j < 4; j++) {
			dst[i][j] = 0.0f;
			for (k = 0; k < 4; k++)
				dst[i][j] += src1[k][j] * src2[i][k];
		}
	}
}

void matrix4x4_init_rotation_x(matrix4x4 m, float rad)
{
	float c = cosf(rad);
	float s = sinf(rad);

	matrix4x4_identity(m);

	m[1][1] = c;
	m[1][2] = s;
	m[2][1] = -s;
	m[2][2] = c;
}

void matrix4x4_init_rotation_y(matrix4x4 m, float rad)
{
	float c = cosf(rad);
	float s = sinf(rad);

	matrix4x4_identity(m);

	m[0][0] = c;
	m[0][2] = -s;
	m[2][0] = s;
	m[2][2] = c;
}


void matrix4x4_init_rotation_z(matrix4x4 m, float rad)
{
	float c = cosf(rad);
	float s = sinf(rad);

	matrix4x4_identity(m);

	m[0][0] = c;
	m[0][1] = s;
	m[1][0] = -s;
	m[1][1] = c;
}

void matrix4x4_rotate_x(matrix4x4 m, float rad)
{
	matrix4x4 m1, m2;

	matrix4x4_init_rotation_x(m1, rad);
	matrix4x4_multiply(m2, m, m1);
	matrix4x4_copy(m, m2);
}


void matrix4x4_rotate_y(matrix4x4 m, float rad)
{
	matrix4x4 m1, m2;

	matrix4x4_init_rotation_y(m1, rad);
	matrix4x4_multiply(m2, m, m1);
	matrix4x4_copy(m, m2);
}

void matrix4x4_rotate_z(matrix4x4 m, float rad)
{
	matrix4x4 m1, m2;

	matrix4x4_init_rotation_z(m1, rad);
	matrix4x4_multiply(m2, m, m1);
	matrix4x4_copy(m, m2);
}

void matrix4x4_init_translation(matrix4x4 m, float x, float y, float z)
{
	matrix4x4_identity(m);

	m[3][0] = x;
	m[3][1] = y;
	m[3][2] = z;
}

void matrix4x4_init_translation_vector3f(matrix4x4 m, const vector3f *v)
{
	matrix4x4_identity(m);

	m[3][0] = v->x;
	m[3][1] = v->y;
	m[3][2] = v->z;
}

void matrix4x4_translate(matrix4x4 m, float x, float y, float z)
{
	matrix4x4 m1, m2;

	matrix4x4_init_translation(m1, x, y, z);
	matrix4x4_multiply(m2, m, m1);
	matrix4x4_copy(m, m2);
}

void matrix4x4_init_scaling(matrix4x4 m, float scale_x, float scale_y, float scale_z)
{
	matrix4x4_identity(m);

	m[0][0] = scale_x;
	m[1][1] = scale_y;
	m[2][2] = scale_z;
}

void matrix4x4_scale(matrix4x4 m, float scale_x, float scale_y, float scale_z)
{
	matrix4x4 m1, m2;

	matrix4x4_init_scaling(m1, scale_x, scale_y, scale_z);
	matrix4x4_multiply(m2, m, m1);
	matrix4x4_copy(m, m2);
}

void matrix4x4_reflect_origin(matrix4x4 m)
{
	matrix4x4_scale(m, -1.0f, -1.0f, -1.0f);
}

void matrix4x4_transpose(matrix4x4 out, const matrix4x4 m)
{
	int i, j;

	for (i = 0; i < 4; i++) {
		for (j = i + 1; j < 4; j++)
			out[i][j] = m[j][i];
	}
}

int matrix4x4_invert(matrix4x4 out, const matrix4x4 m)
{
	int i, j;
	float det;
	matrix4x4 inv;

	inv[0][0] = m[1][1]  * m[2][2] * m[3][3] -
		m[1][1] * m[2][3] * m[3][2] -
		m[2][1] * m[1][2] * m[3][3] +
		m[2][1] * m[1][3] * m[3][2] +
		m[3][1] * m[1][2] * m[2][3] -
		m[3][1] * m[1][3] * m[2][2];

	inv[1][0] = -m[1][0] * m[2][2] * m[3][3] +
		m[1][0] * m[2][3] * m[3][2] +
		m[2][0] * m[1][2] * m[3][3] -
		m[2][0] * m[1][3] * m[3][2] -
		m[3][0] * m[1][2] * m[2][3] +
		m[3][0] * m[1][3] * m[2][2];

	inv[2][0] = m[1][0] * m[2][1] * m[3][3] -
		m[1][0] * m[2][3] * m[3][1] -
		m[2][0] * m[1][1] * m[3][3] +
		m[2][0] * m[1][3] * m[3][1] +
		m[3][0] * m[1][1] * m[2][3] -
		m[3][0] * m[1][3] * m[2][1];

	inv[3][0] = -m[1][0] * m[2][1] * m[3][2] +
		m[1][0] * m[2][2] * m[3][1] +
		m[2][0] * m[1][1] * m[3][2] -
		m[2][0] * m[1][2] * m[3][1] -
		m[3][0] * m[1][1] * m[2][2] +
		m[3][0] * m[1][2] * m[2][1];

	inv[0][1] = -m[0][1] * m[2][2] * m[3][3] +
		m[0][1] * m[2][3] * m[3][2] +
		m[2][1] * m[0][2] * m[3][3] -
		m[2][1] * m[0][3] * m[3][2] -
		m[3][1] * m[0][2] * m[2][3] +
		m[3][1] * m[0][3] * m[2][2];

	inv[1][1] = m[0][0] * m[2][2] * m[3][3] -
		m[0][0] * m[2][3] * m[3][2] -
		m[2][0] * m[0][2] * m[3][3] +
		m[2][0] * m[0][3] * m[3][2] +
		m[3][0] * m[0][2] * m[2][3] -
		m[3][0] * m[0][3] * m[2][2];

	inv[2][1] = -m[0][0] * m[2][1] * m[3][3] +
		m[0][0] * m[2][3] * m[3][1] +
		m[2][0] * m[0][1] * m[3][3] -
		m[2][0] * m[0][3] * m[3][1] -
		m[3][0] * m[0][1] * m[2][3] +
		m[3][0] * m[0][3] * m[2][1];

	inv[3][1] = m[0][0] * m[2][1] * m[3][2] -
		m[0][0] * m[2][2] * m[3][1] -
		m[2][0] * m[0][1] * m[3][2] +
		m[2][0] * m[0][2] * m[3][1] +
		m[3][0] * m[0][1] * m[2][2] -
		m[3][0] * m[0][2] * m[2][1];

	inv[0][2] = m[0][1] * m[1][2] * m[3][3] -
		m[0][1] * m[1][3] * m[3][2] -
		m[1][1] * m[0][2] * m[3][3] +
		m[1][1] * m[0][3] * m[3][2] +
		m[3][1] * m[0][2] * m[1][3] -
		m[3][1] * m[0][3] * m[1][2];

	inv[1][2] = -m[0][0] * m[1][2] * m[3][3] +
		m[0][0] * m[1][3] * m[3][2] +
		m[1][0] * m[0][2] * m[3][3] -
		m[1][0] * m[0][3] * m[3][2] -
		m[3][0] * m[0][2] * m[1][3] +
		m[3][0] * m[0][3] * m[1][2];

	inv[2][2] = m[0][0] * m[1][1] * m[3][3] -
		m[0][0] * m[1][3] * m[3][1] -
		m[1][0] * m[0][1] * m[3][3] +
		m[1][0] * m[0][3] * m[3][1] +
		m[3][0] * m[0][1] * m[1][3] -
		m[3][0] * m[0][3] * m[1][1];

	inv[3][2] = -m[0][0]  * m[1][1] * m[3][2] +
		m[0][0]  * m[1][2] * m[3][1] +
		m[1][0]  * m[0][1] * m[3][2] -
		m[1][0]  * m[0][2] * m[3][1] -
		m[3][0] * m[0][1] * m[1][2] +
		m[3][0] * m[0][2] * m[1][1];

	inv[0][3] = -m[0][1] * m[1][2] * m[2][3] +
		m[0][1] * m[1][3] * m[2][2] +
		m[1][1] * m[0][2] * m[2][3] -
		m[1][1] * m[0][3] * m[2][2] -
		m[2][1] * m[0][2] * m[1][3] +
		m[2][1] * m[0][3] * m[1][2];

	inv[1][3] = m[0][0] * m[1][2] * m[2][3] -
		m[0][0] * m[1][3] * m[2][2] -
		m[1][0] * m[0][2] * m[2][3] +
		m[1][0] * m[0][3] * m[2][2] +
		m[2][0] * m[0][2] * m[1][3] -
		m[2][0] * m[0][3] * m[1][2];

	inv[2][3] = -m[0][0] * m[1][1] * m[2][3] +
		m[0][0] * m[1][3] * m[2][1] +
		m[1][0] * m[0][1] * m[2][3] -
		m[1][0] * m[0][3] * m[2][1] -
		m[2][0] * m[0][1] * m[1][3] +
		m[2][0] * m[0][3] * m[1][1];

	inv[3][3] = m[0][0] * m[1][1] * m[2][2] -
		m[0][0] * m[1][2] * m[2][1] -
		m[1][0] * m[0][1] * m[2][2] +
		m[1][0] * m[0][2] * m[2][1] +
		m[2][0] * m[0][1] * m[1][2] -
		m[2][0] * m[0][2] * m[1][1];

	det = m[0][0] * inv[0][0] + m[0][1] * inv[1][0] + m[0][2] * inv[2][0] + m[0][3] * inv[3][0];

	if (det == 0)
		return 0;

	det = 1.0 / det;

	for (i = 0; i < 4; i++) {
		for (j = 0; j < 4; j++)
			out[i][j] = inv[i][j] * det;
	}

	return 1;
}

void matrix4x4_init_orthographic(matrix4x4 m, float left, float right, float bottom, float top, float near, float far)
{
	m[0][0] = 2.0f / (right - left);
	m[0][1] = 0.0f;
	m[0][2] = 0.0f;
	m[0][3] = 0.0f;

	m[1][0] = 0.0f;
	m[1][1] = 2.0f / (top - bottom);
	m[1][2] = 0.0f;
	m[1][3] = 0.0f;

	m[2][0] = 0.0f;
	m[2][1] = 0.0f;
	m[2][2] = -2.0f / (far - near);
	m[2][3] = 0.0f;

	m[3][0] = -(right + left) / (right - left);
	m[3][1] = -(top + bottom) / (top - bottom);
	m[3][2] = -(far + near) / (far - near);
	m[3][3] = 1.0f;
}

void matrix4x4_init_frustum(matrix4x4 m, float left, float right, float bottom, float top, float near, float far)
{
	m[0][0] = (2.0f * near) / (right - left);
	m[0][1] = 0.0f;
	m[0][2] = 0.0f;
	m[0][3] = 0.0f;

	m[1][0] = 0.0f;
	m[1][1] = (2.0f * near) / (top - bottom);
	m[1][2] = 0.0f;
	m[1][3] = 0.0f;

	m[2][0] = (right + left) / (right - left);
	m[2][1] = (top + bottom) / (top - bottom);
	m[2][2] = -(far + near) / (far - near);
	m[2][3] = -1.0f;

	m[3][0] = 0.0f;
	m[3][1] = 0.0f;
	m[3][2] = (-2.0f * far * near) / (far - near);
	m[3][3] = 0.0f;
}

void matrix4x4_init_perspective(matrix4x4 m, float fov, float aspect, float near, float far)
{
	float half_height = near * tanf(DEG_TO_RAD(fov) * 0.5f);
	float half_width = half_height * aspect;

	matrix4x4_init_frustum(m, -half_width, half_width, -half_height, half_height, near, far);
}

void matrix3x3_normal_matrix(matrix3x3 out, const matrix4x4 m)
{
	matrix4x4 m1, m2;

	matrix4x4_invert(m1, m);
	matrix4x4_transpose(m2, m1);
	matrix3x3_from_matrix4x4(m2, out);
}
