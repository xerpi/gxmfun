#include "camera.h"

void camera_init_zero(struct camera *camera)
{
	vector3f_init(&camera->position, 0.0f, 0.0f, 0.0f);
	vector3f_init(&camera->rotation, 0.0f, 0.0f, 0.0f);
	matrix4x4_identity(camera->view_matrix);
}

void camera_init(struct camera *camera, const vector3f *pos, const vector3f *rot)
{
	vector3f_copy(&camera->position, pos);
	vector3f_copy(&camera->rotation, rot);
	camera_update_view_matrix(camera);
}

void camera_update_view_matrix(struct camera *camera)
{
	vector3f opposite;
	matrix4x4 m1, m2;
	matrix4x4 mt;
	matrix4x4 mx, my, mz;

	vector3f_opposite(&opposite, &camera->position);

	matrix4x4_init_translation_vector3f(mt, &opposite);
	matrix4x4_init_rotation_x(mx, -camera->rotation.x);
	matrix4x4_init_rotation_y(my, -camera->rotation.y);
	matrix4x4_init_rotation_z(mz, -camera->rotation.z);

	matrix4x4_multiply(m1, mx, my);
	matrix4x4_multiply(m2, m1, mz);
	matrix4x4_multiply(camera->view_matrix, m2, mt);
}

void camera_get_direction_vector(const struct camera *camera, vector3f *direction)
{
	direction->x = camera->view_matrix[0][2];
	direction->y = camera->view_matrix[1][2];
	direction->z = camera->view_matrix[2][2];
}

void camera_get_right_vector(const struct camera *camera, vector3f *right)
{
	right->x = camera->view_matrix[0][0];
	right->y = camera->view_matrix[1][0];
	right->z = camera->view_matrix[2][0];
}

void camera_get_up_vector(const struct camera *camera, vector3f *up)
{
	up->x = camera->view_matrix[0][1];
	up->y = camera->view_matrix[1][1];
	up->z = camera->view_matrix[2][1];
}
