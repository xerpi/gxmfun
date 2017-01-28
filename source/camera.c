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
	matrix4x4 mtmp, morient, mtrans;
	matrix4x4 mx, my, mz;

	vector3f_opposite(&opposite, &camera->position);

	matrix4x4_init_translation_vector3f(mtrans, &opposite);
	matrix4x4_init_rotation_x(mx, -camera->rotation.x);
	matrix4x4_init_rotation_y(my, -camera->rotation.y);
	matrix4x4_init_rotation_z(mz, -camera->rotation.z);

	matrix4x4_multiply(mtmp, mx, my);
	matrix4x4_multiply(morient, mtmp, mz);
	matrix4x4_multiply(camera->view_matrix, morient, mtrans);
}

void camera_get_look_vector(const struct camera *camera, vector3f *look)
{
	matrix4x4_get_z_axis(camera->view_matrix, look);
}

void camera_get_right_vector(const struct camera *camera, vector3f *right)
{
	matrix4x4_get_x_axis(camera->view_matrix, right);
}

void camera_get_up_vector(const struct camera *camera, vector3f *up)
{
	matrix4x4_get_y_axis(camera->view_matrix, up);
}
