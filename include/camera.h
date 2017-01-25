#ifndef CAMERA_H
#define CAMERA_H

#include "math_utils.h"

struct camera {
	vector3f position;
	vector3f rotation;
	matrix4x4 view_matrix;
};

void camera_init_zero(struct camera *camera);
void camera_init(struct camera *camera, const vector3f *pos, const vector3f *rot);
void camera_update_view_matrix(struct camera *camera);
void camera_get_direction_vector(const struct camera *camera, vector3f *direction);
void camera_get_right_vector(const struct camera *camera, vector3f *right);
void camera_get_up_vector(const struct camera *camera, vector3f *up);

#endif
