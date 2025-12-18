#ifndef __TEMPERATURE_COMPENSATION_H__
#define __TEMPERATURE_COMPENSATION_H__

#include <vector>
#include <opencv2/core.hpp>
#include "quadmesh.h"
#include "image_view.h"

struct TemperatureCompensation
{
    float offset;
    float b;
};

float get_temperature_adjustment(const TemperatureCompensation *comp, float cos_viewing_angle);

cv::Mat angular_temperature_compensation(const cv::Mat &labels, const QuadMesh &mesh,
                                         const std::vector<ImageView> &image_views, const std::vector<TemperatureCompensation *> comp);

#endif