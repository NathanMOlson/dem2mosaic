#ifndef __LOCAL_SEAM_LEVELING_H__
#define __TEMPERATURE_COMPENSATION_H__

#include <opencv2/core.hpp>

cv::Mat local_seam_leveling(const cv::Mat &labels, const cv::Mat mosaic);

#endif