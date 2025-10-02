#ifndef __SAVE_GEOTIFF_H__
#define __SAVE_GEOTIFF_H__

#include <filesystem>
#include <opencv2/core.hpp>

int save_geotiff(const std::filesystem::path &filepath, const cv::Mat &img);

#endif