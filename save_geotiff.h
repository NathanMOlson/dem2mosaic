#ifndef __SAVE_GEOTIFF_H__
#define __SAVE_GEOTIFF_H__

#include <filesystem>
#include <optional>
#include <opencv2/core.hpp>
#include "geotiff/geovalues.h"

struct GeoInfo
{
    double transform[6];
    std::optional<modeltype_t> model_type;
    std::optional<rastertype_t> raster_type;
    std::optional<geounits_t> projected_linear_units;
    std::optional<geounits_t> geographic_angular_units;
    std::optional<pcstype_t> projected_coordinate_system;
    std::optional<int64_t> gdal_nodata_value;
    std::optional<double> capture_time;
};

int save_geotiff(const std::filesystem::path &filepath, const cv::Mat &img, const GeoInfo& geo);

#endif