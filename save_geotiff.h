/** DEM2mosaic: convert a DEM and an OpenSfM reconstruction file to a orthomosaic.
 Copyright (C) 2025-2026 Lab 308, LLC

 This program is free software: you can redistribute it and/or modify
 it under the terms of the GNU Affero General Public License as
 published by the Free Software Foundation, either version 3 of the
 License, or (at your option) any later version.

 This program is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU Affero General Public License for more details.

 You should have received a copy of the GNU Affero General Public License
 along with this program.  If not, see <http://www.gnu.org/licenses/>.
 **/

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
    std::optional<double> capture_time_utc;
};

int save_geotiff(const std::filesystem::path &filepath, const cv::Mat &img, const GeoInfo& geo);

#endif