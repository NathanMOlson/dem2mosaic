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

#ifndef __PROJ_INFO_H__
#define __PROJ_INFO_H__

#include <filesystem>
#include "geotiff/geovalues.h"

struct ProjInfo
{
    pcstype_t crs;
    double x_offset;
    double y_offset;
};

ProjInfo parse_georef_file(const std::filesystem::path &filepath);

#endif