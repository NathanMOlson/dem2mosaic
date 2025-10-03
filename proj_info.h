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