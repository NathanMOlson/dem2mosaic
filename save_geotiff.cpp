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

#include "save_geotiff.h"
#include "geotiff/xtiffio.h"
#include "geotiff/geotiffio.h"
#include <sstream>

std::string tiff_time_string(double t_utc)
{
    std::time_t t = t_utc;
    std::tm tm = *std::gmtime(&t);
    std::stringstream ss;
    ss << std::put_time(&tm, "%Y:%m:%d %TZ");
    return ss.str();
}

int save_geotiff(const std::filesystem::path &filepath, const cv::Mat &img, const GeoInfo &geo)
{
    if (img.depth() != CV_8U && img.depth() != CV_16U)
    {
        return -1;
    }

    TIFF *tif = nullptr;
    GTIF *gtif = nullptr;

    tif = XTIFFOpen(filepath.c_str(), "w");
    if (!tif)
    {
        return -2;
    }

    gtif = GTIFNew(tif);
    if (!gtif)
    {
        XTIFFClose(tif);
        return -3;
    }

    constexpr uint32_t tile_width = 256;

    size_t bits_per_sample = (img.depth() == CV_16U || img.depth() == CV_16S) ? 16 : 8;

    TIFFSetField(tif, TIFFTAG_IMAGEWIDTH, img.cols);
    TIFFSetField(tif, TIFFTAG_IMAGELENGTH, img.rows);
    TIFFSetField(tif, TIFFTAG_TILEWIDTH, tile_width);
    TIFFSetField(tif, TIFFTAG_TILELENGTH, tile_width);
    TIFFSetField(tif, TIFFTAG_SAMPLESPERPIXEL, img.channels());
    TIFFSetField(tif, TIFFTAG_BITSPERSAMPLE, bits_per_sample);
    TIFFSetField(tif, TIFFTAG_ORIENTATION, ORIENTATION_TOPLEFT);
    TIFFSetField(tif, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
    if (img.depth() == CV_16U || img.depth() == CV_8U)
    {
        TIFFSetField(tif, TIFFTAG_SAMPLEFORMAT, SAMPLEFORMAT_UINT);
    }
    else if (img.depth() == CV_16S)
    {
        TIFFSetField(tif, TIFFTAG_SAMPLEFORMAT, SAMPLEFORMAT_INT);
    }
    if (img.channels() == 3)
    {
        TIFFSetField(tif, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_RGB);
    }
    else
    {
        TIFFSetField(tif, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_MINISBLACK);
    }
    TIFFSetField(tif, TIFFTAG_COMPRESSION, COMPRESSION_DEFLATE);

    size_t bits_per_pixel = bits_per_sample * img.channels();
    size_t bytes_per_line = bits_per_pixel / 8 * img.cols;

    TIFFSetField(tif, TIFFTAG_ROWSPERSTRIP, TIFFDefaultStripSize(tif, bytes_per_line));

    if (geo.capture_time_utc)
    {
        TIFFSetField(tif, TIFFTAG_DATETIME, tiff_time_string(geo.capture_time_utc.value()).c_str());
    }
    if (geo.model_type)
    {
        GTIFKeySet(gtif, GTModelTypeGeoKey, TYPE_SHORT, 1, *geo.model_type);
    }
    if (geo.raster_type)
    {
        GTIFKeySet(gtif, GTRasterTypeGeoKey, TYPE_SHORT, 1, *geo.raster_type);
    }
    if (geo.geographic_angular_units)
    {
        GTIFKeySet(gtif, GeogAngularUnitsGeoKey, TYPE_SHORT, 1, *geo.geographic_angular_units);
    }
    if (geo.projected_linear_units)
    {
        GTIFKeySet(gtif, ProjLinearUnitsGeoKey, TYPE_SHORT, 1, *geo.projected_linear_units);
    }
    if (geo.projected_coordinate_system)
    {
        GTIFKeySet(gtif, ProjectedCSTypeGeoKey, TYPE_SHORT, 1, *geo.projected_coordinate_system);
    }
    // if (geo.transform)
    {
        std::vector<double> tiepoints;
        tiepoints.push_back(0.0);
        tiepoints.push_back(0.0);
        tiepoints.push_back(0.0);
        tiepoints.push_back(geo.transform[0]);
        tiepoints.push_back(geo.transform[3]);
        tiepoints.push_back(0.0);
        TIFFSetField(tif, TIFFTAG_GEOTIEPOINTS, tiepoints.size(), tiepoints.data());
        std::vector<double> scale;
        scale.push_back(geo.transform[1]);
        scale.push_back(geo.transform[5]);
        TIFFSetField(tif, TIFFTAG_GEOPIXELSCALE, scale.size(), scale.data());
    }
    if (geo.gdal_nodata_value)
    {
        TIFFSetField(tif, TIFFTAG_GDAL_NODATA, std::to_string(*geo.gdal_nodata_value).c_str());
    }

    GTIFWriteKeys(gtif);

    cv::Mat chip(tile_width, tile_width, img.type());
    for (int i = 0; i < img.rows; i += tile_width)
    {
        for (int j = 0; j < img.cols; j += tile_width)
        {
            int w = std::min((int)tile_width, img.cols - j);
            int h = std::min((int)tile_width, img.rows - i);
            if (w < (int)tile_width || h < (int)tile_width)
            {
                chip = cv::Mat::zeros(tile_width, tile_width, img.type());
            }
            img(cv::Rect(j, i, w, h)).copyTo(chip(cv::Rect(0, 0, w, h)));
            TIFFWriteTile(tif, chip.data, j, i, 0, 0);
        }
    }
    GTIFFree(gtif);
    XTIFFClose(tif);

    return 0;
}