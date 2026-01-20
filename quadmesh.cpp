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

#include "quadmesh.h"
#include <opencv2/imgcodecs.hpp>
#include <gdal/gdal_priv.h>

QuadMesh::QuadMesh(const std::filesystem::path &filepath)
{
    _image = cv::imread(filepath, cv::IMREAD_ANYDEPTH);
    if (_image.empty())
    {
        throw std::invalid_argument("Invalid file");
    }

    GDALRegister_GTiff();
    GDALDataset *dataset = (GDALDataset *)GDALOpen(filepath.c_str(), GA_ReadOnly);
    if (dataset == nullptr)
    {
        throw std::invalid_argument("Invalid file");
    }

    if (dataset->GetGeoTransform(_geo_transform) != CE_None)
    {
        GDALClose(dataset);
        throw std::invalid_argument("Invalid file");
    }
    GDALClose(dataset);
    GDALDestroyDriverManager();
}

size_t QuadMesh::NumFaces() const
{
    return NumFaceRows() * NumFaceCols();
}

size_t QuadMesh::NumFaceRows() const
{
    return _image.rows - 1;
}

size_t QuadMesh::NumFaceCols() const
{
    return _image.cols - 1;
}

Eigen::Vector3f QuadMesh::GetVertex(size_t i, size_t j) const
{
    return Eigen::Vector3f(_geo_transform[0] + j * _geo_transform[1] + i * _geo_transform[2],
                           _geo_transform[3] + j * _geo_transform[4] + i * _geo_transform[5],
                           _image.at<float>(i, j));
}

void QuadMesh::GetGeoTransform(double* transform) const
{
    memcpy(transform, _geo_transform, sizeof(_geo_transform));
}
