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

#ifndef __QUADMESH_H__
#define __QUADMESH_H__

#include <Eigen/Dense>
#include <opencv2/core.hpp>
#include <filesystem>

class QuadMesh
{
public:
    QuadMesh(const std::filesystem::path& filepath);
    size_t NumFaces() const;
    size_t NumFaceRows() const;
    size_t NumFaceCols() const;
    Eigen::Vector3f GetVertex(size_t i, size_t j) const;
    void GetGeoTransform(double* transform) const;

private:
    cv::Mat _image;
    double _geo_transform[6];
};

#endif
