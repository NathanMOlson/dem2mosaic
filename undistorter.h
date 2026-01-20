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

#ifndef __UNDISTORTER_H__
#define __UNDISTORTER_H__

#include <vector>
#include <opencv2/core.hpp>

class Undistorter
{
public:
    Undistorter(double fx, double fy, double cx, double cy, size_t width, size_t height, const std::vector<double> &dist_coeffs);

    std::vector<cv::Point2f> GetPixelCoords(const std::vector<cv::Point3f> &rays_cam) const;
    cv::Point2f GetPixelCoords(const cv::Point3f &ray_cam) const;
    cv::Mat Undistort(const cv::Mat &img) const;
    float MaxXYLength() const;

private:
    cv::Mat _cam_old;
    cv::Mat _cam_new;
    std::vector<double> _dist_coeffs;
    float _max_xy_length;

    cv::Mat _map1;
    cv::Mat _map2;
};

#endif
