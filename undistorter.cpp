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

#include "undistorter.h"
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>

Undistorter::Undistorter(double fx, double fy,
                         double cx, double cy,
                         size_t width, size_t height,
                         const std::vector<double> &dist_coeffs) : _dist_coeffs(dist_coeffs)
{
    _cam_old = cv::Mat::zeros(3, 3, CV_32F);
    size_t maxdim = std::max(height, width);

    _cam_old.at<float>(0, 0) = fx * maxdim;
    _cam_old.at<float>(1, 1) = fy * maxdim;
    _cam_old.at<float>(0, 2) = cx * maxdim + width / 2.0 - 0.5;
    _cam_old.at<float>(1, 2) = cy * maxdim + height / 2.0 + 0.5; // +0.5 seems like it is the wrong direction, but matches OpenSfM's undistorted images
    _cam_old.at<float>(2, 2) = 1;

    _cam_new = cv::getDefaultNewCameraMatrix(_cam_old, cv::Size(width, height), true);

    float x_out = 0;
    float r = 0.25;
    for (; r < 1.0e9; r*= 1.1)
    {
        cv::Point2f p = GetPixelCoords(cv::Point3f(r, 0, 1));
        if (p.x < x_out)
        {
            break;
        }
        x_out = p.x;
    }
    _max_xy_length = r / 1.1;

    cv::initUndistortRectifyMap(_cam_old, _dist_coeffs, cv::noArray(), _cam_new, cv::Size(width, height), CV_16SC2, _map1, _map2);
}

std::vector<cv::Point2f> Undistorter::GetPixelCoords(const std::vector<cv::Point3f> &rays_cam) const
{
    std::vector<cv::Point2f> coords;
    std::vector<float> zero_vec(3, 0);
    cv::projectPoints(rays_cam, zero_vec, zero_vec, _cam_old, _dist_coeffs, coords);
    return coords;
}

cv::Point2f Undistorter::GetPixelCoords(const cv::Point3f &ray_cam) const
{
    std::vector<cv::Point3f> rays;
    rays.push_back(ray_cam);
    return GetPixelCoords(rays)[0];
}

cv::Mat Undistorter::Undistort(const cv::Mat &img) const
{
    cv::Mat undistorted;
    cv::remap(img, undistorted, _map1, _map2, cv::INTER_LINEAR);
    return undistorted;
}

float Undistorter::MaxXYLength() const
{
    return _max_xy_length;
}
