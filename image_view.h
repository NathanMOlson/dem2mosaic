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

#ifndef __IMAGE_VIEW_H__
#define __IMAGE_VIEW_H__

#include <optional>
#include <Eigen/Dense>
#include "quadmesh.h"
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include "undistorter.h"

struct QuadInfo
{
    uint16_t view_id;
    float quality;
    bool fully_visible;
    uint16_t num_valid_pixels;
    std::vector<float> tl;
    std::vector<float> tr;
    std::vector<float> br;
    std::vector<float> bl;
    float tl_w;
    float tr_w;
    float br_w;
    float bl_w;

    bool operator<(QuadInfo const &other) const
    {
        return view_id < other.view_id;
    }
};

/**
 * Class representing a view with specialized functions for texturing.
 */
class ImageView
{
private:
    std::size_t id;

    Eigen::Vector3f pos;
    Eigen::Vector3f viewdir;
    Eigen::Matrix<float, 4, 4> _world_to_cam;
    std::string image_file;
    double _capture_time;
    cv::Mat image;
    std::string _serial_number;

    cv::Mat _weight_tl;
    cv::Mat _weight_tr;
    cv::Mat _weight_br;
    cv::Mat _weight_bl;

    std::shared_ptr<Undistorter> _undistorter;

    static constexpr size_t _tile_width = 32;

    void initializeCameraPos(const Eigen::Vector3f &trans, const Eigen::Matrix<float, 3, 3> &rot);
    void initializeViewDir(const Eigen::Matrix<float, 3, 3> &rot);
    void initializeWorldToCam(const Eigen::Vector3f &trans, const Eigen::Matrix<float, 3, 3> &rot);

public:
    /** Returns the id of the TexureView which is consistent for every run. */
    std::size_t get_id(void) const;

    /** Returns the 2D pixel coordinates of the given vertex projected into the view. */
    cv::Point2f get_pixel_coords(Eigen::Vector3f const &vertex) const;
    std::vector<cv::Point2f> get_pixel_coords(const std::vector<Eigen::Vector3f> &vertex) const;
    /** Returns the RGB pixel values [0, 1] for the given vertex projected into the view, calculated by linear interpolation. */
    //        Eigen::Vector3f get_pixel_values(Eigen::Vector3f const & vertex) const;

    /** Returns whether the pixel location is valid in this view.
     * The pixel location is valid if its inside the visible area and,
     * if a validity mask has been generated, all surrounding (integer coordinate) pixels are valid in the validity mask.
     */
    bool valid_pixel(cv::Point2f pixel) const;

    bool inside(const std::vector<cv::Point2f> &corners) const;
    bool intersects(const std::vector<cv::Point2f> &corners) const;

    /** Constructs a ImageView from the given metadata */
    ImageView(std::size_t id, const Eigen::Vector3f &translation,
              const Eigen::Vector3f &rotation,
              double capture_time,
              std::shared_ptr<Undistorter> undistorter,
              const std::filesystem::path &image_file);

    cv::Mat GetTile(const std::vector<cv::Point2f> &corners, int interp_type, int border_mode, bool preserve_max) const;

    bool IsImageLoaded() const;

    std::filesystem::path ImagePath() const;

    /** Returns the position. */
    Eigen::Vector3f get_pos(void) const;
    /** Returns the viewing direction. */
    Eigen::Vector3f get_viewing_direction(void) const;

    /** Loads the corresponding image. */
    void load_image(void);

    /** Releases the corresponding image. */
    void release_image(void);

    void get_face_info(const std::vector<cv::Point2f> &corners,
                       QuadInfo *face_info) const;

    double get_capture_time() const;
    std::string get_serial_number() const;
};

std::vector<ImageView> generate_image_views(const std::filesystem::path &json_file);
std::optional<double> get_mean_time(const std::vector<ImageView> &image_views);
float quad_brightness(const QuadInfo &quad_info, int channel);

#endif
