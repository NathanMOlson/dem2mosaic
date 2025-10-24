#include "image_view.h"
#include <opencv2/imgcodecs.hpp>
#include <iostream>
#include <fstream>
#include <nlohmann/json.hpp>
using json = nlohmann::json;

void ImageView::initializeCameraPos(const Eigen::Vector3f &trans, const Eigen::Matrix<float, 3, 3> &rot)
{
    pos[0] = -rot(0, 1) * trans[0] - rot(1, 0) * trans[1] - rot(2, 0) * trans[2];
    pos[1] = -rot(0, 1) * trans[0] - rot(1, 1) * trans[1] - rot(2, 1) * trans[2];
    pos[2] = -rot(0, 2) * trans[0] - rot(1, 2) * trans[1] - rot(2, 2) * trans[2];
}

void ImageView::initializeViewDir(const Eigen::Matrix<float, 3, 3> &rot)
{
    viewdir[0] = rot(2, 0);
    viewdir[1] = rot(2, 1);
    viewdir[2] = rot(2, 2);
}

void ImageView::initializeWorldToCam(const Eigen::Vector3f &trans, const Eigen::Matrix<float, 3, 3> &rot)
{
    _world_to_cam(0, 0) = rot(0, 0);
    _world_to_cam(0, 1) = rot(0, 1);
    _world_to_cam(0, 2) = rot(0, 2);
    _world_to_cam(0, 3) = trans[0];
    _world_to_cam(1, 0) = rot(1, 0);
    _world_to_cam(1, 1) = rot(1, 1);
    _world_to_cam(1, 2) = rot(1, 2);
    _world_to_cam(1, 3) = trans[1];
    _world_to_cam(2, 0) = rot(2, 0);
    _world_to_cam(2, 1) = rot(2, 1);
    _world_to_cam(2, 2) = rot(2, 2);
    _world_to_cam(2, 3) = trans[2];
    _world_to_cam(3, 0) = 0.0f;
    _world_to_cam(3, 1) = 0.0f;
    _world_to_cam(3, 2) = 0.0f;
    _world_to_cam(3, 3) = 1.0f;
}

Eigen::Matrix<float, 3, 3> get_rotation_matrix(const Eigen::Vector3f &rotation)
{
    float len = rotation.norm();
    Eigen::Matrix<float, 3, 3> K;
    K(0, 0) = 0;
    K(0, 1) = -rotation[2] / len;
    K(0, 2) = rotation[1] / len;
    K(1, 0) = rotation[2] / len;
    K(1, 1) = 0;
    K(1, 2) = -rotation[0] / len;
    K(2, 0) = -rotation[1] / len;
    K(2, 1) = rotation[0] / len;
    K(2, 2) = 0;
    Eigen::Matrix<float, 3, 3> I = Eigen::Matrix<float, 3, 3>::Identity();
    return I + K * sin(len) + K * K * (1 - cos(len));
}

ImageView::ImageView(std::size_t id, const Eigen::Vector3f &translation,
                     const Eigen::Vector3f &rotation,
                     double capture_time,
                     std::shared_ptr<Undistorter> undistorter,
                     const std::filesystem::path &image_file)
    : id(id), image_file(image_file), _capture_time(capture_time), _undistorter(undistorter)
{
    Eigen::Matrix<float, 3, 3> rotation_matrix = get_rotation_matrix(rotation);
    initializeCameraPos(translation, rotation_matrix);
    initializeViewDir(rotation_matrix);
    initializeWorldToCam(translation, rotation_matrix);

    _weight_br = cv::Mat(_tile_width, _tile_width, CV_32F);
    for (size_t i = 0; i < _tile_width; i++)
    {
        float y = (i + 0.5) / _tile_width;
        for (size_t j = 0; j < _tile_width; j++)
        {
            float x = (j + 0.5) / _tile_width;
            _weight_br.at<float>(i, j) = x * y;
        }
    }
    _weight_br = _weight_br / cv::sum(_weight_br)[0];

    cv::rotate(_weight_br, _weight_tl, cv::ROTATE_180);
    cv::rotate(_weight_br, _weight_tr, cv::ROTATE_90_COUNTERCLOCKWISE);
    cv::rotate(_weight_br, _weight_bl, cv::ROTATE_90_CLOCKWISE);
}

cv::Point2f ImageView::get_pixel_coords(Eigen::Vector3f const &vertex) const
{
    Eigen::Vector4f ray_cam = _world_to_cam * Eigen::Vector4f(vertex[0], vertex[1], vertex[2], 1.0f);
    ray_cam /= ray_cam[2];
    return _undistorter->GetPixelCoords(cv::Point3f(ray_cam[0], ray_cam[1], ray_cam[2]));
}

std::vector<cv::Point2f> ImageView::get_pixel_coords(const std::vector<Eigen::Vector3f> &vertices) const
{
    std::vector<cv::Point2f> pixels;
    for (const auto &vertex : vertices)
    {
        pixels.push_back(get_pixel_coords(vertex));
    }
    return pixels;
}

cv::Mat ImageView::GetTile(const std::vector<cv::Point2f> &corners, int interp_type, int border_mode, bool preserve_max) const
{
    std::vector<cv::Point2f> tile_corners;
    tile_corners.push_back(cv::Point2f(-0.5, -0.5));
    tile_corners.push_back(cv::Point2f(_tile_width - 0.5, -0.5));
    tile_corners.push_back(cv::Point2f(_tile_width - 0.5, _tile_width - 0.5));
    tile_corners.push_back(cv::Point2f(-0.5, _tile_width - 0.5));

    cv::Mat warp = cv::getPerspectiveTransform(corners, tile_corners);
    cv::Mat tile(_tile_width, _tile_width, image.type());
    cv::warpPerspective(image, tile, warp, tile.size(), interp_type, border_mode);

    if (preserve_max)
    {
        if (tile.type() != CV_8UC1 && tile.type() != CV_16UC1)
        {
            std::cout << "Cannot preserve max, image type unsupported: " << tile.type() << std::endl;
            return tile;
        }
        float max_edge_length = 0;
        for (int i = 0; i < 4; i++)
        {
            float edge_length = cv::norm(corners[(i + 1) % 4] - corners[i]);
            max_edge_length = std::max(edge_length, max_edge_length);
        }
        int tile_width = ceil(1.414 * max_edge_length);
        tile_corners[0] = cv::Point2f(-0.5, -0.5);
        tile_corners[1] = cv::Point2f(tile_width - 0.5, -0.5);
        tile_corners[2] = cv::Point2f(tile_width - 0.5, tile_width - 0.5);
        tile_corners[3] = cv::Point2f(-0.5, tile_width - 0.5);
        warp = cv::getPerspectiveTransform(corners, tile_corners);
        cv::Mat max_preserving_tile(tile_width, tile_width, image.type());
        cv::warpPerspective(image, max_preserving_tile, warp, max_preserving_tile.size(), cv::INTER_NEAREST, cv::BORDER_CONSTANT);
        cv::Point max_loc;
        double max_val;
        cv::minMaxLoc(max_preserving_tile, NULL, &max_val, NULL, &max_loc);
        int i = round((max_loc.y + 0.5) * (float)_tile_width / (float)tile_width - 0.5);
        int j = round((max_loc.x + 0.5) * (float)_tile_width / (float)tile_width - 0.5);
        if (tile.type() == CV_16UC1)
        {
            // std::cout << "tile width: " << tile_width << ", max " << max_val << " at " << max_loc << ": (" << i << "," << j << "), was " << (int)tile.at<uint16_t>(i, j) << std::endl;
            tile.at<uint16_t>(i, j) = max_val;
        }
        else if (tile.type() == CV_8UC1)
        {
            // std::cout << "tile width: " << tile_width << ", max " << max_val << " at " << max_loc << ": (" << i << "," << j << "), was " << (int)tile.at<uint8_t>(i, j) << std::endl;
            tile.at<uint8_t>(i, j) = max_val;
        }
    }

    return tile;
}

bool ImageView::IsImageLoaded() const
{
    return !image.empty();
}

std::filesystem::path ImageView::ImagePath() const
{
    return image_file;
}

inline float quad_area(const std::vector<cv::Point2f> &corners)
{
    if (corners.size() != 4)
    {
        return 0.0;
    }
    return 0.5 * (corners[0].x * corners[1].y - corners[0].y * corners[1].x + corners[1].x * corners[2].y - corners[1].y * corners[2].x + corners[2].x * corners[3].y - corners[2].y * corners[3].x + corners[3].x * corners[0].y - corners[3].y * corners[0].x);
}

bool ImageView::valid_pixel(cv::Point2f pixel) const
{
    return pixel.x >= -0.5 && pixel.x <= image.cols - 0.5 && pixel.y >= -0.5 && pixel.y <= image.rows - 0.5;
}

std::size_t ImageView::get_id(void) const
{
    return id;
}

Eigen::Vector3f ImageView::get_pos(void) const
{
    return pos;
}

Eigen::Vector3f ImageView::get_viewing_direction(void) const
{
    return viewdir;
}

bool ImageView::inside(const std::vector<cv::Point2f> &corners) const
{
    for (const auto &corner : corners)
    {
        if (!valid_pixel(corner))
        {
            return false;
        }
    }
    return true;
}

bool ImageView::intersects(const std::vector<cv::Point2f> &corners) const
{
    for (const auto &corner : corners)
    {
        if (valid_pixel(corner))
        {
            return true;
        }
    }
    return false;
}

void ImageView::load_image(void)
{
    image = cv::imread(image_file, cv::IMREAD_ANYDEPTH | cv::IMREAD_UNCHANGED);
}

void ImageView::release_image(void)
{
    image.release();
}

void ImageView::get_face_info(const std::vector<cv::Point2f> &corners,
                              QuadInfo *face_info) const
{
    assert(!image.empty());
    face_info->fully_visible = inside(corners);

    float area = quad_area(corners);

    if (area < std::numeric_limits<float>::epsilon())
    {
        face_info->quality = 0.0f;
        return;
    }

    float gmi = 0;
    constexpr bool preserve_max = false;
    cv::Mat tile = GetTile(corners, cv::INTER_LINEAR, cv::BORDER_REPLICATE, preserve_max);
    cv::Mat tile_f;
    tile.convertTo(tile_f, CV_32F);

    if (face_info->fully_visible)
    {
        face_info->num_valid_pixels = _tile_width * _tile_width;
        face_info->tl = _weight_tl.dot(tile_f);
        face_info->tr = _weight_tr.dot(tile_f);
        face_info->br = _weight_br.dot(tile_f);
        face_info->bl = _weight_bl.dot(tile_f);
        face_info->tl_w = 1;
        face_info->tr_w = 1;
        face_info->br_w = 1;
        face_info->bl_w = 1;
    }
    else
    {
        cv::Mat mask = GetTile(corners, cv::INTER_NEAREST, cv::BORDER_CONSTANT, preserve_max);
        if (mask.type() != CV_8U)
        {
            mask.convertTo(mask, CV_8U);
        }
        face_info->num_valid_pixels = cv::countNonZero(tile);
        cv::Mat weight_tl, weight_tr, weight_br, weight_bl;
        _weight_tl.copyTo(weight_tl, mask);
        _weight_tr.copyTo(weight_tr, mask);
        _weight_br.copyTo(weight_br, mask);
        _weight_bl.copyTo(weight_bl, mask);

        face_info->tl_w = cv::sum(weight_tl)[0];
        face_info->tr_w = cv::sum(weight_tr)[0];
        face_info->br_w = cv::sum(weight_br)[0];
        face_info->bl_w = cv::sum(weight_bl)[0];

        face_info->tl = weight_tl.dot(tile_f);
        face_info->tr = weight_tr.dot(tile_f);
        face_info->br = weight_br.dot(tile_f);
        face_info->bl = weight_bl.dot(tile_f);
    }

    cv::Mat grad_x;
    cv::Mat grad_y;
    cv::Sobel(tile, grad_x, CV_32F, 1, 0);
    cv::Sobel(tile, grad_y, CV_32F, 0, 1);
    cv::multiply(grad_x, grad_x, grad_x);
    cv::multiply(grad_y, grad_y, grad_y);
    // cv::sqrt(grad_x + grad_y, grad_x);
    gmi = cv::mean(1 - 1 / ((grad_x + grad_y) / 32 + 1))[0];

    face_info->quality = gmi;
}

double ImageView::get_capture_time() const
{
    return _capture_time;
}

std::shared_ptr<Undistorter> create_undistorter_brown(const json &cam)
{
    double fx = cam["focal_x"];
    double fy = cam["focal_y"];
    double cx = cam["c_x"];
    double cy = cam["c_y"];
    size_t width = cam["width"];
    size_t height = cam["height"];
    std::vector<double> dist_coeffs;
    dist_coeffs.push_back(cam["k1"]);
    dist_coeffs.push_back(cam["k2"]);
    dist_coeffs.push_back(cam["p1"]);
    dist_coeffs.push_back(cam["p2"]);
    dist_coeffs.push_back(cam["k3"]);
    return std::make_shared<Undistorter>(fx, fy, cx, cy, width, height, dist_coeffs);
}

std::shared_ptr<Undistorter> create_undistorter_perspective(const json &cam)
{
    double f = cam["focal"];
    size_t width = cam["width"];
    size_t height = cam["height"];
    std::vector<double> dist_coeffs;
    dist_coeffs.push_back(cam["k1"]);
    dist_coeffs.push_back(cam["k2"]);
    dist_coeffs.push_back(0);
    dist_coeffs.push_back(0);
    return std::make_shared<Undistorter>(f, f, 0.0, 0.0, width, height, dist_coeffs);
}

std::shared_ptr<Undistorter> create_undistorter(const json &cam)
{
    if (cam["projection_type"] == "brown")
    {
        return create_undistorter_brown(cam);
    }
    else if (cam["projection_type"] == "perspective")
    {
        return create_undistorter_perspective(cam);
    }
    throw std::invalid_argument("invalid projection type");
}

std::vector<ImageView> generate_image_views(const std::filesystem::path &json_file)
{
    std::ifstream f(json_file);
    json data = json::parse(f);

    std::vector<ImageView> image_views;
    std::map<std::string, std::shared_ptr<Undistorter>> undistorters;

    for (auto &[key, value] : data[0]["cameras"].items())
    {
        undistorters[key] = create_undistorter(value);
    }

    int i = 0;
    for (auto &[key, value] : data[0]["shots"].items())
    {
        Eigen::Vector3f translation(value["translation"][0], value["translation"][1], value["translation"][2]);
        Eigen::Vector3f rotation(value["rotation"][0], value["rotation"][1], value["rotation"][2]);
        double capture_time = value["capture_time"];
        std::filesystem::path image_path = json_file.parent_path() / std::filesystem::path("images") / key;
        image_views.push_back(ImageView(i, translation, rotation, capture_time, undistorters[value["camera"]], image_path));

        i++;
    }
    return image_views;
}

std::optional<double> get_mean_time(const std::vector<ImageView> &image_views)
{
    double sum = 0.0;
    int num = 0;
    for (const auto &image_view : image_views)
    {
        double capture_time = image_view.get_capture_time();
        if (capture_time > 1)
        {
            sum += capture_time;
            num++;
        }
    }
    if (num == 0)
    {
        return std::nullopt;
    }
    return sum / num;
}