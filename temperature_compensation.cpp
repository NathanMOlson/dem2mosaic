#include "temperature_compensation.h"
#include <Eigen/Dense>

float get_temperature_adjustment(const TemperatureCompensation *comp, float cos_viewing_angle)
{
    float k = (1.F - cos_viewing_angle);
    return -comp->offset - comp->b * k * k * k * k;
}

cv::Mat angular_temperature_compensation(const cv::Mat &labels, const QuadMesh &mesh,
                                         const std::vector<ImageView> &image_views, const std::vector<TemperatureCompensation *> comp)
{
    cv::Mat adjustment = cv::Mat::zeros(labels.rows * 2, labels.cols * 2, CV_32F);

    for (size_t k = 0; k < image_views.size(); k++)
    {
        Eigen::Vector3f const &view_pos = image_views[k].get_pos();
        for (int i = 0; i < labels.rows; i++)
        {
            for (int j = 0; j < labels.cols; j++)
            {
                if (labels.at<uint16_t>(i, j) == k + 1)
                {
                    std::vector<Eigen::Vector3f> corner_points;
                    corner_points.push_back(mesh.GetVertex(i, j));
                    corner_points.push_back(mesh.GetVertex(i, j + 1));
                    corner_points.push_back(mesh.GetVertex(i + 1, j + 1));
                    corner_points.push_back(mesh.GetVertex(i + 1, j));

                    std::vector<float> cos_viewing_angle(4);
                    for (size_t i = 0; i < 4; i++)
                    {
                        Eigen::Vector3f corner_normal = (corner_points[(i + 1) % 4] - corner_points[i]).cross((corner_points[i] - corner_points[(i + 3) % 4])).normalized();
                        Eigen::Vector3f corner_to_view_vec = (view_pos - corner_points[i]).normalized();
                        cos_viewing_angle[i] = corner_to_view_vec.dot(corner_normal.normalized());
                    }

                    adjustment.at<float>(2 * i, 2 * j) = get_temperature_adjustment(comp[k], cos_viewing_angle[0]);
                    adjustment.at<float>(2 * i, 2 * j + 1) = get_temperature_adjustment(comp[k], cos_viewing_angle[1]);
                    adjustment.at<float>(2 * i + 1, 2 * j + 1) = get_temperature_adjustment(comp[k], cos_viewing_angle[2]);
                    adjustment.at<float>(2 * i + 1, 2 * j) = get_temperature_adjustment(comp[k], cos_viewing_angle[3]);
                }
            }
        }
    }
    return adjustment;
}