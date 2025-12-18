#include "local_seam_leveling.h"

cv::Mat local_seam_leveling(const cv::Mat &labels, const cv::Mat mosaic)
{
    int tile_width = mosaic.rows / labels.rows;
    constexpr int num_coeffs = 6;
    std::vector<float> coeffs(mosaic.rows * mosaic.cols * num_coeffs, 1.F);

    for (size_t k = 0; k < coeffs.size(); k += num_coeffs)
    {
        coeffs[k] = 4.F;
        coeffs[k + 5] = 0.F;
    }

    for (int i = 0; i < mosaic.rows; i++)
    {
        // Left
        int k = num_coeffs * (i * mosaic.cols);
        coeffs[k] -= 1;
        coeffs[k + 4] = 0;

        // Right
        k = num_coeffs * (i * mosaic.cols + mosaic.cols - 1);
        coeffs[k] -= 1;
        coeffs[k + 2] = 0;
    }

    for (int j = 0; j < mosaic.cols; j++)
    {
        // Top
        int k = num_coeffs * j;
        coeffs[k] -= 1;
        coeffs[k + 1] = 0;

        // Bottom
        k = num_coeffs * ((mosaic.rows - 1) * mosaic.cols + j);
        coeffs[k] -= 1;
        coeffs[k + 3] = 0;
    }

    for (int label_i = 0; label_i < labels.rows; label_i++)
    {
        for (int label_j = 1; label_j < labels.cols; label_j++)
        {
            // Hole
            if (labels.at<uint16_t>(label_i, label_j - 1) == 0 || labels.at<uint16_t>(label_i, label_j) == 0)
            {
                int j = label_j * tile_width;
                for (int i = label_i * tile_width; i < (label_i + 1) * tile_width; i++)
                {
                    int kr = num_coeffs * (i * mosaic.cols + j);
                    int kl = kr - num_coeffs;

                    coeffs[kr] -= 1;
                    coeffs[kr + 4] = 0;

                    coeffs[kl] -= 1;
                    coeffs[kl + 2] = 0;
                }
            }
            // Seam
            else if (labels.at<uint16_t>(label_i, label_j - 1) != labels.at<uint16_t>(label_i, label_j))
            {
                int j = label_j * tile_width;
                for (int i = label_i * tile_width; i < (label_i + 1) * tile_width; i++)
                {
                    int kr = num_coeffs * (i * mosaic.cols + j);
                    int kl = kr - num_coeffs;
                    float seam =  ((int)mosaic.at<uint16_t>(i, j) - (int)mosaic.at<uint16_t>(i, j - 1));

                    coeffs[kr] += 1;
                    coeffs[kr + 4] = 0;
                    coeffs[kr + 5] -= seam;

                    coeffs[kl] += 1;
                    coeffs[kl + 2] = 0;
                    coeffs[kl + 5] += seam;
                }
            }
        }
    }

    for (int label_i = 1; label_i < labels.rows; label_i++)
    {
        for (int label_j = 0; label_j < labels.cols; label_j++)
        {
            // Hole
            if (labels.at<uint16_t>(label_i - 1, label_j) == 0 || labels.at<uint16_t>(label_i, label_j) == 0)
            {
                int i = label_i * tile_width;
                for (int j = label_j * tile_width; j < (label_j + 1) * tile_width; j++)
                {
                    int kb = num_coeffs * (i * mosaic.cols + j);
                    int kt = kb - num_coeffs * mosaic.cols;

                    coeffs[kb] -= 1;
                    coeffs[kb + 1] = 0;

                    coeffs[kt] -= 1;
                    coeffs[kt + 3] = 0;
                }
            }
            // Seam
            else if (labels.at<uint16_t>(label_i - 1, label_j) != labels.at<uint16_t>(label_i, label_j))
            {
                int i = label_i * tile_width;
                for (int j = label_j * tile_width; j < (label_j + 1) * tile_width; j++)
                {
                    int kb = num_coeffs * (i * mosaic.cols + j);
                    int kt = kb - num_coeffs * mosaic.cols;
                    float seam =  ((int)mosaic.at<uint16_t>(i, j) - (int)mosaic.at<uint16_t>(i - 1, j));

                    coeffs[kb] += 1;
                    coeffs[kb + 1] = 0;
                    coeffs[kb + 5] -= seam;

                    coeffs[kt] += 1;
                    coeffs[kt + 3] = 0;
                    coeffs[kt + 5] += seam;
                }
            }
        }
    }

    cv::Mat adjustments = cv::Mat::zeros(mosaic.rows, mosaic.cols, CV_32F);

    for (int n = 0; n < 64; n++)
    {
        cv::Mat tmp = cv::Mat::zeros(adjustments.rows, adjustments.cols, CV_32F);
        for (int i = 1; i < adjustments.rows - 1; i++)
        {
            for (int j = 1; j < adjustments.cols - 1; j++)
            {
                int k = num_coeffs * (i * mosaic.cols + j);
                tmp.at<float>(i, j) = (coeffs[k + 1] * adjustments.at<float>(i - 1, j) +
                                       coeffs[k + 2] * adjustments.at<float>(i, j + 1) +
                                       coeffs[k + 3] * adjustments.at<float>(i + 1, j) +
                                       coeffs[k + 4] * adjustments.at<float>(i, j - 1) +
                                       coeffs[k + 5]) /
                                      coeffs[k];
            }
        }
        adjustments = tmp;
    }

    return adjustments;
}