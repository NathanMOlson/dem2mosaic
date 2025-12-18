#include "local_seam_leveling.h"

#include <iostream>
#include <chrono>
#include <map>
#include <opencv2/imgproc.hpp>

struct SeamPointers
{
    float *out;
    float *t;
    float *r;
    float *b;
    float *l;
    float v;
    float c0;
};

cv::Mat local_seam_leveling(const cv::Mat &labels, const cv::Mat mosaic)
{
    auto t1 = std::chrono::steady_clock::now();
    int tile_width = mosaic.rows / labels.rows;
    std::map<int, std::array<float, 6>> coeffs;
    std::array<float, 6> baseline = {4, 1, 1, 1, 1, 0};

    for (int i = 0; i < mosaic.rows; i++)
    {
        // Left
        int k = i * mosaic.cols;
        coeffs.try_emplace(k, baseline);
        coeffs[k][0] -= 1;
        coeffs[k][4] = 0;

        // Right
        k = i * mosaic.cols + mosaic.cols - 1;
        coeffs.try_emplace(k, baseline);
        coeffs[k][0] -= 1;
        coeffs[k][2] = 0;
    }

    for (int j = 0; j < mosaic.cols; j++)
    {
        // Top
        int k = j;
        coeffs.try_emplace(k, baseline);
        coeffs[k][0] -= 1;
        coeffs[k][1] = 0;

        // Bottom
        k = (mosaic.rows - 1) * mosaic.cols + j;
        coeffs.try_emplace(k, baseline);
        coeffs[k][0] -= 1;
        coeffs[k][3] = 0;
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
                    int kr = i * mosaic.cols + j;
                    int kl = kr - 1;

                    coeffs.try_emplace(kr, baseline);
                    coeffs[kr][0] -= 1;
                    coeffs[kr][4] = 0;

                    coeffs.try_emplace(kl, baseline);
                    coeffs[kl][0] -= 1;
                    coeffs[kl][2] = 0;
                }
            }
            // Seam
            else if (labels.at<uint16_t>(label_i, label_j - 1) != labels.at<uint16_t>(label_i, label_j))
            {
                int j = label_j * tile_width;
                for (int i = label_i * tile_width; i < (label_i + 1) * tile_width; i++)
                {
                    int kr = i * mosaic.cols + j;
                    int kl = kr - 1;
                    float seam = ((int)mosaic.at<uint16_t>(i, j) - (int)mosaic.at<uint16_t>(i, j - 1));

                    coeffs.try_emplace(kr, baseline);
                    coeffs[kr][0] += 1;
                    coeffs[kr][4] = 0;
                    coeffs[kr][5] -= seam;

                    coeffs.try_emplace(kl, baseline);
                    coeffs[kl][0] += 1;
                    coeffs[kl][2] = 0;
                    coeffs[kl][5] += seam;
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
                    int kb = i * mosaic.cols + j;
                    int kt = kb - mosaic.cols;

                    coeffs.try_emplace(kb, baseline);
                    coeffs[kb][0] -= 1;
                    coeffs[kb][1] = 0;

                    coeffs.try_emplace(kt, baseline);
                    coeffs[kt][0] -= 1;
                    coeffs[kt][3] = 0;
                }
            }
            // Seam
            else if (labels.at<uint16_t>(label_i - 1, label_j) != labels.at<uint16_t>(label_i, label_j))
            {
                int i = label_i * tile_width;
                for (int j = label_j * tile_width; j < (label_j + 1) * tile_width; j++)
                {
                    int kb = i * mosaic.cols + j;
                    int kt = kb - mosaic.cols;
                    float seam = ((int)mosaic.at<uint16_t>(i, j) - (int)mosaic.at<uint16_t>(i - 1, j));

                    coeffs.try_emplace(kb, baseline);
                    coeffs[kb][0] += 1;
                    coeffs[kb][1] = 0;
                    coeffs[kb][5] -= seam;

                    coeffs.try_emplace(kt, baseline);
                    coeffs[kt][0] += 1;
                    coeffs[kt][3] = 0;
                    coeffs[kt][5] += seam;
                }
            }
        }
    }

    cv::Mat adjustments = cv::Mat::zeros(mosaic.rows, mosaic.cols, CV_32F);
    cv::Mat poisson_kernel = cv::Mat::zeros(3, 3, CV_32F);
    poisson_kernel.at<float>(0, 1) = 0.25;
    poisson_kernel.at<float>(1, 0) = 0.25;
    poisson_kernel.at<float>(1, 2) = 0.25;
    poisson_kernel.at<float>(2, 1) = 0.25;

    cv::Mat tmp = cv::Mat(adjustments.rows, adjustments.cols, CV_32F);

    std::vector<SeamPointers> pointers(coeffs.size());
    int m = 0;
    float zero = 0;

    for (auto [k, c] : coeffs)
    {
        int i = k / adjustments.cols;
        int j = k % adjustments.cols;
        pointers[m].out = &tmp.at<float>(i, j);
        pointers[m].t = (c[1] == 1) ? &adjustments.at<float>(i - 1, j) : &zero;
        pointers[m].r = (c[2] == 1) ? &adjustments.at<float>(i, j + 1) : &zero;
        pointers[m].b = (c[3] == 1) ? &adjustments.at<float>(i + 1, j) : &zero;
        pointers[m].l = (c[4] == 1) ? &adjustments.at<float>(i, j - 1) : &zero;
        pointers[m].v = c[5];
        pointers[m].c0 = c[0];
        m++;
    }
    coeffs.clear();

    for (int n = 0; n < 64; n++)
    {
        cv::filter2D(adjustments, tmp, CV_32F, poisson_kernel);
        for (const auto &p : pointers)
        {
            *p.out = (*p.t + *p.r + *p.b + *p.l + p.v) / p.c0;
        }
        tmp.copyTo(adjustments);
    }
    auto t2 = std::chrono::steady_clock::now();

    std::cout << "Local leveling time: " << std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1).count() << std::endl;

    return adjustments;
}