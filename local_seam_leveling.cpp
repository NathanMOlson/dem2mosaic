#include "local_seam_leveling.h"

#include <iostream>
#include <chrono>
#include <map>
#include <opencv2/imgproc.hpp>

struct SeamPointers
{
    float *out;
    const float *t;
    const float *r;
    const float *b;
    const float *l;
    float v;
    float c0;
};

cv::Mat local_seam_leveling(const cv::Mat &labels, const cv::Mat mosaic)
{
    auto t1 = std::chrono::steady_clock::now();
    int tile_width = mosaic.rows / labels.rows;
    std::map<int, std::array<float, 6>> coeffs;
    std::array<float, 6> baseline = {4, 1, 1, 1, 1, 0};
    const int num_channels = mosaic.channels();

    for (int i = 0; i < mosaic.rows; i++)
    {
        for (int c = 0; c < num_channels; c++)
        {
            // Left
            int k = i * mosaic.cols * num_channels + c;
            coeffs.try_emplace(k, baseline);
            coeffs[k][0] -= 1;
            coeffs[k][4] = 0;

            // Right
            k = (i * mosaic.cols + mosaic.cols - 1) * num_channels + c;
            coeffs.try_emplace(k, baseline);
            coeffs[k][0] -= 1;
            coeffs[k][2] = 0;
        }
    }

    for (int j = 0; j < mosaic.cols; j++)
    {
        for (int c = 0; c < num_channels; c++)
        {
            // Top
            int k = j * num_channels + c;
            coeffs.try_emplace(k, baseline);
            coeffs[k][0] -= 1;
            coeffs[k][1] = 0;

            // Bottom
            k = ((mosaic.rows - 1) * mosaic.cols + j) * num_channels + c;
            coeffs.try_emplace(k, baseline);
            coeffs[k][0] -= 1;
            coeffs[k][3] = 0;
        }
    }

    for (int label_i = 0; label_i < labels.rows; label_i++)
    {
        for (int label_j = 1; label_j < labels.cols; label_j++)
        {
            for (int c = 0; c < num_channels; c++)
            {
                // Hole
                if (labels.at<uint16_t>(label_i, label_j - 1) == 0 || labels.at<uint16_t>(label_i, label_j) == 0)
                {
                    int j = label_j * tile_width;
                    for (int i = label_i * tile_width; i < (label_i + 1) * tile_width; i++)
                    {
                        int kr = (i * mosaic.cols + j) * num_channels + c;
                        int kl = kr - num_channels;

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
                        int kr = (i * mosaic.cols + j) * num_channels + c;
                        int kl = kr - num_channels;
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
    }

    for (int label_i = 1; label_i < labels.rows; label_i++)
    {
        for (int label_j = 0; label_j < labels.cols; label_j++)
        {
            for (int c = 0; c < num_channels; c++)
            {
                // Hole
                if (labels.at<uint16_t>(label_i - 1, label_j) == 0 || labels.at<uint16_t>(label_i, label_j) == 0)
                {
                    int i = label_i * tile_width;
                    for (int j = label_j * tile_width; j < (label_j + 1) * tile_width; j++)
                    {
                        int kb = (i * mosaic.cols + j) * num_channels + c;
                        int kt = kb - mosaic.cols * num_channels;

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
                        int kb = (i * mosaic.cols + j) * num_channels + c;
                        int kt = kb - mosaic.cols * num_channels;
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
    }

    cv::Mat adjustments = cv::Mat::zeros(mosaic.rows, mosaic.cols, CV_32FC(num_channels));
    cv::Mat poisson_kernel = cv::Mat::zeros(3, 3, CV_32F);
    poisson_kernel.at<float>(0, 1) = 0.25;
    poisson_kernel.at<float>(1, 0) = 0.25;
    poisson_kernel.at<float>(1, 2) = 0.25;
    poisson_kernel.at<float>(2, 1) = 0.25;

    std::vector<SeamPointers> pointers(coeffs.size());
    int m = 0;
    constexpr float zero = 0;

    for (auto [k, c] : coeffs)
    {
        int i = k / (adjustments.cols * num_channels);
        int j = k % (adjustments.cols * num_channels);
        pointers[m].out = &adjustments.at<float>(i, j);
        pointers[m].t = (c[1] == 1) ? &adjustments.at<float>(i - 1, j) : &zero;
        pointers[m].r = (c[2] == 1) ? &adjustments.at<float>(i, j + 1) : &zero;
        pointers[m].b = (c[3] == 1) ? &adjustments.at<float>(i + 1, j) : &zero;
        pointers[m].l = (c[4] == 1) ? &adjustments.at<float>(i, j - 1) : &zero;
        pointers[m].v = c[5];
        pointers[m].c0 = c[0];
        m++;
    }
    coeffs.clear();

    std::vector<float> out_vals(pointers.size());

    for (int n = 0; n < 64; n++)
    {
        for (size_t i = 0; i < pointers.size(); i++)
        {
            SeamPointers &p = pointers[i];
            out_vals[i] = (*p.t + *p.r + *p.b + *p.l + p.v) / p.c0;
        }
        cv::filter2D(adjustments, adjustments, CV_32F, poisson_kernel);
        for (size_t i = 0; i < pointers.size(); i++)
        {
            *pointers[i].out = out_vals[i];
        }
    }
    auto t2 = std::chrono::steady_clock::now();

    std::cout << "Local leveling time: " << std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1).count() << std::endl;

    return adjustments;
}