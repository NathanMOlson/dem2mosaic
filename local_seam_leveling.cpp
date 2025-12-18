#include "local_seam_leveling.h"

#include <iostream>

#include <Eigen/SparseCore>
#include <Eigen/IterativeLinearSolvers>

typedef Eigen::SparseMatrix<float> SpMat;
typedef Eigen::Triplet<float, int> SpCoeff;

enum class BoundaryCondition
{
    ZERO,
    SEAM,
    NEUMANN
};

struct TileBCs
{
    BoundaryCondition top;
    BoundaryCondition right;
    BoundaryCondition bottom;
    BoundaryCondition left;
};

bool all_zero(const TileBCs bcs)
{
    return bcs.top == BoundaryCondition::ZERO && bcs.right == BoundaryCondition::ZERO &&
           bcs.bottom == BoundaryCondition::ZERO && bcs.left == BoundaryCondition::ZERO;
}

#if 0
cv::Mat get_tile_adjustment(const cv::Mat &, int, int, const TileBCs &)
{
    return cv::Mat::ones(32, 32, CV_32F);
}
#else
cv::Mat get_tile_adjustment(const cv::Mat &mosaic, int tile_row, int tile_col, const TileBCs &bcs, size_t tile_width)
{
    std::vector<SpCoeff> coefficients_A;
    std::vector<float> coefficients_b;

    for (size_t i = 1; i < tile_width - 1; i++)
    {
        for (size_t j = 1; j < tile_width - 1; j++)
        {
            int equation_number = i * tile_width + j;
            coefficients_A.push_back(SpCoeff(equation_number, i * tile_width + j, 4));
            coefficients_A.push_back(SpCoeff(equation_number, i * tile_width + j - 1, -1));
            coefficients_A.push_back(SpCoeff(equation_number, i * tile_width + j + 1, -1));
            coefficients_A.push_back(SpCoeff(equation_number, (i - 1) * tile_width + j, -1));
            coefficients_A.push_back(SpCoeff(equation_number, (i + 1) * tile_width + j, -1));
            coefficients_b.push_back(0);
        }
    }

    for (size_t j = 1; j < tile_width - 1; j++)
    {
        // Top
        int equation_number = j;
        float center_coeff = bcs.top == BoundaryCondition::NEUMANN ? 3 : 5;
        coefficients_A.push_back(SpCoeff(equation_number, j, center_coeff));
        coefficients_A.push_back(SpCoeff(equation_number, j - 1, -1));
        coefficients_A.push_back(SpCoeff(equation_number, j + 1, -1));
        coefficients_A.push_back(SpCoeff(equation_number, tile_width + j, -1));
        coefficients_b.push_back(bcs.top == BoundaryCondition::SEAM
                                     ? mosaic.at<uint16_t>(tile_row * tile_width, tile_col * tile_width + j) +
                                           mosaic.at<uint16_t>(tile_row * tile_width - 1, tile_col * tile_width + j)
                                     : 0);
        // Bottom
        equation_number = (tile_width - 1) * tile_width + j;
        center_coeff = bcs.top == BoundaryCondition::NEUMANN ? 3 : 5;
        coefficients_A.push_back(SpCoeff(equation_number, (tile_width - 1) * tile_width + j, center_coeff));
        coefficients_A.push_back(SpCoeff(equation_number, (tile_width - 1) * tile_width + j - 1, -1));
        coefficients_A.push_back(SpCoeff(equation_number, (tile_width - 1) * tile_width + j + 1, -1));
        coefficients_A.push_back(SpCoeff(equation_number, (tile_width - 2) * tile_width + j, -1));
        coefficients_b.push_back(bcs.top == BoundaryCondition::SEAM
                                     ? mosaic.at<uint16_t>((tile_row + 1) * tile_width, tile_col * tile_width + j) +
                                           mosaic.at<uint16_t>((tile_row + 1) * tile_width - 1, tile_col * tile_width + j)
                                     : 0);
    }

    for (size_t i = 1; i < tile_width - 1; i++)
    {
        // Left
        int equation_number = i * tile_width;
        float center_coeff = bcs.left == BoundaryCondition::NEUMANN ? 3 : 5;
        coefficients_A.push_back(SpCoeff(equation_number, i * tile_width, center_coeff));
        coefficients_A.push_back(SpCoeff(equation_number, (i - 1) * tile_width, -1));
        coefficients_A.push_back(SpCoeff(equation_number, (i + 1) * tile_width, -1));
        coefficients_A.push_back(SpCoeff(equation_number, i * tile_width + 1, -1));
        coefficients_b.push_back(bcs.left == BoundaryCondition::SEAM
                                     ? mosaic.at<uint16_t>(tile_row * tile_width + i, tile_col * tile_width) +
                                           mosaic.at<uint16_t>(tile_row * tile_width + i, tile_col * tile_width - 1)
                                     : 0);
        // Right
        equation_number = i * tile_width + tile_width - 1;
        center_coeff = bcs.right == BoundaryCondition::NEUMANN ? 3 : 5;
        coefficients_A.push_back(SpCoeff(equation_number, i * tile_width + tile_width - 1, center_coeff));
        coefficients_A.push_back(SpCoeff(equation_number, (i - 1) * tile_width + tile_width - 1, -1));
        coefficients_A.push_back(SpCoeff(equation_number, (i + 1) * tile_width + tile_width - 1, -1));
        coefficients_A.push_back(SpCoeff(equation_number, i * tile_width + tile_width - 2, -1));
        coefficients_b.push_back(bcs.right == BoundaryCondition::SEAM
                                     ? mosaic.at<uint16_t>(tile_row * tile_width + i, (tile_col + 1) * tile_width) +
                                           mosaic.at<uint16_t>(tile_row * tile_width + i, (tile_col + 1) * tile_width - 1)
                                     : 0);
    }

    // TL
    int equation_number = 0;
    float center_coeff = 4;
    center_coeff += (bcs.left == BoundaryCondition::NEUMANN ? -1 : 1);
    center_coeff += (bcs.top == BoundaryCondition::NEUMANN ? -1 : 1);
    coefficients_A.push_back(SpCoeff(equation_number, 0, center_coeff));
    coefficients_A.push_back(SpCoeff(equation_number, 1, -1));
    coefficients_A.push_back(SpCoeff(equation_number, tile_width, -1));
    coefficients_b.push_back(0);
    if (bcs.left == BoundaryCondition::SEAM)
    {
        coefficients_b.back() += mosaic.at<uint16_t>(tile_row * tile_width, tile_col * tile_width) +
                                 mosaic.at<uint16_t>(tile_row * tile_width, tile_col * tile_width - 1);
    }
    if (bcs.top == BoundaryCondition::SEAM)
    {
        coefficients_b.back() += mosaic.at<uint16_t>(tile_row * tile_width, tile_col * tile_width) +
                                 mosaic.at<uint16_t>(tile_row * tile_width - 1, tile_col * tile_width);
    }

    // TR
    equation_number = tile_width - 1;
    center_coeff = 4;
    center_coeff += (bcs.right == BoundaryCondition::NEUMANN ? -1 : 1);
    center_coeff += (bcs.top == BoundaryCondition::NEUMANN ? -1 : 1);
    coefficients_A.push_back(SpCoeff(equation_number, tile_width - 1, center_coeff));
    coefficients_A.push_back(SpCoeff(equation_number, tile_width - 2, -1));
    coefficients_A.push_back(SpCoeff(equation_number, tile_width + tile_width - 1, -1));
    coefficients_b.push_back(0);
    if (bcs.right == BoundaryCondition::SEAM)
    {
        coefficients_b.back() += mosaic.at<uint16_t>(tile_row * tile_width, (tile_col + 1) * tile_width) +
                                 mosaic.at<uint16_t>(tile_row * tile_width, (tile_col + 1) * tile_width - 1);
    }
    if (bcs.top == BoundaryCondition::SEAM)
    {
        coefficients_b.back() += mosaic.at<uint16_t>(tile_row * tile_width, (tile_col + 1) * tile_width) +
                                 mosaic.at<uint16_t>(tile_row * tile_width - 1, (tile_col + 1) * tile_width);
    }

    // BR
    equation_number = (tile_width - 1) * tile_width + tile_width - 1;
    center_coeff = 4;
    center_coeff += (bcs.bottom == BoundaryCondition::NEUMANN ? -1 : 1);
    center_coeff += (bcs.right == BoundaryCondition::NEUMANN ? -1 : 1);
    coefficients_A.push_back(SpCoeff(equation_number, (tile_width - 1) * tile_width + tile_width - 1, center_coeff));
    coefficients_A.push_back(SpCoeff(equation_number, (tile_width - 1) * tile_width + tile_width - 2, -1));
    coefficients_A.push_back(SpCoeff(equation_number, (tile_width - 2) * tile_width + tile_width - 1, -1));
    coefficients_b.push_back(0);
    if (bcs.bottom == BoundaryCondition::SEAM)
    {
        coefficients_b.back() += mosaic.at<uint16_t>((tile_row + 1) * tile_width, (tile_col + 1) * tile_width) +
                                 mosaic.at<uint16_t>((tile_row + 1) * tile_width, (tile_col + 1) * tile_width - 1);
    }
    if (bcs.right == BoundaryCondition::SEAM)
    {
        coefficients_b.back() += mosaic.at<uint16_t>((tile_row + 1) * tile_width, (tile_col + 1) * tile_width) +
                                 mosaic.at<uint16_t>((tile_row + 1) * tile_width - 1, (tile_col + 1) * tile_width);
    }

    // BL
    equation_number = (tile_width - 1) * tile_width;
    center_coeff = 4;
    center_coeff += (bcs.left == BoundaryCondition::NEUMANN ? -1 : 1);
    center_coeff += (bcs.bottom == BoundaryCondition::NEUMANN ? -1 : 1);
    coefficients_A.push_back(SpCoeff(equation_number, (tile_width - 1) * tile_width, center_coeff));
    coefficients_A.push_back(SpCoeff(equation_number, (tile_width - 1) * tile_width + 1, -1));
    coefficients_A.push_back(SpCoeff(equation_number, (tile_width - 2) * tile_width, -1));
    coefficients_b.push_back(0);
    if (bcs.bottom == BoundaryCondition::SEAM)
    {
        coefficients_b.back() += mosaic.at<uint16_t>((tile_row + 1) * tile_width, tile_col * tile_width) +
                                 mosaic.at<uint16_t>((tile_row + 1) * tile_width, tile_col * tile_width - 1);
    }
    if (bcs.left == BoundaryCondition::SEAM)
    {
        coefficients_b.back() += mosaic.at<uint16_t>((tile_row + 1) * tile_width, tile_col * tile_width) +
                                 mosaic.at<uint16_t>((tile_row + 1) * tile_width - 1, tile_col * tile_width);
    }
}
#endif

cv::Mat local_seam_leveling(const cv::Mat &labels, const cv::Mat mosaic, size_t tile_width)
{
    cv::Mat adjustments = cv::Mat::zeros(mosaic.rows, mosaic.cols, CV_32F);

    for (int i = 0; i < labels.rows; i++)
    {
        for (int j = 0; j < labels.cols; j++)
        {
            cv::Mat input_tile = mosaic(cv::Rect(j * tile_width, i * tile_width, tile_width, tile_width));

            if (labels.at<uint16_t>(i, j) == 0)
            {
                continue;
            }

            TileBCs bcs;
            bcs.top = BoundaryCondition::ZERO;
            if (i == 0 || labels.at<uint16_t>(i - 1, j) == 0)
            {
                bcs.top = BoundaryCondition::NEUMANN;
            }
            else if (labels.at<uint16_t>(i, j) != labels.at<uint16_t>(i - 1, j))
            {
                bcs.top = BoundaryCondition::SEAM;
            }

            bcs.right = BoundaryCondition::ZERO;
            if (j == labels.cols - 1 || labels.at<uint16_t>(i, j + 1) == 0)
            {
                bcs.right = BoundaryCondition::NEUMANN;
            }
            else if (labels.at<uint16_t>(i, j) != labels.at<uint16_t>(i, j + 1))
            {
                bcs.right = BoundaryCondition::SEAM;
            }

            bcs.bottom = BoundaryCondition::ZERO;
            if (i == labels.rows - 1 || labels.at<uint16_t>(i + 1, j) == 0)
            {
                bcs.bottom = BoundaryCondition::NEUMANN;
            }
            else if (labels.at<uint16_t>(i, j) != labels.at<uint16_t>(i + 1, j))
            {
                bcs.bottom = BoundaryCondition::SEAM;
            }

            bcs.left = BoundaryCondition::ZERO;
            if (j == 0 || labels.at<uint16_t>(i, j - 1) == 0)
            {
                bcs.left = BoundaryCondition::NEUMANN;
            }
            else if (labels.at<uint16_t>(i, j) != labels.at<uint16_t>(i, j - 1))
            {
                bcs.left = BoundaryCondition::SEAM;
            }

            if (all_zero(bcs))
            {
                continue;
            }

            cv::Mat output_tile = adjustments(cv::Rect(j * tile_width, i * tile_width, tile_width, tile_width));
            cv::Mat tile_adjustment = get_tile_adjustment(mosaic, i, j, bcs, tile_width);
            tile_adjustment.copyTo(output_tile);
        }
    }
    return adjustments;
}