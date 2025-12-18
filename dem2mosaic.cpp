#include <iostream>
#include <fstream>
#include <vector>
#include <tbb/task_arena.h>
#include <omp.h>
#include <set>

#include <opencv2/imgcodecs.hpp>
#include <Eigen/SparseCore>
#include <Eigen/IterativeLinearSolvers>

#include "mapmap/full.h"
#include "quadmesh.h"
#include "image_view.h"
#include "sparse_table.h"
#include "temperature_compensation.h"
#include "save_geotiff.h"
#include "proj_info.h"

typedef Eigen::SparseMatrix<float> SpMat;
typedef Eigen::Triplet<float, int> SpCoeff;
typedef SparseTable<std::uint32_t, std::uint16_t, float> DataCosts;

cv::Mat view_selection(DataCosts const &data_costs, const QuadMesh &mesh,
                       const cv::Mat &tile_costs,
                       const std::vector<float> &cost_table,
                       size_t num_views)
{
    using uint_t = unsigned int;
    using cost_t = float;
    constexpr uint_t simd_w = 1;
    using unary_t = mapmap::UnaryTable<cost_t, simd_w>;
    using pairwise_t = mapmap::PairwiseTable<cost_t, simd_w>;

    /* Construct graph */
    mapmap::Graph<cost_t> mgraph(mesh.NumFaces());

    size_t rows = mesh.NumFaceRows();
    size_t cols = mesh.NumFaceCols();

    for (std::size_t i = 0; i < rows; ++i)
    {
        for (std::size_t j = 0; j < cols - 1; ++j)
        {
            float edge_cost = (tile_costs.at<float>(i, j) + tile_costs.at<float>(i, j + 1)) / 2.F;
            mgraph.add_edge(i * cols + j, i * cols + j + 1, 1.F + edge_cost * edge_cost);
        }
    }
    for (std::size_t i = 0; i < rows - 1; ++i)
    {
        for (std::size_t j = 0; j < cols; ++j)
        {
            float edge_cost = (tile_costs.at<float>(i, j) + tile_costs.at<float>(i + 1, j)) / 2.F;
            mgraph.add_edge(i * cols + j, (i + 1) * cols + j, 1.F + edge_cost * edge_cost);
        }
    }
    mgraph.update_components();

    mapmap::LabelSet<cost_t, simd_w> label_set(mesh.NumFaces(), false);
    for (std::size_t i = 0; i < data_costs.cols(); ++i)
    {
        DataCosts::Column const &data_costs_for_node = data_costs.col(i);

        std::vector<mapmap::_iv_st<cost_t, simd_w>> labels;
        if (data_costs_for_node.empty())
        {
            labels.push_back(0);
        }
        else
        {
            labels.resize(data_costs_for_node.size());
            for (std::size_t j = 0; j < data_costs_for_node.size(); ++j)
            {
                labels[j] = data_costs_for_node[j].first + 1;
            }
        }

        label_set.set_label_set_for_node(i, labels);
    }

    std::vector<unary_t> unaries;
    unaries.reserve(data_costs.cols());

    mapmap::LabelSet<cost_t, simd_w> table_label_set(1, false);
    {
        std::vector<mapmap::_iv_st<cost_t, simd_w>> labels;
        for (size_t i = 0; i < num_views; i++)
        {
            labels.push_back(i + 1);
        }
        table_label_set.set_label_set_for_node(0, labels);
    }
    pairwise_t pairwise(0, 0, &table_label_set, cost_table);

    for (std::size_t i = 0; i < data_costs.cols(); ++i)
    {
        DataCosts::Column const &data_costs_for_node = data_costs.col(i);

        std::vector<mapmap::_s_t<cost_t, simd_w>> costs;
        if (data_costs_for_node.empty())
        {
            costs.push_back(1.0f);
        }
        else
        {
            costs.resize(data_costs_for_node.size());
            for (std::size_t j = 0; j < data_costs_for_node.size(); ++j)
            {
                float cost = data_costs_for_node[j].second;
                costs[j] = cost;
            }
        }

        unaries.emplace_back(i, &label_set);
        unaries.back().set_costs(costs);
    }

    mapmap::StopWhenReturnsDiminish<cost_t, simd_w> terminate(5, 0.01);
    std::vector<mapmap::_iv_st<cost_t, simd_w>> solution;

    auto display = [](const mapmap::luint_t time_ms,
                      const mapmap::_iv_st<cost_t, simd_w> objective)
    {
        std::cout << "\t\t" << time_ms / 1000 << "\t" << objective << std::endl;
    };

    /* Create mapMAP solver object. */
    mapmap::mapMAP<cost_t, simd_w> solver;
    solver.set_graph(&mgraph);
    solver.set_label_set(&label_set);
    for (std::size_t i = 0; i < mesh.NumFaces(); ++i)
        solver.set_unary(i, &unaries[i]);
    solver.set_pairwise(&pairwise);
    solver.set_logging_callback(display);
    solver.set_termination_criterion(&terminate);

    /* Pass configuration arguments (optional) for solve. */
    mapmap::mapMAP_control ctr;
    ctr.use_multilevel = true;
    ctr.use_spanning_tree = true;
    ctr.use_acyclic = true;
    ctr.spanning_tree_multilevel_after_n_iterations = 5;
    ctr.force_acyclic = true;
    ctr.min_acyclic_iterations = 5;
    ctr.relax_acyclic_maximal = true;
    ctr.tree_algorithm = mapmap::LOCK_FREE_TREE_SAMPLER;

    /* Set false for non-deterministic (but faster) mapMAP execution. */
    ctr.sample_deterministic = true;
    ctr.initial_seed = 548923723;

    std::cout << "\tOptimizing:\n\t\tTime[s]\tEnergy" << std::endl;
    solver.optimize(solution, ctr);

    /* Label 0 is undefined. */
    std::size_t num_labels = sqrt(cost_table.size());
    std::size_t undefined = 0;
    /* Extract resulting labeling from solver. */

    cv::Mat labels(rows, cols, CV_16U);
    for (std::size_t i = 0; i < rows; ++i)
    {
        for (std::size_t j = 0; j < cols; ++j)
        {
            size_t k = i * cols + j;
            int label = label_set.label_from_offset(k, solution[k]);
            if (label < 0 || label > (int)num_labels)
            {
                throw std::runtime_error("Incorrect labeling");
            }
            if (label == 0)
                undefined += 1;
            labels.at<uint16_t>(i, j) = label;
        }
    }
    std::cout << '\t' << undefined << " faces have not been seen" << std::endl;

    return labels;
}

std::vector<std::vector<QuadInfo>> calculate_face_projection_infos(const QuadMesh &mesh,
                                                                   const std::vector<ImageView> &image_views,
                                                                   std::vector<TemperatureCompensation *> &temp_comp_per_view)
{
    std::vector<std::vector<QuadInfo>> face_projection_infos(mesh.NumFaces());
    // std::vector<unsigned int> const & faces = mesh.get_faces();
    // std::vector<Eigen::Vector3f> const & vertices = mesh.get_vertices();
    // mve::TriangleMesh::NormalList const & face_normals = mesh.get_face_normals();

    std::size_t const num_views = image_views.size();
    temp_comp_per_view.resize(num_views);

    // std::cout << "\tBuilding BVH from " << mesh.NumFaces() << " faces... " << std::flush;
    // BVHTree bvh_tree(faces, vertices);
    // std::cout << "done. (Took: " << timer.get_elapsed() << " ms)" << std::endl;

    int next_camera_id = 0;
    std::map<std::string, int> camera_ids;

    for (size_t k = 0; k < num_views; ++k)
    {
        ImageView image_view = image_views.at(k);
        std::string sn = image_view.get_serial_number();
        if (camera_ids.count(sn) == 0)
        {
            camera_ids[sn] = next_camera_id;
            next_camera_id++;
        }
    }
    const size_t num_cameras = camera_ids.size();
    std::vector<TemperatureCompensation *> temp_comps(num_cameras);
    for (auto &temp_comp : temp_comps)
    {
        temp_comp = new TemperatureCompensation;
    }

    Eigen::VectorXd Ab_overall = Eigen::VectorXd::Zero(num_cameras + 1);
    Eigen::MatrixXd AA_overall = Eigen::MatrixXd::Zero(num_cameras + 1, num_cameras + 1);

#pragma omp parallel
    {
        std::vector<std::pair<std::size_t, QuadInfo>> projected_face_view_infos;

        Eigen::VectorXd Ab = Eigen::VectorXd::Zero(num_cameras + 1);
        Eigen::MatrixXd AA = Eigen::MatrixXd::Zero(num_cameras + 1, num_cameras + 1);

#pragma omp for schedule(dynamic)
#if !defined(_MSC_VER)
        for (std::uint16_t k = 0; k < static_cast<std::uint16_t>(num_views); ++k)
        {
#else
        for (std::int32_t k = 0; k < num_views; ++k)
        {
#endif
            ImageView image_view = image_views.at(k);
            std::string sn = image_view.get_serial_number();
            int camera_id = camera_ids[sn];
            temp_comp_per_view[k] = temp_comps[camera_id];

            image_view.load_image();
            if (!image_view.IsImageLoaded())
            {
                std::cout << "Failed to load image: " << image_view.ImagePath() << std::endl;
                continue;
            }

            Eigen::Vector3f const &view_pos = image_view.get_pos();
            Eigen::Vector3f const &viewing_direction = image_view.get_viewing_direction();

            for (std::size_t i = 0; i < mesh.NumFaceRows(); i++)
            {
                for (std::size_t j = 0; j < mesh.NumFaceCols(); j++)
                {
                    std::size_t face_id = i * mesh.NumFaceCols() + j;
                    std::vector<Eigen::Vector3f> corner_points;
                    Eigen::Vector3f const &v1 = mesh.GetVertex(i, j);
                    Eigen::Vector3f const &v2 = mesh.GetVertex(i, j + 1);
                    Eigen::Vector3f const &v3 = mesh.GetVertex(i + 1, j + 1);
                    Eigen::Vector3f const &v4 = mesh.GetVertex(i + 1, j);
                    Eigen::Vector3f const &face_normal = (v3 - v1).cross(v2 - v4);
                    Eigen::Vector3f const face_center = (v1 + v2 + v3 + v4) / 4.0f;

                    corner_points.push_back(v1);
                    corner_points.push_back(v2);
                    corner_points.push_back(v3);
                    corner_points.push_back(v4);

                    std::vector<cv::Point2f> corner_pixels = image_view.get_pixel_coords(corner_points);

                    /* Check visibility and compute quality */
                    Eigen::Vector3f view_to_face_vec = (face_center - view_pos).normalized();
                    Eigen::Vector3f face_to_view_vec = -view_to_face_vec;
                    Eigen::Vector3f up(0, 0, 1);

                    /* Backface and basic frustum culling */
                    float cos_viewing_angle = face_to_view_vec.dot(face_normal.normalized());
                    if (cos_viewing_angle < 0.0f || viewing_direction.dot(view_to_face_vec) < 0.0f)
                        continue;

                    /* Projects into the valid part of the ImageView? */
                    if (!image_view.intersects(corner_pixels))
                        continue;

                    // BVHTree::Ray ray;
                    // ray.dir = view_pos - face_center;
                    // ray.tmax = ray.dir.norm();
                    // ray.tmin = ray.tmax * 0.0001f;
                    // ray.dir.normalize();

                    // BVHTree::Hit hit;
                    // if (bvh_tree.intersect(ray, &hit)) {
                    //     continue;
                    // }

                    QuadInfo info;
                    info.view_id = k;

                    image_view.get_face_info(corner_pixels, &info);

                    if (info.quality <= 0.0)
                        continue;

#if USE_LOCAL_NORMAL
                    float cos_incidence = cos_viewing_angle;
#else
                    float cos_incidence = face_to_view_vec.dot(Eigen::Vector3f(0, 0, 1));
#endif

                    float temp_weight = info.tl_w + info.tr_w + info.br_w + info.bl_w;
                    if (temp_weight > 0)
                    {
                        float temp = (info.tl + info.tr + info.br + info.bl) / temp_weight;

                        double k = 1.0 - cos_incidence;
                        k = k * k;
                        k = k * k; // (1-cos)^4
                        double w = cos_incidence;

                        Eigen::VectorXd A = Eigen::VectorXd::Zero(num_cameras + 1);
                        A[camera_id] = w;
                        A[num_cameras] = w * k;
                        double b = w * temp;

                        AA += A * A.transpose();
                        Ab += A * b;
                    }

                    std::pair<std::size_t, QuadInfo> pair(face_id, info);
                    projected_face_view_infos.push_back(pair);
                }
            }

            image_view.release_image();
        }

        // std::sort(projected_face_view_infos.begin(), projected_face_view_infos.end());

#pragma omp critical
        {
            for (std::size_t i = projected_face_view_infos.size(); 0 < i; --i)
            {
                std::size_t face_id = projected_face_view_infos[i - 1].first;
                QuadInfo const &info = projected_face_view_infos[i - 1].second;
                face_projection_infos.at(face_id).push_back(info);
            }
            projected_face_view_infos.clear();
            AA_overall += AA;
            Ab_overall += Ab;
        }
    }
    Eigen::VectorXd x = AA_overall.ldlt().solve(Ab_overall);
    std::cout << "Temperature Compensation Solution:" << std::endl;
    double mean_temp = x.head(num_cameras).mean();
    for (const auto &[sn, id] : camera_ids)
    {
        std::cout << sn << ": " << mean_temp - x[id] << ", " << x[num_cameras] << std::endl;
        temp_comps[id]->offset = x[id] - mean_temp;
        temp_comps[id]->b = x[num_cameras];
    }

    return face_projection_infos;
}

DataCosts calculate_data_costs(const QuadMesh &mesh, std::vector<std::vector<QuadInfo>> &face_projection_infos)
{
    DataCosts data_costs(mesh.NumFaces());
#pragma omp parallel for schedule(dynamic)
#if !defined(_MSC_VER)
    for (std::size_t i = 0; i < face_projection_infos.size(); ++i)
    {
#else
    for (std::int64_t i = 0; i < face_projection_infos.size(); ++i)
    {
#endif

        std::vector<QuadInfo> &infos = face_projection_infos.at(i);
        // if (settings.outlier_removal != OUTLIER_REMOVAL_NONE) {
        //     photometric_outlier_detection(&infos, settings);

        //     infos.erase(std::remove_if(infos.begin(), infos.end(),
        //         [](FaceProjectionInfo const & info) -> bool {return info.quality == 0.0f;}),
        //         infos.end());
        // }
        std::sort(infos.begin(), infos.end());
    }

    for (std::uint32_t i = 0; i < face_projection_infos.size(); ++i)
    {
        for (QuadInfo const &info : face_projection_infos.at(i))
        {
            if (!info.fully_visible)
            {
                continue;
            }
            data_costs.set_value(i, info.view_id, 1.0f - info.quality);
        }
    }
    return data_costs;
}

cv::Mat best_local_labels(const std::vector<std::vector<QuadInfo>> &quad_infos, const QuadMesh &mesh, cv::Mat &tile_cost)
{
    cv::Mat labels = cv::Mat::zeros(mesh.NumFaceRows(), mesh.NumFaceCols(), CV_16U);
    tile_cost = cv::Mat::ones(mesh.NumFaceRows(), mesh.NumFaceCols(), CV_32F);

    for (int i = 0; i < labels.rows; i++)
    {
        for (int j = 0; j < labels.cols; j++)
        {
            int index = i * labels.cols + j;
            float q = 0;
            uint16_t num_valid = 0;

            for (const QuadInfo &quad_info : quad_infos[index])
            {
                if (quad_info.num_valid_pixels > num_valid || (quad_info.num_valid_pixels == num_valid && quad_info.quality > q))
                {
                    q = quad_info.quality;
                    num_valid = quad_info.num_valid_pixels;
                    labels.at<uint16_t>(i, j) = quad_info.view_id + 1;
                    tile_cost.at<float>(i, j) = quad_info.quality;
                }
            }
        }
    }

    return labels;
}

float calculate_difference(const std::vector<std::vector<QuadInfo>> &quad_infos, cv::Size mesh_size, int row, int col, uint16_t label, uint16_t label2)
{
    float n1 = 0;
    float n2 = 0;
    float m1 = 0;
    float m2 = 0;

    if (row > 0 && col > 0)
    {
        int index = (row - 1) * mesh_size.width + col - 1;
        for (const QuadInfo &quad_info : quad_infos[index])
        {
            if (quad_info.view_id == label - 1)
            {
                m1 += quad_info.br;
                n1 += quad_info.br_w;
            }
            else if (quad_info.view_id == label2 - 1)
            {
                m2 += quad_info.br;
                n2 += quad_info.br_w;
            }
        }
    }

    if (row > 0 && col < mesh_size.width)
    {
        int index = (row - 1) * mesh_size.width + col;
        for (const QuadInfo &quad_info : quad_infos[index])
        {
            if (quad_info.view_id == label - 1)
            {
                m1 += quad_info.bl;
                n1 += quad_info.bl_w;
            }
            else if (quad_info.view_id == label2 - 1)
            {
                m2 += quad_info.bl;
                n2 += quad_info.bl_w;
            }
        }
    }

    if (row < mesh_size.height && col > 0)
    {
        int index = (row)*mesh_size.width + col - 1;
        for (const QuadInfo &quad_info : quad_infos[index])
        {
            if (quad_info.view_id == label - 1)
            {
                m1 += quad_info.tr;
                n1 += quad_info.tr_w;
            }
            else if (quad_info.view_id == label2 - 1)
            {
                m2 += quad_info.tr;
                n2 += quad_info.tr_w;
            }
        }
    }

    if (row < mesh_size.height && col < mesh_size.width)
    {
        int index = row * mesh_size.width + col;
        for (const QuadInfo &quad_info : quad_infos[index])
        {
            if (quad_info.view_id == label - 1)
            {
                m1 += quad_info.tl;
                n1 += quad_info.tl_w;
            }
            else if (quad_info.view_id == label2 - 1)
            {
                m2 += quad_info.tl;
                n2 += quad_info.tl_w;
            }
        }
    }
    assert(n1 > 0 && n2 > 0);

    return m2 / n2 - m1 / n1;
}

cv::Mat global_seam_leveling(const cv::Mat &labels, const std::vector<std::vector<QuadInfo>> &quad_infos, const cv::Mat &existing_adjustments)
{
    cv::Mat index_image = -cv::Mat::ones(labels.rows * 2, labels.cols * 2, CV_32S);
    int next_index = 0;
    for (int i = 0; i < labels.rows; i++)
    {
        for (int j = 0; j < labels.cols; j++)
        {
            if (labels.at<uint16_t>(i, j) == 0)
            {
                continue;
            }

            // Top Left
            if (i > 0 && j > 0 && labels.at<uint16_t>(i, j) == labels.at<uint16_t>(i - 1, j - 1))
            {
                index_image.at<int32_t>(2 * i, 2 * j) = index_image.at<int32_t>(2 * i - 1, 2 * j - 1);
            }
            else if (i > 0 && labels.at<uint16_t>(i, j) == labels.at<uint16_t>(i - 1, j))
            {
                index_image.at<int32_t>(2 * i, 2 * j) = index_image.at<int32_t>(2 * i - 1, 2 * j);
            }
            else if (j > 0 && labels.at<uint16_t>(i, j) == labels.at<uint16_t>(i, j - 1))
            {
                index_image.at<int32_t>(2 * i, 2 * j) = index_image.at<int32_t>(2 * i, 2 * j - 1);
            }
            else
            {
                index_image.at<int32_t>(2 * i, 2 * j) = next_index;
                next_index++;
            }

            // Top Right
            if (i > 0 && labels.at<uint16_t>(i, j) == labels.at<uint16_t>(i - 1, j))
            {
                index_image.at<int32_t>(2 * i, 2 * j + 1) = index_image.at<int32_t>(2 * i - 1, 2 * j + 1);
            }
            else if (i > 0 && j < labels.cols - 1 && labels.at<uint16_t>(i, j) == labels.at<uint16_t>(i - 1, j + 1))
            {
                index_image.at<int32_t>(2 * i, 2 * j + 1) = index_image.at<int32_t>(2 * i - 1, 2 * j + 2);
            }
            else
            {
                index_image.at<int32_t>(2 * i, 2 * j + 1) = next_index;
                next_index++;
            }

            // Bottom Left
            if (j > 0 && labels.at<uint16_t>(i, j) == labels.at<uint16_t>(i, j - 1))
            {
                index_image.at<int32_t>(2 * i + 1, 2 * j) = index_image.at<int32_t>(2 * i + 1, 2 * j - 1);
            }
            else
            {
                index_image.at<int32_t>(2 * i + 1, 2 * j) = next_index;
                next_index++;
            }

            // Bottom Right
            index_image.at<int32_t>(2 * i + 1, 2 * j + 1) = next_index;
            next_index++;
        }
    }
    std::size_t x_rows = next_index;
    std::cout << "faces: " << labels.rows * labels.cols << std::endl;
    std::cout << "possible vertices: " << labels.rows * labels.cols * 4 << std::endl;
    std::cout << "num_indices: " << next_index << std::endl;

    cv::Mat img;
    index_image.convertTo(img, CV_16U);
    cv::imwrite("data/tex/index.png", img);

    std::vector<SpCoeff> coefficients_Gamma;
    coefficients_Gamma.reserve(2 * next_index);
    size_t row = 0;
    constexpr float lambda = 0.1f;
    for (int i = 0; i < labels.rows; i++)
    {
        for (int j = 0; j < labels.cols; j++)
        {
            if (labels.at<uint16_t>(i, j) == 0)
            {
                continue;
            }

            // Top side
            if (i == 0 || labels.at<uint16_t>(i, j) != labels.at<uint16_t>(i - 1, j))
            {
                coefficients_Gamma.push_back(SpCoeff(row, index_image.at<int32_t>(2 * i, 2 * j), lambda));
                coefficients_Gamma.push_back(SpCoeff(row, index_image.at<int32_t>(2 * i, 2 * j + 1), -lambda));
                row++;
            }

            // Left side
            if (j == 0 || labels.at<uint16_t>(i, j) != labels.at<uint16_t>(i, j - 1))
            {
                coefficients_Gamma.push_back(SpCoeff(row, index_image.at<int32_t>(2 * i, 2 * j), lambda));
                coefficients_Gamma.push_back(SpCoeff(row, index_image.at<int32_t>(2 * i + 1, 2 * j), -lambda));
                row++;
            }

            // Bottom side
            coefficients_Gamma.push_back(SpCoeff(row, index_image.at<int32_t>(2 * i + 1, 2 * j), lambda));
            coefficients_Gamma.push_back(SpCoeff(row, index_image.at<int32_t>(2 * i + 1, 2 * j + 1), -lambda));
            row++;

            // Right side
            coefficients_Gamma.push_back(SpCoeff(row, index_image.at<int32_t>(2 * i, 2 * j + 1), lambda));
            coefficients_Gamma.push_back(SpCoeff(row, index_image.at<int32_t>(2 * i + 1, 2 * j + 1), -lambda));
            row++;
        }
    }
    std::cout << "built Gamma" << std::endl;
    std::size_t Gamma_rows = row;
    assert(Gamma_rows < static_cast<std::size_t>(std::numeric_limits<int>::max()));

    SpMat Gamma(Gamma_rows, x_rows);
    std::cout << "Created Gamma: " << Gamma_rows << "x" << x_rows << " from coeffs: " << coefficients_Gamma.size() << std::endl;
    Gamma.setFromTriplets(coefficients_Gamma.begin(), coefficients_Gamma.end());

    std::cout << "Set Gamma from triplets" << std::endl;

    std::vector<SpCoeff> coefficients_A;
    std::vector<float> coefficients_b;
    row = 0;

    for (int i = 0; i <= labels.rows; i++)
    {
        for (int j = 0; j <= labels.cols; j++)
        {
            std::vector<uint16_t> tile_labels;
            std::vector<int32_t> vertex_indices;
            std::vector<cv::Point2i> adjustment_points;

            if (i > 0 && j > 0)
            {
                uint16_t label = labels.at<uint16_t>(i - 1, j - 1);
                if (label > 0)
                {
                    tile_labels.push_back(label);
                    vertex_indices.push_back(index_image.at<int32_t>(2 * i - 1, 2 * j - 1));
                    adjustment_points.push_back(cv::Point2i(2 * j - 1, 2 * i - 1));
                }
            }
            if (i > 0 && j < labels.cols)
            {
                uint16_t label = labels.at<uint16_t>(i - 1, j);
                if (label > 0 && std::find(tile_labels.begin(), tile_labels.end(), label) == tile_labels.end())
                {
                    tile_labels.push_back(label);
                    vertex_indices.push_back(index_image.at<int32_t>(2 * i - 1, 2 * j));
                    adjustment_points.push_back(cv::Point2i(2 * j, 2 * i - 1));
                }
            }
            if (i < labels.rows && j > 0)
            {
                uint16_t label = labels.at<uint16_t>(i, j - 1);
                if (label > 0 && std::find(tile_labels.begin(), tile_labels.end(), label) == tile_labels.end())
                {
                    tile_labels.push_back(label);
                    vertex_indices.push_back(index_image.at<int32_t>(2 * i, 2 * j - 1));
                    adjustment_points.push_back(cv::Point2i(2 * j - 1, 2 * i));
                }
            }
            if (i < labels.rows && j < labels.cols)
            {
                uint16_t label = labels.at<uint16_t>(i, j);
                if (label > 0 && std::find(tile_labels.begin(), tile_labels.end(), label) == tile_labels.end())
                {
                    tile_labels.push_back(label);
                    vertex_indices.push_back(index_image.at<int32_t>(2 * i, 2 * j));
                    adjustment_points.push_back(cv::Point2i(2 * j, 2 * i));
                }
            }

            for (size_t i1 = 0; i1 < tile_labels.size(); i1++)
            {
                for (size_t i2 = i1 + 1; i2 < tile_labels.size(); i2++)
                {
                    coefficients_A.push_back(SpCoeff(row, vertex_indices[i1], 1));
                    coefficients_A.push_back(SpCoeff(row, vertex_indices[i2], -1));
                    coefficients_b.push_back(calculate_difference(quad_infos, labels.size(), i, j, tile_labels[i1], tile_labels[i2]));
                    coefficients_b.back() += existing_adjustments.at<float>(adjustment_points[i2]) - existing_adjustments.at<float>(adjustment_points[i1]);
                    row++;
                }
            }
        }
    }

    std::cout << "built A and b" << std::endl;

    std::size_t A_rows = row;
    assert(A_rows < static_cast<std::size_t>(std::numeric_limits<int>::max()));

    SpMat A(A_rows, x_rows);
    A.setFromTriplets(coefficients_A.begin(), coefficients_A.end());

    SpMat I(x_rows, x_rows);
    I.setIdentity();

    SpMat Lhs = A.transpose() * A + Gamma.transpose() * Gamma + I * 0.0001f;

    /* Only keep lower triangle (CG only uses the lower),
     * prune the rest and compress matrix. */
    Lhs.prune([](const int &row, const int &col, const float &value) -> bool
              { return col <= row && value != 0.0f; }); // value != 0.0f is only to suppress a compiler warning

    std::cout << " done." << std::endl;
    std::cout << "\tLhs dimensionality: " << Lhs.rows() << " x " << Lhs.cols() << std::endl;

    std::cout << "\tCalculating adjustments:" << std::endl;
    /* Prepare solver. */
    Eigen::ConjugateGradient<SpMat, Eigen::Lower> cg;
    cg.setMaxIterations(1000);
    cg.setTolerance(0.0001);
    cg.compute(Lhs);

    /* Prepare right hand side. */
    Eigen::VectorXf b(A_rows);
    for (std::size_t i = 0; i < coefficients_b.size(); ++i)
    {
        b[i] = coefficients_b[i];
    }
    Eigen::VectorXf Rhs = SpMat(A.transpose()) * b;

    /* Solve for x. */
    Eigen::VectorXf x(x_rows);
    x = cg.solve(Rhs);

    /* Subtract mean because system is underconstrained and we seek the solution with minimal adjustments. */
    x = x.array() - x.mean();

    std::cout << "\t\tCG took " << cg.iterations() << " iterations. Residual is " << cg.error() << std::endl;

    cv::Mat adjustments = cv::Mat::zeros(index_image.rows, index_image.cols, CV_32F);
    for (int i = 0; i < index_image.rows; i++)
    {
        for (int j = 0; j < index_image.cols; j++)
        {
            int32_t index = index_image.at<int32_t>(i, j);
            if (index < 0)
            {
                continue;
            }

            adjustments.at<float>(i, j) = x[index];
        }
    }

    return adjustments;
}

cv::Mat create_mosaic(std::vector<ImageView> &image_views, const QuadMesh &mesh, const cv::Mat &labels, const cv::Mat &adjustments)
{
    constexpr size_t tile_width = 32;
    cv::Mat mosaic = cv::Mat::zeros(labels.rows * tile_width, labels.cols * tile_width, CV_16U);

    cv::Mat weight_br(tile_width, tile_width, CV_32F);
    for (size_t i = 0; i < tile_width; i++)
    {
        float y = (i + 0.5) / tile_width;
        for (size_t j = 0; j < tile_width; j++)
        {
            float x = (j + 0.5) / tile_width;
            weight_br.at<float>(i, j) = x * y;
        }
    }

    cv::Mat weight_tl;
    cv::Mat weight_tr;
    cv::Mat weight_bl;
    cv::rotate(weight_br, weight_tl, cv::ROTATE_180);
    cv::rotate(weight_br, weight_tr, cv::ROTATE_90_COUNTERCLOCKWISE);
    cv::rotate(weight_br, weight_bl, cv::ROTATE_90_CLOCKWISE);

    for (size_t k = 0; k < image_views.size(); k++)
    {
        image_views[k].load_image();
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

                    std::vector<cv::Point2f> corner_pixels = image_views[k].get_pixel_coords(corner_points);

                    constexpr bool preserve_max = true;
                    cv::Mat tile = image_views[k].GetTile(corner_pixels, cv::INTER_LINEAR, cv::BORDER_CONSTANT, preserve_max);

                    cv::Mat adjustment = adjustments.at<float>(2 * i, 2 * j) * weight_tl +
                                         adjustments.at<float>(2 * i, 2 * j + 1) * weight_tr +
                                         adjustments.at<float>(2 * i + 1, 2 * j + 1) * weight_br +
                                         adjustments.at<float>(2 * i + 1, 2 * j) * weight_bl;
                    cv::add(tile, adjustment, tile, cv::noArray(), tile.type());
                    tile.copyTo(mosaic(cv::Rect(j * tile_width, i * tile_width, tile_width, tile_width)));
                }
            }
        }
        image_views[k].release_image();
    }
    return mosaic;
}

int main(int argc, char **argv)
{
    if (argc != 5)
    {
        std::cout << "Usage: " << argv[0] << " <reconstruction file (.json)> <dem file (.tiff)> <georef file> <output directory>" << std::endl;
        return -1;
    }
    const std::filesystem::path reconstruction_path(argv[1]);
    const std::filesystem::path dem_path(argv[2]);
    const std::filesystem::path georef_file(argv[3]);
    const std::filesystem::path out_dir(argv[4]);

    const std::filesystem::path labeling_file;
    const bool write_intermediate_results = false;

    if (!std::filesystem::is_directory(out_dir))
    {
        std::filesystem::create_directories(out_dir);
    }

    const std::filesystem::path tmp_dir = out_dir / "tmp";
    if (!std::filesystem::is_directory(tmp_dir))
    {
        std::filesystem::create_directory(tmp_dir);
    }
    else
    {
        std::cout << "Careful! Temporary directory \"tmp\" exists within the destination directory." << std::endl;
    }

    // Set the number of threads to use.
    const int num_threads = -1;
    tbb::task_arena arena(num_threads > 0 ? num_threads : tbb::this_task_arena::max_concurrency());

    if (num_threads > 0)
    {
        omp_set_dynamic(0);
        omp_set_num_threads(num_threads);
    }

    std::cout << "Load and prepare mesh: " << std::endl;
    QuadMesh mesh(dem_path);

    std::cout << "Generating image views: " << std::endl;
    std::vector<ImageView> image_views = generate_image_views(reconstruction_path);
    std::cout << "Generated " << image_views.size() << " image views" << std::endl;

    cv::Mat labels;
    cv::Mat adjustments;
    std::vector<std::vector<QuadInfo>> quad_infos;

    if (labeling_file.empty())
    {
        std::cout << "View selection:" << std::endl;
        std::vector<TemperatureCompensation *> temp_comp_per_view;
        quad_infos = calculate_face_projection_infos(mesh, image_views, temp_comp_per_view);
        DataCosts data_costs = calculate_data_costs(mesh, quad_infos);

        size_t n_views = image_views.size();
        std::vector<float> pairwise_cost(n_views * n_views, 0);
        for (size_t i = 0; i < n_views; i++)
        {
            float max_distance = 0;
            for (size_t j = i + 1; j < n_views; j++)
            {
                pairwise_cost[i * n_views + j] = (image_views[i].get_pos() - image_views[j].get_pos()).norm();
                max_distance = std::max(max_distance, pairwise_cost[i * n_views + j]);
            }
            for (size_t j = i + 1; j < n_views; j++)
            {
                pairwise_cost[i * n_views + j] = 1.F * (1.F + 4.F * pairwise_cost[i * n_views + j] * pairwise_cost[i * n_views + j] / max_distance / max_distance);
                pairwise_cost[j * n_views + i] = pairwise_cost[i * n_views + j];
            }
        }

        cv::Mat tile_cost;
        labels = best_local_labels(quad_infos, mesh, tile_cost);
        if (write_intermediate_results)
        {
            cv::Mat img;
            cv::normalize(tile_cost, img, 255, 0, cv::NORM_MINMAX, CV_8U);
            std::filesystem::path filepath = out_dir / "edge_cost.png";
            cv::imwrite(filepath, img);

            filepath = out_dir / "local_labeling.png";
            cv::imwrite(filepath, labels);
        }

        try
        {
            labels = view_selection(data_costs, mesh, tile_cost, pairwise_cost, n_views);
            adjustments = angular_temperature_compensation(labels, mesh, image_views, temp_comp_per_view);

            if (write_intermediate_results)
            {
                cv::Mat img;
                img = temperature_adjustments + 64;
                img.convertTo(img, CV_8U);
                std::filesystem::path filepath = out_dir / "angular.png";
                cv::imwrite(filepath, img);
            }
            adjustments = adjustments + global_seam_leveling(labels, quad_infos, temperature_adjustments);

            if (write_intermediate_results)
            {
                cv::Mat img;
                img = adjustments + 128;
                img.convertTo(img, CV_8U);
                std::filesystem::path filepath = out_dir / "adjustments.png";
                cv::imwrite(filepath, img);
            }

            // labels = best_local_labels(quad_infos, mesh);
        }
        catch (std::runtime_error &e)
        {
            std::cerr << "\tOptimization failed: " << e.what() << std::endl;
            std::exit(EXIT_FAILURE);
        }

        /* Write labeling to file. */
        if (write_intermediate_results)
        {
            std::filesystem::path filepath = out_dir / "labeling.png";
            cv::imwrite(filepath, labels);
        }
    }
    else
    {
        labels = cv::imread(labeling_file, cv::IMREAD_ANYDEPTH);
    }

    ProjInfo proj = parse_georef_file(georef_file);

    constexpr size_t tile_width = 32;
    GeoInfo geo;
    mesh.GetGeoTransform(geo.transform);
    geo.transform[1] /= tile_width;
    geo.transform[2] /= tile_width;
    geo.transform[4] /= tile_width;
    geo.transform[5] /= tile_width;

    geo.transform[0] += proj.x_offset;
    geo.transform[3] += proj.y_offset;

    geo.model_type = ModelTypeProjected;
    geo.raster_type = RasterPixelIsArea;
    geo.projected_linear_units = Linear_Meter;
    geo.geographic_angular_units = Angular_Degree;
    geo.projected_coordinate_system = proj.crs;
    geo.gdal_nodata_value = 0;
    geo.capture_time_utc = get_mean_time(image_views);

    cv::Mat mosaic = create_mosaic(image_views, mesh, labels, adjustments);
    std::filesystem::path filepath = out_dir / "mosaic.tiff";
    save_geotiff(filepath, mosaic, geo);

    if (write_intermediate_results)
    {
        adjustments = 0;
        mosaic = create_mosaic(image_views, mesh, labels, adjustments);
        std::filesystem::path filepath = out_dir / "mosaic_unadjusted.tiff";
        save_geotiff(filepath, mosaic, geo);
    }

    std::cout << "Created mosaic at " << filepath << std::endl;

    //     /* Remove temporary files. */
    //     for (util::fs::File const & file : util::fs::Directory(tmp_dir)) {
    //         util::fs::unlink(util::fs::join_path(file.path, file.name).c_str());
    //     }
    //     util::fs::rmdir(tmp_dir.c_str());

    return EXIT_SUCCESS;
}
