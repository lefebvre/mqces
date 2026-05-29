#pragma once

#include <mqces/types.hpp>

#include <Eigen/Core>

#include <algorithm>
#include <cstddef>
#include <memory>
#include <numeric>
#include <utility>
#include <vector>

// Balanced k-d tree over a point cloud, with per-node centroid + bounding
// box stored for Barnes-Hut-style traversal.
//
// Build:   O(N log N) — median splits on the largest-variance axis,
//          recursion stops when subtree size <= kd_leaf_size.
// Query:   O(k log N) on average, where k is the number of "near" points
//          opened during traversal. Far subtrees that subtend a small
//          angle from the query are summarized by their centroid via
//          the user-supplied approx callback.
//
// The traversal API takes two callables:
//   exact_fn(idx)                — called for each point opened exactly
//   approx_fn(centroid, count)   — called for each summarized subtree
//
// This lets `spatial_rank` and `inverse_spatial_rank` reuse the same
// tree: both accumulate (query − pt)/||query − pt|| but with different
// outer scaffolding.

namespace mqces::detail {

using RowMajorMatrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

struct KdNode {
    // For leaves: indices into the original cloud are stored in
    // [leaf_begin, leaf_begin + leaf_count) of the tree's flat
    // permutation array.
    int                       split_axis = -1;   // -1 if leaf
    double                    split_value = 0.0;
    std::unique_ptr<KdNode>   left;
    std::unique_ptr<KdNode>   right;
    std::size_t               leaf_begin = 0;
    std::size_t               leaf_count = 0;
    // Aggregated info for Barnes-Hut.
    Eigen::VectorXd           centroid;          // mean of all points in subtree
    std::size_t               count = 0;         // total points in subtree
    Eigen::VectorXd           bbox_min;
    Eigen::VectorXd           bbox_max;
    double                    bbox_extent = 0.0; // ½ · ||bbox_max − bbox_min||
};

class KdTree {
public:
    // Build the tree over the given point cloud. The cloud's row order is
    // preserved; the tree only stores a permutation of row indices.
    KdTree(const RowMajorMatrix& points, std::size_t leaf_size)
        : points_(&points), indices_(static_cast<std::size_t>(points.rows()))
    {
        std::iota(indices_.begin(), indices_.end(), Eigen::Index{0});
        if (indices_.empty()) {
            return;
        }
        root_ = build_recursive(0, indices_.size(), std::max<std::size_t>(1, leaf_size));
    }

    [[nodiscard]] const RowMajorMatrix& points() const { return *points_; }
    [[nodiscard]] std::size_t           size()   const { return indices_.size(); }
    [[nodiscard]] bool                  empty()  const { return indices_.empty(); }

    // Visit nodes for a given query. Subtrees whose extent is small
    // relative to the query distance (extent/dist < opening_theta) are
    // summarized via `approx_fn`; everything else is opened exactly
    // through `exact_fn`.
    template <class ExactFn, class ApproxFn>
    void traverse(
        const Eigen::Ref<const Eigen::VectorXd>& query,
        double                                   opening_theta,
        ExactFn&&                                exact_fn,
        ApproxFn&&                               approx_fn) const
    {
        if (!root_) {
            return;
        }
        traverse_node(*root_, query, opening_theta, exact_fn, approx_fn);
    }

private:
    const RowMajorMatrix*     points_;
    std::vector<Eigen::Index> indices_;
    std::unique_ptr<KdNode>   root_;

    // Pick the splitting axis as the column with the largest extent across
    // the slice [begin, end).
    [[nodiscard]] int largest_extent_axis(std::size_t begin, std::size_t end) const
    {
        const auto d = points_->cols();
        Eigen::VectorXd lo(d), hi(d);
        lo = points_->row(indices_[begin]).transpose();
        hi = lo;
        for (std::size_t k = begin + 1; k < end; ++k) {
            const auto row = points_->row(indices_[k]).transpose();
            lo             = lo.cwiseMin(row);
            hi             = hi.cwiseMax(row);
        }
        Eigen::VectorXd extent = hi - lo;
        Eigen::Index    axis   = 0;
        extent.maxCoeff(&axis);
        return static_cast<int>(axis);
    }

    std::unique_ptr<KdNode> build_recursive(
        std::size_t begin, std::size_t end, std::size_t leaf_size)
    {
        auto node      = std::make_unique<KdNode>();
        node->count    = end - begin;
        node->leaf_begin = begin;
        node->leaf_count = end - begin;

        // Compute centroid and bounding box over [begin, end).
        const auto d = points_->cols();
        node->bbox_min = points_->row(indices_[begin]).transpose();
        node->bbox_max = node->bbox_min;
        Eigen::VectorXd centroid_acc = Eigen::VectorXd::Zero(d);
        for (std::size_t k = begin; k < end; ++k) {
            const auto row = points_->row(indices_[k]).transpose();
            node->bbox_min = node->bbox_min.cwiseMin(row);
            node->bbox_max = node->bbox_max.cwiseMax(row);
            centroid_acc += row;
        }
        node->centroid    = centroid_acc / static_cast<double>(node->count);
        node->bbox_extent = 0.5 * (node->bbox_max - node->bbox_min).norm();

        if (node->count <= leaf_size) {
            // Leaf — keep split_axis = -1, leaf_indices already set.
            return node;
        }

        // Split along largest-extent axis at the median.
        const int axis = largest_extent_axis(begin, end);
        const auto mid_off = (end - begin) / 2;
        std::nth_element(
            indices_.begin() + static_cast<std::ptrdiff_t>(begin),
            indices_.begin() + static_cast<std::ptrdiff_t>(begin + mid_off),
            indices_.begin() + static_cast<std::ptrdiff_t>(end),
            [this, axis](Eigen::Index a, Eigen::Index b) {
                return (*points_)(a, axis) < (*points_)(b, axis);
            });
        const std::size_t mid = begin + mid_off;
        node->split_axis      = axis;
        node->split_value     = (*points_)(indices_[mid], axis);

        // Catch the degenerate case where all points share the same value
        // on the chosen axis (median split would be no-op): fall back to
        // a leaf so we don't recurse forever.
        if (mid == begin || mid == end) {
            node->split_axis = -1;
            return node;
        }

        node->left  = build_recursive(begin, mid, leaf_size);
        node->right = build_recursive(mid,   end, leaf_size);
        return node;
    }

    template <class ExactFn, class ApproxFn>
    void traverse_node(
        const KdNode&                            node,
        const Eigen::Ref<const Eigen::VectorXd>& query,
        double                                   opening_theta,
        ExactFn&                                 exact_fn,
        ApproxFn&                                approx_fn) const
    {
        // Barnes-Hut opening criterion: if the subtree's spatial extent is
        // small compared to its distance from the query, summarize.
        const double dist_to_centroid = (query - node.centroid).norm();
        if (node.split_axis < 0) {
            // Leaf — open every point exactly.
            for (std::size_t k = 0; k < node.leaf_count; ++k) {
                exact_fn(indices_[node.leaf_begin + k]);
            }
            return;
        }
        if (dist_to_centroid > 0.0
            && node.bbox_extent / dist_to_centroid < opening_theta) {
            approx_fn(node.centroid, node.count);
            return;
        }
        // Recurse: visit the nearer child first for better pruning if we
        // ever add an early-exit (we don't today, but the order is
        // deterministic and matches the side of the split that the query
        // sits on).
        const bool go_left_first = query(node.split_axis) < node.split_value;
        const KdNode* first  = go_left_first ? node.left.get()  : node.right.get();
        const KdNode* second = go_left_first ? node.right.get() : node.left.get();
        if (first)  { traverse_node(*first,  query, opening_theta, exact_fn, approx_fn); }
        if (second) { traverse_node(*second, query, opening_theta, exact_fn, approx_fn); }
    }
};

}  // namespace mqces::detail
