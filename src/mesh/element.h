// This file is part of fdaPDE, a C++ library for physics-informed
// spatial and functional data analysis.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

#ifndef __ELEMENT_H__
#define __ELEMENT_H__

#include <Eigen/LU>
#include <array>

#include "../linear_algebra/vector_space.h"
#include "../utils/assert.h"
#include "../utils/compile_time.h"
#include "../utils/symbols.h"

namespace fdapde {
namespace core {

// M local dimension, N embedding dimension, R order of the element (defaulted to linear finite elements)
template <unsigned int M, unsigned int N, unsigned int R = 1> class Element;   // forward declaration

// using declarations for manifold specialization
template <unsigned int R> using SurfaceElement = Element<2, 3, R>;
template <unsigned int R> using NetworkElement = Element<1, 2, R>;

// number of degrees of freedom associated to an M-dimensional element of degree R
constexpr unsigned int ct_nnodes(const unsigned int M, const unsigned int R) {
    return ct_factorial(M + R) / (ct_factorial(M) * ct_factorial(R));
}
// number of vertices of an M-dimensional element
constexpr unsigned int ct_nvertices(const unsigned int M) { return M + 1; }
// number of edges of an M-dimensional element
constexpr unsigned int ct_nedges(const unsigned int M) { return (M * (M + 1)) / 2; }

// A single mesh element. This object represents the main **geometrical** abstraction of a physical element.
// No functional information is carried by instances of this object.
template <unsigned int M, unsigned int N, unsigned int R> class Element {
   private:
    std::size_t ID_;                                         // ID of this element
    std::array<std::size_t, ct_nvertices(M)> node_ids_ {};   // ID of nodes composing the element
    std::array<SVector<N>, ct_nvertices(M)> coords_ {};      // nodes coordinates (1-1 mapped with node_ids_)
    std::vector<int>
      neighbors_ {};       // ID of neighboring elements (for linear networks, size not known at compile time)
    bool boundary_;        // true if the element has at least one vertex on the boundary
    double measure_ = 0;   // measure of the element

    // affine transformations from cartesian to barycentric coordinates and viceversa
    SMatrix<N, M> J_;       // [J_]_ij = (coords_[j][i] - coords_[0][i])
    SMatrix<M, N> inv_J_;   // J^{-1} (Penrose pseudo-inverse for manifold elements)
   public:
    static constexpr unsigned int nodes = ct_nnodes(M, R);
    static constexpr unsigned int vertices = ct_nvertices(M);
    static constexpr unsigned int local_dimension = M;
    static constexpr unsigned int embedding_dimension = N;
    static constexpr unsigned int order = R;

    // constructor
    Element() = default;
    Element(std::size_t ID, const std::array<std::size_t, ct_nvertices(M)>& node_ids,
            const std::array<SVector<N>, ct_nvertices(M)>& coords, const std::vector<int>& neighbors, bool boundary);

    // getters
    const std::array<SVector<N>, ct_nvertices(M)> coords() const { return coords_; }
    const std::vector<int> neighbors() const { return neighbors_; }
    const unsigned int ID() const { return ID_; }
    Eigen::Matrix<double, N, M> barycentric_matrix() const { return J_; }
    Eigen::Matrix<double, M, N> inv_barycentric_matrix() const { return inv_J_; }
    std::array<std::size_t, ct_nvertices(M)> node_ids() const { return node_ids_; }
    double measure() const { return measure_; }   // measure of the element

    // check if x is contained in the element
    template <bool is_manifold = (N != M)>
    typename std::enable_if<!is_manifold, bool>::type contains(const SVector<N>& x) const;
    template <bool is_manifold = (N != M)>   // specialization for manifold case
    typename std::enable_if<is_manifold, bool>::type contains(const SVector<N>& x) const;

    SVector<M + 1> to_barycentric_coords(const SVector<N>& x) const;   // move x into barycentric reference system
    SVector<N> mid_point() const;                                      // computes midpoint of the element
    std::pair<SVector<N>, SVector<N>> bounding_box() const;   // compute the smallest rectangle containing this element
    bool is_on_boundary() const { return boundary_; }   // true if the element has at least one node on the boundary
    VectorSpace<M, N> spanned_space() const;            // vector space passing throught this element

    // allow range for over element's coordinates
    typename std::array<SVector<N>, ct_nvertices(M)>::const_iterator begin() const { return coords_.cbegin(); }
    typename std::array<SVector<N>, ct_nvertices(M)>::const_iterator end() const { return coords_.cend(); }
};

// implementation details

template <unsigned int M, unsigned int N, unsigned int R>
Element<M, N, R>::Element(std::size_t ID, const std::array<std::size_t, ct_nvertices(M)>& node_ids,
                          const std::array<SVector<N>, ct_nvertices(M)>& coords, const std::vector<int>& neighbors,
                          bool boundary) :
    ID_(ID), node_ids_(node_ids), coords_(coords), neighbors_(neighbors), boundary_(boundary) {
    // precompute barycentric coordinate matrix for fast access
    // use first point as reference
    SVector<N> ref = coords_[0];
    for (std::size_t j = 0; j < M; ++j) { J_.col(j) = coords_[j + 1] - ref; }

    // precompute and cache inverse of barycentric matrix
    if constexpr (N == M) {
        inv_J_ = J_.inverse();
    } else {
        // for manifold domains compute the generalized Penrose inverse
        inv_J_ = (J_.transpose() * J_).inverse() * J_.transpose();
    }
    // precompute element measure
    if constexpr (M == N) {   // non-manifold case
        measure_ = std::abs(J_.determinant()) / (ct_factorial(M));
    } else {
        if constexpr (M == 2)   // surface element, compute area of 3D triangle
            measure_ = 0.5 * J_.col(0).cross(J_.col(1)).norm();
        if constexpr (M == 1)   // network element, compute length of 2D segment
            measure_ = J_.col(0).norm();
    }
}

template <unsigned int M, unsigned int N, unsigned int R>
SVector<M + 1> Element<M, N, R>::to_barycentric_coords(const SVector<N>& x) const {
    // solve: barycenitrc_matrix_*z = (x-ref)
    SVector<M> z = inv_J_ * (x - coords_[0]);

    SVector<M + 1> result;
    result << SVector<1>(1 - z.sum()), z;
    return result;
}

template <unsigned int M, unsigned int N, unsigned int R> SVector<N> Element<M, N, R>::mid_point() const {
    // The center of gravity of an element has all its barycentric coordinates equal to 1/(M+1).
    // The midpoint of an element can be computed by mapping it to a cartesian reference system
    SVector<M> barycentric_mid_point;
    barycentric_mid_point.fill(1.0 / (M + 1));   // avoid implicit conversion to int
    return J_ * barycentric_mid_point + coords_[0];
}

template <unsigned int M, unsigned int N, unsigned int R>
std::pair<SVector<N>, SVector<N>> Element<M, N, R>::bounding_box() const {
    // define lower-left and upper-right corner of bounding box
    SVector<N> ll, ur;
    // project each vertex coordinate on the reference axis
    std::array<std::array<double, ct_nvertices(M)>, N> proj_coords;

    for (std::size_t j = 0; j < ct_nvertices(M); ++j) {
        for (std::size_t dim = 0; dim < N; ++dim) { proj_coords[dim][j] = coords_[j][dim]; }
    }
    // take minimum and maximum value along each dimension, i.e. the lower-left and upper-right corner of the box
    for (std::size_t dim = 0; dim < N; ++dim) {
        ll[dim] = *std::min_element(proj_coords[dim].begin(), proj_coords[dim].end());
        ur[dim] = *std::max_element(proj_coords[dim].begin(), proj_coords[dim].end());
    }
    return std::make_pair(ll, ur);
}

// check if a point is contained in the element
template <unsigned int M, unsigned int N, unsigned int R>
template <bool is_manifold>
typename std::enable_if<!is_manifold, bool>::type Element<M, N, R>::contains(const SVector<N>& x) const {
    // A point is inside the element if all its barycentric coordinates are positive
    return (to_barycentric_coords(x).array() >= -10 * std::numeric_limits<double>::epsilon()).all();
}

template <unsigned int M, unsigned int N, unsigned int R> VectorSpace<M, N> Element<M, N, R>::spanned_space() const {
    // build a basis for the space containing the element
    std::array<SVector<N>, M> basis;
    for (size_t i = 0; i < M; ++i) { basis[i] = (coords_[i + 1] - coords_[0]); }
    return VectorSpace<M, N>(basis, coords_[0]);
}

// specialization for manifold elements of contains() routine.
template <unsigned int M, unsigned int N, unsigned int R>
template <bool is_manifold>
typename std::enable_if<is_manifold, bool>::type Element<M, N, R>::contains(const SVector<N>& x) const {
    // check if the point is contained in the affine space spanned by the element
    VectorSpace<M, N> vs = spanned_space();
    if (vs.distance(x) > 10 * std::numeric_limits<double>::epsilon()) {
        return false;   // distance too large, not contained
    }
    // check if its barycentric coordinates are all positive
    return (to_barycentric_coords(x).array() >= -10 * std::numeric_limits<double>::epsilon()).all();
}

}   // namespace core
}   // namespace fdapde

#endif   // __ELEMENT_H__
