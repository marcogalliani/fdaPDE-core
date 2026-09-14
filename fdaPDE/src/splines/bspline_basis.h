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

#ifndef __FDAPDE_BSPLINE_BASIS_H__
#define __FDAPDE_BSPLINE_BASIS_H__

#include "header_check.h"

namespace fdapde {

// given vector of knots u_1, u_2, ..., u_N, this class represents the set of N + order - 1 spline basis functions
// {l_1(x), l_2(x), ..., l_{N + order - 1}(x)} centered at knots u_1, u_2, ..., u_N
class BSplineBasis {
   private:
    int order_;
    std::vector<Spline> basis_ {};
    std::vector<double> knots_ {};
   public:
    static constexpr int StaticInputSize = 1;
    static constexpr int Order = Dynamic;
    // constructors
    constexpr BSplineBasis() : order_(0) { }
    // constructor from user defined knot vector
    template <typename KnotsVectorType>
        requires(requires(KnotsVectorType knots, int i) {
                    { knots[i] } -> std::convertible_to<double>;
                    { knots.size() } -> std::convertible_to<std::size_t>;
                })
    BSplineBasis(KnotsVectorType&& knots, int order) : order_(order) {
        fdapde_assert(std::is_sorted(knots.begin() FDAPDE_COMMA knots.end(), std::less_equal<double>()));
        int n = knots.size();
        knots_.resize(n);
        for (int i = 0; i < n; ++i) { knots_[i] = knots[i]; }
        // define basis system
        basis_.reserve(n - order_ + 1);
        for (int i = 0; i < n - order_ - 1; ++i) { basis_.emplace_back(knots_, i, order_); }
    }
    // constructor from geometric interval (no repeated knots)
    BSplineBasis(const Triangulation<1, 1>& interval, int order) : order_(order) {
        // construct knots vector
        Eigen::Matrix<double, Dynamic, 1> knots = interval.nodes();
        fdapde_assert(std::is_sorted(knots.begin() FDAPDE_COMMA knots.end() FDAPDE_COMMA std::less_equal<double>()));
        knots_ = build_knots(knots, order_, std::vector<int>(knots.size(), 1));
        basis_.reserve(knots_.size() - order_ + 1);
        for (std::size_t i = 0; i < knots_.size() - order_ - 1; ++i) { basis_.emplace_back(knots_, i, order_); }
    }
    /* Constructor from an interval with per-node knot MULTIPLICITY. Repeating an interior knot mu times
    lowers the continuity there to C^(order-mu). `multiplicity` carries one entry per node of
    the interval; the two boundary entries are ignored, since a clamped vector always repeats them
    order + 1 times. Its dimension is order + 1 + sum over INTERIOR nodes of mu_i. */
    BSplineBasis(const Triangulation<1, 1>& interval, int order, const std::vector<int>& multiplicity) :
        order_(order) {
        Eigen::Matrix<double, Dynamic, 1> knots = interval.nodes();
        fdapde_assert(std::is_sorted(knots.begin() FDAPDE_COMMA knots.end() FDAPDE_COMMA std::less_equal<double>()));
        fdapde_assert(static_cast<int>(multiplicity.size()) == knots.size());
        knots_ = build_knots(knots, order_, multiplicity);
        basis_.reserve(knots_.size() - order_ + 1);
        for (std::size_t i = 0; i < knots_.size() - order_ - 1; ++i) { basis_.emplace_back(knots_, i, order_); }
    }
    // getters
    constexpr const Spline& operator[](int i) const { return basis_[i]; }
    constexpr int size() const { return basis_.size(); }
    constexpr const std::vector<double>& knots_vector() const { return knots_; }
    int n_knots() const { return knots_.size(); }
    int order() const { return order_; }
   private:
    
    /* Knot vector over a set of breakpoints: the two boundary breakpoints repeated
    order + 1 times, interior breakpoint i repeated multiplicity[i] times. With every interior
    multiplicity equal to 1 this is the classical open knot vector of n + 2*order knots and n + order - 1
    basis functions; in general it has 2*(order+1) + sum_{interior} mu_i knots. */
    template <typename BreakpointsType>
        requires(requires(BreakpointsType breakpoints, int i) {
                    { breakpoints[i] } -> std::convertible_to<double>;
                    { breakpoints.size() } -> std::convertible_to<std::size_t>;
                })
    static std::vector<double>
    build_knots(const BreakpointsType& breakpoints, int order, const std::vector<int>& multiplicity) {
        int n = breakpoints.size();
        fdapde_assert(n > 1 && order >= 0 && static_cast<int>(multiplicity.size()) == n);
        std::vector<double> knots;
        knots.reserve(n + 2 * order);
        for (int r = 0; r <= order; ++r) { knots.push_back(breakpoints[0]); }
        for (int i = 1; i + 1 < n; ++i) {
            fdapde_assert(multiplicity[i] >= 1 && multiplicity[i] <= order + 1);
            for (int r = 0; r < multiplicity[i]; ++r) { knots.push_back(breakpoints[i]); }
        }
        for (int r = 0; r <= order; ++r) { knots.push_back(breakpoints[n - 1]); }
        return knots;
    }
};

} // namespace fdapde

#endif // __FDAPDE_BSPLINE_BASIS_H__
