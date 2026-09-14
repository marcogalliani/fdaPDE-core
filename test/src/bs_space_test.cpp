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

#include <algorithm>
#include <cmath>
#include <vector>
#include <gtest/gtest.h>   // testing framework

#include <fdaPDE/splines.h>

using namespace fdapde;

namespace {

// deliberately non-uniform interval, so nothing can pass by accident on equal spacing
Eigen::Matrix<double, Dynamic, 1> nodes() {
    Eigen::Matrix<double, Dynamic, 1> t(9);
    t << 0.0, 0.3, 0.55, 1.0, 1.6, 1.65, 2.4, 3.1, 3.5;
    return t;
}

// value at x of the spline of Vh with coefficients c
double eval(BsSpace<Triangulation<1, 1>>& Vh, const Eigen::Matrix<double, Dynamic, 1>& c, double x) {
    return BsFunction<BsSpace<Triangulation<1, 1>>>(Vh, c)(Eigen::Matrix<double, 1, 1>(x));
}

}   // namespace

// raising the knot multiplicity at a node lowers the continuity there to C^(order - mu), and grows the
// space accordingly: dim = order + 1 + sum over interior nodes of mu_i. The classical simple-knot space is
// the mu = 1 case, which the two-argument constructor must keep reproducing exactly.
TEST(bs_space_test, node_multiplicity_controls_dimension_and_continuity) {
    const auto t = nodes();
    const int m = t.size(), N = m - 1;
    for (int p = 1; p <= 4; ++p) {
        Triangulation<1, 1> T(t);
        // the 2-argument constructor is the simple-knot space and is unchanged
        BsSpace simple(T, p);
        EXPECT_EQ(simple.n_dofs(), m + p - 1) << "order " << p;
        BsSpace explicit_simple(T, p, std::vector<int>(m, 1));
        EXPECT_EQ(explicit_simple.n_dofs(), simple.n_dofs());

        // raise every interior node to multiplicity p: the C0 space of degree p
        std::vector<int> mult(m, p);
        BsSpace Vh(T, p, mult);
        EXPECT_EQ(Vh.n_dofs(), p * N + 1) << "order " << p << ": dim of the C0 space is p*N + 1";

        // every cell carries order + 1 dofs, advancing strictly from cell to cell up to the last dof, and the
        // basis remains a partition of unity
        const Eigen::Matrix<double, Dynamic, 1> ones = Eigen::Matrix<double, Dynamic, 1>::Ones(Vh.n_dofs());
        int prev = -1;
        for (int k = 0; k < N; ++k) {
            std::vector<int> dofs = Vh.dof_handler().active_dofs(k);
            ASSERT_EQ(static_cast<int>(dofs.size()), p + 1);
            EXPECT_GT(dofs.front(), prev);
            prev = dofs.front();
            EXPECT_GE(dofs.front(), 0);
            EXPECT_LT(dofs.back(), Vh.n_dofs());
            for (int q = 0; q < 10; ++q) {
                const double th = (q + 0.5) / 10.0, x = t[k] + th * (t[k + 1] - t[k]);
                EXPECT_NEAR(eval(Vh, ones, x), 1.0, 1e-13) << "order " << p << ", cell " << k;
            }
        }
        EXPECT_EQ(Vh.dof_handler().active_dofs(N - 1).back(), Vh.n_dofs() - 1) << "order " << p;
    }
}

/* Multiplicity is per NODE, not global: raising it only where data is observed leaves the space smooth
everywhere else. This is the case the ODE penalty needs -- the collocation trajectory is C0 exactly at the
nodes it steps through, and smooth in between -- so it is pinned directly: a cubic space with multiplicity
3 at the selected nodes must kink there and stay C2 at the rest. */
TEST(bs_space_test, multiplicity_is_raised_only_where_asked) {
    const auto t = nodes();
    const int m = t.size(), p = 3;
    Triangulation<1, 1> T(t);
    std::vector<int> mult(m, 1);
    int n_raised = 0;
    for (int i = 2; i + 1 < m; i += 2) {   // every other interior node
        mult[i] = p;
        ++n_raised;
    }
    BsSpace Vh(T, p, mult);
    // dim = (p + 1) + sum_interior mu_i, with mu = p at the raised nodes and 1 at the rest
    EXPECT_EQ(Vh.n_dofs(), p + 1 + n_raised * p + (m - 2 - n_raised));

    Eigen::Matrix<double, Dynamic, 1> c(Vh.n_dofs());
    for (int i = 0; i < c.size(); ++i) { c[i] = std::sin(1.9 * i); }
    const double h = 1e-5;
    for (int i = 1; i + 1 < m; ++i) {
        const double x = t[i];
        const double dl = (eval(Vh, c, x) - eval(Vh, c, x - h)) / h;
        const double dr = (eval(Vh, c, x + h) - eval(Vh, c, x)) / h;
        const double jump = std::abs(dr - dl);
        // the spline itself is continuous everywhere (C0 at worst)
        EXPECT_NEAR(eval(Vh, c, x - h), eval(Vh, c, x + h), 1e-3) << "node " << i;
        if (mult[i] == p) {
            EXPECT_GT(jump, 1.0) << "node " << i << " has multiplicity p, so it must kink (C0)";
        } else {
            EXPECT_LT(jump, 1e-2) << "node " << i << " keeps multiplicity 1, so it stays C2";
        }
    }
}

/* The multiplicity pattern is per node and arbitrary: it is not a single rule applied uniformly, so a
caller holding a pattern of its own -- a solver that knows where data is observed, say -- expresses it
exactly. Constructing a space is also a read-only act on the geometry: nothing is recorded on the mesh. */
TEST(bs_space_test, multiplicity_vector_carries_an_arbitrary_pattern) {
    const auto t = nodes();
    const int m = t.size(), p = 3;
    Triangulation<1, 1> T(t);
    // a different multiplicity at each interior node, cycling through 1, 2, 3
    std::vector<int> mult(m, 1);
    int expected = p + 1;
    for (int i = 1; i + 1 < m; ++i) {
        mult[i] = 1 + (i % p);
        expected += mult[i];
    }
    BsSpace Vh(T, p, mult);
    EXPECT_EQ(Vh.n_dofs(), expected);
    // the knot vector realises exactly the requested pattern
    const std::vector<double>& u = Vh.physical_basis().knots_vector();
    for (int i = 1; i + 1 < m; ++i) {
        EXPECT_EQ(std::count(u.begin(), u.end(), t[i]), mult[i]) << "node " << i;
    }
    EXPECT_EQ(std::count(u.begin(), u.end(), t[0]), p + 1);         // boundaries are always clamped
    EXPECT_EQ(std::count(u.begin(), u.end(), t[m - 1]), p + 1);
    // the cell -> dof map advances by the multiplicity of the node between consecutive cells, and the basis
    // is still a partition of unity on every cell
    const Eigen::Matrix<double, Dynamic, 1> ones = Eigen::Matrix<double, Dynamic, 1>::Ones(Vh.n_dofs());
    for (int k = 0; k < T.n_cells(); ++k) {
        if (k > 0) {
            EXPECT_EQ(
              Vh.dof_handler().active_dofs(k).front() - Vh.dof_handler().active_dofs(k - 1).front(), mult[k])
              << "cell " << k;
        }
        EXPECT_NEAR(eval(Vh, ones, 0.5 * (t[k] + t[k + 1])), 1.0, 1e-13) << "cell " << k;
    }
    // building a space records nothing on the geometry
    EXPECT_TRUE(T.nodes_markers().empty());
}