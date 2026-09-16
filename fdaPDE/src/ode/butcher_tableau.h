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

#ifndef __FDAPDE_BUTCHER_TABLEAU_H__
#define __FDAPDE_BUTCHER_TABLEAU_H__

#include "header_check.h"

namespace fdapde {

// A Butcher tableau describes an s-stage Runge-Kutta method for the IVP y' = f(t, y):
//
//     k_i = f(t + c_i*dt, y + dt * sum_j A_ij k_j),   i = 1, ..., s
//     y_{n+1} = y_n + dt * sum_i b_i k_i
//
//          c | A
//         ---+---
//            | b^T
//
// A method is explicit iff A is strictly lower triangular (each stage depends only on
// previous ones), otherwise it is (diagonally/fully) implicit and the stage system must
// be solved. Every scheme used by the ODE-penalty solvers (forward Euler, Crank-Nicolson /
// implicit trapezoid, implicit midpoint = GL1, 2-stage Gauss-Legendre = GL2, ...) is a
// single tableau, so one integrator implementation covers all of them.
template <int Stages>
struct ButcherTableau {
    std::array<std::array<double, Stages>, Stages> A_ {};
    std::array<double, Stages> b_ {}, c_ {};
    bool is_explicit_ = false;
    bool is_collocation_ = false, is_symplectic_ = false;

    // default (zero) tableau: a placeholder for default-constructed integrators/solvers, overwritten
    // before use. The meaningful tableaux are built by the ode_schemes factory functions below.
    constexpr ButcherTableau() = default;
    constexpr ButcherTableau(
        std::array<std::array<double, Stages>, Stages> A,
        std::array<double, Stages> b,
        std::array<double, Stages> c
    ) : A_(A), b_(b), c_(c) {
        // detect explicit tableaux (A strictly lower triangular)
        is_explicit_ = true;
        for (int i = 0; i < Stages && is_explicit_; ++i) {
            for (int j = i; j < Stages; ++j) {
                if (A_[i][j] != 0.0) { is_explicit_ = false; break; }
            }
        }
        // a method is a COLLOCATION method iff its nodes c_i are pairwise distinct: the s stage values then
        // determine a unique degree-s polynomial through the step, of which the k_i are the derivatives at
        // the c_i. This is what makes the stage values samples of one polynomial at all -- rk4 fails it
        // (c = 0, 1/2, 1/2, 1), so no polynomial interpolates its stages.
        is_collocation_ = true;
        for (int i = 0; i < Stages && is_collocation_; ++i) {
            for (int j = i + 1; j < Stages; ++j) {
                if (abs_(c_[i] - c_[j]) < 1e-14) { is_collocation_ = false; break; }
            }
        }
        // a method is SYMPLECTIC iff b_i a_ij + b_j a_ji = b_i b_j. Equivalently (Hager's transformed
        // adjoint tableau collapses to A itself) the discrete adjoint of the method is the same method run
        // backward in time -- so the stage adjoints are collocation values of the continuous adjoint
        // equation, on the same nodes when the c_i are symmetric about 1/2. Gauss-Legendre schemes satisfy
        // both; Radau IIA / Lobatto IIIA / the explicit schemes do not. Only under this flag may stage
        // costates (and the optimal control they define) be interpolated on the forward nodes.
        is_symplectic_ = true;
        for (int i = 0; i < Stages && is_symplectic_; ++i) {
            for (int j = 0; j < Stages; ++j) {
                if (abs_(b_[i] * A_[i][j] + b_[j] * A_[j][i] - b_[i] * b_[j]) > 1e-13) {
                    is_symplectic_ = false;
                    break;
                }
            }
        }
    }
    // observers
    static constexpr int n_stages() { return Stages; }
    constexpr const std::array<std::array<double, Stages>, Stages>& A() const { return A_; }
    constexpr const std::array<double, Stages>& b() const { return b_; }
    constexpr const std::array<double, Stages>& c() const { return c_; }
    constexpr bool is_explicit() const { return is_explicit_; }
    constexpr bool is_collocation() const { return is_collocation_; }
    constexpr bool is_symplectic() const { return is_symplectic_; }

   private:
    // std::abs is not usable in a constant expression before C++23
    static constexpr double abs_(double x) { return x < 0 ? -x : x; }
};

// named tableaux ----------------------------------------------------------------------------
namespace ode_schemes {

// explicit Euler (order 1)
inline constexpr ButcherTableau<1> forward_euler() {
    return ButcherTableau<1>(
        {{{0.0}}},  // A
        {1.0},      // b
        {0.0});     // c
}
// backward / implicit Euler (order 1)
inline constexpr ButcherTableau<1> backward_euler() {
    return ButcherTableau<1>(
        {{{1.0}}},  // A
        {1.0},      // b
        {1.0});     // c
}
// Crank-Nicolson / implicit trapezoidal rule (order 2)
inline constexpr ButcherTableau<2> crank_nicolson() {
    return ButcherTableau<2>(
        {{{0.0, 0.0}, {0.5, 0.5}}},  // A
        {0.5, 0.5},      // b
        {0.0, 1.0});     // c
}
// implicit midpoint = 1-stage Gauss-Legendre, GL1 (order 2)
inline constexpr ButcherTableau<1> implicit_midpoint() {
    return ButcherTableau<1>(
        {{{0.5}}},  // A
        {1.0},      // b
        {0.5});     // c
}
// 2-stage Gauss-Legendre, GL2 (order 4). Not constexpr: the nodes/weights involve std::sqrt, which
// is not a constant expression in C++20 (the ButcherTableau itself is constexpr, just built at runtime).
// TODO: make it constexpr
inline ButcherTableau<2> gauss_legendre_2() {
    const double s3 = std::sqrt(3.0) / 6.0;
    return ButcherTableau<2>(
        {{{0.25, 0.25 - s3}, {0.25 + s3, 0.25}}},   // A
        {0.5, 0.5},                                 // b
        {0.5 - s3, 0.5 + s3});                      // c
}
// 3-stage Gauss-Legendre, GL3 (order 6). Like GL2, not constexpr: the nodes/weights involve std::sqrt.
inline ButcherTableau<3> gauss_legendre_3() {
    const double s15 = std::sqrt(15.0);
    return ButcherTableau<3>(
        {{{5.0 / 36.0, 2.0 / 9.0 - s15 / 15.0, 5.0 / 36.0 - s15 / 30.0},
          {5.0 / 36.0 + s15 / 24.0, 2.0 / 9.0, 5.0 / 36.0 - s15 / 24.0},
          {5.0 / 36.0 + s15 / 30.0, 2.0 / 9.0 + s15 / 15.0, 5.0 / 36.0}}},   // A
        {5.0 / 18.0, 4.0 / 9.0, 5.0 / 18.0},                                 // b
        {0.5 - s15 / 10.0, 0.5, 0.5 + s15 / 10.0});                          // c
}
/* 4-stage Gauss-Legendre, GL4 (order 8). Nodes are the roots of the shifted Legendre polynomial of degree
4, weights those of the matching quadrature rule. Unlike GL2 / GL3 the entries of A have no compact closed
form -- they involve nested radicals -- so rather than transcribe sixteen irrational constants they are
built from the DEFINING property of a collocation method,

    A_ij = int_0^{c_i} l_j(tau) dtau,

l_j being the Lagrange basis on the nodes. That identity is what makes the scheme a collocation method at
all (it is also what makes each stage value the step's polynomial at c_i), so deriving A from it is exact and
self-checking: ode_test verifies the result against the order and symplecticity conditions.

GL4 is the smallest Gauss scheme whose stage space carries a CUBIC control: a control basis of degree p is
represented exactly by the s stage values only when p <= s - 1, so degree 3 needs s = 4. */
inline ButcherTableau<4> gauss_legendre_4() {
    // 4-point Gauss-Legendre on [-1, 1], mapped to [0, 1]
    const double x_out = std::sqrt(3.0 / 7.0 + (2.0 / 7.0) * std::sqrt(6.0 / 5.0));
    const double x_in  = std::sqrt(3.0 / 7.0 - (2.0 / 7.0) * std::sqrt(6.0 / 5.0));
    const double w_out = (18.0 - std::sqrt(30.0)) / 36.0;
    const double w_in  = (18.0 + std::sqrt(30.0)) / 36.0;
    std::array<double, 4> c {0.5 * (1 - x_out), 0.5 * (1 - x_in), 0.5 * (1 + x_in), 0.5 * (1 + x_out)};
    std::array<double, 4> b {0.5 * w_out, 0.5 * w_in, 0.5 * w_in, 0.5 * w_out};
    std::array<std::array<double, 4>, 4> A {};
    for (int j = 0; j < 4; ++j) {
        // l_j in the monomial basis: expand prod_{k != j} (tau - c_k) / (c_j - c_k) by repeated
        // multiplication, high coefficient first so each factor is applied in place
        std::array<double, 5> p {1.0, 0.0, 0.0, 0.0, 0.0};
        int deg = 0;
        double denom = 1.0;
        for (int k = 0; k < 4; ++k) {
            if (k == j) { continue; }
            for (int t = deg + 1; t > 0; --t) { p[t] = p[t - 1] - c[k] * p[t]; }
            p[0] = -c[k] * p[0];
            ++deg;
            denom *= (c[j] - c[k]);
        }
        for (int t = 0; t <= deg; ++t) { p[t] /= denom; }
        // integrate term by term: int_0^{c_i} tau^t = c_i^(t+1) / (t + 1)
        for (int i = 0; i < 4; ++i) {
            double acc = 0.0, pow_c = c[i];
            for (int t = 0; t <= deg; ++t) {
                acc += p[t] * pow_c / (t + 1);
                pow_c *= c[i];
            }
            A[i][j] = acc;
        }
    }
    return ButcherTableau<4>(A, b, c);
}
// classic explicit Runge-Kutta (order 4)
inline constexpr ButcherTableau<4> rk4() {
    return ButcherTableau<4>(
        {{
            {0.0, 0.0, 0.0, 0.0}, 
            {0.5, 0.0, 0.0, 0.0},
            {0.0, 0.5, 0.0, 0.0},
            {0.0, 0.0, 1.0, 0.0}
        }},  // A
        {1.0 / 6.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0},      // b
        {0.0, 0.5, 0.5, 1.0});     // c
}

}   // namespace ode_schemes
}   // namespace fdapde

#endif   // __FDAPDE_BUTCHER_TABLEAU_H__
