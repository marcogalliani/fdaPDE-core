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
    }
    // observers
    static constexpr int n_stages() { return Stages; }
    constexpr const std::array<std::array<double, Stages>, Stages>& A() const { return A_; }
    constexpr const std::array<double, Stages>& b() const { return b_; }
    constexpr const std::array<double, Stages>& c() const { return c_; }
    constexpr bool is_explicit() const { return is_explicit_; }
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
