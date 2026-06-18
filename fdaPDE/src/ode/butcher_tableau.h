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
class ButcherTableau {
   public:
    using matrix_t = Eigen::Matrix<double, Dynamic, Dynamic>;
    using vector_t = Eigen::Matrix<double, Dynamic, 1>;

    ButcherTableau() = default;
    ButcherTableau(const matrix_t& A, const vector_t& b, const vector_t& c) : A_(A), b_(b), c_(c) {
        n_stages_ = b_.size();
        fdapde_assert(A_.rows() == n_stages_ && A_.cols() == n_stages_ && c_.size() == n_stages_);
        // detect explicit tableaux (A strictly lower triangular)
        is_explicit_ = true;
        for (int i = 0; i < n_stages_ && is_explicit_; ++i) {
            for (int j = i; j < n_stages_; ++j) {
                if (A_(i, j) != 0.0) { is_explicit_ = false; break; }
            }
        }
    }
    // observers
    int n_stages() const { return n_stages_; }
    const matrix_t& A() const { return A_; }
    const vector_t& b() const { return b_; }
    const vector_t& c() const { return c_; }
    bool is_explicit() const { return is_explicit_; }
   private:
    matrix_t A_;
    vector_t b_, c_;
    int n_stages_ = 0;
    bool is_explicit_ = false;
};

// named tableaux ----------------------------------------------------------------------------
namespace ode_schemes {

// explicit Euler (order 1)
inline ButcherTableau forward_euler() {
    ButcherTableau::matrix_t A(1, 1);
    ButcherTableau::vector_t b(1), c(1);
    A << 0.0;
    b << 1.0;
    c << 0.0;
    return ButcherTableau(A, b, c);
}
// backward / implicit Euler (order 1)
inline ButcherTableau backward_euler() {
    ButcherTableau::matrix_t A(1, 1);
    ButcherTableau::vector_t b(1), c(1);
    A << 1.0;
    b << 1.0;
    c << 1.0;
    return ButcherTableau(A, b, c);
}
// Crank-Nicolson / implicit trapezoidal rule (order 2)
inline ButcherTableau crank_nicolson() {
    ButcherTableau::matrix_t A(2, 2);
    ButcherTableau::vector_t b(2), c(2);
    A << 0.0, 0.0,
         0.5, 0.5;
    b << 0.5, 0.5;
    c << 0.0, 1.0;
    return ButcherTableau(A, b, c);
}
// implicit midpoint = 1-stage Gauss-Legendre, GL1 (order 2)
inline ButcherTableau implicit_midpoint() {
    ButcherTableau::matrix_t A(1, 1);
    ButcherTableau::vector_t b(1), c(1);
    A << 0.5;
    b << 1.0;
    c << 0.5;
    return ButcherTableau(A, b, c);
}
// 2-stage Gauss-Legendre, GL2 (order 4)
inline ButcherTableau gauss_legendre_2() {
    const double s3 = std::sqrt(3.0) / 6.0;
    ButcherTableau::matrix_t A(2, 2);
    ButcherTableau::vector_t b(2), c(2);
    A << 0.25,      0.25 - s3,
         0.25 + s3, 0.25;
    b << 0.5, 0.5;
    c << 0.5 - s3, 0.5 + s3;
    return ButcherTableau(A, b, c);
}
// classic explicit Runge-Kutta (order 4)
inline ButcherTableau rk4() {
    ButcherTableau::matrix_t A(4, 4);
    ButcherTableau::vector_t b(4), c(4);
    A.setZero();
    A(1, 0) = 0.5;
    A(2, 1) = 0.5;
    A(3, 2) = 1.0;
    b << 1.0 / 6.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0;
    c << 0.0, 0.5, 0.5, 1.0;
    return ButcherTableau(A, b, c);
}

}   // namespace ode_schemes
}   // namespace fdapde

#endif   // __FDAPDE_BUTCHER_TABLEAU_H__
