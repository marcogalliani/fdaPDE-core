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

#ifndef __ODE_RHS_H__
#define __ODE_RHS_H__

/* File Description
This file implements a set of abstractions to represent systems of first-order Ordinary Differential Equations (ODEs). As only first-order ODE systems of the form dx/dt = f(t,x) are represented throughout the library, ODEs can be identified by their righ-hand side field. The choice of considering only first-order ODEs may appear limiting, but higher order ODEs can be encoded in a first-order system by defining fictitious variables, e.g., d^2x/dt^2 is equal to dv/dt with v = dx/dt.
*/

#include "header_check.h"

namespace fdapde {

/* RHS concepts
A right-hand side is either plain, f(t, y), or theta-parameterized, f(t, y, theta), the latter for the inverse (parameter-estimation) solvers, whose coefficients are the unknowns. ode_rhs_field wraps both: a plain rhs is just the theta-parameterized one with an empty theta bound, so a single field template and a single set of downstream machinery cover them. The concepts below only classify a user functor's capabilities so the wrapper can dispatch to the right calls; the plain functor never has to know about theta.
*/

// f models the right-hand side of an ODE y' = f(t, y): callable as f(t, y) -> R^d. The system
// dimension d is a compile-time property (see ode_rhs_dim), not a runtime method.
template <typename F>
concept is_ode_rhs = std::is_invocable_r_v<
  Eigen::Matrix<double, Dynamic, 1>, const F&, double, const Eigen::Matrix<double, Dynamic, 1>&>;

// f additionally exposes an analytic state Jacobian state_jacobian(t, y) -> R^{d x d}; when absent the
// field falls back to central finite differences.
template <typename F>
concept ode_rhs_has_state_jacobian = requires(const F& f, double t, const Eigen::Matrix<double, Dynamic, 1>& y) {
    { f.state_jacobian(t, y) } -> std::convertible_to<Eigen::Matrix<double, Dynamic, Dynamic>>;
};

// f models a theta-parameterized ODE rhs: callable as f(t, y, theta) -> R^d.
template <typename F>
concept is_parameterized_ode_rhs = std::is_invocable_r_v<
  Eigen::Matrix<double, Dynamic, 1>, const F&, double, const Eigen::Matrix<double, Dynamic, 1>&,
  const Eigen::Matrix<double, Dynamic, 1>&>;

// f additionally exposes the analytic state Jacobian state_jacobian(t, y, theta) -> R^{d x d}
template <typename F>
concept parameterized_ode_rhs_has_state_jacobian = requires(
  const F& f, double t, const Eigen::Matrix<double, Dynamic, 1>& y, const Eigen::Matrix<double, Dynamic, 1>& th) {
    { f.state_jacobian(t, y, th) } -> std::convertible_to<Eigen::Matrix<double, Dynamic, Dynamic>>;
};
// f additionally exposes the analytic parameter Jacobian param_jacobian(t, y, theta) -> R^{d x n_theta};
// when absent the parameter sensitivity falls back to central finite differences.
template <typename F>
concept parameterized_ode_rhs_has_param_jacobian = requires(
  const F& f, double t, const Eigen::Matrix<double, Dynamic, 1>& y, const Eigen::Matrix<double, Dynamic, 1>& th) {
    { f.param_jacobian(t, y, th) } -> std::convertible_to<Eigen::Matrix<double, Dynamic, Dynamic>>;
};

/* Computing the dimensionality of the ODE system at compile-time
ode_rhs_dim<F>: the system dimension of an ODE rhs, read statically. A field that knows its dimension exposes it as `static constexpr int dim` (ode_rhs_field does); otherwise it is taken from the compile-time row count of the return type (Dim for a fixed-size return, Dynamic for a dynamic VectorXd) of f(t, y) for a plain rhs, or f(t, y, theta) for a parameterized one. So a user functor opts into the static path simply by returning a fixed-size vector; VectorXd keeps the dynamic path.
*/
template <typename F> constexpr int ode_rhs_dim() {
    using G = std::decay_t<F>;
    using vec = Eigen::Matrix<double, Dynamic, 1>;
    if constexpr (requires { G::dim; }) {
        return G::dim;
    } else if constexpr (is_ode_rhs<G>) {
        return std::invoke_result_t<G, double, vec>::RowsAtCompileTime;
    } else {
        return std::invoke_result_t<G, double, vec, vec>::RowsAtCompileTime;
    }
}
template <typename F> inline constexpr int ode_rhs_dim_v = ode_rhs_dim<F>();

/* RHS field of a Dim-dimensional system of ODEs
The field is templated on both the system dimension Dim and the concrete rhs functor type F, which it
stores by value (no type erasure). Storing F directly lets the integrator's stage evaluations and
Jacobians inline through it -- the hot path of the implicit Newton solve -- instead of routing through a
std::function. The rhs is either a plain functor f(t, y) or a theta-parameterized one f(t, y, theta); the
parameterized case stores the current theta and binds it internally, so operator()(t, y) is always a
2-argument callable and the field is a drop-in is_ode_rhs consumed by the integrator / engines unchanged.
A plain field is exactly the parameterized one with an empty theta (n_params() == 0). It exposes:
  - operator()(t, y)   : f(t, y) [or f(t, y, theta) at the bound theta]
  - state_jacobian(t, y): d x d state Jacobian (from F's state_jacobian method if present, else finite diff)
  - param_jacobian(t, y): d x n_theta parameter Jacobian (parameterized fields only; analytic or finite diff)

Field algebra is not closed on the type: the sum wraps a delegating ode_rhs_sum functor, so the result is a new ode_rhs_field of a different F. That is fine because such sums (e.g. the control forcing f + u) are only ever temporaries handed to the integrator, never stored.

Dim is normally deduced from the wrapped functor (see ode_rhs_dim / the deduction guides below): a fixed-size return type gives a static Dim, a VectorXd return gives Dim = Dynamic.
*/
template <int Dim, typename F>
class ode_rhs_field {
   public:
    using vector_t = Eigen::Matrix<double, Dynamic, 1>;
    using matrix_t = Eigen::Matrix<double, Dynamic, Dynamic>;
    static constexpr int dim = Dim;   // read by ode_rhs_dim and the integrator's stage math

    ode_rhs_field() = default;   // requires F default-constructible; only instantiated where actually used
    explicit ode_rhs_field(F f) : f_(std::move(f)) { }
    // convenience: a parameterized functor together with an initial theta
    ode_rhs_field(F f, vector_t theta) : f_(std::move(f)), theta_(std::move(theta)) { }

    // ODE rhs interface: f(t, y), binding the current theta for a parameterized functor
    vector_t operator()(double t, const vector_t& y) const {
        if constexpr (is_parameterized_ode_rhs<F>) return f_(t, y, theta_);
        else                                        return f_(t, y);
    }

    // state Jacobian d f / d y (d x d): F's analytic state_jacobian when it exposes one, otherwise central
    // finite differences of operator() in y.
    matrix_t state_jacobian(double t, const vector_t& y) const {
        if constexpr (is_parameterized_ode_rhs<F>) {
            if constexpr (parameterized_ode_rhs_has_state_jacobian<F>) return f_.state_jacobian(t, y, theta_);
            else                                                       return fd_state_jacobian_(t, y);
        } else {
            if constexpr (ode_rhs_has_state_jacobian<F>) return f_.state_jacobian(t, y);
            else                                         return fd_state_jacobian_(t, y);
        }
    }

    // parameter Jacobian d f / d theta (d x n_theta) at the current theta: F's analytic param_jacobian when
    // present, otherwise central finite differences in theta. Parameterized fields only.
    matrix_t param_jacobian(double t, const vector_t& y) const {
        static_assert(is_parameterized_ode_rhs<F>, "param_jacobian is defined only for parameterized fields");
        fdapde_assert(theta_.size() > 0);
        if constexpr (parameterized_ode_rhs_has_param_jacobian<F>) return f_.param_jacobian(t, y, theta_);
        else                                                       return fd_param_jacobian_(t, y);
    }

    // parameter binding: rebind the theta a parameterized field evaluates at (an in-place update)
    void set_theta(const vector_t& theta) { theta_ = theta; }
    const vector_t& theta() const { return theta_; }
    int n_params() const { return static_cast<int>(theta_.size()); }
    static constexpr bool is_parametric() { return is_parameterized_ode_rhs<F>; }
    const F& functor() const { return f_; }

   private:
    F f_ {};
    vector_t theta_;   // current parameter (empty when plain)

    // central finite-difference state Jacobian: column j = (f(t, y + h e_j) - f(t, y - h e_j)) / 2h,
    // so entry (i, j) = d f_i / d y_j (matching the analytic convention)
    matrix_t fd_state_jacobian_(double t, const vector_t& y) const {
        const int d = y.size();
        matrix_t J(d, d);
        vector_t yp = y, ym = y;
        for (int j = 0; j < d; ++j) {
            const double h = 1e-6 * std::max(1.0, std::abs(y[j]));
            yp[j] = y[j] + h;
            ym[j] = y[j] - h;
            J.col(j) = ((*this)(t, yp) - (*this)(t, ym)) / (2 * h);
            yp[j] = y[j];
            ym[j] = y[j];
        }
        return J;
    }
    // central finite-difference parameter Jacobian: column k = (f(t, y, theta + h e_k) - ...) / 2h
    matrix_t fd_param_jacobian_(double t, const vector_t& y) const {
        const int n = theta_.size();
        const vector_t f0 = f_(t, y, theta_);
        matrix_t J(f0.size(), n);
        vector_t thp = theta_, thm = theta_;
        for (int k = 0; k < n; ++k) {
            const double h = 1e-6 * std::max(1.0, std::abs(theta_[k]));
            thp[k] = theta_[k] + h;
            thm[k] = theta_[k] - h;
            J.col(k) = (f_(t, y, thp) - f_(t, y, thm)) / (2 * h);
            thp[k] = theta_[k];
            thm[k] = theta_[k];
        }
        return J;
    }
};

// deduce (Dim, F) from the wrapped functor: a fixed-size return type -> static Dim, VectorXd -> Dynamic
// (works for both plain f(t, y) and parameterized f(t, y, theta), see ode_rhs_dim)
template <typename F> ode_rhs_field(F&&) -> ode_rhs_field<ode_rhs_dim_v<F>, std::decay_t<F>>;
template <typename F>
ode_rhs_field(F&&, const Eigen::Matrix<double, Dynamic, 1>&) -> ode_rhs_field<ode_rhs_dim_v<F>, std::decay_t<F>>;

/* f1 + f2
f1 + f2 as a plain (t, y) functor delegating value and state Jacobian to the two operand fields. It is
used to form perturbed dynamics (e.g. the control forcing f + u); it never needs param_jacobian (the
parameter sensitivity is fed to the integrator separately). Because each operand field already resolves
analytic-vs-finite-difference internally, delegating inherits that for free.
*/
template <typename LHS, typename RHS>
struct ode_rhs_sum {
    using vector_t = Eigen::Matrix<double, Dynamic, 1>;
    using matrix_t = Eigen::Matrix<double, Dynamic, Dynamic>;
    static constexpr int dim = LHS::dim;   // == RHS::dim by construction
    LHS a;
    RHS b;
    vector_t operator()(double t, const vector_t& y) const { return a(t, y) + b(t, y); }
    matrix_t state_jacobian(double t, const vector_t& y) const {
        return a.state_jacobian(t, y) + b.state_jacobian(t, y);
    }
};

// field algebra: g = f1 + f2, an ode_rhs_field wrapping the delegating ode_rhs_sum. The result is a new
// field type (its F is the sum functor) -- only ever a temporary fed to the integrator, never stored.
template <int Dim, typename F1, typename F2>
auto operator+(const ode_rhs_field<Dim, F1>& lhs, const ode_rhs_field<Dim, F2>& rhs) {
    using sum_t = ode_rhs_sum<ode_rhs_field<Dim, F1>, ode_rhs_field<Dim, F2>>;
    return ode_rhs_field<Dim, sum_t>(sum_t {lhs, rhs});
}

}   // namespace fdapde

#endif   //__ODE_RHS_H__
