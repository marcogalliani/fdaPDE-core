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

#ifndef __FDAPDE_AD_ODE_RHS_H__
#define __FDAPDE_AD_ODE_RHS_H__

/* Automatic differentiation of an ODE right-hand side

An ODE rhs enters the library through ode_rhs_field, which asks the wrapped functor for state_jacobian
(df/dy) and, for a theta-parameterized field, param_jacobian (df/dtheta), and central-differences them
when the functor does not expose them. Both alternatives cost the user something: the analytic route is
error-prone hand calculus that has to be redone at every change of the model, the finite-difference route
is inexact (it sets the accuracy floor of every sensitivity built on top of it) and, for an implicit
scheme, sits on the hot path of a Newton stage solve.

ad_ode_rhs is the third route: the user writes the VALUE ONLY, generic in the scalar type, and the
adapter derives both Jacobians by forward-mode automatic differentiation (autodiff::dual). The result is
exact to machine precision at the same asymptotic cost as central differences (one sweep per input:
d sweeps for df/dy, n_theta for df/dtheta -- vs 2d and 2*n_theta *field evaluations* for central
differences, so it is in fact the cheaper of the two).

The adapter is a drop-in: because it EXPOSES state_jacobian / param_jacobian as ordinary methods, the
existing ode_rhs_field concepts see an analytically-differentiable functor and route to it, with no
change anywhere downstream (integrator, controlled solver, the smoothing / estimation solvers).

    // the user's model: one templated operator(), no derivatives
    struct fhn {
        template <typename Scalar>
        Eigen::Matrix<Scalar, 2, 1> operator()(
          double, const Eigen::Matrix<Scalar, Dynamic, 1>& y, const Eigen::Matrix<Scalar, Dynamic, 1>& th) const {
            Eigen::Matrix<Scalar, 2, 1> o;
            o[0] = th[2] * (y[0] - y[0] * y[0] * y[0] / 3.0 + y[1]);
            o[1] = -(y[0] - th[0] + th[1] * y[1]) / th[2];
            return o;
        }
    };
    ode_rhs_field field(ad_ode_rhs {fhn {}}, theta);   // df/dy and df/dtheta now come from autodiff
    field.state_jacobian(t, y);                        // exact, not central differences

REQUIRED SIGNATURE. The functor must be callable at BOTH double and autodiff::dual, i.e. templated on the
scalar type (or on the vector type). Time stays a plain double: nothing differentiates w.r.t. t. A
parameterized functor takes (t, y, theta) and gets both Jacobians; a plain one takes (t, y) and gets
state_jacobian alone.

STATIC DIMENSION. operator() forwards to the functor at Scalar = double, so the RETURN TYPE is the
functor's own: a fixed-size return keeps ode_rhs_dim static (and with it the integrator's heap-free stage
math, see ode_rhs.h), a VectorXd return keeps the dynamic path -- the adapter is transparent to that
choice. Note that the *stage math* is always in double: AD is applied to the field only, never through
the integrator, so the propagation of these Jacobians is the library's existing (double) sensitivity
machinery.
*/

#include "header_check.h"

namespace fdapde {

namespace internals {

// first-order forward-mode scalar and the vector type the seeded sweeps run on. First order is all the
// rhs Jacobians need; the second-order type (dual2nd) is used only by ad_objective's hessian().
using ad_dual_t = autodiff::dual;
using ad_vector_t = Eigen::Matrix<ad_dual_t, Dynamic, 1>;

// double -> AD vector lift, elementwise (not Eigen's cast<>, which would require a NumTraits-level
// conversion the AD scalar does not advertise)
template <typename Scalar> Eigen::Matrix<Scalar, Dynamic, 1> ad_lift(const Eigen::Matrix<double, Dynamic, 1>& x) {
    Eigen::Matrix<Scalar, Dynamic, 1> out(x.size());
    for (int i = 0; i < x.size(); ++i) { out[i] = x[i]; }
    return out;
}

}   // namespace internals

// f is a theta-parameterized rhs differentiable by AD: a valid f(t, y, theta) at double (so it is an
// ode rhs at all) which is ALSO callable at the AD scalar (so the Jacobians can be taken).
template <typename F>
concept is_ad_parameterized_ode_rhs =
  is_parameterized_ode_rhs<F> &&
  std::is_invocable_v<const F&, double, const internals::ad_vector_t&, const internals::ad_vector_t&>;
// f is a plain rhs f(t, y) differentiable by AD
template <typename F>
concept is_ad_ode_rhs = is_ode_rhs<F> && std::is_invocable_v<const F&, double, const internals::ad_vector_t&>;

/* AD-differentiated rhs wrapper
Stores the user's functor by value and adds the two Jacobian methods on top of it. Everything else --
the value, the return type, the system dimension -- passes through unchanged.
*/
template <typename F> class ad_ode_rhs {
   public:
    using vector_t = Eigen::Matrix<double, Dynamic, 1>;
    using matrix_t = Eigen::Matrix<double, Dynamic, Dynamic>;
    // a parameterized functor gets both Jacobians, a plain one only df/dy
    static constexpr bool parameterized = is_ad_parameterized_ode_rhs<F>;
    fdapde_static_assert(
      parameterized || is_ad_ode_rhs<F>,
      AD_ODE_RHS_REQUIRES_A_FUNCTOR_CALLABLE_AT_BOTH_DOUBLE_AND_AUTODIFF_DUAL__TEMPLATE_IT_ON_THE_SCALAR_TYPE);

    ad_ode_rhs() = default;
    explicit ad_ode_rhs(F f) : f_(std::move(f)) { }

    // value: the functor at Scalar = double, forwarded verbatim (return type, and hence ode_rhs_dim, preserved)
    auto operator()(double t, const vector_t& y, const vector_t& theta) const
        requires(parameterized) {
        return f_(t, y, theta);
    }
    auto operator()(double t, const vector_t& y) const
        requires(!parameterized) {
        return f_(t, y);
    }

    // df/dy (d x d): one seeded forward sweep per state component, at the current (y, theta)
    matrix_t state_jacobian(double t, const vector_t& y, const vector_t& theta) const
        requires(parameterized) {
        auto y_ad = internals::ad_lift<internals::ad_dual_t>(y);
        auto th_ad = internals::ad_lift<internals::ad_dual_t>(theta);
        auto g = [&](const internals::ad_vector_t& y_, const internals::ad_vector_t& th_) { return f_(t, y_, th_); };
        decltype(g(y_ad, th_ad)) out;   // the functor's return type at the AD scalar (see note below)
        return autodiff::jacobian(g, autodiff::wrt(y_ad), autodiff::at(y_ad, th_ad), out);
    }
    matrix_t state_jacobian(double t, const vector_t& y) const
        requires(!parameterized) {
        auto y_ad = internals::ad_lift<internals::ad_dual_t>(y);
        auto g = [&](const internals::ad_vector_t& y_) { return f_(t, y_); };
        decltype(g(y_ad)) out;
        return autodiff::jacobian(g, autodiff::wrt(y_ad), autodiff::at(y_ad), out);
    }
    // df/dtheta (d x n_theta): one seeded forward sweep per parameter
    matrix_t param_jacobian(double t, const vector_t& y, const vector_t& theta) const
        requires(parameterized) {
        auto y_ad = internals::ad_lift<internals::ad_dual_t>(y);
        auto th_ad = internals::ad_lift<internals::ad_dual_t>(theta);
        auto g = [&](const internals::ad_vector_t& y_, const internals::ad_vector_t& th_) { return f_(t, y_, th_); };
        decltype(g(y_ad, th_ad)) out;
        return autodiff::jacobian(g, autodiff::wrt(th_ad), autodiff::at(y_ad, th_ad), out);
    }

    const F& functor() const { return f_; }
    F& functor() { return f_; }
   private:
    // NOTE. The type autodiff::jacobian writes the seeded evaluation into -- the functor's return type at
    // the AD scalar, fixed- or dynamic-size alike -- is spelled INSIDE each method (decltype of the call)
    // rather than as a member alias: a member alias is instantiated with the class, so the plain-rhs
    // spelling would be a hard error on a parameterized functor and vice versa.
    F f_ {};
};

template <typename F> ad_ode_rhs(F&&) -> ad_ode_rhs<std::decay_t<F>>;
// f -> AD-differentiated rhs (spelling-free alternative to the deduction guide, e.g. in a return statement)
template <typename F> auto make_ad_ode_rhs(F&& f) { return ad_ode_rhs<std::decay_t<F>>(std::forward<F>(f)); }
// f -> ready-to-integrate field with AD Jacobians. The system dimension is deduced from the functor's
// return type exactly as for a hand-differentiated field (see ode_rhs_field's deduction guides).
template <typename F> auto make_ad_ode_rhs_field(F&& f) {
    using G = ad_ode_rhs<std::decay_t<F>>;
    return ode_rhs_field<ode_rhs_dim_v<G>, G>(G(std::forward<F>(f)));
}
template <typename F> auto make_ad_ode_rhs_field(F&& f, const Eigen::Matrix<double, Dynamic, 1>& theta) {
    using G = ad_ode_rhs<std::decay_t<F>>;
    return ode_rhs_field<ode_rhs_dim_v<G>, G>(G(std::forward<F>(f)), theta);
}

}   // namespace fdapde

#endif   // __FDAPDE_AD_ODE_RHS_H__
