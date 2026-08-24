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

#ifndef __FDAPDE_AD_OBJECTIVE_H__
#define __FDAPDE_AD_OBJECTIVE_H__

/* Automatic differentiation of an optimization objective

The optimization module consumes an objective through a fixed, minimal interface, identical across
GradientDescent / ConjugateGradient / BFGS / LBFGS (and Newton, which additionally asks for the Hessian):

    double     obj(x)         // value
    auto grad = obj.gradient();   grad(x) -> vector      // gradient, as a callable
    auto hess = obj.hessian();    hess(x) -> matrix      // Newton only

ad_objective supplies all three from a single scalar-generic value function, so a user objective needs no
hand-written derivatives to be optimized by any of the gradient-based algorithms (and none of them needs
to know it is being differentiated automatically -- the adapter satisfies the interface they already
expect, line searches included: WolfeLineSearch queries obj and obj.gradient() the same way).

    struct rosenbrock {
        template <typename Scalar> Scalar operator()(const Eigen::Matrix<Scalar, Dynamic, 1>& x) const {
            return (1 - x[0]) * (1 - x[0]) + 100 * (x[1] - x[0] * x[0]) * (x[1] - x[0] * x[0]);
        }
    };
    BFGS<Dynamic> opt(500, 1e-8, 1.0);
    opt.optimize(ad_objective<Dynamic, rosenbrock> {rosenbrock {}}, x0, WolfeLineSearch());

WHEN NOT TO USE IT. AD differentiates the code it is given, so it is the right tool exactly when the
objective IS that code. An objective whose value comes out of a solve -- a nested optimization, a
smoothing fit, a forward ODE integration -- must instead be differentiated by the sensitivity/adjoint
identity of that solve (the library's ODE solvers do this: their outer gradients are adjoint sweeps, and
AD enters one level below, on the vector field, see ad_ode_rhs). Differentiating the solver's iterations
would be both far more expensive and, at a non-converged iterate, not the gradient of the object of
interest.

COST. Forward mode: the gradient costs n seeded sweeps and the Hessian n(n+1)/2, so this is meant for
the low-dimensional parameter vectors the estimation solvers optimize over, not for large-scale problems
(there reverse mode -- autodiff::var -- is the appropriate tool, and a reverse-mode adapter would slot in
here the same way).
*/

#include "header_check.h"

namespace fdapde {

namespace internals {

// second-order forward-mode scalar, needed for the Hessian (dual2nd carries the second derivative that
// autodiff::hessian reads off each seeded pair)
using ad_dual2nd_t = autodiff::dual2nd;
using ad_vector2nd_t = Eigen::Matrix<ad_dual2nd_t, Dynamic, 1>;

}   // namespace internals

// f is a scalar objective differentiable by AD: callable at double (the value) and at the first-order AD
// scalar (the gradient)
template <typename F>
concept is_ad_objective =
  std::is_invocable_r_v<double, const F&, const Eigen::Matrix<double, Dynamic, 1>&> &&
  std::is_invocable_v<const F&, const internals::ad_vector_t&>;
// f additionally supports the second-order scalar, so a Hessian can be taken (Newton)
template <typename F>
concept is_ad_twice_differentiable_objective =
  is_ad_objective<F> && std::is_invocable_v<const F&, const internals::ad_vector2nd_t&>;

/* AD-differentiated objective wrapper
N is the static input size, mirroring the optimizers' own template parameter (Dynamic for a runtime-sized
parameter vector); it fixes the vector/matrix types gradient() and hessian() hand back, so they match
what the optimizer stores.
*/
template <int N, typename F> class ad_objective {
   public:
    using vector_t = std::conditional_t<N == Dynamic, Eigen::Matrix<double, Dynamic, 1>, Eigen::Matrix<double, N, 1>>;
    using matrix_t =
      std::conditional_t<N == Dynamic, Eigen::Matrix<double, Dynamic, Dynamic>, Eigen::Matrix<double, N, N>>;
    static constexpr int static_input_size = N;
    fdapde_static_assert(
      is_ad_objective<F>,
      AD_OBJECTIVE_REQUIRES_A_SCALAR_FUNCTOR_CALLABLE_AT_BOTH_DOUBLE_AND_AUTODIFF_DUAL__TEMPLATE_IT_ON_THE_SCALAR_TYPE);

    ad_objective() = default;
    explicit ad_objective(F f) : f_(std::move(f)) { }

    // value. Returns double exactly (not the functor's own scalar type): the optimizers static_assert on it.
    double operator()(const vector_t& x) const {
        return static_cast<double>(f_(Eigen::Matrix<double, Dynamic, 1>(x)));
    }
    // gradient, as the callable the optimizers bind once and evaluate per iterate
    auto gradient() const {
        return [this](const vector_t& x) -> vector_t {
            auto x_ad = internals::ad_lift<internals::ad_dual_t>(Eigen::Matrix<double, Dynamic, 1>(x));
            auto g = [this](const internals::ad_vector_t& x_) { return f_(x_); };
            decltype(g(x_ad)) out;
            return autodiff::gradient(g, autodiff::wrt(x_ad), autodiff::at(x_ad), out);
        };
    }
    // Hessian (Newton). Instantiated only if actually asked for, so a once-differentiable objective is
    // still a valid ad_objective for every gradient-based algorithm.
    auto hessian() const
        requires(is_ad_twice_differentiable_objective<F>) {
        return [this](const vector_t& x) -> matrix_t {
            auto x_ad = internals::ad_lift<internals::ad_dual2nd_t>(Eigen::Matrix<double, Dynamic, 1>(x));
            auto g = [this](const internals::ad_vector2nd_t& x_) { return f_(x_); };
            decltype(g(x_ad)) out;
            Eigen::Matrix<double, Dynamic, 1> grad;
            return autodiff::hessian(g, autodiff::wrt(x_ad), autodiff::at(x_ad), out, grad);
        };
    }

    const F& functor() const { return f_; }
    F& functor() { return f_; }
   private:
    F f_ {};
};

// f -> AD-differentiated objective (N defaults to a runtime-sized input, the estimation solvers' case)
template <int N = Dynamic, typename F> auto make_ad_objective(F&& f) {
    return ad_objective<N, std::decay_t<F>>(std::forward<F>(f));
}

}   // namespace fdapde

#endif   // __FDAPDE_AD_OBJECTIVE_H__
