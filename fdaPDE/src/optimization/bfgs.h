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

#ifndef __FDAPDE_BFGS_H__
#define __FDAPDE_BFGS_H__

#include "header_check.h"

namespace fdapde {

template <int N> class BFGS {
   private:
    using vector_t = std::conditional_t<N == Dynamic, Eigen::Matrix<double, Dynamic, 1>, Eigen::Matrix<double, N, 1>>;
    using matrix_t =
      std::conditional_t<N == Dynamic, Eigen::Matrix<double, Dynamic, Dynamic>, Eigen::Matrix<double, N, N>>;

    vector_t optimum_;
    double value_;                 // objective value at optimum
    int n_iter_ = 0;               // current iteration number
    // why the loop stopped. The stopping test below reads `error > tol_`, and a NON-FINITE error makes that
    // comparison false, so an objective whose gradient is NaN (routine when a search leaves the region where
    // the problem is defined) exits the loop exactly as a converged run does. Recording the reason keeps the
    // two apart: `converged()` is true only for a genuinely small gradient.
    bool converged_ = false;
    bool nonfinite_ = false;
    std::vector<double> values_;   // explored objective values during optimization
    matrix_t inv_hessian_;
  
    int max_iter_;     // maximum number of iterations before forced stop
    double tol_;       // tolerance on error before forced stop
    double step_;      // update step
   public:
    static constexpr bool gradient_free = false;
    static constexpr int static_input_size = N;
    vector_t x_old, x_new, update, grad_old, grad_new;
    double h;
    // constructor
    BFGS() : max_iter_(500), tol_(1e-5), step_(1e-2) { }
    BFGS(int max_iter, double tol, double step) : max_iter_(max_iter), tol_(tol), step_(step) { }
    BFGS(const BFGS& other) : max_iter_(other.max_iter_), tol_(other.tol_), step_(other.step_) { }
    BFGS& operator=(const BFGS& other) {
        max_iter_ = other.max_iter_;
        tol_ = other.tol_;
        step_ = other.step_;
        return *this;
    }
    template <typename ObjectiveT, typename... Callbacks>
    vector_t optimize(ObjectiveT&& objective, const vector_t& x0, Callbacks&&... callbacks) {
        fdapde_static_assert(
          std::is_same<decltype(std::declval<ObjectiveT>().operator()(vector_t())) FDAPDE_COMMA double>::value,
          INVALID_CALL_TO_OPTIMIZE__OBJECTIVE_FUNCTOR_NOT_CALLABLE_AT_VECTOR_TYPE);
        constexpr double NaN = std::numeric_limits<double>::quiet_NaN();
        std::tuple<Callbacks...> callbacks_ {callbacks...};
        bool stop = false;   // asserted true in case of forced stop
        double error = std::numeric_limits<double>::max();
        int size = N == Dynamic ? x0.rows() : N;
        auto grad = objective.gradient();
        h = step_;
        n_iter_ = 0;
        x_old = x0, x_new = vector_t::Constant(size, NaN);
        grad_old = grad(x_old), grad_new = vector_t::Constant(size, NaN);
        if constexpr (N == Dynamic) {   // inv_hessian approximated with identity matrix
            inv_hessian_ = matrix_t::Identity(size, size);
        } else {
            inv_hessian_ = matrix_t::Identity();
	}
	update = -inv_hessian_ * grad_old;
	stop |= internals::exec_grad_hooks(*this, objective, callbacks_);
        error = grad_old.norm();
        values_.push_back(objective(x_old));

        while (n_iter_ < max_iter_ && error > tol_ && std::isfinite(error) && !stop) {
            stop |= internals::exec_adapt_hooks(*this, objective, callbacks_);
            // update along descent direction
            x_new = x_old + h * update;
            grad_new = grad(x_new);
            // update inverse hessian approximation
            vector_t delta_x = x_new - x_old;
            vector_t delta_grad = grad_new - grad_old;
            double xg = delta_x.dot(delta_grad);
            // SKIP the secant update unless the curvature condition holds. The update divides by xg, and a
            // rejected step (h = 0), a repeated point or an evaluation in a region where the objective is a
            // flat sentinel with zero gradient all give xg ~ 0 -- which takes the inverse Hessian, and every
            // iterate after it, to NaN. Keeping the previous approximation is the standard remedy.
            const double curvature_floor = 1e-12 * delta_x.norm() * delta_grad.norm();
            if (std::isfinite(xg) && xg > curvature_floor && delta_x.allFinite() && delta_grad.allFinite()) {
                vector_t hx = inv_hessian_ * delta_grad;
                matrix_t U = (1 + (delta_grad.dot(hx)) / xg) * ((delta_x * delta_x.transpose()) / xg);
                matrix_t V = ((hx * delta_x.transpose() + delta_x * hx.transpose())) / xg;
                inv_hessian_ += (U - V);
            }
            // prepare next iteration
            update = -inv_hessian_ * grad_new;
            stop |=
              (internals::exec_grad_hooks(*this, objective, callbacks_) || internals::exec_stop_if(*this, objective));
            x_old = x_new;
            grad_old = grad_new;
            error = grad_new.norm();
            values_.push_back(objective(x_old));
            n_iter_++;
        }	
        nonfinite_ = !std::isfinite(error);
        converged_ = !nonfinite_ && error <= tol_;
        optimum_ = x_old;
        value_ = values_.back();
        return optimum_;
    }
    // observers
    const vector_t& optimum() const { return optimum_; }
    double value() const { return value_; }
    int n_iter() const { return n_iter_; }
    // true only when the loop stopped on a small gradient. Distinct from `n_iter() < max_iter()`, which is
    // also true when the loop stopped on a NaN gradient -- see gradient_nonfinite().
    bool converged() const { return converged_; }
    // true when the loop stopped because the gradient was not finite: the returned point is wherever the
    // search happened to be, and is not an optimum of anything.
    bool gradient_nonfinite() const { return nonfinite_; }
    int max_iter() const { return max_iter_; }
    const std::vector<double>& values() const { return values_; }
};

}   // namespace fdapde

#endif   // __FDAPDE_BFGS_H__
