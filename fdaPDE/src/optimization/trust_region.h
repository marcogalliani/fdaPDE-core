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

#ifndef __FDAPDE_TRUST_REGION_H__
#define __FDAPDE_TRUST_REGION_H__

#include "header_check.h"

namespace fdapde {

// Trust-region quasi-Newton (BFGS Hessian approximation, dogleg step).
//
// WHY, next to BFGS + line search. A line search commits to a direction and then looks for a step along it:
// it cannot reject the direction, only shorten it, and the Wolfe implementation here grows alpha while it
// has not bracketed a minimum. On criteria with a region where the dynamics diverge -- every ODE inverse
// problem in this library -- that is how a search leaves the region the model is defined on, in one step,
// and comes back with a non-finite objective or gradient.
//
// A trust region inverts the order: it fixes how far it is willing to move (the radius Delta), minimizes a
// local quadratic model inside that ball, and then asks whether the objective agreed with the model,
//
//     rho = [ f(x) - f(x + p) ] / [ m(0) - m(p) ] ,
//
// shrinking Delta when it did not and growing it when it did. A step into the divergent region simply
// yields rho <= 0 (or a non-finite f) and is REJECTED, with the radius halved -- no sentinel cost, no
// zeroed gradient, no poisoned secant pair. The guards that a line-search version needs (cap the first
// step, sanitize NaN gradients, keep the best point evaluated) are all subsumed by that one test.
//
// Minimal by design: BFGS update for the model Hessian B (skipped when the curvature pair is unusable),
// dogleg step (Cauchy point when B is not positive definite along the direction), and the standard radius
// rules. No Steihaug-CG, no scaling heuristics -- p is small in these problems.
template <int N> class TrustRegion {
   private:
    using vector_t = std::conditional_t<N == Dynamic, Eigen::Matrix<double, Dynamic, 1>, Eigen::Matrix<double, N, 1>>;
    using matrix_t =
      std::conditional_t<N == Dynamic, Eigen::Matrix<double, Dynamic, Dynamic>, Eigen::Matrix<double, N, N>>;

    vector_t optimum_;
    double value_ = std::numeric_limits<double>::quiet_NaN();
    int n_iter_ = 0;
    int n_rejected_ = 0;      // steps the model test threw away (includes non-finite evaluations)
    bool converged_ = false;
    bool nonfinite_ = false;  // the gradient at the STARTING point was not finite: nothing to optimize
    std::vector<double> values_;

    int max_iter_;
    double tol_;              // on the gradient norm
    double delta0_;           // initial radius, in the optimizer's own variables
    double delta_max_;
    double delta_min_;        // give up when the region is this small: no usable step exists
    static constexpr double eta_ = 0.1;   // accept a step when rho exceeds this

    // dogleg: the minimizer of the model over the ball of radius delta, along the path Cauchy -> Newton
    vector_t dogleg_(const vector_t& g, const matrix_t& B, double delta) const {
        const double gBg = g.dot(B * g);
        const vector_t pu = (gBg > 0) ? vector_t(-(g.squaredNorm() / gBg) * g)   // unconstrained Cauchy point
                                      : vector_t(-(delta / g.norm()) * g);      // no curvature: go to the border
        if (!(gBg > 0)) { return pu; }
        if (pu.norm() >= delta) { return vector_t((delta / pu.norm()) * pu); }
        // Newton step, if the model is convex enough to have one
        Eigen::LLT<matrix_t> llt(B);
        if (llt.info() != Eigen::Success) { return vector_t((delta / pu.norm()) * pu); }
        const vector_t pb = llt.solve(-g);
        if (!pb.allFinite()) { return vector_t((delta / pu.norm()) * pu); }
        if (pb.norm() <= delta) { return pb; }
        // the dogleg leg: pu + t (pb - pu) with || . || = delta, t in [0,1]
        const vector_t d = pb - pu;
        const double a = d.squaredNorm(), b = 2.0 * pu.dot(d), c = pu.squaredNorm() - delta * delta;
        const double disc = b * b - 4 * a * c;
        if (!(a > 0) || disc < 0) { return vector_t((delta / pu.norm()) * pu); }
        const double t = (-b + std::sqrt(disc)) / (2 * a);
        return vector_t(pu + std::clamp(t, 0.0, 1.0) * d);
    }
   public:
    static constexpr bool gradient_free = false;
    static constexpr int static_input_size = N;

    TrustRegion() : max_iter_(500), tol_(1e-6), delta0_(1.0), delta_max_(1e3), delta_min_(1e-12) { }
    TrustRegion(int max_iter, double tol, double delta0) :
        max_iter_(max_iter), tol_(tol), delta0_(delta0), delta_max_(1e3 * delta0), delta_min_(1e-12) { }
    void set_radius_bounds(double delta_min, double delta_max) {
        delta_min_ = delta_min;
        delta_max_ = delta_max;
    }

    // `objective` follows the same protocol as the other optimizers here: operator()(x) and .gradient().
    // Callbacks are accepted and their stop_if/grad hooks honoured, so a driver can trace a run; there is
    // no adapt hook to run, the radius rule replaces the line search.
    template <typename ObjectiveT, typename... Callbacks>
    vector_t optimize(ObjectiveT&& objective, const vector_t& x0, Callbacks&&... callbacks) {
        std::tuple<Callbacks...> callbacks_ {callbacks...};
        auto grad = objective.gradient();
        const int size = (N == Dynamic) ? x0.rows() : N;

        vector_t x = x0, g = grad(x);
        double f = objective(x);
        matrix_t B = matrix_t::Identity(size, size);
        double delta = delta0_;
        n_iter_ = 0;
        n_rejected_ = 0;
        values_.clear();
        values_.push_back(f);
        bool stop = false;
        // a non-finite starting point is not something a radius can fix
        if (!std::isfinite(f) || !g.allFinite()) {
            nonfinite_ = true;
            optimum_ = x;
            value_ = f;
            return optimum_;
        }
        while (n_iter_ < max_iter_ && g.norm() > tol_ && delta > delta_min_ && !stop) {
            const vector_t p = dogleg_(g, B, delta);
            const double pred = -(g.dot(p) + 0.5 * p.dot(B * p));   // model decrease, > 0 for a descent step
            const vector_t x_trial = x + p;
            const double f_trial = objective(x_trial);
            // REJECT anything the model cannot account for, including a non-finite evaluation: that is the
            // whole point of the method here. No sentinel cost is needed for the search to back off.
            const double rho = (std::isfinite(f_trial) && pred > 0) ? (f - f_trial) / pred
                                                                    : -std::numeric_limits<double>::infinity();
            if (rho < 0.25) {
                delta *= 0.25;
            } else if (rho > 0.75 && p.norm() > 0.99 * delta) {
                delta = std::min(2 * delta, delta_max_);
            }
            if (rho > eta_) {
                const vector_t g_new = grad(x_trial);
                if (g_new.allFinite()) {
                    const vector_t s = p, y = g_new - g;
                    const double sy = s.dot(y);
                    // same curvature test as the BFGS class: skip the update rather than divide by ~0
                    if (std::isfinite(sy) && sy > 1e-12 * s.norm() * y.norm()) {
                        const vector_t Bs = B * s;
                        B += (y * y.transpose()) / sy - (Bs * Bs.transpose()) / s.dot(Bs);
                    }
                    x = x_trial;
                    f = f_trial;
                    g = g_new;
                    values_.push_back(f);
                } else {
                    // the objective improved but the gradient there is unusable: treat as a rejection
                    ++n_rejected_;
                    delta *= 0.25;
                }
            } else {
                ++n_rejected_;
            }
            n_iter_++;
            stop |= internals::exec_stop_if(*this, objective);
        }
        converged_ = g.allFinite() && g.norm() <= tol_;
        nonfinite_ = !g.allFinite();
        optimum_ = x;
        value_ = f;
        return optimum_;
    }
    // observers, named as in BFGS so a caller can swap the two
    const vector_t& optimum() const { return optimum_; }
    double value() const { return value_; }
    int n_iter() const { return n_iter_; }
    int n_rejected() const { return n_rejected_; }
    bool converged() const { return converged_; }
    bool gradient_nonfinite() const { return nonfinite_; }
    int max_iter() const { return max_iter_; }
    const std::vector<double>& values() const { return values_; }
};

}   // namespace fdapde

#endif   // __FDAPDE_TRUST_REGION_H__
