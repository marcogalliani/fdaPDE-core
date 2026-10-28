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

#ifndef __FDAPDE_LEVENBERG_MARQUARDT_H__
#define __FDAPDE_LEVENBERG_MARQUARDT_H__

#include "header_check.h"

namespace fdapde {

/* Levenberg-Marquardt for least squares:  minimise  S(x) = ||r(x)||^2.
================================================================================================
DIFFERENT PROTOCOL FROM THE OTHER OPTIMIZERS HERE. BFGS, TrustRegion and the rest take a scalar
objective and its gradient. This one takes the RESIDUAL VECTOR and its Jacobian,

    problem.residuals(x) -> r      (n_res)          problem.jacobian(x) -> J = dr/dx   (n_res x n)

because that is what makes it Levenberg-Marquardt rather than a damped quasi-Newton: the curvature
model J^T J is assembled from first derivatives alone, exactly, instead of accumulated from secant
pairs. A caller that has only f and grad f wants TrustRegion.

THE STEP. Each iteration solves the damped normal equations

    (J^T J + mu D) p = -J^T r ,        D = diag(J^T J)   (Marquardt)  or  I  (Levenberg),

which interpolates between Gauss-Newton (mu -> 0) and steepest descent with step 1/mu (mu -> inf),
and is accepted or rejected on the gain ratio

    rho = [ S(x) - S(x+p) ] / p^T ( mu D p - J^T r ) ,

the denominator being the reduction the quadratic model predicted (always > 0 for the p above). mu
follows Nielsen's update: on success mu *= max(1/3, 1 - (2 rho - 1)^3) and the growth factor resets;
on failure mu *= nu, nu *= 2. A non-finite S(x+p) is simply a rejection -- as in TrustRegion, no
sentinel cost is needed for the method to back away from a region where the model is undefined.

WHY THE DIAGONAL SCALING MATTERS HERE. `D = diag(J^T J)` is Marquardt's own modification and it is the
reason this is worth having on badly conditioned problems: damping with I adds the SAME ridge to every
direction, so it is only meaningful when the columns of J are comparably scaled, whereas diag(J^T J)
damps each direction in proportion to the curvature already present -- i.e. it is invariant to a
rescaling of the parameters. On a criterion with a long flat valley the undamped Gauss-Newton step
shoots along the valley floor and overshoots; the damping shortens it in the flat directions
specifically, which is what lets LM crawl along a valley that a line search oscillates across.

WHAT IT DOES NOT DO. The damping regularises the STEP, not the criterion: it changes the path, never
the location of the minimum. On a problem whose parameters are genuinely redundant, LM reaches the same
flat valley floor the other optimizers reach, faster and more reliably -- and the estimates are just as
dispersed along the valley. Ill-CONDITIONING is an optimizer problem; non-IDENTIFIABILITY is not.

The reference is Madsen, Nielsen & Tingleff, "Methods for Non-Linear Least Squares Problems" (2004),
sections 3.2 and 3.3, with the objective written as ||r||^2 rather than their ||r||^2/2 so that a caller
can hand this class the same tolerance it hands BFGS.
*/
template <int N> class LevenbergMarquardt {
   private:
    using vector_t = std::conditional_t<N == Dynamic, Eigen::Matrix<double, Dynamic, 1>, Eigen::Matrix<double, N, 1>>;
    using square_t =
      std::conditional_t<N == Dynamic, Eigen::Matrix<double, Dynamic, Dynamic>, Eigen::Matrix<double, N, N>>;
    using jacobian_t = Eigen::Matrix<double, Dynamic, N>;

    vector_t optimum_;
    double value_ = std::numeric_limits<double>::quiet_NaN();
    int n_iter_ = 0;
    int n_rejected_ = 0;
    bool converged_ = false;
    bool nonfinite_ = false;   // the residuals or the Jacobian at the STARTING point were not usable
    std::vector<double> values_;

    int max_iter_;
    double tol_;               // on ||grad S|| = ||2 J^T r||, so it matches the other optimizers' tol
    double tau_;               // mu_0 = tau * max(diag(J^T J))
    double step_tol_ = 1e-14;  // give up when the accepted step is this small relative to x
    double mu_max_ = 1e16;     // give up when damping this heavy still fails: no usable step exists
    bool marquardt_ = true;    // D = diag(J^T J); false gives plain Levenberg, D = I

   public:
    static constexpr bool gradient_free = false;
    static constexpr int static_input_size = N;

    LevenbergMarquardt() : max_iter_(500), tol_(1e-6), tau_(1e-3) { }
    LevenbergMarquardt(int max_iter, double tol, double tau = 1e-3) :
        max_iter_(max_iter), tol_(tol), tau_(tau) { }
    // D = diag(J^T J) (the default) or D = I
    void use_marquardt_scaling(bool on) { marquardt_ = on; }
    void set_step_tolerance(double t) { step_tol_ = t; }

    template <typename ProblemT, typename... Callbacks>
    vector_t optimize(ProblemT&& problem, const vector_t& x0, Callbacks&&... callbacks) {
        std::tuple<Callbacks...> callbacks_ {callbacks...};
        const int size = (N == Dynamic) ? x0.rows() : N;

        vector_t x = x0;
        Eigen::Matrix<double, Dynamic, 1> r = problem.residuals(x);
        jacobian_t J = problem.jacobian(x);
        double S = r.squaredNorm();
        n_iter_ = 0;
        n_rejected_ = 0;
        values_.clear();
        // a starting point the model cannot even be built at is not something damping can fix
        if (!r.allFinite() || !J.allFinite() || !std::isfinite(S)) {
            nonfinite_ = true;
            converged_ = false;
            optimum_ = x;
            value_ = S;
            return optimum_;
        }
        values_.push_back(S);
        square_t A = J.transpose() * J;
        vector_t g = J.transpose() * r;          // = grad S / 2
        double mu = tau_ * A.diagonal().maxCoeff();
        if (!(mu > 0)) { mu = tau_; }            // J = 0 at the start: nothing to scale by
        double nu = 2.0;
        bool stop = (2.0 * g.norm() <= tol_);

        while (n_iter_ < max_iter_ && !stop && mu < mu_max_) {
            // damped normal equations. D is rebuilt each iteration: the curvature the damping is scaled
            // against is the one at the CURRENT point, not at the start.
            square_t D = square_t::Zero(size, size);
            if (marquardt_) {
                for (int i = 0; i < size; ++i) { D(i, i) = std::max(A(i, i), 1e-12 * A.diagonal().maxCoeff()); }
            } else {
                D = square_t::Identity(size, size);
            }
            Eigen::LDLT<square_t> ldlt(A + mu * D);
            const vector_t p = ldlt.solve(-g);
            if (ldlt.info() != Eigen::Success || !p.allFinite()) {   // heavier damping will make it solvable
                ++n_rejected_;
                mu *= nu;
                nu *= 2.0;
                n_iter_++;
                continue;
            }
            if (p.norm() <= step_tol_ * (x.norm() + step_tol_)) { break; }   // converged on the step

            const vector_t x_trial = x + p;
            const Eigen::Matrix<double, Dynamic, 1> r_trial = problem.residuals(x_trial);
            const double S_trial = r_trial.allFinite() ? r_trial.squaredNorm()
                                                       : std::numeric_limits<double>::infinity();
            // predicted reduction of ||r||^2 (see the class comment); > 0 whenever p solves the system
            const double pred = p.dot(mu * (D * p) - g);
            const double rho = (std::isfinite(S_trial) && pred > 0) ? (S - S_trial) / pred
                                                                    : -std::numeric_limits<double>::infinity();
            if (rho > 0) {
                const jacobian_t J_trial = problem.jacobian(x_trial);
                if (J_trial.allFinite()) {
                    x = x_trial;
                    r = r_trial;
                    J = J_trial;
                    S = S_trial;
                    A = J.transpose() * J;
                    g = J.transpose() * r;
                    values_.push_back(S);
                    mu *= std::max(1.0 / 3.0, 1.0 - std::pow(2.0 * rho - 1.0, 3));
                    nu = 2.0;
                    stop = (2.0 * g.norm() <= tol_);
                } else {
                    // the residuals improved but the model cannot be rebuilt there: treat as a rejection
                    ++n_rejected_;
                    mu *= nu;
                    nu *= 2.0;
                }
            } else {
                ++n_rejected_;
                mu *= nu;
                nu *= 2.0;
            }
            n_iter_++;
            stop |= internals::exec_stop_if(*this, problem);
        }
        converged_ = g.allFinite() && 2.0 * g.norm() <= tol_;
        nonfinite_ = !g.allFinite();
        optimum_ = x;
        value_ = S;
        return optimum_;
    }
    // observers, named as in BFGS / TrustRegion so a caller can swap the three
    const vector_t& optimum() const { return optimum_; }
    double value() const { return value_; }          // ||r||^2 at the optimum
    int n_iter() const { return n_iter_; }
    int n_rejected() const { return n_rejected_; }
    bool converged() const { return converged_; }
    bool gradient_nonfinite() const { return nonfinite_; }
    int max_iter() const { return max_iter_; }
    const std::vector<double>& values() const { return values_; }
};

}   // namespace fdapde

#endif   // __FDAPDE_LEVENBERG_MARQUARDT_H__
