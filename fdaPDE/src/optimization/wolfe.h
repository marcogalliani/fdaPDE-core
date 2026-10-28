
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

#ifndef __FDAPDE_WOLFE_LINE_SEARCH_H__
#define __FDAPDE_WOLFE_LINE_SEARCH_H__

#include "header_check.h"

namespace fdapde {

// wolfe line search method for adaptive step size
template <int MaxIter = 10>
class WolfeLineSearch {
   private:
    double alpha_ = 1.0;
    double alpha_max_ = std::numeric_limits<double>::infinity(), alpha_min_ = 0;
    double c1_ = 1e-4, c2_ = 0.9;
   public:
    WolfeLineSearch() = default;
    WolfeLineSearch(double alpha, double c1, double c2) : alpha_(alpha), c1_(c1), c2_(c2) { }

    // bisection method for the weak Wolfe conditions. check "Jorge Nocedal, Stephen J. Wright (2006), Numerical
    // Optimization, pag 58"
    template <typename Opt, typename Obj> bool adapt_hook(Opt& opt, Obj& obj) {
        fdapde_static_assert(is_gradient_based_opt_v<Opt>, THIS_METHOD_IS_FOR_GRADIENT_BASED_OPTIMIZATION_ONLY);
        double grad_0 = opt.grad_old.dot(opt.update), val_0 = obj(opt.x_old);
        auto grad = obj.gradient();

        bool zooming = false;
        double c1 = c1_, c2 = c2_;
        double alpha_max = alpha_max_, alpha_min = alpha_min_;
        double alpha_prev = 0;
        double alpha_curr = alpha_;
        double val_prev = val_0;
        // the largest step tried that met the Armijo (sufficient decrease) test, and the objective there.
        // Without this the loop can run out of iterations and hand back `alpha_curr` -- a step it has just
        // REJECTED -- which on a criterion with a divergent region means accepting a step into it.
        double alpha_ok = 0;
        double val_at_alpha_max = std::numeric_limits<double>::quiet_NaN();   // evaluated at most once

        for (int i = 0; i < MaxIter; ++i) {
            double val_curr = obj(opt.x_old + alpha_curr * opt.update);
            double grad_curr = grad(opt.x_old + alpha_curr * opt.update).dot(opt.update);
            if (std::isfinite(val_curr) && val_curr <= val_0 + c1 * grad_0 * alpha_curr && alpha_curr > alpha_ok) {
                alpha_ok = alpha_curr;
            }
            if (!zooming) {
                if (val_curr > val_0 + c1 * grad_0 * alpha_curr || (i > 0 && val_curr >= val_prev)) {
                    // upper bound found, zoom
                    alpha_min = alpha_prev;
                    alpha_max = alpha_curr;
                    val_at_alpha_max = std::numeric_limits<double>::quiet_NaN();
                    zooming = true;
                    continue;
                }
                if (std::abs(grad_curr) <= -c2 * grad_0) {
                    opt.h = alpha_curr;
                    return false;
                }
                if (grad_curr >= 0) {
                    // the function at alpha is increasing, zoom with alpha_min and alpha_max reversed
                    alpha_min = alpha_curr;
                    alpha_max = alpha_prev;
                    val_at_alpha_max = std::numeric_limits<double>::quiet_NaN();
                    zooming = true;
                    continue;
                }
            } else {
                if (!std::isfinite(val_at_alpha_max) && std::isfinite(alpha_max)) {
                    val_at_alpha_max = obj(opt.x_old + alpha_max * opt.update);
                }
                if (val_curr > val_0 + c1 * alpha_curr * grad_0 || val_curr > val_at_alpha_max) {
                    alpha_max = alpha_curr;
                    val_at_alpha_max = val_curr;
                } else {
                    if (std::abs(grad_curr) <= -c2 * grad_0) {
                        opt.h = alpha_curr;
                        return false;
                    }
                    if (grad_curr * (alpha_max - alpha_min) >= 0) {
                        alpha_max = alpha_min;
                        val_at_alpha_max = std::numeric_limits<double>::quiet_NaN();
                    }
                    alpha_min = alpha_curr;
                }
            }
            alpha_prev = alpha_curr;
            val_prev = val_curr;
            // proper upper bound not found, try by multiplying alpha_max by two
            alpha_curr = std::isinf(alpha_max) ? 2 * alpha_curr : (alpha_max + alpha_min) / 2;
        }
        // out of iterations without a point meeting both Wolfe conditions. Fall back on the largest step
        // that at least decreased the objective sufficiently. If there is none, take NO step (h = 0) and
        // ask the optimizer to stop: every step tried was rejected, so continuing means accepting one of
        // them. Returning true is the callbacks' stop signal.
        opt.h = alpha_ok;
        return alpha_ok == 0;
    }
};

}   // namespace fdapde

#endif   // __FDAPDE_WOLFE_LINE_SEARCH_H__
