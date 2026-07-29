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

#ifndef __FDAPDE_ODE_SOLVER_H__
#define __FDAPDE_ODE_SOLVER_H__

#include "header_check.h"

namespace fdapde {


/* General-purpose ODE solver
It bundles a right-hand side ode_rhs_field<Dim> with an RKIntegrator<Stages> and exposes the forward initial-value solve together with the local sensitivities of single forward and adjoint steps.
*/

template <int Stages, int Dim, typename F>
class ode_solver {
   public:
    using vector_t = Eigen::Matrix<double, Dynamic, 1>;
    using matrix_t = Eigen::Matrix<double, Dynamic, Dynamic>;

    ode_solver() = default;
    ode_solver(ode_rhs_field<Dim, F> field, RKIntegrator<Stages> integrator) :
        field_(std::move(field)), integrator_(std::move(integrator)) { }

    // Forward integration
    // single forward step y_{n+1} = y + dt * Phi over [t, t + dt]
    vector_t step(double t, const vector_t& y, double dt) const {
        return integrator_.step(field_, t, y, dt);
    }
    // full-horizon IVP solve on the grid `times` from y0: the nodal trajectory Y (m x d)
    matrix_t solve(const vector_t& times, const vector_t& y0) const {
        return integrator_.integrate(field_, times, y0);
    }

    // Forward-mode sensitivities
    // state (flow) Jacobian d y_{n+1}/d y = I + dt * dPhi/dy
    matrix_t flow_jacobian(double t, const vector_t& y, double dt) const {
        return integrator_.flow_jacobian(field_, t, y, dt);
    }
    // forward step together with its flow Jacobian, from a single stage solve
    rk_fwd_step_t step_with_flow_jacobian(double t, const vector_t& y, double dt) const {
        return integrator_.step_with_flow_jacobian(field_, t, y, dt);
    }
    // parameter sensitivity d y_{n+1}/d theta (d x n_theta) for theta-dependent dynamics bound into the
    // field; param_jacobian(t, y) -> R^{d x n_theta} is the parameter Jacobian of the rhs
    template <typename ParamJacobian>
    matrix_t step_param_jacobian(
      double t, const vector_t& y, double dt, const ParamJacobian& param_jacobian, int n_theta) const {
        return integrator_.step_param_jacobian(field_, t, y, dt, param_jacobian, n_theta);
    }
    // forward step with BOTH its state and parameter Jacobians, from a single stage solve
    template <typename ParamJacobian>
    rk_fwd_step_t step_with_state_param_jacobians(
      double t, const vector_t& y, double dt, const ParamJacobian& param_jacobian, int n_theta) const {
        return integrator_.step_with_state_param_jacobians(field_, t, y, dt, param_jacobian, n_theta);
    }

    // Reverse-mode sensitivity 
    // state-transition adjoint of one step: p_curr = (d y_{n+1}/d y_n)^T p_next
    vector_t adjoint_step(double t, const vector_t& y, double dt, const vector_t& p_next) const {
        return integrator_.adjoint_step(field_, t, y, dt, p_next);
    }
    // adjoint with parameter gradient: returns {costate, dC/dtheta} (see RKIntegrator::adjoint_step)
    template <typename ParamJacobian>
    rk_adj_step_t adjoint_step(
      double t, const vector_t& y, double dt, const vector_t& p_next, const ParamJacobian& param_jacobian) const {
        return integrator_.adjoint_step(field_, t, y, dt, p_next, param_jacobian);
    }

    // observers (mutable field() lets a consumer rebind parameters -- e.g. theta -- in place)
    const ode_rhs_field<Dim, F>& field() const { return field_; }
    ode_rhs_field<Dim, F>& field() { return field_; }
    const RKIntegrator<Stages>& integrator() const { return integrator_; }

   private:
    ode_rhs_field<Dim, F> field_;
    RKIntegrator<Stages> integrator_;
};

// deduce all three parameters from a ready-made field + integrator (used to build a solver over a
// derived field -- e.g. a control-forced sum -- without spelling out its functor type)
template <int Stages, int Dim, typename F>
ode_solver(ode_rhs_field<Dim, F>, RKIntegrator<Stages>) -> ode_solver<Stages, Dim, F>;

// build an ode_solver from a scheme (ButcherTableau) and any is_ode_rhs callable: the stage count comes
// from the tableau, the system dimension is deduced from the rhs (fixed-size return -> static Dim,
// VectorXd -> Dynamic), mirroring the ode_rhs_field deduction guide.
template <int Stages, typename F>
    requires(is_ode_rhs<F>)
auto make_ode_solver(const ButcherTableau<Stages>& tableau, F&& f) {
    constexpr int Dim = ode_rhs_dim_v<F>;
    using G = std::decay_t<F>;
    return ode_solver<Stages, Dim, G>(ode_rhs_field<Dim, G>(std::forward<F>(f)), RKIntegrator<Stages>(tableau));
}

}   // namespace fdapde

#endif   // __FDAPDE_ODE_SOLVER_H__
