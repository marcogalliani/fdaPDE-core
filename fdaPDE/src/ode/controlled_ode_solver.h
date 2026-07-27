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

#ifndef __FDAPDE_CONTROLLED_ODE_SOLVER_H__
#define __FDAPDE_CONTROLLED_ODE_SOLVER_H__

#include "header_check.h"

namespace fdapde {

// A constant additive control u expressed as an ODE rhs g(t, y) = u, with state Jacobian d g/d y = 0. The
// control-forced dynamics f + u are formed by adding this to the prior field through ode_rhs_field's
// field-field addition (the only place the f + u control specialization is expressed). The zero
// state_jacobian keeps the sum's state Jacobian analytic whenever the prior field's is.
struct constant_rhs {
    Eigen::Matrix<double, Dynamic, 1> u;
    Eigen::Matrix<double, Dynamic, 1> operator()(double, const Eigen::Matrix<double, Dynamic, 1>&) const {
        return u;
    }
    Eigen::Matrix<double, Dynamic, Dynamic> state_jacobian(double, const Eigen::Matrix<double, Dynamic, 1>& y) const {
        return Eigen::Matrix<double, Dynamic, Dynamic>::Zero(y.size(), y.size());
    }
};

// Control-aware ODE solver: the control-forced sibling of ode_solver -- literally an ode_solver applied to
// the dynamics forced by an additive control (g = f + u).
// controlled_ode_solver<Stages, Dim, F> bundles a prior field ode_rhs_field<Dim, F> with an
// RKIntegrator<Stages> (held as an ode_solver over the prior field) and exposes exactly the single-step
// operations an optimal-control solver needs, with the per-interval control entering as an additive forcing
// of the dynamics. The stage count Stages, the system dimension Dim and the rhs functor type F are all
// concrete here, so the wrapped integrator keeps its fixed-stage / fixed-dimension fast paths and inlines
// the rhs; the type-erased any_controlled_ode_solver then hides all three parameters so any consumer holding
// it is free of the Stages, Dim and F template parameters.
template <int Stages, int Dim, typename F>
struct controlled_ode_solver {
    using vector_t = Eigen::Matrix<double, Dynamic, 1>;
    using matrix_t = Eigen::Matrix<double, Dynamic, Dynamic>;

    controlled_ode_solver() = default;
    controlled_ode_solver(ode_rhs_field<Dim, F> field, RKIntegrator<Stages> integrator) :
        solver_(std::move(field), std::move(integrator)) { }

    // forced forward step: integrate field + u over [t, t + dt] (u = 0 recovers the prior dynamics)
    vector_t step(double t, const vector_t& y, double dt, const vector_t& u) const {
        return forced_solver_(u).step(t, y, dt);
    }
    // forced discrete adjoint of one step: u is the constant additive control on the interval, p_next
    // the incoming costate; returns {p_curr, dC/du}. The control is the special case of a parameter with
    // an identity Jacobian (d f/d u = I), so dC/du is the core adjoint's parameter gradient under that map.
    std::pair<vector_t, vector_t> adjoint_step(
      double t, const vector_t& y, double dt, const vector_t& p_next, const vector_t& u) const {
        return forced_solver_(u).adjoint_step(t, y, dt, p_next, identity_df_du_(y.size()));
    }
    // unforced step together with the flow Jacobian of the prior dynamics (no control); used by edf()
    std::pair<vector_t, matrix_t> step_with_flow_jacobian(double t, const vector_t& y, double dt) const {
        return solver_.step_with_flow_jacobian(t, y, dt);
    }
    // forced step together with its state (flow) and control Jacobians (control-aware; u = 0 recovers the
    // prior dynamics). The forward-mode primitive the full-space SQP solver assembles its KKT system from.
    // The control Jacobian d y_{n+1}/d u is the core parameter Jacobian under the identity map d f/d u = I.
    std::tuple<vector_t, matrix_t, matrix_t> step_with_jacobians(
      double t, const vector_t& y, double dt, const vector_t& u) const {
        const int d = y.size();
        return forced_solver_(u).step_with_state_param_jacobians(t, y, dt, identity_df_du_(d), d);
    }
    // forced parameter sensitivity: d y_{n+1}/d theta on the control-forced dynamics f + u for a
    // theta-dependent prior field. param_jacobian(t, y) -> R^{d x n_theta} is the parameter Jacobian of the
    // *unforced* rhs (the control u carries no theta), while the stage system is built on the forced
    // dynamics. This is the theta-analogue of step_with_jacobians' control block.
    template <typename ParamJacobian>
    matrix_t step_param_jacobian(
      double t, const vector_t& y, double dt, const vector_t& u, const ParamJacobian& param_jacobian,
      int n_theta) const {
        return forced_solver_(u).step_param_jacobian(t, y, dt, param_jacobian, n_theta);
    }

    // Parametric operations (meaningful when the prior field is theta-parameterized; the type-erased
    // any_controlled_ode_solver carries them for every field, so they are guarded to compile and no-op /
    // assert on a non-parametric field, which never calls them).
    // rebind the current parameter vector theta of the prior dynamics in place
    void set_theta(const vector_t& theta) {
        if constexpr (ode_rhs_field<Dim, F>::is_parametric()) { solver_.field().set_theta(theta); }
    }
    // number of parameters theta (0 for a non-parametric field)
    int n_params() const { return solver_.field().n_params(); }
    // d y_{n+1}/d theta on the control-forced dynamics f + u, using the field's own parameter Jacobian. The
    // concrete (non-templated) counterpart of step_param_jacobian, so it can cross the type-erasure boundary.
    matrix_t param_jacobian(double t, const vector_t& y, double dt, const vector_t& u) const {
        if constexpr (ode_rhs_field<Dim, F>::is_parametric()) {
            return step_param_jacobian(
              t, y, dt, u, [this](double tt, const vector_t& yy) { return solver_.field().param_jacobian(tt, yy); },
              solver_.field().n_params());
        } else {
            fdapde_assert(false && "param_jacobian called on a non-parametric controlled_ode_solver");
            return matrix_t {};
        }
    }

    // the prior (unforced) field; the mutable overload lets a parametric consumer rebind theta in place
    const ode_rhs_field<Dim, F>& field() const { return solver_.field(); }
    ode_rhs_field<Dim, F>& field() { return solver_.field(); }

   private:
    // an ode_solver over the prior dynamics forced by the constant additive control u: f + u, formed as
    // ode_rhs_field field-field addition (the constant field carries d(u)/dy = 0, so the sum keeps f's
    // analytic Jacobians). The RKIntegrator is a fixed-size tableau, so this per-call solver is a trivially
    // copied stack temporary handed straight to the forced-dynamics step -- the same object the control-free
    // ode_solver drives, only over f + u instead of f.
    auto forced_solver_(const vector_t& u) const {
        return ode_solver(
          solver_.field() + ode_rhs_field<Dim, constant_rhs>(constant_rhs {u}), solver_.integrator());
    }
    // the additive control enters the dynamics as g = f + u, i.e. as a parameter u with d g/d u = I. This
    // is the identity parameter map the core integrator consumes to yield the control Jacobian / dC/du,
    // the only place the f + u control specialization is expressed.
    static auto identity_df_du_(int d) {
        return [d](double, const vector_t&) -> matrix_t { return matrix_t::Identity(d, d); };
    }

    ode_solver<Stages, Dim, F> solver_;   // the control-free solver over the prior (unforced) dynamics
};

// type-erased control-aware solver: the solver-facing interface with Stages, Dim and F erased. Beyond the
// control-aware step/adjoint/jacobian primitives it also carries the parametric operations (set_theta,
// n_params, param_jacobian) so a single erased engine serves both the forward optimal-control solver and the
// parameter-estimation solver; on a non-parametric field the parametric methods are simply never called.
struct IControlledOdeSolver {
    using vector_t = Eigen::Matrix<double, Dynamic, 1>;
    using matrix_t = Eigen::Matrix<double, Dynamic, Dynamic>;
    template <typename T> using fn_ptrs =
      fdapde::bindings<&T::step, &T::adjoint_step, &T::step_with_flow_jacobian, &T::step_with_jacobians,
                       &T::set_theta, &T::n_params, &T::param_jacobian>;
    vector_t step(double t, const vector_t& y, double dt, const vector_t& u) const {
        return fdapde::invoke<vector_t, 0>(*this, t, y, dt, u);
    }
    std::pair<vector_t, vector_t> adjoint_step(
      double t, const vector_t& y, double dt, const vector_t& p_next, const vector_t& u) const {
        return fdapde::invoke<std::pair<vector_t, vector_t>, 1>(*this, t, y, dt, p_next, u);
    }
    std::pair<vector_t, matrix_t> step_with_flow_jacobian(double t, const vector_t& y, double dt) const {
        return fdapde::invoke<std::pair<vector_t, matrix_t>, 2>(*this, t, y, dt);
    }
    std::tuple<vector_t, matrix_t, matrix_t> step_with_jacobians(
      double t, const vector_t& y, double dt, const vector_t& u) const {
        return fdapde::invoke<std::tuple<vector_t, matrix_t, matrix_t>, 3>(*this, t, y, dt, u);
    }
    void set_theta(const vector_t& theta) { fdapde::invoke<void, 4>(*this, theta); }
    int n_params() const { return fdapde::invoke<int, 5>(*this); }
    matrix_t param_jacobian(double t, const vector_t& y, double dt, const vector_t& u) const {
        return fdapde::invoke<matrix_t, 6>(*this, t, y, dt, u);
    }
};

using any_controlled_ode_solver = fdapde::erase<fdapde::heap_storage, IControlledOdeSolver>;

}   // namespace fdapde

#endif   // __FDAPDE_CONTROLLED_ODE_SOLVER_H__
