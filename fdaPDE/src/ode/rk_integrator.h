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

#ifndef __FDAPDE_RK_INTEGRATOR_H__
#define __FDAPDE_RK_INTEGRATOR_H__

#include <array>
#include <functional>
#include <memory>
#include <tuple>
#include <utility>

#include "header_check.h"

namespace fdapde {
    
/* Generic Runge-Kutta integrator driven by a ButcherTableau. 
A single implementation covers explicit and implicit methods: explicit tableaux evaluate the stages by forward substitution, implicit ones solve the coupled stage system by Newton's method. 

Besides the forward step it exposes the local sensitivities of one step, which are needed by the ODE smoothing solvers:

- increment(.): 
    k_i = f(t + c_i*dt, y + dt * sum_j A_ij k_j),   i = 1, ..., s
    Phi = sum_i b_i k_i
    y_{n+1} = y_n + dt * Phi
- increment_jacobian(.): d Phi / d y  (d x d)   
- flow_jacobian(.): 
    d y_{n+1} / d y = I + dt*dPhi  (state-transition matrix)   
- step_param_jacobian(.) : d y_{n+1} / d theta            (d x n_theta)
    for parameter-dependent dynamics y' = f(t, y, theta)   
- adjoint_step(.)        : 
    the exact discrete adjoint of one forward step. Given the incoming costate p_{n+1} = dC/dy_{n+1}, it returns dC/dy_n; a  parameter-aware overload additionally returns dC/dtheta, the cost sensitivity to the rhs parameters, contracted from the same stage adjoints. Derived from the tableau via stage adjoints, a single implementation covers every scheme (the C++ analogue of the per-scheme DtO adjoints). 

The stage math is fixed-size when the field carries a static dimension (ode_rhs_field<Dim>): each method reads the dimension off the field type (stage_types<Field>) and uses Eigen objects of size Stages*Dim, so the Newton / sensitivity / adjoint solves allocate no heap and the loops unroll. For Dim = Dynamic the same code falls back to dynamically-sized Eigen. 
 */
template <int Stages>
class RKIntegrator {
   public:
    using vector_t = Eigen::Matrix<double, Dynamic, 1>;
    using matrix_t = Eigen::Matrix<double, Dynamic, Dynamic>;

    RKIntegrator() = default;
    // the compile-time ButcherTableau<Stages> is stored as-is: the constexpr coefficients drive the
    // fixed-stage stage loops directly, with no runtime copy of the tableau.
    explicit RKIntegrator(const ButcherTableau<Stages>& tableau) : tableau_(tableau) { }
    RKIntegrator(const ButcherTableau<Stages>& tableau, int newton_max_iter, double newton_tol) :
        tableau_(tableau), newton_max_iter_(newton_max_iter), newton_tol_(newton_tol) { }

    // tableau observers
    static constexpr int n_stages() { return Stages; }
    bool is_explicit() const { return tableau_.is_explicit(); }
    const ButcherTableau<Stages>& tableau() const { return tableau_; }

    // Each method operates on an ode_rhs_field<Dim, F>, which supplies f(t, y) and its state_jacobian
    // (analytically when the wrapped functor exposes one, by finite differences otherwise). A raw is_ode_rhs
    // callable is turned into a field at the call site (ode_rhs_field{f}); the engines already hold fields.

    // forward step: y_{n+1} = y + dt * Phi
    template <int Dim, typename F>
    vector_t step(const ode_rhs_field<Dim, F>& f, double t, const vector_t& y, double dt) const {
        return y + dt * increment(f, t, y, dt);
    }
    // full-horizon forward integration of y' = f(t, y) on the grid `times` from y0: chains step over the
    // intervals and returns the nodal trajectory Y (m x d). This is the general IVP solve on a fixed time
    // grid, built on the single-step utilities; it has no notion of any cost/objective.
    template <int Dim, typename F>
    matrix_t integrate(const ode_rhs_field<Dim, F>& f, const vector_t& times, const vector_t& y0) const {
        const int m = times.size(), d = y0.size();
        fdapde_assert(m >= 1 && d > 0);
        matrix_t Y(m, d);
        Y.row(0) = y0.transpose();
        for (int t = 0; t + 1 < m; ++t) {
            vector_t yc = Y.row(t).transpose();
            Y.row(t + 1) = step(f, times[t], yc, times[t + 1] - times[t]).transpose();
        }
        return Y;
    }
    // increment Phi = sum_i b_i k_i  (so y_{n+1} = y + dt * Phi)
    template <int Dim, typename F>
    vector_t increment(const ode_rhs_field<Dim, F>& f, double t, const vector_t& y, double dt) const {
        const int d = y.size();
        auto K = solve_stages_(f, t, y, dt);
        vector_t phi = vector_t::Zero(d);
        for (int i = 0; i < Stages; ++i) { phi += tableau_.b()[i] * K.segment(i * d, d); }
        return phi;
    }
    // d(Phi)/d(y): differentiate the stage system w.r.t. the initial state and reuse its
    // (already factorized) Jacobian G.  S_i = dk_i/dy solves  G S = [J_1; ...; J_s].
    template <int Dim, typename F>
    matrix_t increment_jacobian(const ode_rhs_field<Dim, F>& f, double t, const vector_t& y, double dt) const {
        using Field = ode_rhs_field<Dim, F>;
        using ST = stage_types<Field>;
        const int d = y.size(), ns = Stages;
        auto K = solve_stages_(f, t, y, dt);
        typename ST::jac_array J;
        typename ST::stage_lu G;
        stage_jacobians_(f, t, y, dt, K, J, G);
        matrix_t rhs(ns * d, d);
        for (int i = 0; i < ns; ++i) { rhs.block(i * d, 0, d, d) = J[i]; }
        matrix_t S = G.solve(rhs);   // (ns*d) x d, S_i = dk_i/dy
        matrix_t dphi = matrix_t::Zero(d, d);
        for (int i = 0; i < ns; ++i) { dphi += tableau_.b()[i] * S.block(i * d, 0, d, d); }
        return dphi;
    }
    // d(y_{n+1})/d(y) = I + dt * dPhi/dy
    template <int Dim, typename F>
    matrix_t flow_jacobian(const ode_rhs_field<Dim, F>& f, double t, const vector_t& y, double dt) const {
        return matrix_t::Identity(y.size(), y.size()) + dt * increment_jacobian(f, t, y, dt);
    }
    // forward step y_{n+1} together with its flow Jacobian d y_{n+1}/d y, from a single stage
    // solve (the combination a time-stepping smoother needs once per interval).
    // TODO: better naming than step_with_flow_jacobian 
    template <int Dim, typename F>
    std::pair<vector_t, matrix_t> step_with_flow_jacobian(
      const ode_rhs_field<Dim, F>& f, double t, const vector_t& y, double dt) const {
        using Field = ode_rhs_field<Dim, F>;
        using ST = stage_types<Field>;
        const int d = y.size(), ns = Stages;
        auto K = solve_stages_(f, t, y, dt);
        vector_t phi = vector_t::Zero(d);
        for (int i = 0; i < ns; ++i) { phi += tableau_.b()[i] * K.segment(i * d, d); }
        typename ST::jac_array J;
        typename ST::stage_lu G;
        stage_jacobians_(f, t, y, dt, K, J, G);
        matrix_t rhs(ns * d, d);
        for (int i = 0; i < ns; ++i) { rhs.block(i * d, 0, d, d) = J[i]; }
        matrix_t S = G.solve(rhs);
        matrix_t dphi = matrix_t::Zero(d, d);
        for (int i = 0; i < ns; ++i) { dphi += tableau_.b()[i] * S.block(i * d, 0, d, d); }
        return {y + dt * phi, matrix_t::Identity(d, d) + dt * dphi};
    }
    // forward step y_{n+1} together with BOTH its state (flow) Jacobian d y_{n+1}/d y and its parameter
    // Jacobian d y_{n+1}/d theta (d x n_theta), from a single stage solve. param_jacobian(t, y) -> R^{d x n_theta}
    // is the parameter Jacobian of the dynamics; the two sensitivity right-hand sides (per-stage state
    // Jacobians for the flow block, per-stage param_jacobian for the parameter block) share the one factorized
    // stage system. (the forward-mode / tangent-linear counterpart of adjoint_step: it returns the Jacobian
    // matrices themselves, not their transposed action on a costate.) A constant additive forcing g = f + u
    // is the special case param_jacobian = I (n_theta = d): passing an identity map yields d y_{n+1}/d u.
    template <int Dim, typename F, typename ParamJacobina>
    std::tuple<vector_t, matrix_t, matrix_t> step_with_state_param_jacobians(
      const ode_rhs_field<Dim, F>& f, double t, const vector_t& y, double dt, const ParamJacobina& param_jacobian, int n_theta) const {
        // Reuses the single stage-system factorization G for two block solves: S_y = dK/dy from
        // G S_y = [J_1; ...; J_s] (flow = I + dt sum b_i S_y,i) and S_theta = dK/dtheta from
        // G S_theta = [param_jacobian_1; ...; param_jacobian_s] (param jac = dt sum b_i S_theta,i).
        using Field = ode_rhs_field<Dim, F>;
        using ST = stage_types<Field>;
        const int d = y.size(), ns = Stages;
        auto K = solve_stages_(f, t, y, dt);
        vector_t phi = vector_t::Zero(d);
        for (int i = 0; i < ns; ++i) { phi += tableau_.b()[i] * K.segment(i * d, d); }
        typename ST::jac_array J;
        typename ST::stage_lu G;
        stage_jacobians_(f, t, y, dt, K, J, G);
        matrix_t rhs_y(ns * d, d), rhs_p(ns * d, n_theta);
        for (int i = 0; i < ns; ++i) {
            rhs_y.block(i * d, 0, d, d) = J[i];
            rhs_p.block(i * d, 0, d, n_theta) =
              param_jacobian(t + tableau_.c()[i] * dt, stage_arg_(y, K, dt, i, d));
        }
        matrix_t Sy = G.solve(rhs_y);   // (ns*d) x d,       S_y,i = dk_i/dy
        matrix_t Sp = G.solve(rhs_p);   // (ns*d) x n_theta, S_theta,i = dk_i/dtheta
        matrix_t dphi_y = matrix_t::Zero(d, d), dphi_p = matrix_t::Zero(d, n_theta);
        for (int i = 0; i < ns; ++i) {
            dphi_y += tableau_.b()[i] * Sy.block(i * d, 0, d, d);
            dphi_p += tableau_.b()[i] * Sp.block(i * d, 0, d, n_theta);
        }
        return {y + dt * phi, matrix_t::Identity(d, d) + dt * dphi_y, dt * dphi_p};
    }
    // parameter sensitivity of one step, d y_{n+1}/d theta (d x n_theta), for dynamics
    // y' = f(t, y, theta). f is the already theta-bound field and param_jacobian(t, y) -> R^{d x n_theta} its
    // parameter Jacobian. Same stage machinery as step_with_state_param_jacobians: the stage system is
    // solved once and its Jacobian reused, only the sensitivity right-hand side changes.
    template <int Dim, typename F, typename ParamJacobian>
    matrix_t step_param_jacobian(
      const ode_rhs_field<Dim, F>& f, double t, const vector_t& y, double dt, const ParamJacobian& param_jacobian, int n_theta) const {
        // the stage sensitivities S_theta = dK/dtheta solve the same stage system with the stacked parameter
        // Jacobians as rhs: G S_theta = [df/dtheta(t_1, arg_1); ...], d y_{n+1}/d theta = dt sum_i b_i S_theta,i.
        using Field = ode_rhs_field<Dim, F>;
        using ST = stage_types<Field>;
        const int d = y.size(), ns = Stages;
        auto K = solve_stages_(f, t, y, dt);
        typename ST::jac_array J;
        typename ST::stage_lu G;
        stage_jacobians_(f, t, y, dt, K, J, G);
        matrix_t rhs(ns * d, n_theta);
        for (int i = 0; i < ns; ++i) {
            rhs.block(i * d, 0, d, n_theta) =
              param_jacobian(t + tableau_.c()[i] * dt, stage_arg_(y, K, dt, i, d));
        }
        matrix_t S = G.solve(rhs);   // (ns*d) x n_theta, S_i = dk_i/dtheta
        matrix_t dphi = matrix_t::Zero(d, n_theta);
        for (int i = 0; i < ns; ++i) { dphi += tableau_.b()[i] * S.block(i * d, 0, d, n_theta); }
        return dt * dphi;
    }
    // discrete adjoint of one forward step. Inputs: the field f, the step (t, y, dt) of the *forward*
    // trajectory, and the incoming costate p_next = dC/dy_{n+1}. Returns the state-transition adjoint
    //   p_curr = dC/dy_n   (external node sources added by the caller).
    template <int Dim, typename F>
    vector_t adjoint_step(const ode_rhs_field<Dim, F>& f, double t, const vector_t& y, double dt, const vector_t& p_next) const {
        return std::get<0>(adjoint_stage_solve_(f, t, y, dt, p_next));   // p_curr
    }
    // parameter-aware overload: additionally returns dC/dtheta, the cost sensitivity to the rhs parameters,
    // formed as sum_i (param_jacobian(t_i, arg_i))^T lam_i from the same stage adjoints lam_i. A constant additive
    // forcing g = f + u is the special case param_jacobian = I (n_theta = d): an identity map yields dC/du = sum_i lam_i.
    template <int Dim, typename F, typename ParamJacobian>
    std::pair<vector_t, vector_t> adjoint_step(
      const ode_rhs_field<Dim, F>& f, double t, const vector_t& y, double dt, const vector_t& p_next, const ParamJacobian& param_jacobian) const {
        // dC/dtheta = sum_i (param_jacobian(t_i, arg_i))^T lam_i, from the same stage adjoints lam_i
        const int d = y.size(), ns = Stages;
        auto [p_curr, lam, K] = adjoint_stage_solve_(f, t, y, dt, p_next);
        vector_t g;
        for (int i = 0; i < ns; ++i) {
            matrix_t D = param_jacobian(t + tableau_.c()[i] * dt, stage_arg_(y, K, dt, i, d));   // d x n_theta
            if (i == 0) { g = vector_t::Zero(D.cols()); }
            g += D.transpose() * vector_t(lam.segment(i * d, d));
        }
        return {p_curr, g};
    }

   private:
    // compile-time stage-system Eigen types for a given field. When the field carries a static
    // dimension (ode_rhs_field<Dim>) the stage vector / stage matrix / per-stage Jacobians are
    // fixed-size, so the Newton and sensitivity solves allocate no heap and the stage loops are fully
    // unrolled; for Dim = Dynamic these reduce to the dynamic types (Matrix<double, Dynamic, ...>), i.e.
    // the previous behaviour. d-shaped boundary/return objects (y, phi, dphi, p, ...) stay dynamic.
    template <typename Field> struct stage_types {
        static constexpr int D = ode_rhs_dim_v<Field>;
        static constexpr int SD = (D == Dynamic ? Dynamic : Stages * D);
        using mat_d = Eigen::Matrix<double, D, D>;       // d x d (a single-stage Jacobian / Identity)
        using vec_sd = Eigen::Matrix<double, SD, 1>;     // Stages*d stage vector
        using mat_sd = Eigen::Matrix<double, SD, SD>;    // (Stages*d) x (Stages*d) stage system
        using jac_array = std::array<mat_d, Stages>;     // the per-stage Jacobians J_i
        using stage_lu = Eigen::PartialPivLU<mat_sd>;
    };

    // exact discrete adjoint of one RK step. Recovers the forward stages and their Jacobians
    // J_i = df_dy(t + c_i dt, Y_i), then solves the stage-adjoint block system
    //   M lam = rhs,   M_ij = delta_ij I - dt A_ji J_j^T,   rhs_i = dt b_i p_next
    // and forms  p_curr = p_next + sum_i J_i^T lam_i. The stage adjoints lam and the stage vector K (needed
    // to evaluate a parameter Jacobian at the stage arguments) are returned for the parameter-aware overload.
    // For forward Euler this reduces to p_curr = (I + dt J^T) p_next, lam = dt p_next.
    template <typename Field>
    std::tuple<vector_t, typename stage_types<Field>::vec_sd, typename stage_types<Field>::vec_sd>
    adjoint_stage_solve_(const Field& f, double t, const vector_t& y, double dt, const vector_t& p_next) const {
        using ST = stage_types<Field>;
        const int d = y.size(), ns = Stages;
        const typename ST::mat_d Id = ST::mat_d::Identity(d, d);
        auto K = solve_stages_(f, t, y, dt);
        typename ST::jac_array J;
        typename ST::stage_lu G;   // forward stage-system factorization (unused here)
        stage_jacobians_(f, t, y, dt, K, J, G);
        typename ST::mat_sd M(ns * d, ns * d);
        typename ST::vec_sd rhs(ns * d);
        for (int i = 0; i < ns; ++i) {
            for (int j = 0; j < ns; ++j) {
                typename ST::mat_d blk = -dt * tableau_.A()[j][i] * J[j].transpose();
                if (i == j) { blk += Id; }
                M.block(i * d, j * d, d, d) = blk;
            }
            rhs.segment(i * d, d) = dt * tableau_.b()[i] * p_next;
        }
        typename ST::vec_sd lam = M.partialPivLu().solve(rhs);
        vector_t p_curr = p_next;
        for (int i = 0; i < ns; ++i) { p_curr += J[i].transpose() * vector_t(lam.segment(i * d, d)); }
        return {p_curr, lam, K};
    }
    // stage argument of stage i:  y + dt * sum_j A_ij k_j (K may be a fixed- or dynamic-size stage vector)
    template <typename KVec>
    vector_t stage_arg_(const vector_t& y, const KVec& K, double dt, int i, int d) const {
        vector_t arg = y;
        for (int j = 0; j < Stages; ++j) { arg += dt * tableau_.A()[i][j] * K.segment(j * d, d); }
        return arg;
    }
    // stage derivatives K = [k_1; ...; k_s], k_i = f(t + c_i dt, y + dt sum_j A_ij k_j): forward
    // substitution for explicit tableaux (no Jacobian needed), Newton for implicit ones.
    template <typename Field>
    typename stage_types<Field>::vec_sd solve_stages_(const Field& f, double t, const vector_t& y, double dt) const {
        using ST = stage_types<Field>;
        const int d = y.size(), ns = Stages;
        typename ST::vec_sd K = ST::vec_sd::Zero(ns * d);
        if (tableau_.is_explicit()) {
            for (int i = 0; i < ns; ++i) {
                vector_t arg = stage_arg_(y, K, dt, i, d);
                K.segment(i * d, d) = f(t + tableau_.c()[i] * dt, arg);
            }
            return K;
        }
        // implicit: Newton on the coupled stage residual R_i = k_i - f(t + c_i dt, arg_i)
        const typename ST::mat_d Id = ST::mat_d::Identity(d, d);
        vector_t f0 = f(t, y);
        for (int i = 0; i < ns; ++i) { K.segment(i * d, d) = f0; }   // initial guess
        typename ST::mat_sd G(ns * d, ns * d);
        typename ST::vec_sd R(ns * d);
        typename ST::stage_lu lu;
        for (int it = 0; it < newton_max_iter_; ++it) {
            typename ST::jac_array J;
            for (int i = 0; i < ns; ++i) {
                double ti = t + tableau_.c()[i] * dt;
                vector_t arg = stage_arg_(y, K, dt, i, d);
                R.segment(i * d, d) = K.segment(i * d, d) - f(ti, arg);
                J[i] = f.state_jacobian(ti, arg);
            }
            if (R.norm() < newton_tol_) { break; }
            for (int i = 0; i < ns; ++i) {
                for (int j = 0; j < ns; ++j) {
                    typename ST::mat_d blk = -dt * tableau_.A()[i][j] * J[i];
                    if (i == j) { blk += Id; }
                    G.block(i * d, j * d, d, d) = blk;
                }
            }
            lu.compute(G);
            K += lu.solve(-R);
        }
        return K;
    }
    // per-stage Jacobians J_i = df_dy at the (converged) stage arguments and the factorized stage
    // Jacobian G_{ij} = delta_ij I - dt A_ij J_i, reused to propagate the stage sensitivities.
    template <typename Field>
    void stage_jacobians_(
      const Field& f, double t, const vector_t& y, double dt, const typename stage_types<Field>::vec_sd& K,
      typename stage_types<Field>::jac_array& J, typename stage_types<Field>::stage_lu& G) const {
        using ST = stage_types<Field>;
        const int d = y.size(), ns = Stages;
        const typename ST::mat_d Id = ST::mat_d::Identity(d, d);
        for (int i = 0; i < ns; ++i) { J[i] = f.state_jacobian(t + tableau_.c()[i] * dt, stage_arg_(y, K, dt, i, d)); }
        typename ST::mat_sd Gm(ns * d, ns * d);
        for (int i = 0; i < ns; ++i) {
            for (int j = 0; j < ns; ++j) {
                typename ST::mat_d blk = -dt * tableau_.A()[i][j] * J[i];
                if (i == j) { blk += Id; }
                Gm.block(i * d, j * d, d, d) = blk;
            }
        }
        G.compute(Gm);
    }

    ButcherTableau<Stages> tableau_;
    int newton_max_iter_ = 50;
    double newton_tol_ = 1e-12;
};

}   // namespace fdapde

#endif   // __FDAPDE_RK_INTEGRATOR_H__
