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

#include <functional>
#include <memory>
#include <utility>

#include "header_check.h"

namespace fdapde {

// TODO: change the trait to is ode_rhs_constructible or something like that

// f models the right-hand side of an ODE y' = f(t, y): callable as f(t, y) -> R^d. This is the
// minimal interface RKIntegrator requires from a field; the state dimension is read from y.
template <typename F>
concept ode_rhs = std::is_invocable_r_v<
  Eigen::Matrix<double, Dynamic, 1>, const F&, double, const Eigen::Matrix<double, Dynamic, 1>&>;

// f additionally exposes an analytic Jacobian df_dy(t, y) -> R^{d x d}; when absent the integrator
// falls back to central finite differences.
template <typename F>
concept provides_jacobian = requires(const F& f, double t, const Eigen::Matrix<double, Dynamic, 1>& y) {
    { f.df_dy(t, y) } -> std::convertible_to<Eigen::Matrix<double, Dynamic, Dynamic>>;
};

// Type-erased vector field f : (t, y) -> R^d, adapting the library MatrixField machinery. It stores
// the value callable f(t, y) and exposes df/dy: an analytic Jacobian is used when the wrapped functor
// provides df_dy (provides_jacobian), otherwise df/dy is obtained from the library finite-difference
// Jacobian of a MatrixField view of y -> f(t, y). The state dimension d is read from the argument y
// at evaluation (the MatrixField backend is assembled lazily on first use), so no n_components() is
// required. Constructible from any ode_rhs callable; models ode_rhs and provides_jacobian.
class ode_field {
   public:
    using vector_t = Eigen::Matrix<double, Dynamic, 1>;
    using matrix_t = Eigen::Matrix<double, Dynamic, Dynamic>;

    ode_field() = default;
    template <typename F>
        requires(ode_rhs<F> && !std::is_same_v<std::decay_t<F>, ode_field>)
    ode_field(F&& field) {
        auto ptr = std::make_shared<std::decay_t<F>>(std::forward<F>(field));
        f_ = [ptr](double t, const vector_t& y) -> vector_t { return (*ptr)(t, y); };
        if constexpr (provides_jacobian<std::decay_t<F>>) {
            analytic_df_dy_ = [ptr](double t, const vector_t& y) -> matrix_t { return ptr->df_dy(t, y); };
        }
    }
    vector_t operator()(double t, const vector_t& y) const { return f_(t, y); }
    // df/dy: the supplied analytic Jacobian when available, otherwise the finite-difference Jacobian
    // of the MatrixField view of y -> f(t, y), assembled lazily and reused.
    matrix_t df_dy(double t, const vector_t& y) const {
        if (analytic_df_dy_) { return analytic_df_dy_(t, y); }
        const int d = y.size();
        ensure_fd_backend_(d);
        vector_t p(d + 1);   // augmented input [t; y]: the field is differentiated w.r.t. the y-block
        p[0] = t;
        p.tail(d) = y;
        matrix_t J = (*fd_jacobian_)(p);          // (d+1) x d, entry (i, j) = d f_j / d x_i
        return J.middleRows(1, d).transpose();    // d x d,     entry (i, j) = d f_i / d y_j
    }
    explicit operator bool() const { return static_cast<bool>(f_); }
    // the field g(t, y) = f(t, y) + c, i.e. this field with its value offset by the constant c. A
    // constant shift leaves the Jacobian unchanged, so the Jacobian source (analytic callback or the
    // lazily-built finite-difference backend) is reused. Used by the ODE-penalty solvers to inject a
    // control value u that is constant over an integration interval (g = f + u).
    ode_field shifted(const vector_t& c) const {
        ode_field g;
        // explicit return type: evaluate f(t, y) + c into a vector_t before the temporary f(t, y)
        // dies (a deduced return type would yield a dangling Eigen expression).
        g.f_ = [f = f_, c](double t, const vector_t& y) -> vector_t { return f(t, y) + c; };
        g.analytic_df_dy_ = analytic_df_dy_;
        return g;
    }
   private:
    // scalar component y -> f(t, y)[i] over the augmented input [t; y]; one type for every entry so
    // they live in a single MatrixField (a dynamic VectorField), the source for the library Jacobian.
    using fd_component_t = std::function<double(const vector_t&)>;
    using fd_field_t = MatrixField<Dynamic, Dynamic, 1, fd_component_t>;

    // assemble the MatrixField view of y -> f(t, y) and its finite-difference Jacobian the first time
    // a Jacobian is needed (d is read from y). Cached and shared across copies / shifted() fields.
    void ensure_fd_backend_(int d) const {
        if (fd_jacobian_ && fd_input_size_ == d) { return; }
        auto field = std::make_shared<fd_field_t>();
        field->resize(d + 1, d);
        for (int i = 0; i < d; ++i) {
            (*field)[i] =
              fd_component_t([f = f_, d, i](const vector_t& p) -> double { return f(p[0], p.tail(d))[i]; });
        }
        fd_field_ = field;
        fd_jacobian_ = std::make_shared<Jacobian<fd_field_t>>(*field);
        fd_input_size_ = d;
    }

    std::function<vector_t(double, const vector_t&)> f_;                // value f(t, y)
    std::function<matrix_t(double, const vector_t&)> analytic_df_dy_;   // analytic df/dy, or empty
    mutable std::shared_ptr<fd_field_t> fd_field_;                      // MatrixField source of the FD Jacobian
    mutable std::shared_ptr<Jacobian<fd_field_t>> fd_jacobian_;         // library finite-difference Jacobian
    mutable int fd_input_size_ = 0;
};

// Generic Runge-Kutta integrator driven by a ButcherTableau. A single implementation covers
// explicit and implicit methods: explicit tableaux evaluate the stages by forward
// substitution, implicit ones solve the coupled stage system by Newton's method. Besides the
// forward step it exposes the local sensitivities needed by the ODE-penalty solvers:
//   - increment(.)           : Phi = sum_i b_i k_i           (so y_{n+1} = y + dt*Phi)
//   - increment_jacobian(.)  : d Phi / d y                   (d x d)
//   - flow_jacobian(.)       : d y_{n+1} / d y = I + dt*dPhi  (state-transition matrix)
// from which a solver assembles the control defect u = (y_n - y_c)/dt - Phi and its exact
// linearization (d u/d y_n = I/dt, d u/d y_c = -I/dt - dPhi/dy_c).
//
// For reduced (adjoint-method) optimal control it additionally exposes
//   - adjoint_step(.)        : the exact discrete adjoint of one forward step. Given the incoming
//                              costate p_{n+1} = dC/dy_{n+1}, it returns dC/dy_n together with the
//                              sensitivity dC/du of the cost to a constant additive control u that
//                              enters the dynamics over the interval (g = f + u). Derived from the
//                              tableau via stage adjoints, a single implementation covers every
//                              scheme (the C++ analogue of the per-scheme DtO adjoints).
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

    // Each method accepts any ode_rhs callable f(t, y); it is wrapped into an ode_field, which
    // supplies df_dy analytically when f exposes it and by finite differences otherwise.

    // forward step: y_{n+1} = y + dt * Phi
    template <typename F>
        requires(ode_rhs<F>)
    vector_t step(const F& f, double t, const vector_t& y, double dt) const {
        return step_(ode_field(f), t, y, dt);
    }
    // increment Phi = sum_i b_i k_i  (so y_{n+1} = y + dt * Phi)
    template <typename F>
        requires(ode_rhs<F>)
    vector_t increment(const F& f, double t, const vector_t& y, double dt) const {
        return increment_(ode_field(f), t, y, dt);
    }
    // d(Phi)/d(y): differentiate the stage system w.r.t. the initial state and reuse its
    // (already factorized) Jacobian G.  S_i = dk_i/dy solves  G S = [J_1; ...; J_s].
    template <typename F>
        requires(ode_rhs<F>)
    matrix_t increment_jacobian(const F& f, double t, const vector_t& y, double dt) const {
        return increment_jacobian_(ode_field(f), t, y, dt);
    }
    // d(y_{n+1})/d(y) = I + dt * dPhi/dy
    template <typename F>
        requires(ode_rhs<F>)
    matrix_t flow_jacobian(const F& f, double t, const vector_t& y, double dt) const {
        return matrix_t::Identity(y.size(), y.size()) + dt * increment_jacobian_(ode_field(f), t, y, dt);
    }
    // forward step y_{n+1} together with its flow Jacobian d y_{n+1}/d y, from a single stage
    // solve (the combination a time-stepping smoother needs once per interval).
    template <typename F>
        requires(ode_rhs<F>)
    std::pair<vector_t, matrix_t> step_with_flow_jacobian(
      const F& f, double t, const vector_t& y, double dt) const {
        return step_with_flow_jacobian_(ode_field(f), t, y, dt);
    }
    // discrete adjoint of one forward step. Inputs: the field f (the dynamics already shifted by the
    // interval control, if any), the step (t, y, dt) of the *forward* trajectory, and the incoming
    // costate p_next = dC/dy_{n+1}. Returns {p_curr, grad_contrib} with
    //   p_curr       = dC/dy_n   (state-transition adjoint; external node sources added by caller)
    //   grad_contrib = dC/du     (sensitivity to a constant additive control u over the interval)
    template <typename F>
        requires(ode_rhs<F>)
    std::pair<vector_t, vector_t> adjoint_step(
      const F& f, double t, const vector_t& y, double dt, const vector_t& p_next) const {
        return adjoint_step_(ode_field(f), t, y, dt, p_next);
    }

   private:
    // implementations operating on the (already wrapped) type-erased field
    vector_t step_(const ode_field& f, double t, const vector_t& y, double dt) const {
        return y + dt * increment_(f, t, y, dt);
    }
    vector_t increment_(const ode_field& f, double t, const vector_t& y, double dt) const {
        const int d = y.size();
        vector_t K = solve_stages_(f, t, y, dt);
        vector_t phi = vector_t::Zero(d);
        for (int i = 0; i < Stages; ++i) { phi += tableau_.b()[i] * K.segment(i * d, d); }
        return phi;
    }
    matrix_t increment_jacobian_(const ode_field& f, double t, const vector_t& y, double dt) const {
        const int d = y.size(), ns = Stages;
        vector_t K = solve_stages_(f, t, y, dt);
        std::vector<matrix_t> J;
        Eigen::PartialPivLU<matrix_t> G;
        stage_jacobians_(f, t, y, dt, K, J, G);
        matrix_t rhs(ns * d, d);
        for (int i = 0; i < ns; ++i) { rhs.block(i * d, 0, d, d) = J[i]; }
        matrix_t S = G.solve(rhs);   // (ns*d) x d, S_i = dk_i/dy
        matrix_t dphi = matrix_t::Zero(d, d);
        for (int i = 0; i < ns; ++i) { dphi += tableau_.b()[i] * S.block(i * d, 0, d, d); }
        return dphi;
    }
    std::pair<vector_t, matrix_t> step_with_flow_jacobian_(
      const ode_field& f, double t, const vector_t& y, double dt) const {
        const int d = y.size(), ns = Stages;
        vector_t K = solve_stages_(f, t, y, dt);
        vector_t phi = vector_t::Zero(d);
        for (int i = 0; i < ns; ++i) { phi += tableau_.b()[i] * K.segment(i * d, d); }
        std::vector<matrix_t> J;
        Eigen::PartialPivLU<matrix_t> G;
        stage_jacobians_(f, t, y, dt, K, J, G);
        matrix_t rhs(ns * d, d);
        for (int i = 0; i < ns; ++i) { rhs.block(i * d, 0, d, d) = J[i]; }
        matrix_t S = G.solve(rhs);
        matrix_t dphi = matrix_t::Zero(d, d);
        for (int i = 0; i < ns; ++i) { dphi += tableau_.b()[i] * S.block(i * d, 0, d, d); }
        return {y + dt * phi, matrix_t::Identity(d, d) + dt * dphi};
    }
    // exact discrete adjoint of one RK step. Recovers the forward stages and their Jacobians
    // J_i = df_dy(t + c_i dt, Y_i), then solves the stage-adjoint block system
    //   M lam = rhs,   M_ij = delta_ij I - dt A_ji J_j^T,   rhs_i = dt b_i p_next
    // and forms  p_curr = p_next + sum_i J_i^T lam_i,  grad_contrib = sum_i lam_i.
    // For forward Euler this reduces to p_curr = (I + dt J^T) p_next, grad_contrib = dt p_next.
    std::pair<vector_t, vector_t> adjoint_step_(
      const ode_field& f, double t, const vector_t& y, double dt, const vector_t& p_next) const {
        const int d = y.size(), ns = Stages;
        const matrix_t Id = matrix_t::Identity(d, d);
        vector_t K = solve_stages_(f, t, y, dt);
        std::vector<matrix_t> J;
        Eigen::PartialPivLU<matrix_t> G;   // forward stage-system factorization (unused here)
        stage_jacobians_(f, t, y, dt, K, J, G);
        matrix_t M(ns * d, ns * d);
        vector_t rhs(ns * d);
        for (int i = 0; i < ns; ++i) {
            for (int j = 0; j < ns; ++j) {
                matrix_t blk = -dt * tableau_.A()[j][i] * J[j].transpose();
                if (i == j) { blk += Id; }
                M.block(i * d, j * d, d, d) = blk;
            }
            rhs.segment(i * d, d) = dt * tableau_.b()[i] * p_next;
        }
        vector_t lam = M.partialPivLu().solve(rhs);
        vector_t p_curr = p_next;
        vector_t grad_contrib = vector_t::Zero(d);
        for (int i = 0; i < ns; ++i) {
            const vector_t lam_i = lam.segment(i * d, d);
            p_curr += J[i].transpose() * lam_i;
            grad_contrib += lam_i;
        }
        return {p_curr, grad_contrib};
    }
    // stage argument of stage i:  y + dt * sum_j A_ij k_j
    vector_t stage_arg_(const vector_t& y, const vector_t& K, double dt, int i, int d) const {
        vector_t arg = y;
        for (int j = 0; j < Stages; ++j) { arg += dt * tableau_.A()[i][j] * K.segment(j * d, d); }
        return arg;
    }
    // stage derivatives K = [k_1; ...; k_s], k_i = f(t + c_i dt, y + dt sum_j A_ij k_j): forward
    // substitution for explicit tableaux (no Jacobian needed), Newton for implicit ones.
    vector_t solve_stages_(const ode_field& f, double t, const vector_t& y, double dt) const {
        const int d = y.size(), ns = Stages;
        vector_t K = vector_t::Zero(ns * d);
        if (tableau_.is_explicit()) {
            for (int i = 0; i < ns; ++i) {
                vector_t arg = stage_arg_(y, K, dt, i, d);
                K.segment(i * d, d) = f(t + tableau_.c()[i] * dt, arg);
            }
            return K;
        }
        // implicit: Newton on the coupled stage residual R_i = k_i - f(t + c_i dt, arg_i)
        const matrix_t Id = matrix_t::Identity(d, d);
        vector_t f0 = f(t, y);
        for (int i = 0; i < ns; ++i) { K.segment(i * d, d) = f0; }   // initial guess
        matrix_t G(ns * d, ns * d);
        vector_t R(ns * d);
        Eigen::PartialPivLU<matrix_t> lu;
        for (int it = 0; it < newton_max_iter_; ++it) {
            std::vector<matrix_t> J(ns);
            for (int i = 0; i < ns; ++i) {
                double ti = t + tableau_.c()[i] * dt;
                vector_t arg = stage_arg_(y, K, dt, i, d);
                R.segment(i * d, d) = K.segment(i * d, d) - f(ti, arg);
                J[i] = f.df_dy(ti, arg);
            }
            if (R.norm() < newton_tol_) { break; }
            for (int i = 0; i < ns; ++i) {
                for (int j = 0; j < ns; ++j) {
                    matrix_t blk = -dt * tableau_.A()[i][j] * J[i];
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
    void stage_jacobians_(
      const ode_field& f, double t, const vector_t& y, double dt, const vector_t& K, std::vector<matrix_t>& J,
      Eigen::PartialPivLU<matrix_t>& G) const {
        const int d = y.size(), ns = Stages;
        const matrix_t Id = matrix_t::Identity(d, d);
        J.resize(ns);
        for (int i = 0; i < ns; ++i) { J[i] = f.df_dy(t + tableau_.c()[i] * dt, stage_arg_(y, K, dt, i, d)); }
        matrix_t Gm(ns * d, ns * d);
        for (int i = 0; i < ns; ++i) {
            for (int j = 0; j < ns; ++j) {
                matrix_t blk = -dt * tableau_.A()[i][j] * J[i];
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

// Type-erased, non-templated handle over a RKIntegrator<Stages>. The time-stepping ODE solvers need
// only three stage-count-independent operations on an ode_field (forward step, discrete adjoint, and
// step + flow Jacobian); erasing the integrator behind these lets the solver and the model wrapper
// drop the Stages template parameter, while the wrapped RKIntegrator<Stages> keeps its constexpr
// ButcherTableau and fixed-stage loops. Built on the library type-erasure facility (fdapde::erase):
// the interface binds RKIntegrator's field-templated operations through their ode_field instantiation.
struct IRKIntegrator {
    using vector_t = Eigen::Matrix<double, Dynamic, 1>;
    using matrix_t = Eigen::Matrix<double, Dynamic, Dynamic>;
    template <typename T> using fn_ptrs = fdapde::bindings<
      &T::template step<ode_field>, &T::template adjoint_step<ode_field>,
      &T::template step_with_flow_jacobian<ode_field>>;
    // interface: forwarded to the wrapped integrator through the type-erasure virtual table
    vector_t step(const ode_field& f, double t, const vector_t& y, double dt) const {
        return fdapde::invoke<vector_t, 0>(*this, f, t, y, dt);
    }
    std::pair<vector_t, vector_t> adjoint_step(
      const ode_field& f, double t, const vector_t& y, double dt, const vector_t& p_next) const {
        return fdapde::invoke<std::pair<vector_t, vector_t>, 1>(*this, f, t, y, dt, p_next);
    }
    std::pair<vector_t, matrix_t> step_with_flow_jacobian(
      const ode_field& f, double t, const vector_t& y, double dt) const {
        return fdapde::invoke<std::pair<vector_t, matrix_t>, 2>(*this, f, t, y, dt);
    }
};
using any_rk_integrator = fdapde::erase<fdapde::heap_storage, IRKIntegrator>;

}   // namespace fdapde

#endif   // __FDAPDE_RK_INTEGRATOR_H__
