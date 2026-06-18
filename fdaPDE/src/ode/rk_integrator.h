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

#include "header_check.h"

namespace fdapde {

// is_ode_field<F> holds when F models a (non-autonomous) vector field f : (t, y) -> R^d,
// y in R^d. Any dependence on parameters theta is captured inside the field object. A field must
// expose
//
//   int      n_components() const;                                  // d
//   vector_t operator()(double t, const vector_t& y) const;         // f(t, y)        in R^d
//   matrix_t df_dy      (double t, const vector_t& y) const;        // d f / d y      in R^{dxd}
//
// The Jacobian df_dy may be analytic or finite-difference (see fd_ode_field, which adapts a
// bare callable into a valid field by central differences). The type-erased holder `ode_field`
// (below) is the canonical type a consumer stores.
template <typename F>
concept is_ode_field = requires(const F& f, double t, const Eigen::Matrix<double, Dynamic, 1>& y) {
    { f.n_components() } -> std::convertible_to<int>;
    { f(t, y) } -> std::convertible_to<Eigen::Matrix<double, Dynamic, 1>>;
    { f.df_dy(t, y) } -> std::convertible_to<Eigen::Matrix<double, Dynamic, Dynamic>>;
};

// Adapts a bare callable f(t, y) -> R^d into a valid field, supplying df_dy by central finite
// differences. Lets callers use the integrator without hand-coding a Jacobian.
template <typename Callable> class fd_ode_field {
   public:
    using vector_t = Eigen::Matrix<double, Dynamic, 1>;
    using matrix_t = Eigen::Matrix<double, Dynamic, Dynamic>;

    fd_ode_field(Callable f, int n_components, double h = 1e-6) :
        f_(std::move(f)), n_components_(n_components), h_(h) { }
    int n_components() const { return n_components_; }
    vector_t operator()(double t, const vector_t& y) const { return f_(t, y); }
    matrix_t df_dy(double t, const vector_t& y) const {
        matrix_t J(n_components_, n_components_);
        vector_t yp = y, ym = y;
        for (int j = 0; j < n_components_; ++j) {
            yp[j] = y[j] + h_;
            ym[j] = y[j] - h_;
            J.col(j) = (f_(t, yp) - f_(t, ym)) / (2 * h_);
            yp[j] = y[j];
            ym[j] = y[j];
        }
        return J;
    }
   private:
    Callable f_;
    int n_components_;
    double h_;
};

// Type-erased field: stores the vector field behind std::function so a consumer (e.g. a solver)
// can hold a field member without being templated on its concrete type. Constructible from any
// is_ode_field (reusing its analytic Jacobian) or from a bare callable f(t, y) plus the component
// count (Jacobian then supplied by central finite differences). It itself models is_ode_field.
class ode_field {
   public:
    using vector_t = Eigen::Matrix<double, Dynamic, 1>;
    using matrix_t = Eigen::Matrix<double, Dynamic, Dynamic>;

    ode_field() = default;
    template <typename Field>
        requires(is_ode_field<std::decay_t<Field>> && !std::is_same_v<std::decay_t<Field>, ode_field>)
    ode_field(Field&& field) : n_components_(field.n_components()) {
        auto ptr = std::make_shared<std::decay_t<Field>>(std::forward<Field>(field));
        f_ = [ptr](double t, const vector_t& y) { return (*ptr)(t, y); };
        df_dy_ = [ptr](double t, const vector_t& y) { return ptr->df_dy(t, y); };
    }
    template <typename Callable>
        requires(std::is_invocable_r_v<vector_t, Callable, double, const vector_t&>)
    ode_field(Callable f, int n_components, double h = 1e-6) : n_components_(n_components) {
        auto fp = std::make_shared<Callable>(std::move(f));
        f_ = [fp](double t, const vector_t& y) { return (*fp)(t, y); };
        df_dy_ = [fp, n_components, h](double t, const vector_t& y) {
            matrix_t J(n_components, n_components);
            vector_t yp = y, ym = y;
            for (int j = 0; j < n_components; ++j) {
                yp[j] = y[j] + h;
                ym[j] = y[j] - h;
                J.col(j) = ((*fp)(t, yp) - (*fp)(t, ym)) / (2 * h);
                yp[j] = y[j];
                ym[j] = y[j];
            }
            return J;
        };
    }
    int n_components() const { return n_components_; }
    vector_t operator()(double t, const vector_t& y) const { return f_(t, y); }
    matrix_t df_dy(double t, const vector_t& y) const { return df_dy_(t, y); }
    explicit operator bool() const { return static_cast<bool>(f_); }
   private:
    std::function<vector_t(double, const vector_t&)> f_;
    std::function<matrix_t(double, const vector_t&)> df_dy_;
    int n_components_ = 0;
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
class RKIntegrator {
   public:
    using vector_t = Eigen::Matrix<double, Dynamic, 1>;
    using matrix_t = Eigen::Matrix<double, Dynamic, Dynamic>;

    RKIntegrator() = default;
    explicit RKIntegrator(ButcherTableau tableau) : tableau_(std::move(tableau)) { }
    RKIntegrator(ButcherTableau tableau, int newton_max_iter, double newton_tol) :
        tableau_(std::move(tableau)), newton_max_iter_(newton_max_iter), newton_tol_(newton_tol) { }

    const ButcherTableau& tableau() const { return tableau_; }

    // forward step: y_{n+1} = y + dt * Phi
    template <typename Field>
        requires(is_ode_field<Field>)
    vector_t step(const Field& f, double t, const vector_t& y, double dt) const {
        return y + dt * increment(f, t, y, dt);
    }
    // increment Phi = sum_i b_i k_i
    template <typename Field>
        requires(is_ode_field<Field>)
    vector_t increment(const Field& f, double t, const vector_t& y, double dt) const {
        stage_solution s = solve_stages_(f, t, y, dt);
        const int d = f.n_components();
        vector_t phi = vector_t::Zero(d);
        for (int i = 0; i < tableau_.n_stages(); ++i) { phi += tableau_.b()(i) * s.K.segment(i * d, d); }
        return phi;
    }
    // d(Phi)/d(y): differentiate the stage system w.r.t. the initial state and reuse its
    // (already factorized) Jacobian G.  S_i = dk_i/dy solves  G S = [J_1; ...; J_s].
    template <typename Field>
        requires(is_ode_field<Field>)
    matrix_t increment_jacobian(const Field& f, double t, const vector_t& y, double dt) const {
        stage_solution s = solve_stages_(f, t, y, dt);
        const int d = f.n_components(), ns = tableau_.n_stages();
        matrix_t rhs(ns * d, d);
        for (int i = 0; i < ns; ++i) { rhs.block(i * d, 0, d, d) = s.J[i]; }
        matrix_t S = s.G.solve(rhs);   // (ns*d) x d, S_i = dk_i/dy
        matrix_t dphi = matrix_t::Zero(d, d);
        for (int i = 0; i < ns; ++i) { dphi += tableau_.b()(i) * S.block(i * d, 0, d, d); }
        return dphi;
    }
    // d(y_{n+1})/d(y) = I + dt * dPhi/dy
    template <typename Field>
        requires(is_ode_field<Field>)
    matrix_t flow_jacobian(const Field& f, double t, const vector_t& y, double dt) const {
        const int d = f.n_components();
        return matrix_t::Identity(d, d) + dt * increment_jacobian(f, t, y, dt);
    }
    // forward step y_{n+1} together with its flow Jacobian d y_{n+1}/d y, from a single stage
    // solve (the combination a time-stepping smoother needs once per interval).
    template <typename Field>
        requires(is_ode_field<Field>)
    std::pair<vector_t, matrix_t> step_with_flow_jacobian(
      const Field& f, double t, const vector_t& y, double dt) const {
        stage_solution s = solve_stages_(f, t, y, dt);
        const int d = f.n_components(), ns = tableau_.n_stages();
        vector_t phi = vector_t::Zero(d);
        for (int i = 0; i < ns; ++i) { phi += tableau_.b()(i) * s.K.segment(i * d, d); }
        matrix_t rhs(ns * d, d);
        for (int i = 0; i < ns; ++i) { rhs.block(i * d, 0, d, d) = s.J[i]; }
        matrix_t S = s.G.solve(rhs);
        matrix_t dphi = matrix_t::Zero(d, d);
        for (int i = 0; i < ns; ++i) { dphi += tableau_.b()(i) * S.block(i * d, 0, d, d); }
        return {y + dt * phi, matrix_t::Identity(d, d) + dt * dphi};
    }

   private:
    // converged stage system: stage derivatives K = [k_1; ...; k_s], the per-stage Jacobians
    // J_i = df_dy at the stage argument, and the factorized stage Jacobian
    //   G_{ij} = delta_ij * I - dt * A_ij * J_i      (size (s*d) x (s*d)).
    struct stage_solution {
        vector_t K;
        std::vector<matrix_t> J;
        Eigen::PartialPivLU<matrix_t> G;
    };
    // stage argument of stage i:  y + dt * sum_j A_ij k_j
    vector_t stage_arg_(const vector_t& y, const vector_t& K, double dt, int i, int d) const {
        vector_t arg = y;
        for (int j = 0; j < tableau_.n_stages(); ++j) { arg += dt * tableau_.A()(i, j) * K.segment(j * d, d); }
        return arg;
    }
    template <typename Field>
    stage_solution solve_stages_(const Field& f, double t, const vector_t& y, double dt) const {
        const int d = f.n_components(), ns = tableau_.n_stages();
        const matrix_t Id = matrix_t::Identity(d, d);
        stage_solution s;
        s.K = vector_t::Zero(ns * d);
        s.J.resize(ns);

        if (tableau_.is_explicit()) {
            // forward substitution: stage i depends only on stages j < i
            for (int i = 0; i < ns; ++i) {
                vector_t arg = stage_arg_(y, s.K, dt, i, d);
                s.K.segment(i * d, d) = f(t + tableau_.c()(i) * dt, arg);
            }
        } else {
            // Newton on the coupled stage residual R_i = k_i - f(t + c_i dt, arg_i)
            vector_t f0 = f(t, y);
            for (int i = 0; i < ns; ++i) { s.K.segment(i * d, d) = f0; }   // initial guess
            matrix_t G(ns * d, ns * d);
            vector_t R(ns * d);
            for (int it = 0; it < newton_max_iter_; ++it) {
                for (int i = 0; i < ns; ++i) {
                    double ti = t + tableau_.c()(i) * dt;
                    vector_t arg = stage_arg_(y, s.K, dt, i, d);
                    R.segment(i * d, d) = s.K.segment(i * d, d) - f(ti, arg);
                    s.J[i] = f.df_dy(ti, arg);
                }
                for (int i = 0; i < ns; ++i) {
                    for (int j = 0; j < ns; ++j) {
                        matrix_t blk = -dt * tableau_.A()(i, j) * s.J[i];
                        if (i == j) { blk += Id; }
                        G.block(i * d, j * d, d, d) = blk;
                    }
                }
                s.G.compute(G);
                if (R.norm() < newton_tol_) { return s; }
                s.K += s.G.solve(-R);
            }
        }
        // (re)build per-stage Jacobians and the factorized stage Jacobian at the converged K
        matrix_t G(ns * d, ns * d);
        for (int i = 0; i < ns; ++i) {
            s.J[i] = f.df_dy(t + tableau_.c()(i) * dt, stage_arg_(y, s.K, dt, i, d));
        }
        for (int i = 0; i < ns; ++i) {
            for (int j = 0; j < ns; ++j) {
                matrix_t blk = -dt * tableau_.A()(i, j) * s.J[i];
                if (i == j) { blk += Id; }
                G.block(i * d, j * d, d, d) = blk;
            }
        }
        s.G.compute(G);
        return s;
    }

    ButcherTableau tableau_;
    int newton_max_iter_ = 50;
    double newton_tol_ = 1e-12;
};

}   // namespace fdapde

#endif   // __FDAPDE_RK_INTEGRATOR_H__
