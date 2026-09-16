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

#include <cmath>
#include <vector>
#include <gtest/gtest.h>   // testing framework

#include <fdaPDE/ode.h>

using fdapde::ButcherTableau;
using fdapde::ode_rhs_field;
using fdapde::RKIntegrator;
namespace ode_schemes = fdapde::ode_schemes;

using vector_t = Eigen::Matrix<double, fdapde::Dynamic, 1>;
using matrix_t = Eigen::Matrix<double, fdapde::Dynamic, fdapde::Dynamic>;

namespace {

// scalar linear field f(t, y) = a*y  ->  exact solution y(T) = y0 * exp(a*(T - t0))
struct scalar_linear_field {
    double a;
    int n_components() const { return 1; }
    vector_t operator()(double, const vector_t& y) const {
        vector_t out(1);
        out << a * y[0];
        return out;
    }
    matrix_t state_jacobian(double, const vector_t&) const {
        matrix_t J(1, 1);
        J << a;
        return J;
    }
};

// nonlinear, non-autonomous field with analytic Jacobian:
//   f(t, y) = [ y0*y1 + sin(t) ; y0 - y1^2 ]
//   state_jacobian   = [ [ y1, y0 ] ; [ 1, -2*y1 ] ]
struct nonlinear_field {
    int n_components() const { return 2; }
    vector_t operator()(double t, const vector_t& y) const {
        vector_t out(2);
        out << y[0] * y[1] + std::sin(t), y[0] - y[1] * y[1];
        return out;
    }
    matrix_t state_jacobian(double, const vector_t& y) const {
        matrix_t J(2, 2);
        J << y[1], y[0], 1.0, -2.0 * y[1];
        return J;
    }
};

// central finite-difference Jacobian of a vector map g : R^d -> R^d
template <typename Map> matrix_t fd_jacobian(Map&& g, const vector_t& y, double h = 1e-6) {
    const int d = y.size();
    matrix_t J(d, d);
    vector_t yp = y, ym = y;
    for (int j = 0; j < d; ++j) {
        yp[j] = y[j] + h;
        ym[j] = y[j] - h;
        J.col(j) = (g(yp) - g(ym)) / (2 * h);
        yp[j] = y[j];
        ym[j] = y[j];
    }
    return J;
}

template <int Stages>
double integrate_scalar(const RKIntegrator<Stages>& integrator, const scalar_linear_field& f, double t0, double T,
                        int n, double y0) {
    const double dt = (T - t0) / n;
    vector_t y(1);
    y << y0;
    double t = t0;
    ode_rhs_field field {f};
    for (int k = 0; k < n; ++k) {
        y = integrator.step(field, t, y, dt);
        t += dt;
    }
    return y[0];
}

// apply visit(name, tableau, order, is_explicit) to every named scheme. The tableaux have distinct
// types (ButcherTableau<Stages> with different Stages), so they cannot share a container; instead the
// visitor is generic (templated on the tableau) and invoked once per scheme.
template <typename Visitor> void for_each_scheme(Visitor&& visit) {
    visit("forward_euler",     ode_schemes::forward_euler(),     1, true );
    visit("backward_euler",    ode_schemes::backward_euler(),    1, false);
    visit("crank_nicolson",    ode_schemes::crank_nicolson(),    2, false);
    visit("implicit_midpoint", ode_schemes::implicit_midpoint(), 2, false);
    visit("gauss_legendre_2",  ode_schemes::gauss_legendre_2(),  4, false);
    visit("rk4",               ode_schemes::rk4(),               4, true );
}

// the Gauss-Legendre family: the symplectic collocation schemes, the only ones for which the collocation
// machinery (stage values and costates on the nodes, stage-wise controls) is valid. Same generic-visitor trick as
// for_each_scheme -- the tableaux have distinct types and cannot share a container.
template <typename Visitor> void for_each_gauss_scheme(Visitor&& visit) {
    visit("implicit_midpoint", ode_schemes::implicit_midpoint());
    visit("gauss_legendre_2",  ode_schemes::gauss_legendre_2());
    visit("gauss_legendre_3",  ode_schemes::gauss_legendre_3());
    visit("gauss_legendre_4",  ode_schemes::gauss_legendre_4());
}

}   // namespace

// tableau structural properties: row-sum consistency (sum_j A_ij = c_i), sum_i b_i = 1,
// and correct explicit/implicit classification.
TEST(ode_test, tableau_consistency) {
    for_each_scheme([](const char* name, auto tab, int /*order*/, bool is_explicit) {
        constexpr int S = std::decay_t<decltype(tab)>::n_stages();
        double b_sum = 0;
        for (double bi : tab.b()) { b_sum += bi; }
        EXPECT_NEAR(b_sum, 1.0, 1e-12) << name;
        for (int i = 0; i < S; ++i) {
            double row_sum = 0;
            for (int j = 0; j < S; ++j) { row_sum += tab.A()[i][j]; }
            EXPECT_NEAR(row_sum, tab.c()[i], 1e-12) << name << " row " << i;
        }
        EXPECT_EQ(tab.is_explicit(), is_explicit) << name;
    });
}

// empirical convergence order on a scalar linear ODE: halving dt should shrink the global
// error by ~2^order.
TEST(ode_test, convergence_order) {
    scalar_linear_field f {-1.0};
    const double t0 = 0.0, T = 1.0, y0 = 1.0;
    const double exact = y0 * std::exp(f.a * (T - t0));
    for_each_scheme([&](const char* name, auto tab, int order, bool /*is_explicit*/) {
        RKIntegrator integrator(tab);
        double err_coarse = std::abs(integrate_scalar(integrator, f, t0, T, 20, y0) - exact);
        double err_fine = std::abs(integrate_scalar(integrator, f, t0, T, 40, y0) - exact);
        double p_est = std::log2(err_coarse / err_fine);
        EXPECT_NEAR(p_est, order, 0.5) << name << " (p_est = " << p_est << ")";
    });
}

// implicit one-step exactness on the linear test problem: a single GL2 step matches the
// (4,4)-Pade-like stability function to high accuracy, well beyond an order-1 scheme.
TEST(ode_test, single_step_linear) {
    scalar_linear_field f {-0.7};
    vector_t y(1);
    y << 1.0;
    const double dt = 0.1;
    const double exact = std::exp(f.a * dt);
    RKIntegrator gl2(ode_schemes::gauss_legendre_2());
    double y_next = gl2.step(ode_rhs_field {f}, 0.0, y, dt)[0];
    EXPECT_NEAR(y_next, exact, 1e-8);
}

// flow Jacobian d y_{n+1}/d y and increment Jacobian d Phi/d y against finite differences of
// the corresponding maps, for every tableau, on the nonlinear field.
TEST(ode_test, jacobians_match_finite_differences) {
    nonlinear_field f;
    ode_rhs_field field {f};
    vector_t y(2);
    y << 0.4, -0.6;
    const double t = 0.3, dt = 0.1;
    for_each_scheme([&](const char* name, auto tab, int /*order*/, bool /*is_explicit*/) {
        RKIntegrator integrator(tab);

        matrix_t flow_analytic = integrator.flow_jacobian(field, t, y, dt);
        matrix_t flow_fd = fd_jacobian([&](const vector_t& yy) { return integrator.step(field, t, yy, dt); }, y);
        EXPECT_LT((flow_analytic - flow_fd).cwiseAbs().maxCoeff(), 1e-6) << "flow " << name;

        matrix_t incr_analytic = integrator.increment_jacobian(field, t, y, dt);
        matrix_t incr_fd = fd_jacobian([&](const vector_t& yy) { return integrator.increment(field, t, yy, dt); }, y);
        EXPECT_LT((incr_analytic - incr_fd).cwiseAbs().maxCoeff(), 1e-6) << "increment " << name;
    });
}

// the integrator accepts a bare callable f(t, y): the state dimension is read from y and the
// Jacobian (needed by the implicit GL2 stage solve / sensitivities) falls back to finite
// differences. No n_components()/state_jacobian required.
TEST(ode_test, generic_callable_field) {
    auto f = [](double t, const vector_t& y) {
        vector_t out(2);
        out << y[0] * y[1] + std::sin(t), y[0] - y[1] * y[1];
        return out;
    };
    static_assert(fdapde::is_ode_rhs<decltype(f)>);
    static_assert(!fdapde::ode_rhs_has_state_jacobian<decltype(f)>);
    nonlinear_field analytic;   // same field, but with analytic Jacobian + n_components

    vector_t y(2);
    y << 0.4, -0.6;
    const double t = 0.3, dt = 0.1;
    RKIntegrator gl2(ode_schemes::gauss_legendre_2());   // implicit: exercises the Jacobian path
    ode_rhs_field field {f};                     // bare lambda -> finite-difference state Jacobian
    ode_rhs_field analytic_field {analytic};     // same dynamics, analytic state Jacobian
    // the converged step is independent of how the Jacobian is obtained (same stage solution)
    EXPECT_LT((gl2.step(field, t, y, dt) - gl2.step(analytic_field, t, y, dt)).cwiseAbs().maxCoeff(), 1e-10);
    // the flow Jacobian via the FD fallback matches the analytic-field one
    EXPECT_LT((gl2.flow_jacobian(field, t, y, dt) - gl2.flow_jacobian(analytic_field, t, y, dt)).cwiseAbs().maxCoeff(), 1e-5);
}

namespace {

// the coupled forward-adjoint system z = (y, p):  y' = f(y),  p' = -J_f(t, y)^T p, for the dynamics of
// nonlinear_field. Integrated on a DECREASING time grid it yields the exact continuous costate along the
// exact trajectory without interpolating y anywhere -- the reference for the convergence test below.
struct coupled_adjoint_field {
    int n_components() const { return 4; }
    vector_t operator()(double t, const vector_t& z) const {
        vector_t y = z.head(2), p = z.tail(2);
        vector_t out(4);
        out << nonlinear_field {}(t, y), -(nonlinear_field {}.state_jacobian(t, y).transpose() * p);
        return out;
    }
    matrix_t state_jacobian(double t, const vector_t& z) const {
        vector_t y = z.head(2), p = z.tail(2);
        matrix_t J = nonlinear_field {}.state_jacobian(t, y), H(2, 2), G = matrix_t::Zero(4, 4);
        H << 0.0, p[0], p[0], -2.0 * p[1];   // d(J^T p)/dy for these dynamics
        G.block(0, 0, 2, 2) = J;
        G.block(2, 0, 2, 2) = -H;
        G.block(2, 2, 2, 2) = -J.transpose();
        return G;
    }
};

// RK stage derivatives k_i, solved by an independent Newton iteration (deliberately NOT the integrator's
// own solve_stages_, so the collocation identity below is checked against a separate derivation).
template <int Stages>
std::vector<vector_t> independent_stages(const ButcherTableau<Stages>& tab, double t, const vector_t& y, double dt) {
    nonlinear_field f;
    const int d = y.size();
    std::vector<vector_t> K(Stages, f(t, y));
    for (int it = 0; it < 100; ++it) {
        matrix_t G(Stages * d, Stages * d);
        vector_t R(Stages * d);
        for (int i = 0; i < Stages; ++i) {
            vector_t arg = y;
            for (int j = 0; j < Stages; ++j) { arg += dt * tab.A()[i][j] * K[j]; }
            R.segment(i * d, d) = K[i] - f(t + tab.c()[i] * dt, arg);
            matrix_t J = f.state_jacobian(t + tab.c()[i] * dt, arg);
            for (int j = 0; j < Stages; ++j) {
                matrix_t blk = -dt * tab.A()[i][j] * J;
                if (i == j) { blk += matrix_t::Identity(d, d); }
                G.block(i * d, j * d, d, d) = blk;
            }
        }
        if (R.norm() < 1e-14) { break; }
        vector_t s = G.partialPivLu().solve(-R);
        for (int i = 0; i < Stages; ++i) { K[i] += s.segment(i * d, d); }
    }
    return K;
}

// residual of the collocation identity for the stage costates of one adjoint step. If psi_i are the values
// at t + c_i*dt of the collocation polynomial of p' = -J^T p (the polynomial pinned by p(t + dt) = p_next),
// then, since int_1^{c_i} l_j = a_ij - b_j,
//     psi_i = p_next + dt * sum_j (b_j - a_ij) J_j^T psi_j.
// This holds iff the tableau is symplectic; the residual is returned so both the positive and the negative
// case can be asserted.
template <int Stages> double collocation_residual(const ButcherTableau<Stages>& tab) {
    nonlinear_field f;
    ode_rhs_field field {f};
    RKIntegrator<Stages> integrator(tab);
    vector_t y(2), p_next(2);
    y << 0.7, -0.4;
    p_next << 1.3, 0.6;
    const double t = 0.3, dt = 0.05;
    fdapde::rk_adj_step_t r = integrator.adjoint_step_with_stages(field, t, y, dt, p_next);
    std::vector<vector_t> K = independent_stages(tab, t, y, dt);
    std::vector<matrix_t> J(Stages);
    for (int i = 0; i < Stages; ++i) {
        vector_t arg = y;
        for (int j = 0; j < Stages; ++j) { arg += dt * tab.A()[i][j] * K[j]; }
        J[i] = f.state_jacobian(t + tab.c()[i] * dt, arg);
    }
    double residual = 0;
    for (int i = 0; i < Stages; ++i) {
        vector_t rhs = p_next;
        for (int j = 0; j < Stages; ++j) {
            rhs += dt * (tab.b()[j] - tab.A()[i][j]) * (J[j].transpose() * vector_t(r.stages.col(j)));
        }
        residual = std::max(residual, (vector_t(r.stages.col(i)) - rhs).norm());
    }
    return residual;
}

}   // namespace

// The stage costates psi_i = lam_i/(dt b_i) exposed by adjoint_step_with_stages are the collocation values,
// at the stage times t + c_i*dt, of the adjoint equation p' = -J_f^T p -- for SYMPLECTIC (Gauss) tableaux
// only. This identity is what licenses interpolating the costate (and hence the optimal control
// u = psi/(2 lambda)) on the forward nodes between mesh nodes, so it is asserted to hold to round-off on the
// Gauss schemes and asserted to FAIL off them: a non-Gauss tableau must never be read that way.
TEST(ode_test, adjoint_stage_costates_are_collocation_values) {
    EXPECT_LT(collocation_residual(ode_schemes::implicit_midpoint()), 1e-13);
    EXPECT_LT(collocation_residual(ode_schemes::gauss_legendre_2()), 1e-13);
    // negative controls: non-symplectic tableaux (Lobatto IIIA, Radau IIA) do not satisfy it
    EXPECT_GT(collocation_residual(ode_schemes::crank_nicolson()), 1e-3);
    EXPECT_GT(collocation_residual(ode_schemes::backward_euler()), 1e-3);
}

// the stage costates converge to the CONTINUOUS costate with the orders collocation theory predicts:
// O(dt^(s+1)) at the interior stage points and O(dt^(2s)) at the mesh nodes (superconvergence). Checked
// against a reference obtained by integrating the coupled (y, p) system backward on a refined grid that
// contains every stage time, so no interpolation enters the comparison.
TEST(ode_test, adjoint_stage_costates_converge_to_continuous_costate) {
    const double T_end = 2.0;
    vector_t y0(2), pT(2);
    y0 << 0.7, -0.4;
    pT << 1.3, 0.6;
    RKIntegrator<2> reference(ode_schemes::gauss_legendre_2(), 200, 1e-15);
    ode_rhs_field coupled {coupled_adjoint_field {}};
    ode_rhs_field forward {nonlinear_field {}};

    auto run = [&](auto tab, int expected_stage_order, int expected_node_order) {
        constexpr int S = std::decay_t<decltype(tab)>::n_stages();
        RKIntegrator<S> integrator(tab);
        ode_rhs_field field {nonlinear_field {}};
        double prev_stage = 0, prev_node = 0;
        for (int m : {21, 41, 81}) {
            vector_t time = vector_t::LinSpaced(m, 0.0, T_end);
            matrix_t Y = integrator.integrate(field, time, y0);
            // discrete adjoint sweep, recording the stage costates and the nodal ones with their times
            std::vector<double> stage_t, node_t;
            std::vector<vector_t> stage_p, node_p;
            vector_t p = pT;
            for (int n = m - 2; n >= 0; --n) {
                const double dt = time[n + 1] - time[n];
                vector_t yn = Y.row(n).transpose();
                fdapde::rk_adj_step_t r = integrator.adjoint_step_with_stages(field, time[n], yn, dt, p);
                for (int i = 0; i < S; ++i) {
                    stage_t.push_back(time[n] + tab.c()[i] * dt);
                    stage_p.push_back(r.stages.col(i));
                }
                p = r.costate;
                node_t.push_back(time[n]);
                node_p.push_back(p);
            }
            // build a refinement of [0, T] containing every time we need to compare at
            std::vector<double> knots = stage_t;
            knots.insert(knots.end(), node_t.begin(), node_t.end());
            knots.push_back(0.0);
            knots.push_back(T_end);
            std::sort(knots.begin(), knots.end());
            knots.erase(std::unique(knots.begin(), knots.end()), knots.end());
            std::vector<double> fine;
            const int sub = 16;
            for (std::size_t i = 0; i + 1 < knots.size(); ++i) {
                for (int k = 0; k < sub; ++k) { fine.push_back(knots[i] + (knots[i + 1] - knots[i]) * k / sub); }
            }
            fine.push_back(T_end);
            vector_t ascending = Eigen::Map<vector_t>(fine.data(), static_cast<int>(fine.size()));
            vector_t descending = ascending.reverse();
            // y(T) from an accurate forward solve, then the coupled system backward from (y(T), p(T))
            matrix_t Yf = reference.integrate(forward, ascending, y0);
            vector_t z0(4);
            z0 << Yf.row(Yf.rows() - 1).transpose(), pT;
            matrix_t Z = reference.integrate(coupled, descending, z0);
            auto err_at = [&](const std::vector<double>& ts, const std::vector<vector_t>& ps) {
                double e = 0;
                for (std::size_t j = 0; j < ts.size(); ++j) {
                    for (int r2 = 0; r2 < descending.size(); ++r2) {
                        if (std::abs(descending[r2] - ts[j]) < 1e-12) {
                            e = std::max(e, (vector_t(Z.row(r2).tail(2).transpose()) - ps[j]).norm());
                            break;
                        }
                    }
                }
                return e;
            };
            const double e_stage = err_at(stage_t, stage_p), e_node = err_at(node_t, node_p);
            if (prev_stage > 0) {
                EXPECT_GT(std::log2(prev_stage / e_stage), expected_stage_order - 0.3) << "stage order, m = " << m;
                EXPECT_GT(std::log2(prev_node / e_node), expected_node_order - 0.3) << "node order, m = " << m;
            }
            prev_stage = e_stage;
            prev_node = e_node;
        }
    };
    run(ode_schemes::implicit_midpoint(), 2, 2);   // GL1: s + 1 = 2, 2s = 2
    run(ode_schemes::gauss_legendre_2(), 3, 4);    // GL2: s + 1 = 3, 2s = 4
}

/* GL4's coefficient matrix is DERIVED, not transcribed: A_ij = int_0^{c_i} l_j, the defining relation of a
collocation method (see gauss_legendre_4). That makes the factory short and unambiguous, but it also means
nothing pins A unless the order conditions do -- which is what this checks. B(k) fixes the quadrature order,
C(q) and D(r) the simplifying assumptions a Gauss scheme of s stages must satisfy, and their combination is
what gives order 2s = 8. */
TEST(ode_test, gauss_legendre_4_satisfies_its_order_conditions) {
    const auto tab = ode_schemes::gauss_legendre_4();
    const auto& A = tab.A();
    const auto& b = tab.b();
    const auto& c = tab.c();
    // B(k): sum_i b_i c_i^(k-1) = 1/k, for k = 1..2s. Order 8 exactly: it must FAIL at k = 9.
    for (int k = 1; k <= 8; ++k) {
        double sum = 0;
        for (int i = 0; i < 4; ++i) { sum += b[i] * std::pow(c[i], k - 1); }
        EXPECT_NEAR(sum, 1.0 / k, 1e-14) << "B(" << k << ")";
    }
    double sum9 = 0;
    for (int i = 0; i < 4; ++i) { sum9 += b[i] * std::pow(c[i], 8); }
    EXPECT_GT(std::abs(sum9 - 1.0 / 9), 1e-8) << "GL4 is order 8, not more";
    // C(s): sum_j A_ij c_j^(k-1) = c_i^k / k -- stage order s, what makes it a collocation method
    for (int k = 1; k <= 4; ++k) {
        for (int i = 0; i < 4; ++i) {
            double sum = 0;
            for (int j = 0; j < 4; ++j) { sum += A[i][j] * std::pow(c[j], k - 1); }
            EXPECT_NEAR(sum, std::pow(c[i], k) / k, 1e-14) << "C(" << k << ") row " << i;
        }
    }
    // D(s): sum_i b_i c_i^(k-1) A_ij = b_j (1 - c_j^k) / k
    for (int k = 1; k <= 4; ++k) {
        for (int j = 0; j < 4; ++j) {
            double sum = 0;
            for (int i = 0; i < 4; ++i) { sum += b[i] * std::pow(c[i], k - 1) * A[i][j]; }
            EXPECT_NEAR(sum, b[j] * (1 - std::pow(c[j], k)) / k, 1e-14) << "D(" << k << ") col " << j;
        }
    }
    // row-sum consistency and the Gauss nodes' symmetry about 1/2
    for (int i = 0; i < 4; ++i) {
        double row = 0;
        for (int j = 0; j < 4; ++j) { row += A[i][j]; }
        EXPECT_NEAR(row, c[i], 1e-14);
        EXPECT_NEAR(c[i] + c[3 - i], 1.0, 1e-14);
        EXPECT_NEAR(b[i], b[3 - i], 1e-14);
    }
}

// the tableau predicates that gate the collocation machinery. is_collocation (distinct nodes) decides
// whether the stage values lie on one polynomial at all; is_symplectic decides whether the stage costates
// may be read on the same nodes (see adjoint_stage_costates_are_collocation_values).
TEST(ode_test, tableau_collocation_and_symplecticity_predicates) {
    EXPECT_TRUE(ode_schemes::implicit_midpoint().is_symplectic());
    EXPECT_TRUE(ode_schemes::gauss_legendre_2().is_symplectic());
    EXPECT_TRUE(ode_schemes::gauss_legendre_3().is_symplectic());
    EXPECT_TRUE(ode_schemes::gauss_legendre_4().is_symplectic());
    EXPECT_FALSE(ode_schemes::forward_euler().is_symplectic());
    EXPECT_FALSE(ode_schemes::backward_euler().is_symplectic());
    EXPECT_FALSE(ode_schemes::crank_nicolson().is_symplectic());
    EXPECT_FALSE(ode_schemes::rk4().is_symplectic());
    // every scheme here has distinct nodes except rk4 (c = 0, 1/2, 1/2, 1)
    EXPECT_TRUE(ode_schemes::gauss_legendre_2().is_collocation());
    EXPECT_TRUE(ode_schemes::gauss_legendre_3().is_collocation());
    EXPECT_TRUE(ode_schemes::gauss_legendre_4().is_collocation());
    EXPECT_TRUE(ode_schemes::crank_nicolson().is_collocation());
    EXPECT_FALSE(ode_schemes::rk4().is_collocation());
    // Gauss nodes are symmetric about 1/2, so the adjoint's reflected nodes 1 - c_i permute the node set --
    // the second ingredient (with symplecticity) behind reading stage costates as collocation data
    for_each_gauss_scheme([](const char* name, auto tab) {
        constexpr int S = std::decay_t<decltype(tab)>::n_stages();
        for (int i = 0; i < S; ++i) {
            bool found = false;
            for (int j = 0; j < S; ++j) {
                if (std::abs((1.0 - tab.c()[i]) - tab.c()[j]) < 1e-14) { found = true; }
            }
            EXPECT_TRUE(found) << "node reflection " << name << " i = " << i;
        }
    });
}

// The stage values returned by step_with_stage_values are the step's collocation polynomial at the nodes:
// they come with the same step as step(), equal y + dt sum_j a_ij k_j for stage derivatives from an
// independent Newton iteration, and match the solution at t + c_i*dt at the collocation stage order s + 1.
// That order is what lets a fitted trajectory be rebuilt from its stage values between the mesh nodes.
TEST(ode_test, stage_values_lie_on_the_collocation_polynomial) {
    ode_rhs_field field {nonlinear_field {}};
    vector_t y0(2);
    y0 << 0.7, -0.4;
    for_each_gauss_scheme([&](const char* name, auto tab) {
        constexpr int S = std::decay_t<decltype(tab)>::n_stages();
        RKIntegrator<S> integrator(tab);
        const double t = 0.3, dt = 0.1;
        fdapde::rk_stage_step_t s = integrator.step_with_stage_values(field, t, y0, dt);
        EXPECT_LT((s.state - integrator.step(field, t, y0, dt)).cwiseAbs().maxCoeff(), 1e-14) << name;
        const std::vector<vector_t> K = independent_stages(tab, t, y0, dt);
        for (int i = 0; i < S; ++i) {
            vector_t Yi = y0;
            for (int j = 0; j < S; ++j) { Yi += dt * tab.A()[i][j] * K[j]; }
            EXPECT_LT((vector_t(s.values.col(i)) - Yi).cwiseAbs().maxCoeff(), 1e-12) << name << " stage " << i;
        }
        // accuracy at the stage times: O(dt^(s+1)) against a finely resolved reference
        RKIntegrator<3> reference(ode_schemes::gauss_legendre_3(), 200, 1e-15);
        double prev = 0;
        for (double h : {0.4, 0.2, 0.1}) {
            fdapde::rk_stage_step_t step = integrator.step_with_stage_values(field, t, y0, h);
            double err = 0;
            for (int i = 0; i < S; ++i) {
                vector_t exact = reference.integrate(
                  field, vector_t::LinSpaced(2001, t, t + tab.c()[i] * h), y0).row(2000).transpose();
                err = std::max(err, (vector_t(step.values.col(i)) - exact).norm());
            }
            if (prev > 0) { EXPECT_GT(std::log2(prev / err), S + 1 - 0.4) << "stage order " << name; }
            prev = err;
        }
    });
}

// The stage representation makes the control's L2 norm EXACT, not a quadrature approximation: on Gauss
// nodes the Gram matrix of the Lagrange basis is diagonal, int_0^1 l_i l_j = b_i delta_ij, so
// int_0^1 ||u(theta)||^2 dtheta = sum_i b_i ||u_i||^2 for ANY stage values. Verified against a fine
// numerical integration of the actual control polynomial.
TEST(ode_test, stage_control_penalty_is_the_exact_l2_norm) {
    for_each_gauss_scheme([](const char* name, auto tab) {
        constexpr int S = std::decay_t<decltype(tab)>::n_stages();
        const int d = 2;
        Eigen::Matrix<double, fdapde::Dynamic, fdapde::Dynamic> U(d, S);   // deliberately non-constant stage values
        for (int i = 0; i < S; ++i) {
            U(0, i) = 0.4 - 0.9 * i + 0.2 * i * i;
            U(1, i) = -0.3 + 0.7 * i;
        }
        double b_form = 0;
        for (int i = 0; i < S; ++i) { b_form += tab.b()[i] * U.col(i).squaredNorm(); }
        // fine composite Simpson integration of ||sum_i l_i(theta) u_i||^2 over [0, 1]
        const int n = 20001;
        double quad = 0;
        for (int k = 0; k < n; ++k) {
            const double theta = double(k) / (n - 1);
            const double w = (k == 0 || k == n - 1) ? 1.0 : (k % 2 ? 4.0 : 2.0);
            Eigen::Matrix<double, S, 1> l;   // the Lagrange basis on the tableau nodes, l_i(theta)
            for (int i = 0; i < S; ++i) {
                l[i] = 1.0;
                for (int j = 0; j < S; ++j) {
                    if (j != i) { l[i] *= (theta - tab.c()[j]) / (tab.c()[i] - tab.c()[j]); }
                }
            }
            quad += w * (U * l).squaredNorm();
        }
        quad *= 1.0 / (3.0 * (n - 1));
        EXPECT_NEAR(b_form, quad, 1e-9) << name << ": sum b_i ||u_i||^2 = " << b_form << " vs int = " << quad;
    });
}