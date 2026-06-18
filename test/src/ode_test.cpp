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
using fdapde::RKIntegrator;
using fdapde::fd_ode_field;
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
    matrix_t df_dy(double, const vector_t&) const {
        matrix_t J(1, 1);
        J << a;
        return J;
    }
};

// nonlinear, non-autonomous field with analytic Jacobian:
//   f(t, y) = [ y0*y1 + sin(t) ; y0 - y1^2 ]
//   df_dy   = [ [ y1, y0 ] ; [ 1, -2*y1 ] ]
struct nonlinear_field {
    int n_components() const { return 2; }
    vector_t operator()(double t, const vector_t& y) const {
        vector_t out(2);
        out << y[0] * y[1] + std::sin(t), y[0] - y[1] * y[1];
        return out;
    }
    matrix_t df_dy(double, const vector_t& y) const {
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

double integrate_scalar(const RKIntegrator& integrator, const scalar_linear_field& f, double t0, double T, int n,
                        double y0) {
    const double dt = (T - t0) / n;
    vector_t y(1);
    y << y0;
    double t = t0;
    for (int k = 0; k < n; ++k) {
        y = integrator.step(f, t, y, dt);
        t += dt;
    }
    return y[0];
}

struct named_tableau {
    const char* name;
    ButcherTableau tableau;
    int order;
    bool is_explicit;
};

std::vector<named_tableau> all_tableaux() {
    return {
      {"forward_euler",     ode_schemes::forward_euler(),     1, true },
      {"backward_euler",    ode_schemes::backward_euler(),    1, false},
      {"crank_nicolson",    ode_schemes::crank_nicolson(),    2, false},
      {"implicit_midpoint", ode_schemes::implicit_midpoint(), 2, false},
      {"gauss_legendre_2",  ode_schemes::gauss_legendre_2(),  4, false},
      {"rk4",               ode_schemes::rk4(),               4, true }
    };
}

}   // namespace

// tableau structural properties: row-sum consistency (sum_j A_ij = c_i), sum_i b_i = 1,
// and correct explicit/implicit classification.
TEST(ode_test, tableau_consistency) {
    for (const auto& nt : all_tableaux()) {
        const ButcherTableau& T = nt.tableau;
        EXPECT_NEAR(T.b().sum(), 1.0, 1e-12) << nt.name;
        for (int i = 0; i < T.n_stages(); ++i) {
            EXPECT_NEAR(T.A().row(i).sum(), T.c()(i), 1e-12) << nt.name << " row " << i;
        }
        EXPECT_EQ(T.is_explicit(), nt.is_explicit) << nt.name;
    }
}

// empirical convergence order on a scalar linear ODE: halving dt should shrink the global
// error by ~2^order.
TEST(ode_test, convergence_order) {
    scalar_linear_field f {-1.0};
    const double t0 = 0.0, T = 1.0, y0 = 1.0;
    const double exact = y0 * std::exp(f.a * (T - t0));
    for (const auto& nt : all_tableaux()) {
        RKIntegrator integrator(nt.tableau);
        double err_coarse = std::abs(integrate_scalar(integrator, f, t0, T, 20, y0) - exact);
        double err_fine = std::abs(integrate_scalar(integrator, f, t0, T, 40, y0) - exact);
        double p_est = std::log2(err_coarse / err_fine);
        EXPECT_NEAR(p_est, nt.order, 0.5) << nt.name << " (p_est = " << p_est << ")";
    }
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
    double y_next = gl2.step(f, 0.0, y, dt)[0];
    EXPECT_NEAR(y_next, exact, 1e-8);
}

// flow Jacobian d y_{n+1}/d y and increment Jacobian d Phi/d y against finite differences of
// the corresponding maps, for every tableau, on the nonlinear field.
TEST(ode_test, jacobians_match_finite_differences) {
    nonlinear_field f;
    vector_t y(2);
    y << 0.4, -0.6;
    const double t = 0.3, dt = 0.1;
    for (const auto& nt : all_tableaux()) {
        RKIntegrator integrator(nt.tableau);

        matrix_t flow_analytic = integrator.flow_jacobian(f, t, y, dt);
        matrix_t flow_fd = fd_jacobian([&](const vector_t& yy) { return integrator.step(f, t, yy, dt); }, y);
        EXPECT_LT((flow_analytic - flow_fd).cwiseAbs().maxCoeff(), 1e-6) << "flow " << nt.name;

        matrix_t incr_analytic = integrator.increment_jacobian(f, t, y, dt);
        matrix_t incr_fd = fd_jacobian([&](const vector_t& yy) { return integrator.increment(f, t, yy, dt); }, y);
        EXPECT_LT((incr_analytic - incr_fd).cwiseAbs().maxCoeff(), 1e-6) << "increment " << nt.name;
    }
}

// fd_ode_field adaptor: its finite-difference Jacobian matches the analytic one, and it
// satisfies the ode_field concept so it drives the integrator.
TEST(ode_test, fd_ode_field_adaptor) {
    nonlinear_field analytic;
    auto bare = [](double t, const vector_t& y) {
        vector_t out(2);
        out << y[0] * y[1] + std::sin(t), y[0] - y[1] * y[1];
        return out;
    };
    fd_ode_field fd_field(bare, 2);
    static_assert(fdapde::is_ode_field<decltype(fd_field)>);

    vector_t y(2);
    y << 0.4, -0.6;
    const double t = 0.3;
    EXPECT_LT((fd_field.df_dy(t, y) - analytic.df_dy(t, y)).cwiseAbs().maxCoeff(), 1e-6);

    // the adaptor integrates to the same step as the analytic field
    RKIntegrator integrator(ode_schemes::gauss_legendre_2());
    const double dt = 0.1;
    EXPECT_LT((integrator.step(fd_field, t, y, dt) - integrator.step(analytic, t, y, dt)).cwiseAbs().maxCoeff(), 1e-7);
}
