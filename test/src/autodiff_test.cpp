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

// Exercises the autodiff adapter (fdaPDE/autodiff.h). The whole file is inert unless the autodiff headers
// are on the include path (configure the test build with -DAUTODIFF_INC=/path/to/autodiff), which is the
// same optionality the module itself has.

#include <cmath>
#include <gtest/gtest.h>   // testing framework

#include <fdaPDE/autodiff.h>
#include <fdaPDE/optimization.h>

#ifdef FDAPDE_HAS_AUTODIFF

using fdapde::ButcherTableau;
using fdapde::RKIntegrator;
using fdapde::ad_ode_rhs;
using fdapde::ode_rhs_field;
namespace ode_schemes = fdapde::ode_schemes;

using vector_t = Eigen::Matrix<double, fdapde::Dynamic, 1>;
using matrix_t = Eigen::Matrix<double, fdapde::Dynamic, fdapde::Dynamic>;

namespace ad_test {

// a nonlinear, non-autonomous, theta-parameterized field written ONCE, generic in the scalar type:
//   f(t, y, th) = [ th0 * y0 * y1 + sin(t) ; th1 * y0 - th2 * y1^2 ]
struct generic_field {
    template <typename Scalar>
    Eigen::Matrix<Scalar, 2, 1> operator()(
      double t, const Eigen::Matrix<Scalar, fdapde::Dynamic, 1>& y,
      const Eigen::Matrix<Scalar, fdapde::Dynamic, 1>& th) const {
        Eigen::Matrix<Scalar, 2, 1> out;
        out[0] = th[0] * y[0] * y[1] + std::sin(t);
        out[1] = th[1] * y[0] - th[2] * y[1] * y[1];
        return out;
    }
};
// the same dynamics, hand-differentiated: the reference every AD result is checked against
struct analytic_field {
    vector_t operator()(double t, const vector_t& y, const vector_t& th) const {
        vector_t out(2);
        out << th[0] * y[0] * y[1] + std::sin(t), th[1] * y[0] - th[2] * y[1] * y[1];
        return out;
    }
    matrix_t state_jacobian(double, const vector_t& y, const vector_t& th) const {
        matrix_t J(2, 2);
        J << th[0] * y[1], th[0] * y[0], th[1], -2.0 * th[2] * y[1];
        return J;
    }
    matrix_t param_jacobian(double, const vector_t& y, const vector_t&) const {
        matrix_t J = matrix_t::Zero(2, 3);
        J(0, 0) = y[0] * y[1];
        J(1, 1) = y[0];
        J(1, 2) = -y[1] * y[1];
        return J;
    }
};
// a plain (non-parameterized) generic field: f(t, y) = [ y0 * y1 ; -y1^2 ]
struct generic_plain_field {
    template <typename Scalar>
    Eigen::Matrix<Scalar, 2, 1> operator()(double, const Eigen::Matrix<Scalar, fdapde::Dynamic, 1>& y) const {
        Eigen::Matrix<Scalar, 2, 1> out;
        out[0] = y[0] * y[1];
        out[1] = -y[1] * y[1];
        return out;
    }
};
// a scalar objective, generic in the scalar type: the Rosenbrock function
struct rosenbrock {
    template <typename Scalar> Scalar operator()(const Eigen::Matrix<Scalar, fdapde::Dynamic, 1>& x) const {
        return (1 - x[0]) * (1 - x[0]) + 100 * (x[1] - x[0] * x[0]) * (x[1] - x[0] * x[0]);
    }
};

vector_t vec2(double a, double b) {
    vector_t v(2);
    v << a, b;
    return v;
}
vector_t vec3(double a, double b, double c) {
    vector_t v(3);
    v << a, b, c;
    return v;
}

}   // namespace ad_test

// the adapter must present itself to the rest of the library as an ordinary, analytically differentiable
// rhs -- that is what makes it a drop-in replacement
TEST(ad_ode_rhs, models_the_ode_rhs_concepts) {
    using ad_field_t = ad_ode_rhs<ad_test::generic_field>;
    static_assert(fdapde::is_parameterized_ode_rhs<ad_field_t>);
    static_assert(fdapde::parameterized_ode_rhs_has_state_jacobian<ad_field_t>);
    static_assert(fdapde::parameterized_ode_rhs_has_param_jacobian<ad_field_t>);
    // a fixed-size return type must survive the wrapper, or the integrator silently loses its heap-free
    // fixed-size stage math
    static_assert(fdapde::ode_rhs_dim_v<ad_field_t> == 2);

    using ad_plain_t = ad_ode_rhs<ad_test::generic_plain_field>;
    static_assert(fdapde::is_ode_rhs<ad_plain_t>);
    static_assert(fdapde::ode_rhs_has_state_jacobian<ad_plain_t>);
    static_assert(fdapde::ode_rhs_dim_v<ad_plain_t> == 2);
}

// the AD Jacobians ARE the analytic ones (to machine precision), for both the parameterized and the plain
// field, at several points
TEST(ad_ode_rhs, jacobians_match_the_hand_written_ones) {
    ad_ode_rhs ad {ad_test::generic_field {}};
    ad_test::analytic_field analytic;
    for (double s : {-1.0, 0.0, 0.7}) {
        vector_t y = ad_test::vec2(0.4 + s, -0.2 - s), th = ad_test::vec3(0.8, 1.2, 0.9);
        const double t = 0.3 + s;
        EXPECT_LT((vector_t(ad(t, y, th)) - analytic(t, y, th)).cwiseAbs().maxCoeff(), 1e-15);
        EXPECT_LT((ad.state_jacobian(t, y, th) - analytic.state_jacobian(t, y, th)).cwiseAbs().maxCoeff(), 1e-15);
        EXPECT_LT((ad.param_jacobian(t, y, th) - analytic.param_jacobian(t, y, th)).cwiseAbs().maxCoeff(), 1e-15);
    }
    // plain field: df/dy = [ y1  y0 ; 0  -2 y1 ]
    ad_ode_rhs ad_plain {ad_test::generic_plain_field {}};
    vector_t y = ad_test::vec2(0.4, -0.2);
    matrix_t expected(2, 2);
    expected << y[1], y[0], 0.0, -2 * y[1];
    EXPECT_LT((ad_plain.state_jacobian(0.0, y) - expected).cwiseAbs().maxCoeff(), 1e-15);
}

// the reason to prefer AD over the finite-difference fallback: same interface, same cost class, but the
// truncation error is gone. ode_rhs_field routes a functor without Jacobians through central differences.
TEST(ad_ode_rhs, is_exact_where_finite_differences_are_not) {
    struct value_only_field {   // the same dynamics with NO Jacobian methods -> finite differences
        vector_t operator()(double t, const vector_t& y, const vector_t& th) const {
            return ad_test::analytic_field {}(t, y, th);
        }
    };
    vector_t y = ad_test::vec2(0.4, -0.2), th = ad_test::vec3(0.8, 1.2, 0.9);
    matrix_t exact = ad_test::analytic_field {}.param_jacobian(0.3, y, th);
    ode_rhs_field fd_field {value_only_field {}, th};
    ode_rhs_field ad_field {ad_ode_rhs {ad_test::generic_field {}}, th};
    double fd_err = (fd_field.param_jacobian(0.3, y) - exact).cwiseAbs().maxCoeff();
    double ad_err = (ad_field.param_jacobian(0.3, y) - exact).cwiseAbs().maxCoeff();
    EXPECT_LT(ad_err, 1e-15);
    EXPECT_GT(fd_err, 1e-12);      // central differences are nowhere near machine precision ...
    EXPECT_LT(ad_err, fd_err);     // ... and AD is strictly better on the same call
}

// an AD field integrates exactly like the hand-differentiated one, under an implicit scheme (where the
// state Jacobian drives the Newton stage solve, so a wrong one would show up in the trajectory)
TEST(ad_ode_rhs, integrates_like_the_analytic_field) {
    vector_t th = ad_test::vec3(1.0, 1.0, 1.0), y0 = ad_test::vec2(0.5, -0.3);
    vector_t time(21);
    for (int i = 0; i < 21; ++i) { time[i] = 2.0 * i / 20; }
    auto integrator = RKIntegrator(ode_schemes::gauss_legendre_2());
    matrix_t Y_ad = integrator.integrate(fdapde::make_ad_ode_rhs_field(ad_test::generic_field {}, th), time, y0);
    matrix_t Y_an = integrator.integrate(ode_rhs_field {ad_test::analytic_field {}, th}, time, y0);
    ASSERT_EQ(Y_ad.rows(), 21);
    EXPECT_LT((Y_ad - Y_an).cwiseAbs().maxCoeff(), 1e-12);
}

// the objective adapter supplies value, gradient and Hessian, and they are the analytic ones
TEST(ad_objective, gradient_and_hessian_are_exact) {
    auto objective = fdapde::make_ad_objective(ad_test::rosenbrock {});
    vector_t x = ad_test::vec2(-1.2, 1.0);
    // d/dx0 = -2(1 - x0) - 400 x0 (x1 - x0^2),  d/dx1 = 200 (x1 - x0^2)
    vector_t expected_grad = ad_test::vec2(
      -2 * (1 - x[0]) - 400 * x[0] * (x[1] - x[0] * x[0]), 200 * (x[1] - x[0] * x[0]));
    matrix_t expected_hess(2, 2);
    expected_hess << 2 - 400 * (x[1] - 3 * x[0] * x[0]), -400 * x[0], -400 * x[0], 200;
    EXPECT_NEAR(objective(x), (1 - x[0]) * (1 - x[0]) + 100 * (x[1] - x[0] * x[0]) * (x[1] - x[0] * x[0]), 1e-15);
    EXPECT_LT((objective.gradient()(x) - expected_grad).cwiseAbs().maxCoeff(), 1e-12);
    EXPECT_LT((objective.hessian()(x) - expected_hess).cwiseAbs().maxCoeff(), 1e-12);
}

// and it is consumed unchanged by the optimization module: every gradient-based algorithm accepts it
TEST(ad_objective, drives_the_optimization_algorithms) {
    auto objective = fdapde::make_ad_objective(ad_test::rosenbrock {});
    vector_t x0 = ad_test::vec2(-1.2, 1.0), optimum = ad_test::vec2(1.0, 1.0);

    fdapde::BFGS<fdapde::Dynamic> bfgs(500, 1e-8, 1.0);
    EXPECT_LT((bfgs.optimize(objective, x0, fdapde::WolfeLineSearch()) - optimum).cwiseAbs().maxCoeff(), 1e-4);

    fdapde::LBFGS<fdapde::Dynamic> lbfgs(500, 1e-8, 1.0, 10);
    EXPECT_LT((lbfgs.optimize(objective, x0, fdapde::WolfeLineSearch()) - optimum).cwiseAbs().maxCoeff(), 1e-4);

    fdapde::Newton<fdapde::Dynamic> newton(200, 1e-8, 1.0);   // also exercises the AD Hessian
    EXPECT_LT(
      (newton.optimize(objective, x0, fdapde::BacktrackingLineSearch()) - optimum).cwiseAbs().maxCoeff(), 1e-4);
}

#endif   // FDAPDE_HAS_AUTODIFF
