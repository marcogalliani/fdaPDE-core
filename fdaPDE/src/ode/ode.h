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

#ifndef __ODE_H__
#define __ODE_H__

#include "header_check.h"

namespace fdapde {

// f models the right-hand side of an ODE y' = f(t, y): callable as f(t, y) -> R^d. The system
// dimension d is a compile-time property (see ode_rhs_dim), not a runtime method.
template <typename F>
concept is_ode_rhs = std::is_invocable_r_v<
  Eigen::Matrix<double, Dynamic, 1>, const F&, double, const Eigen::Matrix<double, Dynamic, 1>&>;

// f additionally exposes an analytic Jacobian df_dy(t, y) -> R^{d x d}; when absent the integrator
// falls back to central finite differences.
template <typename F>
concept has_jacobian = requires(const F& f, double t, const Eigen::Matrix<double, Dynamic, 1>& y) {
    { f.df_dy(t, y) } -> std::convertible_to<Eigen::Matrix<double, Dynamic, Dynamic>>;
};

// ode_rhs_dim<F>: the system dimension of an ODE rhs, read statically. A field that knows its
// dimension exposes it as `static constexpr int dim` (ode_rhs_field does); otherwise it is taken from
// the compile-time row count of f(t, y)'s return type (Dim for a fixed-size return, Dynamic for a
// dynamic VectorXd). So a user functor opts into the static path simply by returning a fixed-size
// vector; returning VectorXd keeps the (default) dynamic path.
template <typename F> constexpr int ode_rhs_dim() {
    using G = std::decay_t<F>;
    if constexpr (requires { G::dim; }) {
        return G::dim;
    } else {
        return std::invoke_result_t<G, double, Eigen::Matrix<double, Dynamic, 1>>::RowsAtCompileTime;
    }
}
template <typename F> inline constexpr int ode_rhs_dim_v = ode_rhs_dim<F>();

// Vector field f : (t, y) -> R^d for the right-hand side of an ODE y' = f(t, y), templated on the
// system dimension Dim (Dynamic by default). It is a library MatrixField (a VectorField over the
// augmented input [t; y]) extended with the ODE rhs interface f(t, y), df/dy (the analytic Jacobian
// of the wrapped functor when present, otherwise the library finite-difference Jacobian restricted to
// the y-block), and field algebra: f + c (a constant control) and f1 + f2 (two dynamics) are again
// ode_rhs_field objects, preserving the analytic Jacobian.
//
// Dim is normally deduced from the wrapped functor (see ode_rhs_dim / the deduction guide below): a
// fixed-size return type gives a static Dim (fixed-size MatrixField, fixed-size stage math in the
// integrator), a VectorXd return gives Dim = Dynamic. With a static Dim the inherited MatrixField
// components are built at construction (so the inherited field expressions are usable); under
// Dim = Dynamic the dimension is known only at evaluation, so those inherited operators are not
// populated (operator() / df_dy / operator+ work regardless).
// Dim has no default: it is supplied explicitly (ode_rhs_field<2>) or deduced by the guide below. A
// defaulted Dim would let the implicit constructor deduction guide win over the explicit one and
// always deduce Dynamic. The dynamic field is spelled ode_rhs_field<Dynamic>.
template <int Dim>
class ode_rhs_field :
    public MatrixField<
      (Dim == Dynamic ? Dynamic : Dim + 1), Dim, 1, std::function<double(const Eigen::Matrix<double, Dynamic, 1>&)>> {
   public:
    using vector_t = Eigen::Matrix<double, Dynamic, 1>;
    using matrix_t = Eigen::Matrix<double, Dynamic, Dynamic>;
    static constexpr int dim = Dim;   // read by ode_rhs_dim and the integrator's stage math

   private:
    using component_t = std::function<double(const vector_t&)>;
    using Base = MatrixField<(Dim == Dynamic ? Dynamic : Dim + 1), Dim, 1, component_t>;
    using fd_field_t = MatrixField<Dynamic, Dynamic, 1, component_t>;   // dynamic FD source (uniform path)

    // the scalar component y -> f(t, y)[i] over the augmented input [t; y]
    component_t component_(int i) const {
        return component_t(
          [value = value_, i](const vector_t& ty) -> double { return value(ty[0], ty.tail(ty.size() - 1))[i]; });
    }
    // populate the inherited (statically-sized) MatrixField base so the inherited field expressions are
    // usable; a no-op under Dim = Dynamic, where the base size is unknown until a y is seen.
    void build_components_() {
        if constexpr (Dim != Dynamic) {
            for (int i = 0; i < Dim; ++i) { (*this)[i] = component_(i); }
        }
    }

   public:
    ode_rhs_field() = default;
    template <typename F>
        requires(is_ode_rhs<F> && !std::is_same_v<std::decay_t<F>, ode_rhs_field>)
    ode_rhs_field(F&& field) {
        auto ptr = std::make_shared<std::decay_t<F>>(std::forward<F>(field));
        value_ = [ptr](double t, const vector_t& y) -> vector_t { return (*ptr)(t, y); };
        if constexpr (has_jacobian<std::decay_t<F>>) {
            analytic_df_dy_ = [ptr](double t, const vector_t& y) -> matrix_t { return ptr->df_dy(t, y); };
        }
        build_components_();
    }

    using Base::operator();   // inherited MatrixField evaluation / component access (static Dim)
    // ODE rhs interface: f(t, y), through the value callable (a single functor call)
    vector_t operator()(double t, const vector_t& y) const { return value_(t, y); }

    // df/dy: analytic when available, otherwise the library finite-difference Jacobian of the field over
    // [t; y] restricted to the y-block. The FD source is a dynamic MatrixField built lazily from value_,
    // so the path is identical for static and dynamic Dim (d is read from y).
    matrix_t df_dy(double t, const vector_t& y) const {
        if (analytic_df_dy_) { return analytic_df_dy_(t, y); }
        const int d = y.size();
        if (!fd_jacobian_ || fd_dim_ != d) {
            auto field = std::make_shared<fd_field_t>();
            field->resize(d + 1, d);
            for (int i = 0; i < d; ++i) { (*field)[i] = component_(i); }
            fd_field_ = field;
            fd_jacobian_ = std::make_shared<Jacobian<fd_field_t>>(*field);
            fd_dim_ = d;
        }
        vector_t ty(d + 1);
        ty[0] = t;
        ty.tail(d) = y;
        matrix_t J = (*fd_jacobian_)(ty);         // (d+1) x d, entry (i, j) = d f_j / d x_i
        return J.middleRows(1, d).transpose();    // d x d,     entry (i, j) = d f_i / d y_j
    }
    explicit operator bool() const { return static_cast<bool>(value_); }

    // TODO: check MartixField implementation, it should be mirrored here
    // field algebra preserving the analytic Jacobian (the ODE-penalty solvers inject the control as
    // f + u_t): g = f + c (constant), and g = f1 + f2 (the Jacobian is analytic only when both are).
    ode_rhs_field operator+(const vector_t& c) const {
        ode_rhs_field g;
        g.value_ = [v = value_, c](double t, const vector_t& y) -> vector_t { return v(t, y) + c; };
        g.analytic_df_dy_ = analytic_df_dy_;   // a constant shift leaves the Jacobian unchanged
        g.build_components_();
        return g;
    }
    ode_rhs_field operator+(const ode_rhs_field& rhs) const {
        ode_rhs_field g;
        g.value_ = [a = value_, b = rhs.value_](double t, const vector_t& y) -> vector_t { return a(t, y) + b(t, y); };
        if (analytic_df_dy_ && rhs.analytic_df_dy_) {
            g.analytic_df_dy_ = [a = analytic_df_dy_, b = rhs.analytic_df_dy_](
                                  double t, const vector_t& y) -> matrix_t { return a(t, y) + b(t, y); };
        }
        g.build_components_();
        return g;
    }

   private:
    std::function<vector_t(double, const vector_t&)> value_;            // f(t, y)
    std::function<matrix_t(double, const vector_t&)> analytic_df_dy_;   // analytic df/dy, or empty
    mutable std::shared_ptr<fd_field_t> fd_field_;                      // dynamic FD source (lazy)
    mutable std::shared_ptr<Jacobian<fd_field_t>> fd_jacobian_;        // library finite-difference Jacobian
    mutable int fd_dim_ = -1;
};

// deduce Dim from the wrapped functor: a fixed-size return type -> static Dim, VectorXd -> Dynamic
template <typename F> ode_rhs_field(F&&) -> ode_rhs_field<ode_rhs_dim_v<F>>;

} // namespace fdapde

#endif //__ODE_H__