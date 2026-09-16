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

#ifndef __FDAPDE_ODE_SYSTEM_H__
#define __FDAPDE_ODE_SYSTEM_H__

#include "header_check.h"

namespace fdapde {

/* First-order ODE systems y' = f(t, y), identified by their right-hand side.

An rhs is either plain, f(t, y), or theta-parameterized, f(t, y, theta), the latter for the inverse solvers.
It is written either as a functor, or in strong form as equations over unknowns of a function space:

    BsSpace Vh(T, 3, std::vector<int>(T.n_nodes(), 3));
    ode_unknown x(Vh, "x"), v(Vh, "v");
    ode_parameter c(0.5), k(2.0);
    ode_system sys {dx(x) == v, dx(v) == -c * v - k * sin(x)};

Either way ode_rhs_field wraps it into the one interface the integrators consume. */

// f(t, y) -> R^d
template <typename F>
concept is_ode_rhs = std::is_invocable_r_v<
  Eigen::Matrix<double, Dynamic, 1>, const F&, double, const Eigen::Matrix<double, Dynamic, 1>&>;
// f(t, y, theta) -> R^d
template <typename F>
concept is_parameterized_ode_rhs = std::is_invocable_r_v<
  Eigen::Matrix<double, Dynamic, 1>, const F&, double, const Eigen::Matrix<double, Dynamic, 1>&,
  const Eigen::Matrix<double, Dynamic, 1>&>;
// analytic Jacobians; when absent, ode_rhs_field falls back to central differences
template <typename F>
concept ode_rhs_has_state_jacobian = requires(const F& f, double t, const Eigen::Matrix<double, Dynamic, 1>& y) {
    { f.state_jacobian(t, y) } -> std::convertible_to<Eigen::Matrix<double, Dynamic, Dynamic>>;
};
template <typename F>
concept parameterized_ode_rhs_has_state_jacobian = requires(
  const F& f, double t, const Eigen::Matrix<double, Dynamic, 1>& y, const Eigen::Matrix<double, Dynamic, 1>& th) {
    { f.state_jacobian(t, y, th) } -> std::convertible_to<Eigen::Matrix<double, Dynamic, Dynamic>>;
};
template <typename F>
concept parameterized_ode_rhs_has_param_jacobian = requires(
  const F& f, double t, const Eigen::Matrix<double, Dynamic, 1>& y, const Eigen::Matrix<double, Dynamic, 1>& th) {
    { f.param_jacobian(t, y, th) } -> std::convertible_to<Eigen::Matrix<double, Dynamic, Dynamic>>;
};

namespace internals {
template <typename G> constexpr int ode_rhs_dim() {
    using vec = Eigen::Matrix<double, Dynamic, 1>;
    if constexpr (requires { G::dim; }) return G::dim;
    else if constexpr (is_ode_rhs<G>)   return std::invoke_result_t<G, double, vec>::RowsAtCompileTime;
    else                                return std::invoke_result_t<G, double, vec, vec>::RowsAtCompileTime;
}
}   // namespace internals

// the system dimension, read statically: a `dim` member if present, else the rows of the return type (a
// fixed-size return gives a static dimension, VectorXd gives Dynamic)
template <typename F> inline constexpr int ode_rhs_dim_v = internals::ode_rhs_dim<std::decay_t<F>>();

/* The rhs of a Dim-dimensional system, as the integrators consume it.

F is stored by value, so evaluations and Jacobians inline through it. A parameterized F is evaluated at the
bound theta, so operator()(t, y) is always two-argument; a plain F is the parameterized case at an empty theta. */
template <int Dim, typename F> class ode_rhs_field {
   public:
    using vector_t = Eigen::Matrix<double, Dynamic, 1>;
    using matrix_t = Eigen::Matrix<double, Dynamic, Dynamic>;
    static constexpr int dim = Dim;

    ode_rhs_field() = default;
    explicit ode_rhs_field(F f) : f_(std::move(f)) { }
    ode_rhs_field(F f, vector_t theta) : f_(std::move(f)), theta_(std::move(theta)) { }

    vector_t operator()(double t, const vector_t& y) const { return eval_(t, y, theta_); }
    // d f / d y (d x d)
    matrix_t state_jacobian(double t, const vector_t& y) const {
        if constexpr (is_parameterized_ode_rhs<F> && parameterized_ode_rhs_has_state_jacobian<F>) {
            return f_.state_jacobian(t, y, theta_);
        } else if constexpr (!is_parameterized_ode_rhs<F> && ode_rhs_has_state_jacobian<F>) {
            return f_.state_jacobian(t, y);
        } else {
            return fd_jacobian_([&](const vector_t& y_) { return eval_(t, y_, theta_); }, y, y.size());
        }
    }
    // d f / d theta (d x n_theta) at the bound theta
    matrix_t param_jacobian(double t, const vector_t& y) const {
        static_assert(is_parameterized_ode_rhs<F>, "param_jacobian is defined only for parameterized fields");
        fdapde_assert(theta_.size() > 0);
        if constexpr (parameterized_ode_rhs_has_param_jacobian<F>) {
            return f_.param_jacobian(t, y, theta_);
        } else {
            return fd_jacobian_([&](const vector_t& th) { return eval_(t, y, th); }, theta_, y.size());
        }
    }

    void set_theta(const vector_t& theta) { theta_ = theta; }
    int n_params() const { return static_cast<int>(theta_.size()); }
    static constexpr bool is_parametric() { return is_parameterized_ode_rhs<F>; }

   private:
    F f_ {};
    vector_t theta_;   // empty when plain

    // in F's own return type: a fixed-size one spares the finite differences an allocation per evaluation
    auto eval_(double t, const vector_t& y, const vector_t& theta) const {
        if constexpr (is_parameterized_ode_rhs<F>) return f_(t, y, theta);
        else                                        return f_(t, y);
    }
    // central differences of g at x: entry (i, k) = d g_i / d x_k
    template <typename G> static matrix_t fd_jacobian_(const G& g, const vector_t& x, int rows) {
        matrix_t J(rows, x.size());
        vector_t xp = x, xm = x;
        for (int k = 0; k < x.size(); ++k) {
            const double h = 1e-6 * std::max(1.0, std::abs(x[k]));
            xp[k] = x[k] + h;
            xm[k] = x[k] - h;
            J.col(k) = (g(xp) - g(xm)) / (2 * h);
            xp[k] = x[k];
            xm[k] = x[k];
        }
        return J;
    }
};
template <typename F> ode_rhs_field(F&&) -> ode_rhs_field<ode_rhs_dim_v<F>, std::decay_t<F>>;
template <typename F>
ode_rhs_field(F&&, const Eigen::Matrix<double, Dynamic, 1>&) -> ode_rhs_field<ode_rhs_dim_v<F>, std::decay_t<F>>;

// f1 + f2, e.g. a prior field forced by a control. A temporary handed to an integrator, never stored.
template <typename LHS, typename RHS> struct ode_rhs_sum {
    using vector_t = Eigen::Matrix<double, Dynamic, 1>;
    using matrix_t = Eigen::Matrix<double, Dynamic, Dynamic>;
    static constexpr int dim = LHS::dim;
    LHS a;
    RHS b;
    vector_t operator()(double t, const vector_t& y) const { return a(t, y) + b(t, y); }
    matrix_t state_jacobian(double t, const vector_t& y) const {
        return a.state_jacobian(t, y) + b.state_jacobian(t, y);
    }
};
template <int Dim, typename F1, typename F2>
auto operator+(const ode_rhs_field<Dim, F1>& lhs, const ode_rhs_field<Dim, F2>& rhs) {
    using sum_t = ode_rhs_sum<ode_rhs_field<Dim, F1>, ode_rhs_field<Dim, F2>>;
    return ode_rhs_field<Dim, sum_t>(sum_t {lhs, rhs});
}

/* Strong form.

An equation's rhs is an ordinary scalar-field expression whose leaves evaluate on an ode_packet. The system
models the parameterized-rhs concept, so ode_rhs_field wraps it like any functor; its unknowns carry the
trajectory space, which a solver reads from the system instead of taking it separately.

Every equation is enforced softly. That is the intended model for a physical law, not for the definitional
equation of an auxiliary variable: {dx(x) == v, dx(v) == -c v - k sin x} penalizes (x' - v)^2 as well, so v is
not the derivative of x and the fit differs from one penalizing x'' + c x' + k sin x. Such reductions of
higher-order ODEs are out of scope. */

struct ode_packet {
    double t = 0;
    const Eigen::Matrix<double, Dynamic, 1>* y = nullptr;
    const Eigen::Matrix<double, Dynamic, 1>* theta = nullptr;
};

namespace internals {

// scalar-field traits shared by every leaf. The input size must be static: the field operators check a
// Dynamic one against p.rows(), which a packet does not have.
template <typename Derived> struct ode_leaf : public ScalarFieldBase<1, Derived> {
    using InputType = ode_packet;
    using Scalar = double;
    static constexpr int StaticInputSize = 1;
    static constexpr int NestAsRef = 0;
    static constexpr int XprBits = 0;
    constexpr int input_size() const { return StaticInputSize; }
};

/* A leaf that ode_system assigns an index to. Expressions hold leaves by copy, so the index lives in a slot
shared by every copy. Binding a different index is an error: the leaf would change meaning in the system it
already belongs to. */
template <typename Derived> class ode_indexed_leaf : public ode_leaf<Derived> {
   public:
    ode_indexed_leaf() = default;
    explicit ode_indexed_leaf(std::string name) : name_(std::move(name)) { }
    const std::string& name() const { return name_; }
    int index() const { return *slot_; }
    void bind(int index) const {
        fdapde_assert((*slot_ < 0 || *slot_ == index) && "this leaf already belongs to another ode_system");
        *slot_ = index;
    }
   private:
    std::shared_ptr<int> slot_ = std::make_shared<int>(-1);
    std::string name_;
};

}   // namespace internals

// one state component, indexed by the position of the equation that differentiates it
template <typename Space> class ode_unknown : public internals::ode_indexed_leaf<ode_unknown<Space>> {
    using Base = internals::ode_indexed_leaf<ode_unknown<Space>>;
   public:
    ode_unknown() = default;
    explicit ode_unknown(const Space& space, std::string name = "") : Base(std::move(name)), space_(&space) { }
    double operator()(const ode_packet& p) const {
        fdapde_assert(Base::index() >= 0 && "this unknown is not part of any ode_system");
        return (*p.y)[Base::index()];
    }
    const Space& function_space() const { return *space_; }
   private:
    const Space* space_ = nullptr;
};

/* One entry of theta, indexed by with_parameters. It reads its declared value when unbound or when theta does
not reach its index, i.e. in a forward fit. A plain double in an equation is a constant instead. */
class ode_parameter : public internals::ode_indexed_leaf<ode_parameter> {
    using Base = internals::ode_indexed_leaf<ode_parameter>;
   public:
    ode_parameter() = default;
    explicit ode_parameter(double value, std::string name = "") : Base(std::move(name)), value_(value) { }
    double operator()(const ode_packet& p) const {
        return index() < 0 || index() >= p.theta->size() ? value_ : (*p.theta)[index()];
    }
   private:
    double value_ = 0;
};

// the independent variable, for a non-autonomous rhs
struct ode_time : public internals::ode_leaf<ode_time> {
    constexpr double operator()(const ode_packet& p) const { return p.t; }
};

template <typename Unknown> struct ode_derivative {
    Unknown unknown;
};
template <typename Space> ode_derivative<ode_unknown<Space>> dx(const ode_unknown<Space>& x) { return {x}; }

template <typename Unknown, typename Rhs> struct ode_equation {
    Unknown unknown;
    Rhs rhs;
};
template <typename Unknown, typename Rhs>
ode_equation<Unknown, Rhs> operator==(const ode_derivative<Unknown>& lhs, const ScalarFieldBase<1, Rhs>& rhs) {
    return {lhs.unknown, rhs.derived()};
}

// the equations, in the order that fixes the state layout
template <typename... Eqs> class ode_system {
    fdapde_static_assert(sizeof...(Eqs) > 0, AN_ODE_SYSTEM_NEEDS_AT_LEAST_ONE_EQUATION);
   public:
    using vector_t = Eigen::Matrix<double, Dynamic, 1>;
    static constexpr int dim = sizeof...(Eqs);
    using value_t = Eigen::Matrix<double, dim, 1>;

    explicit ode_system(const Eqs&... eqs) : eqs_(eqs...) {
        // one space for the whole system: its degree is the only discretization choice
        bool one_space = true;
        internals::apply_index_pack<dim>([&]<int... Ns_>() {
            ((std::get<Ns_>(eqs_).unknown.bind(Ns_),
              one_space = one_space && &std::get<Ns_>(eqs_).unknown.function_space() == &function_space()),
             ...);
        });
        fdapde_assert(one_space && "every unknown of an ode_system must be declared over the same space");
    }
    // theta's layout is the declaration order here, which the caller of an inverse solver must know
    template <typename... Ps> ode_system& with_parameters(const Ps&... ps) {
        n_params_ = sizeof...(Ps);
        int k = 0;
        (ps.bind(k++), ...);
        return *this;
    }
    int n_params() const { return n_params_; }
    // in state order; a solver binds the response to them by name
    std::vector<std::string> component_names() const {
        std::vector<std::string> names(dim);
        internals::apply_index_pack<dim>(
          [&]<int... Ns_>() { ((names[Ns_] = std::get<Ns_>(eqs_).unknown.name()), ...); });
        return names;
    }
    const auto& function_space() const { return std::get<0>(eqs_).unknown.function_space(); }

    value_t operator()(double t, const vector_t& y, const vector_t& theta) const {
        fdapde_assert(y.size() == dim);
        const ode_packet p {t, &y, &theta};
        value_t out;
        internals::apply_index_pack<dim>([&]<int... Ns_>() { ((out[Ns_] = std::get<Ns_>(eqs_).rhs(p)), ...); });
        return out;
    }
   private:
    std::tuple<Eqs...> eqs_;
    int n_params_ = 0;
};
template <typename... Eqs> ode_system(const Eqs&...) -> ode_system<Eqs...>;

template <typename T> struct is_ode_system : std::false_type { };
template <typename... Eqs> struct is_ode_system<ode_system<Eqs...>> : std::true_type { };
template <typename T> inline constexpr bool is_ode_system_v = is_ode_system<std::decay_t<T>>::value;

}   // namespace fdapde

#endif   // __FDAPDE_ODE_SYSTEM_H__
