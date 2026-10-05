# Apple Clang build patches (fdaPDE-core)

This document records the source changes, on top of upstream `fdaPDE-core`
`stable` (`724d81f`), that make the library compile on macOS with the system
Apple Clang. It is the core-library part of `APPLE_CLANG_PATCH.md` in the
fdaPDE-R repository; the fixes there that concern `fdaPDE-cpp` (`distributions.h`,
`fpca.h`, `sr.h`, `gsr.h`, `solvers/utility.h`) and the R wrappers are not
repeated here.

**Compilers.** The patches were written for Apple Clang 15 (`clang-1500.3.9.4`,
LLVM 16). Apple Clang 16 (`clang-1600.0.26.6`) still crashes on upstream: building a
`SparseBlockMatrix<double, 2, 2>` from four sparse matrices ends in
`clang++: error: unable to execute command: Segmentation fault: 11`. The patched
tree builds with `-Wall -Wextra -Wpedantic -Werror` on Apple Clang 16 and GCC 15;
Apple Clang 15 was not re-tested after the changes marked *new* below.

The fixes fall into two categories:

- **A. C++20 conformance.** Code that a conforming C++20 compiler must reject.
- **C. Apple Clang crashes (ICE).** Valid C++20 on which Apple Clang crashes during
  template instantiation. Each rewrite keeps the original behaviour.

Almost all Category C fixes remove one of two patterns:

1. a **templated generic lambda that expands a parameter pack**
   (`[&]<int... Ns>() { ... }`, or a fold expression over a pack), instantiated
   inside another context that has its own parameter packs. Clang crashes in
   `Sema::collectUnexpandedParameterPacks` / `TransformCXXFoldExpr`;
2. a **requires-expression with a return-type-requirement inside a lambda body**
   (`{ t.size() } -> std::convertible_to<int>`). Clang crashes in
   `Sema::BuildExprRequirement` / `TransformLambdaExpr`. Moving the requirement
   into a namespace-scope concept avoids it: clang then sees a
   `ConceptSpecializationExpr` instead of a `RequiresExpr`.

---

## 1. `assembly.h`: `static_assert(false)` in a discarded `if constexpr` branch [A]

**File:** `fdaPDE/src/assembly.h`

### Old
```cpp
if constexpr (Quadrature::order == 0) {
    fdapde_static_assert(false, THIS_METHOD_REQUIRES_A_QUADRATURE_RULE);
} else { … }
```

### Issue
In C++20 a `static_assert` whose condition is not value-dependent is checked
when the template is defined, even inside a discarded `if constexpr` branch
(P2593 relaxes this only in C++23). The build fails as soon as `assembly.h` is
included.

### Fix
Use a dependent condition. Inside the branch `Quadrature::order == 0`, so it is
still `false` when the branch is instantiated:
```cpp
fdapde_static_assert(Quadrature::order != 0, THIS_METHOD_REQUIRES_A_QUADRATURE_RULE);
```

---

## 2. `sparse_block_matrix.h`: variadic constructor [C]

**File:** `fdaPDE/src/linear_algebra/sparse_block_matrix.h` (and `fdaPDE/src/utility/meta.h`)

### Old
```cpp
template <typename T> class is_matrix_blk {                // private trait
    using T_ = std::decay_t<T>;
   public:
    static constexpr bool value = []() {                    // requires-expression inside a lambda
        if constexpr (requires(T_ t) {
                          typename T_::Scalar;
                          { t.rows() } -> std::convertible_to<std::size_t>;
                          { t.cols() } -> std::convertible_to<std::size_t>;
                      }) {
            return std::convertible_to<typename T_::Scalar, Scalar_>;
        } else { return false; }
    }();
};

template <typename... Block> SparseBlockMatrix(Block&&... m) … {
    std::array<int, Rows_ * Cols_> row_dims, col_dims;
    internals::for_each_index_and_args<sizeof...(Block)>(   // templated generic lambda
        [&]<int Ns_, typename Arg_>(const Arg_& arg) { … }, m...);
    …
    internals::for_each_index_and_args<sizeof...(Block)>(   // second one, stores the blocks
        [&]<int Ns_, typename Arg_>(const Arg_& arg) { … }, m...);
}
```

### Issue
Both patterns occur here, and removing only one of them is not enough. The
trait crashes clang every time it is evaluated (pattern 2). The two
`for_each_index_and_args` calls, whose generic lambdas get Eigen expression
types as arguments, crash it as well (pattern 1). Upstream's constructor still
crashes Apple Clang 16 (see *Compilers* above); which of the two patterns
triggers that crash was not isolated.

### Fix
- The requires-expression becomes the namespace-scope concept
  `internals::is_matrix_blk` in `meta.h`. *New:* the scalar check is the concept
  `internals::is_matrix_blk_of<T, Scalar>`, and the class uses
  `is_matrix_blk_v<T> = internals::is_matrix_blk_of<T, Scalar_>`.
- The constructor uses no templated generic lambdas. Block sizes come from
  per-argument static helpers in an array initializer, and the blocks are stored
  by a plain comma fold at function scope:
```cpp
std::array<int, Rows_ * Cols_> row_dims = { rows_of_(m)... };
std::array<int, Rows_ * Cols_> col_dims = { cols_of_(m)... };
…
int idx_ = 0;
(emplace_single_block_(idx_++, std::forward<Block>(m)), ...);
```

*New, regression fix:* the first version of this patch defined
`is_matrix_blk_v<T> = is_matrix_blk<T> && std::convertible_to<typename std::decay_t<T>::Scalar, Scalar_>`.
In an ordinary expression `&&` does not stop the compiler from forming
`typename T::Scalar`. So the `0` placeholder for an empty block, which upstream
accepts (`SparseBlockMatrix<double, 2, 2> M(A, B, C, 0)`), became a hard error
("'int' is not a class"). In a concept definition the conjunction does
short-circuit, so `is_matrix_blk_of<int, double>` is simply `false`, and the
placeholder works again.

---

## 3. `meta.h`: `apply_index_pack`, `for_each_index_in_pack`, `for_each_index_and_args` [C]

**File:** `fdaPDE/src/utility/meta.h`

### Old
```cpp
template <int N_, typename F_> constexpr decltype(auto) apply_index_pack(F_&& f) {
    return [&]<int... Ns_>(std::integer_sequence<int, Ns_...>) -> decltype(auto) {
        return f.template operator()<Ns_...>();
    }(std::make_integer_sequence<int, N_> {});
}
template <int N_, typename F_> constexpr void for_each_index_in_pack(F_&& f) {
    [&]<int... Ns_>(std::integer_sequence<int, Ns_...>) {
        (f.template operator()<Ns_>(), ...);
    }(std::make_integer_sequence<int, N_> {});
}
template <int N_, typename F_, typename... Args_> requires(sizeof...(Args_) == N_)
constexpr void for_each_index_and_args(F_&& f, Args_&&... args) {
    [&]<int... Ns_>(std::integer_sequence<int, Ns_...>) {
        (f.template operator()<Ns_, Args_>(std::forward<Args_>(args)), ...);
    }(std::make_integer_sequence<int, N_> {});
}
```

### Issue
Pattern 1: each helper wraps its body in a templated generic lambda with a fold
expression. Called from a context with outer packs (for example
`insert_scalar_layer_<GeoInfo...>`), clang crashes during return-type deduction.

### Fix
Dispatch through named function templates, and replace the folds with
compile-time recursion (same left-to-right order):
```cpp
template <typename F_, int... Ns_>
constexpr decltype(auto) apply_index_pack_impl_(F_&& f, std::integer_sequence<int, Ns_...>) {
    return f.template operator()<Ns_...>();
}
template <int N_, int I_, typename F_> constexpr void for_each_index_in_pack_impl_(F_&& f) {
    if constexpr (I_ < N_) {
        f.template operator()<I_>();
        for_each_index_in_pack_impl_<N_, I_ + 1>(std::forward<F_>(f));
    }
}
template <int I_, int N_, typename F_, typename Tuple_>
constexpr void for_each_index_and_args_impl_(F_&& f, Tuple_&& tuple) {
    if constexpr (I_ < N_) {
        using elem_t = std::tuple_element_t<I_, std::remove_reference_t<Tuple_>>;
        f.template operator()<I_, elem_t>(std::get<I_>(std::forward<Tuple_>(tuple)));
        for_each_index_and_args_impl_<I_ + 1, N_>(std::forward<F_>(f), std::forward<Tuple_>(tuple));
    }
}
// for_each_index_and_args packs the arguments with std::forward_as_tuple
```

**Behaviour note.** For an rvalue argument of type `A`, the callback of
`for_each_index_and_args` now gets `A&&` as its type template parameter, where
upstream passed `A`. Lvalues are unchanged (`A&`). All current callers either
take the argument by value / `const&` or test the type with traits that decay it
first (`is_integer_v`, `is_pair_v`, `is_vector_like_v`), so nothing changes
today. New callers should apply `std::decay_t` before using `std::is_same_v` and
similar on that parameter. The note is also in the source.

---

## 4. `meta.h`: `is_vector_like` [C]

**File:** `fdaPDE/src/utility/meta.h`

### Old
```cpp
static constexpr bool value = []() {
    …
    return (is_subscriptable<T_, int> || …) && requires(T_ t) {
        { t.size() } -> std::convertible_to<int>;
    };
}();
```

### Issue
Pattern 2: a return-type-requirement inside the lambda body.

### Fix
Move it into the namespace-scope concept `internals::is_int_sized`, which the
lambda references:
```cpp
template <typename T>
concept is_int_sized = requires(T t) {
    { t.size() } -> std::convertible_to<int>;
};
…
    return (is_subscriptable<T_, int> || …) && is_int_sized<T_>;
```
`is_vector_like` stays a class trait with an `is_vector_like_v` alias. An
intermediate version that turned it into a concept made `is_vector_like_v`
ill-formed (`is_vector_like<T>::value` on a concept) and broke the build on both
Clang and GCC.

---

## 5. `data_layer.h`: nested `is_vector_like` [C]

**File:** `fdaPDE/src/geoframe/data_layer.h`

### Issue / Fix
Same as section 4, for the private `scalar_data_layer::is_vector_like`. The
inline `requires(T t) { { t.size() } -> std::convertible_to<size_t>; }` is
replaced by `internals::is_int_sized<T_>`. The required return type changes from
`size_t` to `int`; both hold for any integral `size()`.

---

## 6. `data_layer.h`: two packs expanded together in `operator()` [C]

**File:** `fdaPDE/src/geoframe/data_layer.h` (`random_access_col_view`, both overloads)

### Old
```cpp
return internals::apply_index_pack<sizeof...(Idxs)>([&]<int... Ns_>() -> decltype(auto) {
    return data_((Ns_ == 0 ? static_cast<index_t>(idxs_[idxs]) : idxs)...);
});
```

### Issue
Pattern 1: `Ns_` (the lambda's pack) and `idxs` (the function's pack) are
expanded in the same pattern.

### Fix
Copy the outer pack into an array first, so only `Ns_` is expanded:
```cpp
const std::array<index_t, Order> idxs_arr_ = {static_cast<index_t>(idxs)...};
return internals::apply_index_pack<Order>([&]<int... Ns_>() -> decltype(auto) {
    return data_((Ns_ == 0 ? static_cast<index_t>(idxs_[idxs_arr_[0]]) : idxs_arr_[Ns_])...);
});
```
The values are the same; indices are passed as `index_t`.

---

## 7. `data_layer.h`: fold over `types` in the row-filter constructor [C]

**File:** `fdaPDE/src/geoframe/data_layer.h` (`scalar_data_layer(row_filter, cols)`)

### Old
```cpp
std::apply([&](const auto&... ts) { ([&]() { using T = std::decay_t<decltype(ts)>; … }(), ...); }, types {});
```

### Issue
Pattern 1: the fold operand is a lambda that refers to the outer pack `ts`.

### Fix
`types` is a fixed tuple of six types, so the fold becomes six explicit calls of
one templated lambda:
```cpp
auto copy_typed_block_ = [&]<typename T>() { … };   // body of the former fold
copy_typed_block_.template operator()<double>();
copy_typed_block_.template operator()<float>();
copy_typed_block_.template operator()<std::int64_t>();
copy_typed_block_.template operator()<std::int32_t>();
copy_typed_block_.template operator()<bool>();
copy_typed_block_.template operator()<std::string>();
```
*New:* the explicit list duplicates `types`. If upstream added a type, columns
of that type would silently not be copied. A `static_assert` now checks that
`types` is exactly `std::tuple<double, float, std::int64_t, std::int32_t, bool, std::string>`.

---

## 8. `data_layer.h`: extent checks in `resize` / `conservative_resize` [C]

**File:** `fdaPDE/src/geoframe/data_layer.h`

### Old
```cpp
if (internals::apply_index_pack<Order>([&]<int... Ns_>() { return ((std::cmp_equal(exts, data.extent(Ns_))) && ...); })) …
```

### Issue / Fix
Pattern 1, with the same two-pack expansion as section 6. Replaced by copying
`exts...` into a `std::array` and comparing in a plain loop. `resize` keeps
`std::cmp_equal`, and `conservative_resize` keeps `==`, as upstream.

---

## 9. `geoframe.h`: redundant `apply_index_pack` wrappers [C]

**File:** `fdaPDE/src/geoframe/geoframe.h` (`insert_scalar_layer_`)

### Old
```cpp
layers_.emplace_back(
  name,
  internals::apply_index_pack<sizeof...(GeoInfo)>([&]<int... Ns>() {
      return std::array<ltype, sizeof...(GeoInfo)> {ltype_from_layer_tag<typename GeoInfo::layer_tag>()...};
  }),
  internals::apply_index_pack<sizeof...(GeoInfo)>([&, this]<int... Ns>() { return geo_layer_t(std::forward<Args>(args)...); }));
```

### Issue / Fix
Neither lambda uses `Ns`, but declaring it next to the outer packs `GeoInfo...`
and `args...` is enough to crash clang (pattern 1). The wrappers are dropped and
both arguments are built directly:
```cpp
layers_.emplace_back(
  name,
  std::array<ltype, sizeof...(GeoInfo)> {ltype_from_layer_tag<typename GeoInfo::layer_tag>()...},
  geo_layer_t(std::forward<Args>(args)...));
```

---

## 10. `geo_layer.h`: `decltype` over a templated lambda [C]

**File:** `fdaPDE/src/geoframe/geo_layer.h` (row-filter constructor)

### Old
```cpp
using mem_t = decltype(internals::apply_index_pack<Order>([]<int... Ns>() {
    return std::tuple<std::vector<
      typename std::tuple_element_t<Ns, GeoInfo>::template value_type<local_dim[Ns], embed_dim[Ns]>>...> {};
}));
```

### Issue / Fix
`decltype` forces the templated lambda to be instantiated (pattern 1). The same
type now comes from a declared, never defined, static helper:
```cpp
template <std::size_t... Is>
static auto geo_data_storage_helper_(std::index_sequence<Is...>)
  -> std::tuple<std::vector<
    typename std::tuple_element_t<Is, GeoInfo>::template value_type<local_dim[Is], embed_dim[Is]>>...>;
…
using mem_t = decltype(geo_data_storage_helper_(std::make_index_sequence<Order> {}));
```

---

## Keeping the patch maintainable

- Each rewritten spot carries an `Apple-Clang-15 workaround` comment in the
  source. *New:* the comments on the `meta.h` helpers, removed by an
  intermediate commit, are restored.
- When merging upstream, check whether upstream reintroduced either crash
  pattern in new code, especially generic lambdas with `<int... Ns>` packs and
  requires-expressions inside lambdas.
- To check for the crash directly, compile a `SparseBlockMatrix<double, 2, 2>`
  construction and a `GeoFrame` layer insertion with Apple Clang.
