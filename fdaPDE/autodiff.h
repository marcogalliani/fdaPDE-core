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

#ifndef __FDAPDE_AUTODIFF_MODULE_H__
#define __FDAPDE_AUTODIFF_MODULE_H__

/* Adapter for the autodiff library (https://github.com/autodiff/autodiff)

fdaPDE asks the user for derivatives in two places: an ODE rhs is expected to expose state_jacobian /
param_jacobian (falling back to central finite differences when it does not), and the optimization
algorithms ask an objective for .gradient() (and .hessian(), for Newton). Both are hand-written today.
This module lets forward-mode automatic differentiation supply them instead: the user writes the *value*
once, generic in the scalar type, and the adapter derives the Jacobians exactly, at finite-difference
cost but without the truncation/cancellation error of a difference quotient.

  ad_ode_rhs<F>    : wraps a scalar-generic rhs functor into an ODE rhs exposing analytic (AD) Jacobians,
                     so ode_rhs_field picks them up through the existing concepts. A drop-in replacement
                     for a hand-differentiated field.
  ad_objective<N,F>: wraps a scalar-generic objective into the functor interface the optimization module
                     consumes (operator(), .gradient(), .hessian()).

DEPENDENCY. autodiff is header-only and OPTIONAL: this module is inert unless its headers are on the
include path (detected with __has_include; define FDAPDE_NO_AUTODIFF to disable the detection). Nothing
else in fdaPDE depends on it, so a build without autodiff compiles unchanged and keeps the analytic /
finite-difference paths. When present, FDAPDE_HAS_AUTODIFF is defined and the adapters below exist.

    clang++ -std=c++20 -I /path/to/autodiff -I /path/to/eigen3 ...
*/

// clang-format off

// include required modules
#include "linear_algebra.h"    // pull Eigen first
#include "utility.h"
#include "fields.h"
#include "ode.h"

#if !defined(FDAPDE_NO_AUTODIFF) && __has_include(<autodiff/forward/dual.hpp>) &&                              \
  __has_include(<autodiff/forward/dual/eigen.hpp>)
#define FDAPDE_HAS_AUTODIFF
#endif

#ifdef FDAPDE_HAS_AUTODIFF

// autodiff gates its Eigen API (jacobian(), gradient(), hessian() over Eigen vectors) behind this macro,
// which its own CMake package defines. Eigen is a hard fdaPDE dependency and is already included above,
// so the API is always available here: turn it on unconditionally rather than requiring the consumer's
// build system to know about it.
#ifndef AUTODIFF_EIGEN_FOUND
#define AUTODIFF_EIGEN_FOUND
#endif
#include <autodiff/forward/dual.hpp>
#include <autodiff/forward/dual/eigen.hpp>

#include "src/autodiff/ad_ode_rhs.h"
#include "src/autodiff/ad_objective.h"

#endif   // FDAPDE_HAS_AUTODIFF

// clang-format on

#endif   // __FDAPDE_AUTODIFF_MODULE_H__
