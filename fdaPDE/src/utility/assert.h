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

#ifndef __FDAPDE_ASSERT_H__
#define __FDAPDE_ASSERT_H__

// standalone (no header_check.h): the execution module includes it directly, as upstream's typed assertions
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <type_traits>

namespace fdapde {

#define FDAPDE_COMMA ,
  
namespace internals {
  
inline void fdapde_assert_failed_(const char* str, const char* file, int line) {
    std::cerr << file << ":" << line << ". Assertion: '" << str << "' failed." << std::endl;
    abort();
}

}   // namespace internals

// throws the requested exception on failure, also when debug checks are disabled (upstream's typed assertions)
#define fdapde_strong_assert(condition, exception_type, message)                                                       \
    do {                                                                                                               \
        if (!(condition)) { throw exception_type(message); }                                                           \
    } while (false)

// fdapde_assert(condition): the legacy form, aborts with a message; fdapde_assert(condition, exception_type, message):
// upstream's typed form, throws. Both are disabled by FDAPDE_NO_DEBUG
#define fdapde_assert_pick_(_1, _2, _3, name, ...) name
#ifdef FDAPDE_NO_DEBUG
#    define fdapde_assert_1_(condition) (void)0
#    define fdapde_assert_3_(condition, exception_type, message) (void)0
#else
#    define fdapde_assert_1_(condition)                                                                                \
        if (!(condition)) { ::fdapde::internals::fdapde_assert_failed_(#condition, __FILE__, __LINE__); }
#    define fdapde_assert_3_(condition, exception_type, message)                                                       \
        fdapde_strong_assert((condition), exception_type, (message))
#endif   // FDAPDE_NO_DEBUG
#define fdapde_assert(...) fdapde_assert_pick_(__VA_ARGS__, fdapde_assert_3_, fdapde_assert_2_, fdapde_assert_1_)(__VA_ARGS__)

#define fdapde_static_assert(condition, message) static_assert(condition, #message)

#ifdef FDAPDE_NO_DEBUG
#    define fdapde_constexpr_assert(condition) (void)0
#else
#    define fdapde_constexpr_assert(condition)                                                                         \
        if (std::is_constant_evaluated()) {                                                                            \
            if (!(condition)) { throw std::logic_error(#condition); }                                                  \
        } else {                                                                                                       \
            fdapde_assert(condition);                                                                                  \
        }

#endif   // FDAPDE_NO_DEBUG

}   // namespace fdapde

#endif   // __FDAPDE_ASSERT_H__
