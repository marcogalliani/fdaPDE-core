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

#ifndef __UTILS_H__
#define __UTILS_H__

#include <fdaPDE/utility.h>
#include <unsupported/Eigen/SparseExtra>
#include <string>
#include "constants.h"

template<typename T>
using matrix_t = Eigen::Matrix<T, Dynamic, Dynamic>;

using sparse_matrix_t = Eigen::SparseMatrix<double>;

// a set of useful utilities
namespace fdapde {
namespace testing {

// this function is an implementation of the test for floating point equality based on relative error. There is
// an huge literature about floating point comparison, refer to it for details
template <typename T>
typename std::enable_if<!std::numeric_limits<T>::is_integer, bool>::type almost_equal(T a, T b, T epsilon) {
    return std::fabs(a - b) < epsilon ||
           std::fabs(a - b) < ((std::fabs(a) < std::fabs(b) ? std::fabs(b) : std::fabs(a)) * epsilon);
}

// set default epsilon to DOUBLE_TOLERANCE
template <typename T> typename std::enable_if<!std::numeric_limits<T>::is_integer, bool>::type almost_equal(T a, T b) {
    return almost_equal(a, b, DOUBLE_TOLERANCE);
}

// test if two matrices are equal testing the relative error of the infinte norm of their difference
inline bool almost_equal(const matrix_t<double>& op1, const matrix_t<double>& op2, double epsilon) {
    return (op1 - op2).lpNorm<Eigen::Infinity>() < epsilon ||
           (op1 - op2).lpNorm<Eigen::Infinity>() <
             (std::max(op1.lpNorm<Eigen::Infinity>(), op2.lpNorm<Eigen::Infinity>()) * epsilon);
}
inline bool almost_equal(const matrix_t<double>& op1, const matrix_t<double>& op2) {
    return almost_equal(op1, op2, DOUBLE_TOLERANCE);
}
// sparse operands
inline bool almost_equal(const sparse_matrix_t& op1, const sparse_matrix_t& op2, double epsilon) {
    return almost_equal(matrix_t<double>(op1), matrix_t<double>(op2), epsilon);
}
inline bool almost_equal(const sparse_matrix_t& op1, const sparse_matrix_t& op2) {
    return almost_equal(matrix_t<double>(op1), matrix_t<double>(op2));
}

  // load rhs from file
  bool almost_equal(const sparse_matrix_t& op1, std::string op2) {
    sparse_matrix_t mem_buff;
    Eigen::loadMarket(mem_buff, op2);
    return almost_equal(op1, mem_buff);
  }
  bool almost_equal(const matrix_t<double>& op1, std::string op2) {
    sparse_matrix_t mem_buff;
    Eigen::loadMarket(mem_buff, op2);
    return almost_equal(op1, matrix_t<double>(mem_buff));
  }

  bool almost_equal(const std::vector<double>& op1, std::string op2) {
    matrix_t<double> m;
    m.resize(op1.size(), 1);
    for (std::size_t i = 0; i < op1.size(); ++i) m(i, 0) = op1[i];
    return almost_equal(m, op2);
  }

  template <int N> bool almost_equal(const SVector<N>& op1, const SVector<N>& op2) {
    bool equal = true;
    for (int i = 0; i < N; ++i) equal &= almost_equal(op1[i], op2[i]);
    return equal;
  }

  /*
  // utility to import .mtx files
  template <typename T>
  DMatrix<T> read_mtx(const std::string& file_name) {
    sparse_matrix_t buff;
    Eigen::loadMarket(buff, file_name);
    return buff;
  }
  // utility to import .csv files
  template <typename T>
  matrix_t<T> read_csv(const std::string& file_name) {
    CSVReader<T> reader {};
    return reader.template parse_file<Eigen::Dense>(file_name);
  }
  
}}
*/

#endif // __UTILS_H__
