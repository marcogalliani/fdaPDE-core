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

#include <gtest/gtest.h>   // testing framework

#include <fdaPDE/linear_algebra.h>
#include <fdaPDE/utility.h>

using fdapde::RSI;
using fdapde::NysRSI;
using fdapde::RBKI;
using fdapde::NysRBKI;
using fdapde::RpChol;

using fdapde::almost_equal;

using matrix_t = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;

// TODO: correct bug in RBKI
TEST(rand_svd_test, square_test){
    matrix_t A = matrix_t::Random(20,20);
    int tr_rank = 3;
    unsigned int seed = fdapde::random_seed;
    double tol = 1e-3;

    RSI<matrix_t> rsi(A, tr_rank, tol, 30, seed);
    //RBKI<matrix_t> rbki(A, tr_rank, tol, 30, seed);
    Eigen::JacobiSVD<matrix_t> jacobi_svd;

    jacobi_svd.compute(A, Eigen::ComputeThinU | Eigen::ComputeThinV);

    EXPECT_TRUE((jacobi_svd.singularValues().head(tr_rank)-rsi.singularValues()).template lpNorm<2>() < tol);
    //EXPECT_TRUE((jacobi_svd.singularValues().head(tr_rank)-rbki.singularValues()).template lpNorm<2>() < tol);
}

TEST(rand_svd_test, rect_test){
    matrix_t A = matrix_t::Random(10,20);
    int tr_rank = 3;
    unsigned int seed = fdapde::random_seed;
    double tol = 1e-3;

    RSI<matrix_t> rsi(A, tr_rank, tol, 30, seed);
    //RBKI<matrix_t> rbki(A, tr_rank, tol, 30, seed);

    Eigen::JacobiSVD<matrix_t> jacobi_svd;
    jacobi_svd.compute(A, Eigen::ComputeThinU | Eigen::ComputeThinV);

    EXPECT_TRUE((jacobi_svd.singularValues().head(tr_rank)-rsi.singularValues()).template lpNorm<2>() < tol);
    //EXPECT_TRUE((jacobi_svd.singularValues().head(tr_rank)-rbki.singularValues()).template lpNorm<2>() < tol);

    rsi.compute(A.transpose(),tr_rank);
    //rbki.compute(A.transpose(),tr_rank);
    jacobi_svd.compute(A, Eigen::ComputeThinU | Eigen::ComputeThinV);

    EXPECT_TRUE((jacobi_svd.singularValues().head(tr_rank)-rsi.singularValues()).template lpNorm<2>() < tol);
    //EXPECT_TRUE((jacobi_svd.singularValues().head(tr_rank)-rbki.singularValues()).template lpNorm<2>() < tol);
}

TEST(rand_evd_test, full_rank){
    matrix_t A = matrix_t::Random(20,20);
    A = A*A.transpose();
    int tr_rank = 3;
    unsigned int seed = fdapde::random_seed; double tol = 1e-4;

    NysRSI<matrix_t> nys_rsi(A, tr_rank, tol, 30, seed);
    NysRBKI<matrix_t> nys_rbki(A, tr_rank, tol, 30, seed);
    
    Eigen::JacobiSVD<matrix_t> jacobi_svd;
    jacobi_svd.compute(A, Eigen::ComputeThinU | Eigen::ComputeThinV);

    EXPECT_TRUE((jacobi_svd.singularValues().head(tr_rank)-nys_rsi.eigenValues()).template lpNorm<2>() < tol);
    EXPECT_TRUE((jacobi_svd.singularValues().head(tr_rank)-nys_rbki.eigenValues()).template lpNorm<2>() < tol);
}

TEST(rand_evd_test, rank_deficient){
    matrix_t A = matrix_t::Random(40,20);
    A = A*A.transpose();
    int tr_rank = 3;
    unsigned int seed = fdapde::random_seed; double tol = 1e-4;

    NysRSI<matrix_t> nys_rsi(A, tr_rank, tol, 30, seed);
    NysRBKI<matrix_t> nys_rbki(A, tr_rank, tol, 30, seed);
    
    Eigen::JacobiSVD<matrix_t> jacobi_svd;
    jacobi_svd.compute(A, Eigen::ComputeThinU | Eigen::ComputeThinV);

    EXPECT_TRUE((jacobi_svd.singularValues().head(tr_rank)-nys_rsi.eigenValues()).template lpNorm<2>() < tol);
    EXPECT_TRUE((jacobi_svd.singularValues().head(tr_rank)-nys_rbki.eigenValues()).template lpNorm<2>() < tol);
}

TEST(nys_approximation, block_equal_one){
    matrix_t A = matrix_t::Random(40,20);
    A = A*A.transpose();
    int block_sz = 1;
    unsigned int seed = fdapde::random_seed; double tol = 1e-3;

    RpChol<matrix_t> rp_chol(A, tol, block_sz, /*max_iter*/30, seed);

    EXPECT_TRUE((A-rp_chol.matrixL()*rp_chol.matrixL().transpose()).norm() < tol*A.norm());
}

TEST(nys_approximation, block_larger_than_one){
    matrix_t A = matrix_t::Random(40,40);
    A = A*A.transpose();
    int block_sz = 7;
    unsigned int seed = fdapde::random_seed; double tol = 1e-3;

    RpChol<matrix_t> rp_chol(A, tol, block_sz, /*max_iter*/30, seed);

    EXPECT_TRUE((A-rp_chol.matrixL()*rp_chol.matrixL().transpose()).norm() < tol*A.norm());
}
