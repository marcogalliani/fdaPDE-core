//
// Created by Marco Galliani on 23/10/24.
//

#ifndef TEST_TRUNCATED_SVD_H
#define TEST_TRUNCATED_SVD_H

#include <Eigen/SVD>
#include "randomized_algorithms/randomized_svd.h"

namespace fdapde{
namespace core{

enum SVDPolicy{JacobiSVD, RandSVD_SI, RandSVD_BKI};

template<typename MatrixType, SVDPolicy SVDpol>

class TruncatedSVD{
private:
    DMatrix<double> U_, V_;
    DVector<double> Sigma_;
    int tr_rank_;
public:
    TruncatedSVD(const MatrixType &A, int tr_rank) : tr_rank_(tr_rank){
        compute(A);
    }
    void compute(const MatrixType &A){
        if constexpr(SVDpol==JacobiSVD){
            Eigen::JacobiSVD<DMatrix<double>> svd(A, Eigen::ComputeThinU | Eigen::ComputeThinV);
            U_ = svd.matrixU().leftCols(tr_rank_);
            V_ = svd.matrixV().leftCols(tr_rank_);
            Sigma_ = svd.singularValues().head(tr_rank_);
        }else if constexpr(SVDpol==RandSVD_SI){
            RandomizedSVD<MatrixType,SubspaceIterations> svd(A);
            U_ = svd.matrixU().leftCols(tr_rank_);
            V_ = svd.matrixV().leftCols(tr_rank_);
            Sigma_ = svd.singularValues().head(tr_rank_);
        }else if constexpr(SVDpol==RandSVD_BKI){
            RandomizedSVD<MatrixType,BlockKrylovIterations> svd(A);
            U_ = svd.matrixU().leftCols(tr_rank_);
            V_ = svd.matrixV().leftCols(tr_rank_);
            Sigma_ = svd.singularValues().head(tr_rank_);
        }
    }
    const DMatrix<double> &matrixU() const{ return U_;}
    const DMatrix<double> &matrixV() const{ return V_;}
    const DVector<double> &singularValues() const{ return Sigma_; }
};

}//core
}//fdapde

#endif //TEST_TRUNCATED_SVD_H
