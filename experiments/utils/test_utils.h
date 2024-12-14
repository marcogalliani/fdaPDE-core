//
// Created by Marco Galliani on 29/10/24.
//

#ifndef TEST_UTILS_H
#define TEST_UTILS_H

#include "utils/symbols.h"
#include <Eigen/QR>

namespace test_utils{

struct TestSVD{
    DMatrix<double> U_,V_;
    DVector<double> SingVals_;
    TestSVD(size_t row, size_t col, DVector<double> sing_vals){
        Eigen::HouseholderQR<DMatrix<double>> qr;
        SingVals_ = sing_vals;
        U_ = DMatrix<double>::Random(row, sing_vals.size());
        V_ = DMatrix<double>::Random(col, sing_vals.size());
        U_ = qr.compute(U_).householderQ() * DMatrix<double>::Identity(U_.rows(),sing_vals.size());
        V_ = qr.compute(V_).householderQ() * DMatrix<double>::Identity(V_.rows(),sing_vals.size());
    }
    DMatrix<double> matrixU() const{ return U_;}
    DMatrix<double> matrixV() const{ return V_;}
    DVector<double> singularValues() const{ return SingVals_;}
};

struct TestEVD{
    DMatrix<double> U_;
    DVector<double> EigenVals_;
    TestEVD(int mat_size, DVector<double> eigen_vals){
        Eigen::HouseholderQR<DMatrix<double>> qr;
        EigenVals_ = eigen_vals;
        U_ = DMatrix<double>::Random(mat_size, eigen_vals.size());
        U_ = qr.compute(U_).householderQ() * DMatrix<double>::Identity(U_.rows(),eigen_vals.size());
    }
    DMatrix<double> matrixU() const{ return U_;}
    DVector<double> eigenValues() const{ return EigenVals_;}
};


}

#endif //TEST_UTILS_H
