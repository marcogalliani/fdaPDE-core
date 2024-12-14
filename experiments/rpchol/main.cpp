//
// Created by Marco Galliani on 03/10/24.
//
#include <iostream>
#include <chrono>
#include "nlohmann/json.hpp"

#include <Eigen/SVD>
#include "fdaPDE/linear_algebra.h"
#include "test/src/utils/utils.h"

//Nystrom approximation
using fdapde::core::NystromApproximation;
using fdapde::core::RPChol;

//json
using nlohmann::json;

//Building a test Matrix
DMatrix<double> TestMatrix(int size, DVector<double> sing_vals){
    Eigen::HouseholderQR<DMatrix<double>> qr;
    DMatrix<double> U = DMatrix<double>::Random(size, sing_vals.size());
    // orthogonalization
    U = qr.compute(U).householderQ() * DMatrix<double>::Identity(U.rows(),sing_vals.size());
    // target matrix to be decomposed
    return U * sing_vals.asDiagonal() * U.transpose();
}

int main(){
    //parsing data
    std::ifstream input("params.json");
    json data = json::parse(input);

    int size = data["TestMatrix"].value("size", 1000);

    DVector<double> sing_vals(data["TestMatrix"].value("rank", 1000));
    //singular values decay
    for (int i = 0; i < sing_vals.size(); ++i){
        //sing_vals(i) = pow(0.9f, i); //fast-decaying
        sing_vals(i) = 1/std::log(i+2); //slow-decaying
    }
    DMatrix<double> A = TestMatrix(size, sing_vals);
    int seed = data["RunParams"].value("seed",7050);
    Eigen::setNbThreads(data["RunParams"].value("n_cores",1));
    std::cout << "n_cores: " << Eigen::nbThreads() << std::endl;

    NystromApproximation<DMatrix<double>> rp_chol(std::make_unique<RPChol<DMatrix<double>>>(seed,1e-3));

    int block_sz = data["RPChol"].value("block_size",10);
    rp_chol.compute(A,block_sz);

    std::cout << "Error:" << std::endl;
    std::cout << (A-rp_chol.factor()*rp_chol.factor().transpose()).norm() << std::endl;

    return 0;
}