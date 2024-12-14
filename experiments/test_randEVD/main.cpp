//
// Created by Marco Galliani on 03/10/24.
//
#include <iostream>
#include <chrono>
#include "nlohmann/json.hpp"

#include <Eigen/SVD>
#include "fdaPDE/linear_algebra.h"
#include "test/src/utils/utils.h"

//Randomized SVD
using fdapde::core::RandomizedEVD;

using fdapde::core::IterationPolicy;
using fdapde::core::StoppingPolicy;

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
        sing_vals(i) = pow(0.9f, i); //fast-decaying
        //sing_vals(i) = 1/std::log(i+2); //slow-decaying
    }
    DMatrix<double> A = TestMatrix(size, sing_vals);
    int seed = data["RunParams"].value("seed",7050);
    Eigen::setNbThreads(data["RunParams"].value("n_cores",1));
    std::cout << "n_cores: " << Eigen::nbThreads() << std::endl;

    if(data["RBKI"].value("enabled",1)){
        //params
        int block_sz = data["RBKI"].value("block_size",10);
        std::cout << "RBKI" << std::endl;
        const auto start{std::chrono::steady_clock::now()};
        RandomizedEVD<decltype(A),IterationPolicy::BlockKrylovIterations> rbki(A,block_sz,1e-1,seed);
        const auto end{std::chrono::steady_clock::now()};

        std::cout << "Rank: " << rbki.rank() << std::endl;
        std::cout << (std::chrono::duration<double>{end - start}).count() << std::endl;
        std::cout << (A-rbki.matrixU()*rbki.eigenValues().asDiagonal()*rbki.matrixU().transpose()).norm() << std::endl;
    }
    if(data["Rchol"].value("enabled",1)){
        //params
        int block_sz = data["Rchol"].value("block_size",10);

        std::cout << "Randomly Pivoted Cholesky" << std::endl;
        const auto start{std::chrono::steady_clock::now()};
        RandomizedEVD<decltype(A),IterationPolicy::RandomlyPivotedCholesky> rpchol(A,block_sz,1e-1);
        const auto end{std::chrono::steady_clock::now()};

        std::cout << "Rank: " << rpchol.rank() << std::endl;
        std::cout << "Time: " << (std::chrono::duration<double>{end - start}).count() << std::endl;
        std::cout << (A-rpchol.matrixU()*rpchol.eigenValues().asDiagonal()*rpchol.matrixU().transpose()).norm() << std::endl;
    }
    return 0;
}