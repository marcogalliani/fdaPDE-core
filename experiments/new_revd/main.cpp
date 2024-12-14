//
// Created by Marco Galliani on 29/10/24.
//

//
// Created by Marco Galliani on 28/09/24.
//

#include <iostream>
#include <chrono>

#include "nlohmann/json.hpp"
#include "test_utils.h"

#include <Eigen/SVD>
#include "fdaPDE/linear_algebra.h"
#include "test/src/utils/utils.h"

//RSVD
using fdapde::core::REVD;
using fdapde::core::NysRSI;
using fdapde::core::NysRBKI;
using fdapde::core::RPChol;

//json
using nlohmann::json;

//Performance metrics
void performance_metrics(test_utils::TestEVD &test, int tr_rank, REVD<DMatrix<double>> &evd){
    DMatrix<double> A = test.matrixU()*test.eigenValues().asDiagonal()*test.matrixU().transpose();
    //Original matrix
    const auto start{std::chrono::steady_clock::now()};
    evd.compute(A,tr_rank);
    const auto end{std::chrono::steady_clock::now()};

    std::ofstream exe_times("results/exe_times.csv");
    exe_times << (std::chrono::duration<double>{end - start}).count() << std::endl;
    exe_times.close();

    std::ofstream singVal_err("results/sing_val_err.csv");
    singVal_err << (test.eigenValues().head(tr_rank)-evd.eigenValues()).template lpNorm<2>() << std::endl;
    singVal_err.close();

    return;
}

int main(int argc, char **argv){
    //parsing data
    std::ifstream input("params.json");
    json data = json::parse(input);

    //cores
    Eigen::setNbThreads(data["RunParams"].value("n_cores",1));
    std::cout << "N cores: " << Eigen::nbThreads() << std::endl;

    //test matrix
    int size = data["TestMatrix"].value("size", 1000);
    DVector<double> eigen_vals(size);
    //singular values decay
    for (int i = 0; i < eigen_vals.size(); ++i){
        //sing_vals(i) = pow(0.997f, i); //fast-decaying
        eigen_vals(i) = 1/std::log(i+2); //slow-decaying
    }
    test_utils::TestEVD testEvd(size, eigen_vals);
    int tr_rank = data["RunParams"].value("truncation_param",3);
    int seed = data["RunParams"].value("seed",7050);

    //methods
    if(data["RunParams"].value("method","rsi") == "rsi"){
        REVD<DMatrix<double>> rsi_evd(std::make_unique<NysRSI<DMatrix<double>>>());
        performance_metrics(testEvd,tr_rank,rsi_evd);
    }
    else if(data["RunParams"].value("method","rbki") == "rbki"){
        REVD<DMatrix<double>> rbki_evd(std::make_unique<NysRBKI<DMatrix<double>>>());
        performance_metrics(testEvd,tr_rank,rbki_evd);
    }
    else if(data["RunParams"].value("method","rpchol") == "rpchol"){
        REVD<DMatrix<double>> rpchol_evd(std::make_unique<RPChol<DMatrix<double>>>());
        performance_metrics(testEvd,tr_rank,rpchol_evd);
    }else{
        DMatrix<double> A = testEvd.matrixU()*testEvd.eigenValues().asDiagonal()*testEvd.matrixU().transpose();
        const auto start{std::chrono::steady_clock::now()};
        Eigen::JacobiSVD<DMatrix<double>> svd(A, Eigen::ComputeThinU | Eigen::ComputeThinV);
        const auto end{std::chrono::steady_clock::now()};

        std::ofstream exe_times("results/exe_times.csv");
        exe_times << (std::chrono::duration<double>{end - start}).count() << std::endl;
        exe_times.close();
    }
    return 0;
}

