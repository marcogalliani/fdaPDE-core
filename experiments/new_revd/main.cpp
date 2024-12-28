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
using fdapde::core::is_rand_svd;
using fdapde::core::RSVD;
using fdapde::core::RSI;
using fdapde::core::RBKI;

using fdapde::core::is_rand_evd;
using fdapde::core::REVD;
using fdapde::core::NysRSI;
using fdapde::core::NysRBKI;

//json
using nlohmann::json;

template<typename SVDType>
void performance_metrics(test_utils::TestEVD orig_evd,
                         int tr_rank,
                         SVDType &evd_device){
    //Original matrix
    DMatrix<double> A = orig_evd.matrixU()*orig_evd.eigenValues().asDiagonal()*orig_evd.matrixU().transpose();

    DVector<double> eigen_values;

    const auto start{std::chrono::steady_clock::now()};
    if constexpr(is_rand_svd<SVDType>{}){
        evd_device.compute(A,tr_rank);
        eigen_values = evd_device.singularValues();
    }else if constexpr(is_rand_evd<SVDType>{}){
        evd_device.compute(A,tr_rank);
        eigen_values = evd_device.eigenValues();
    }else{
        evd_device.compute(A, Eigen::ComputeThinU | Eigen::ComputeThinV);
        eigen_values = evd_device.singularValues();
    }
    const auto end{std::chrono::steady_clock::now()};
    DMatrix<double> U = evd_device.matrixU();

    std::ofstream test_report("results/test_report.csv");
    //(1) execution time
    test_report << (std::chrono::duration<double>{end - start}).count() << std::endl;
    //(2) reconstruction error
    test_report << (A - U.leftCols(tr_rank)*eigen_values.head(tr_rank).asDiagonal()*U.leftCols(tr_rank).transpose()).norm() << std::endl;
    //(3) eigenvalue error
    test_report << (orig_evd.eigenValues().head(tr_rank)-eigen_values.head(tr_rank)).template lpNorm<2>() << std::endl;
    //(4) eigenvectors error
    DMatrix<double> resU_m = orig_evd.matrixU().leftCols(tr_rank)-U.leftCols(tr_rank);
    DMatrix<double> resU_p = (orig_evd.matrixU().leftCols(tr_rank)+U.leftCols(tr_rank));
    DVector<double> eigVect_errors = resU_m.colwise().maxCoeff().array().min(resU_p.colwise().maxCoeff().array());
    test_report << eigVect_errors.maxCoeff() << std::endl;
    //(5) angle between left subspaces
    test_report << test_utils::subspace(U.leftCols(tr_rank), orig_evd.matrixU().leftCols(tr_rank)) << std::endl;

    test_report.close();
}

int main(int argc, char **argv){
    //parsing data
    std::ifstream input("params.json");
    json data = json::parse(input);

    //test matrix
    int size = data["TestMatrix"].value("size", 1000);
    DVector<double> eigen_vals(size);
   //singular values decay
    if(data["TestMatrix"].value("decay","slow") == "slow"){
        for (int i = 0; i < eigen_vals.size(); ++i){
            eigen_vals(i) = 1/std::log(i+2); //slow-decaying
        }
    }else{
        for (int i = 0; i < eigen_vals.size(); ++i){
            eigen_vals(i) = pow(0.95f, i); //fast-decaying
        }
    }
    test_utils::TestEVD testEvd(size, eigen_vals);
    int tr_rank = data["RunParams"].value("truncation_param",3);
    int seed = data["RunParams"].value("seed",7050);

    //methods
    if(data["RunParams"].value("method","nys_rsi") == "nys_rsi"){
        REVD<DMatrix<double>> nys_rsi(std::make_unique<NysRSI<DMatrix<double>>>());
        performance_metrics(testEvd,tr_rank,nys_rsi);
    }
    else if(data["RunParams"].value("method","nys_rbki") == "nys_rbki"){
        REVD<DMatrix<double>> nys_rbki(std::make_unique<NysRBKI<DMatrix<double>>>());
        performance_metrics(testEvd,tr_rank,nys_rbki);
    }
    else if(data["RunParams"].value("method","rbki") == "rsi"){
        RSVD<DMatrix<double>> rsi(std::make_unique<RSI<DMatrix<double>>>());
        performance_metrics(testEvd,tr_rank,rsi);
    }
    else if(data["RunParams"].value("method","rbki") == "rbki"){
        RSVD<DMatrix<double>> rbki(std::make_unique<RBKI<DMatrix<double>>>());
        performance_metrics(testEvd,tr_rank,rbki);
    }else{
        Eigen::JacobiSVD<DMatrix<double>> svd;
        performance_metrics(testEvd,tr_rank,svd);
    }
    return 0;
}

