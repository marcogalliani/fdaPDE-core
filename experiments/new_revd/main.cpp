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
void performance_metrics(const DMatrix<double> &true_U, const DVector<double> &true_eigen_vals,
                         int tr_rank, int max_iter,
                         SVDType &evd_device){
    //Original matrix
    DMatrix<double> A = true_U*true_eigen_vals.asDiagonal()*true_U.transpose();
    DMatrix<double> A_truncated = true_U.leftCols(tr_rank)*true_eigen_vals.head(tr_rank).asDiagonal()*true_U.leftCols(tr_rank).transpose();

    DVector<double> eigen_values;

    const auto start{std::chrono::steady_clock::now()};
    if constexpr(is_rand_svd<SVDType>{}){
        evd_device.compute(A,tr_rank,max_iter);
        eigen_values = evd_device.singularValues();
    }else if constexpr(is_rand_evd<SVDType>{}){
        evd_device.compute(A,tr_rank,max_iter);
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
    //(2) reconstruction error w.r.t. optimal
    test_report << (A_truncated - U.leftCols(tr_rank)*eigen_values.head(tr_rank).asDiagonal()*U.leftCols(tr_rank).transpose()).lpNorm<Eigen::Infinity>() << std::endl;
    //(3) eigenvalue error
    test_report << (true_eigen_vals.head(tr_rank)-eigen_values.head(tr_rank)).template lpNorm<2>() << std::endl;
    //(4) eigenvectors error
    DMatrix<double> resU_m = true_U.leftCols(tr_rank)-U.leftCols(tr_rank);
    DMatrix<double> resU_p = (true_U.leftCols(tr_rank)+U.leftCols(tr_rank));
    DVector<double> eigVect_errors = resU_m.colwise().maxCoeff().array().min(resU_p.colwise().maxCoeff().array());
    test_report << eigVect_errors.maxCoeff() << std::endl;
    //(5) angle between left subspaces
    test_report << test_utils::subspace(true_U.leftCols(tr_rank), U.leftCols(tr_rank)) << std::endl;

    test_report.close();
}

int main(int argc, char **argv){

    //parsing data
    std::ifstream input("params.json");
    json data = json::parse(input);
    int tr_rank = data["RunParams"].value("truncation_param",3);
    double epsilon = data["RunParams"].value("epsilon",1e-4);
    int seed = data["RunParams"].value("seed",7050);
    int max_iter = data["RunParams"].value("max_iter",100);

    //generating test SVD
    int size = data["TestMatrix"].value("size",1000);

    DVector<double> sing_vals(size);

    //singular values decay
    for (int i = 0; i < sing_vals.size(); ++i){
        sing_vals(i) = 1/std::log(i+2); //slow-decaying
    }
    test_utils::TestEVD TrueEVD(size, sing_vals);

    //methods
    if(data["RunParams"].value("method","nys_rsi") == "nys_rsi"){
        REVD<DMatrix<double>> nys_rsi(std::make_unique<NysRSI<DMatrix<double>>>());
        nys_rsi.setSeed(seed);
        nys_rsi.setTol(epsilon);
        performance_metrics(TrueEVD.matrixU(),TrueEVD.eigenValues(),tr_rank,max_iter,nys_rsi);
    }
    else if(data["RunParams"].value("method","nys_rbki") == "nys_rbki"){
        REVD<DMatrix<double>> nys_rbki(std::make_unique<NysRBKI<DMatrix<double>>>());
        nys_rbki.setSeed(seed);
        performance_metrics(TrueEVD.matrixU(),TrueEVD.eigenValues(),tr_rank,max_iter,nys_rbki);
    }
    else if(data["RunParams"].value("method","rbki") == "rsi"){
        RSVD<DMatrix<double>> rsi(std::make_unique<RSI<DMatrix<double>>>());
        rsi.setSeed(seed);
        rsi.setTol(epsilon);
        performance_metrics(TrueEVD.matrixU(),TrueEVD.eigenValues(),tr_rank,max_iter,rsi);
    }
    else if(data["RunParams"].value("method","rbki") == "rbki"){
        RSVD<DMatrix<double>> rbki(std::make_unique<RBKI<DMatrix<double>>>());
        rbki.setSeed(seed);
        rbki.setTol(epsilon);
        performance_metrics(TrueEVD.matrixU(),TrueEVD.eigenValues(),tr_rank,max_iter,rbki);
    }else{
        Eigen::JacobiSVD<DMatrix<double>> svd;
        performance_metrics(TrueEVD.matrixU(),TrueEVD.eigenValues(),tr_rank,max_iter,svd);
    }
    return 0;
}

