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
using fdapde::core::RSVD;
using fdapde::core::RSI;
using fdapde::core::RBKI;
using fdapde::core::GeneralizedRSI;
using fdapde::core::GeneralizedRBKI;

using fdapde::core::is_rand_svd;

//json
using nlohmann::json;

//Performance metrics

template<typename SVDType>
void performance_metrics(test_utils::TestSVD gen_svd,
                           int tr_rank,
                           SVDType &svd_device){
    //Original matrix
    DMatrix<double> A = gen_svd.matrixU()*gen_svd.singularValues().asDiagonal()*gen_svd.matrixV().transpose();

    const auto start{std::chrono::steady_clock::now()};
    if constexpr(is_rand_svd<SVDType>{}){
        svd_device.compute(A,tr_rank);
    }else{
        svd_device.compute(A, Eigen::ComputeThinU | Eigen::ComputeThinV);
    }
    const auto end{std::chrono::steady_clock::now()};

    std::ofstream test_report("results/test_report.csv");
    //(1) execution time
    test_report << (std::chrono::duration<double>{end - start}).count() << std::endl;
    //(2) reconstruction error
    test_report << (A - svd_device.matrixU().leftCols(tr_rank)*svd_device.singularValues().head(tr_rank).asDiagonal()*svd_device.matrixV().leftCols(tr_rank).transpose()).norm() << std::endl;
    //(3) singular value error
    test_report << (gen_svd.singularValues().head(tr_rank)-svd_device.singularValues().head(tr_rank)).template lpNorm<2>() << std::endl;
    //(4) left singular vectors error
    DMatrix<double> resU_m = gen_svd.matrixU().leftCols(tr_rank)-svd_device.matrixU().leftCols(tr_rank);
    DMatrix<double> resU_p = gen_svd.matrixU().leftCols(tr_rank)+svd_device.matrixU().leftCols(tr_rank);
    DVector<double> l_singVect_errors = resU_m.colwise().maxCoeff().array().min(resU_p.colwise().maxCoeff().array());
    test_report << l_singVect_errors.maxCoeff() << std::endl;
    //(5) right singular vectors error
    DMatrix<double> resV_m = gen_svd.matrixV().leftCols(tr_rank)-svd_device.matrixV().leftCols(tr_rank);
    DMatrix<double> resV_p = gen_svd.matrixV().leftCols(tr_rank)+svd_device.matrixV().leftCols(tr_rank);
    DVector<double> r_singVect_errors = resV_m.colwise().maxCoeff().array().min(resV_p.colwise().maxCoeff().array());
    test_report << r_singVect_errors.maxCoeff() << std::endl;
    //(6) angle between left subspaces
    test_report << test_utils::subspace(svd_device.matrixU().leftCols(tr_rank), gen_svd.matrixU().leftCols(tr_rank)) << std::endl;
    //(7) angle between right subspaces
    test_report << test_utils::subspace(svd_device.matrixV().leftCols(tr_rank), gen_svd.matrixV().leftCols(tr_rank)) << std::endl;

    test_report.close();
}

int main(int argc, char **argv){
    //parsing data
    std::ifstream input("params.json");
    json data = json::parse(input);

    //test matrix
    size_t rows, cols;
    rows = data["TestMatrix"].value("rows", 1000);
    cols = data["TestMatrix"].value("cols", 1000);
    DVector<double> sing_vals(std::min(rows,cols));

    //singular values decay
    if(data["TestMatrix"].value("decay","slow") == "slow"){
        for (int i = 0; i < sing_vals.size(); ++i){
            sing_vals(i) = 1/std::log(i+2); //slow-decaying
        }
    }else{
        for (int i = 0; i < sing_vals.size(); ++i){
            sing_vals(i) = pow(0.95f, i); //fast-decaying
        }
    }
    test_utils::TestSVD test_svd(rows, cols, sing_vals);
    int tr_rank = data["RunParams"].value("truncation_param",3);
    int seed = data["RunParams"].value("seed",7050);

    //methods
    if(data["RunParams"].value("method","rsi") == "rsi"){
        RSVD<DMatrix<double>> rsi(std::make_unique<RSI<DMatrix<double>>>());
        performance_metrics(test_svd,tr_rank,rsi);
    }
    else if(data["RunParams"].value("method","rbki") == "rbki"){
        RSVD<DMatrix<double>> rbki(std::make_unique<RBKI<DMatrix<double>>>());
        performance_metrics(test_svd,tr_rank,rbki);
    }
    else if(data["RunParams"].value("method","ext_rsi") == "ext_rsi"){
        RSVD<DMatrix<double>> ext_rsi(std::make_unique<GeneralizedRSI<DMatrix<double>>>());
        performance_metrics(test_svd,tr_rank,ext_rsi);
    }
    else if(data["RunParams"].value("method","ext_rbki") == "ext_rbki"){
        RSVD<DMatrix<double>> ext_rbki(std::make_unique<GeneralizedRBKI<DMatrix<double>>>());
        performance_metrics(test_svd,tr_rank,ext_rbki);
    }else{
        Eigen::JacobiSVD<DMatrix<double>> svd;
        performance_metrics(test_svd,tr_rank,svd);
    }
    return 0;
}

