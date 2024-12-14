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

//json
using nlohmann::json;

//Performance metrics

double performance_metrics(test_utils::TestSVD gen_svd, int tr_rank,
                         RSVD<DMatrix<double>> &svd_device){
    //Original matrix
    DMatrix<double> A = gen_svd.matrixU()*gen_svd.singularValues().asDiagonal()*gen_svd.matrixV().transpose();

    const auto start{std::chrono::steady_clock::now()};
    svd_device.compute(A, tr_rank);
    const auto end{std::chrono::steady_clock::now()};

    std::ofstream exe_times("results/exe_times.csv");
    exe_times << (std::chrono::duration<double>{end - start}).count() << std::endl;
    exe_times.close();

    std::ofstream singVal_err("results/sing_val_err.csv");
    singVal_err << (gen_svd.singularValues().head(tr_rank)-svd_device.singularValues()).template lpNorm<2>() << std::endl;
    singVal_err.close();

    auto resU_m = gen_svd.matrixU().leftCols(svd_device.rank())-svd_device.matrixU();
    auto resU_p = gen_svd.matrixU().leftCols(svd_device.rank())+svd_device.matrixU();
    Eigen::saveMarket(resU_m.colwise().maxCoeff().array().min(resU_p.colwise().maxCoeff().array()),"results/l_sing_vecs_err.mtx");

    auto resV_m = gen_svd.matrixV().leftCols(svd_device.rank())-svd_device.matrixV();
    auto resV_p = gen_svd.matrixV().leftCols(svd_device.rank())+svd_device.matrixV();
    Eigen::saveMarket(resV_m.colwise().maxCoeff().array().min(resV_p.colwise().maxCoeff().array()),"results/r_sing_vecs_err.mtx");

    return (std::chrono::duration<double>{end - start}).count();
}

int main(int argc, char **argv){
    //parsing data
    std::ifstream input("params.json");
    json data = json::parse(input);

    //cores
    Eigen::setNbThreads(data["RunParams"].value("n_cores",1));
    std::cout << "N cores: " << Eigen::nbThreads() << std::endl;

    //test matrix
    size_t rows, cols;
    rows = data["TestMatrix"].value("rows", 1000);
    cols = data["TestMatrix"].value("cols", 1000);
    DVector<double> sing_vals(std::min(rows,cols));
    //singular values decay
    for (int i = 0; i < sing_vals.size(); ++i){
        //sing_vals(i) = pow(0.997f, i); //fast-decaying
        sing_vals(i) = 1/std::log(i+2); //slow-decaying
    }
    test_utils::TestSVD test_svd(rows, cols, sing_vals);
    int tr_rank = data["RunParams"].value("truncation_param",3);
    int seed = data["RunParams"].value("seed",7050);

    //methods
    if(data["RunParams"].value("method","rsi") == "rsi"){
        RSVD<DMatrix<double>> rsi_svd(std::make_unique<RSI<DMatrix<double>>>());
        performance_metrics(test_svd,tr_rank,rsi_svd);
    }
    else if(data["RunParams"].value("method","rbki") == "rbki"){
        RSVD<DMatrix<double>> rbki_svd(std::make_unique<RBKI<DMatrix<double>>>());
        performance_metrics(test_svd,tr_rank,rbki_svd);
    }else{
        DMatrix<double> A = test_svd.matrixU()*test_svd.singularValues().asDiagonal()*test_svd.matrixV().transpose();

        const auto start{std::chrono::steady_clock::now()};
        Eigen::JacobiSVD<DMatrix<double>> svd(A, Eigen::ComputeThinU | Eigen::ComputeThinV);
        const auto end{std::chrono::steady_clock::now()};

        std::ofstream exe_times("results/exe_times.csv");
        exe_times << (std::chrono::duration<double>{end - start}).count() << std::endl;
        exe_times.close();
    }
    return 0;
}

