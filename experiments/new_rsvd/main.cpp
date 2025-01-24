#include <iostream>
#include <chrono>

#include "nlohmann/json.hpp"
#include "test_utils.h"

#include <Eigen/Dense>
#include <Eigen/SVD>
#include "fdaPDE/linear_algebra.h"
#include "test/src/utils/utils.h"

#include <unsupported/Eigen/SparseExtra>

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
void performance_metrics(const DMatrix<double> &U, const DVector<double> &sing_vals, const DMatrix<double> &V,
                         int tr_rank, int max_iter,
                         SVDType &svd_device){
    //Original matrix
    DMatrix<double> A = U*sing_vals.asDiagonal()*V.transpose();
    DMatrix<double> A_truncated = U.leftCols(tr_rank)*sing_vals.head(tr_rank).asDiagonal()*V.leftCols(tr_rank).transpose();

    const auto start{std::chrono::steady_clock::now()};
    if constexpr(is_rand_svd<SVDType>{}){
        svd_device.compute(A,tr_rank,max_iter);
    }else{
        svd_device.compute(A, Eigen::ComputeThinU | Eigen::ComputeThinV);
    }
    const auto end{std::chrono::steady_clock::now()};

    std::ofstream test_report("results/test_report.csv");
    //(1) execution time
    test_report << (std::chrono::duration<double>{end - start}).count() << std::endl;
    //(2) reconstruction error w.r.t. the optimal reconstruction error
    test_report << (A_truncated - svd_device.matrixU().leftCols(tr_rank)*svd_device.singularValues().head(tr_rank).asDiagonal()*svd_device.matrixV().leftCols(tr_rank).transpose()).template lpNorm<Eigen::Infinity>() << std::endl;
    //(3) singular value error
    test_report << (sing_vals.head(tr_rank)-svd_device.singularValues().head(tr_rank)).template lpNorm<2>() << std::endl;
    //(4) left singular vectors error
    DMatrix<double> resU_m = U.leftCols(tr_rank)-svd_device.matrixU().leftCols(tr_rank);
    DMatrix<double> resU_p = U.leftCols(tr_rank)+svd_device.matrixU().leftCols(tr_rank);
    DVector<double> l_singVect_errors = resU_m.colwise().maxCoeff().array().min(resU_p.colwise().maxCoeff().array());
    test_report << l_singVect_errors.maxCoeff() << std::endl;
    //(5) right singular vectors error
    DMatrix<double> resV_m = V.leftCols(tr_rank)-svd_device.matrixV().leftCols(tr_rank);
    DMatrix<double> resV_p = V.leftCols(tr_rank)+svd_device.matrixV().leftCols(tr_rank);
    DVector<double> r_singVect_errors = resV_m.colwise().maxCoeff().array().min(resV_p.colwise().maxCoeff().array());
    test_report << r_singVect_errors.maxCoeff() << std::endl;
    //(6) angle between left subspaces
    test_report << test_utils::subspace(svd_device.matrixU().leftCols(tr_rank), U.leftCols(tr_rank)) << std::endl;
    //(7) angle between right subspaces
    test_report << test_utils::subspace(svd_device.matrixV().leftCols(tr_rank), V.leftCols(tr_rank)) << std::endl;

    test_report.close();
}

int main(int argc, char **argv){
    //parsing data
    std::ifstream input("params.json");
    json data = json::parse(input);

    int tr_rank = data["RunParams"].value("truncation_param",3);
    int seed = data["RunParams"].value("seed",1412);
    double epsilon = data["RunParams"].value("epsilon",1e-4);
    int max_iter = data["RunParams"].value("max_iter",100);

    //generating test SVD
    int rows = data["TestMatrix"].value("rows",1000);
    int cols = data["TestMatrix"].value("cols",1000);

    DVector<double> sing_vals(std::min(rows,cols));

    //singular values decay
    for (int i = 0; i < sing_vals.size(); ++i){
        sing_vals(i) = 1/std::log(i+2); //slow-decaying
    }
    test_utils::TestSVD TrueSVD(rows,cols,sing_vals);

    //methods
    if(data["RunParams"].value("method","rsi") == "rsi"){
        RSVD<DMatrix<double>> rsi(std::make_unique<RSI<DMatrix<double>>>());
        rsi.setTol(epsilon);
        performance_metrics(TrueSVD.matrixU(), sing_vals, TrueSVD.matrixV(),tr_rank,max_iter,rsi);
    }
    else if(data["RunParams"].value("method","rbki") == "rbki"){
        RSVD<DMatrix<double>> rbki(std::make_unique<RBKI<DMatrix<double>>>());
        rbki.setTol(epsilon);
        performance_metrics(TrueSVD.matrixU(), sing_vals, TrueSVD.matrixV(),tr_rank,max_iter,rbki);
    }
    else if(data["RunParams"].value("method","gen_rsi") == "gen_rsi"){
        RSVD<DMatrix<double>> ext_rsi(std::make_unique<GeneralizedRSI<DMatrix<double>>>());
        ext_rsi.setTol(epsilon);
        performance_metrics(TrueSVD.matrixU(), sing_vals, TrueSVD.matrixV(),tr_rank,max_iter,ext_rsi);
    }
    else if(data["RunParams"].value("method","gen_rbki") == "gen_rbki"){
        RSVD<DMatrix<double>> ext_rbki(std::make_unique<GeneralizedRBKI<DMatrix<double>>>());
        ext_rbki.setTol(epsilon);
        performance_metrics(TrueSVD.matrixU(), sing_vals, TrueSVD.matrixV(),tr_rank,max_iter,ext_rbki);
    }else{
        Eigen::JacobiSVD<DMatrix<double>> svd;
        performance_metrics(TrueSVD.matrixU(), sing_vals, TrueSVD.matrixV(),tr_rank,max_iter,svd);
    }
    return 0;
}

