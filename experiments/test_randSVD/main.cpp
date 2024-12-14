//
// Created by Marco Galliani on 28/09/24.
//

#include <iostream>
#include <chrono>
#include "nlohmann/json.hpp"

#include <Eigen/SVD>
#include "fdaPDE/linear_algebra.h"
#include "test/src/utils/utils.h"

//Randomized SVD
using fdapde::core::RandomizedSVD;

using fdapde::core::IterationPolicy;
using fdapde::core::StoppingPolicy;

//json
using nlohmann::json;

//Gaussian Matrix
using fdapde::core::GaussianMatrix;

//Building a test Matrix
struct TestMatrix{
    DMatrix<double> U_,V_;
    DVector<double> SingVals_;

    TestMatrix(size_t row, size_t col, DVector<double> sing_vals){
        Eigen::HouseholderQR<DMatrix<double>> qr;
        SingVals_ = sing_vals;

        U_ = DMatrix<double>::Random(row, sing_vals.size());
        V_ = DMatrix<double>::Random(col, sing_vals.size());
        // orthogonalization
        U_ = qr.compute(U_).householderQ() * DMatrix<double>::Identity(U_.rows(),sing_vals.size());
        V_ = qr.compute(V_).householderQ() * DMatrix<double>::Identity(V_.rows(),sing_vals.size());
    }
    DMatrix<double> matrix() const{ return U_ * SingVals_.asDiagonal()* V_.transpose();}
};

//Performance metrics
template<typename SVDType>
void performance_metrics(TestMatrix A, int tr_rank){
    //Original matrix
    const auto start{std::chrono::steady_clock::now()};
    SVDType svd(A.matrix(),tr_rank);
    const auto end{std::chrono::steady_clock::now()};

    std::ofstream exe_times("results/exe_times.csv");
    exe_times << (std::chrono::duration<double>{end - start}).count() << std::endl;
    exe_times.close();

    std::ofstream singVal_err("results/sing_val_err.csv");
    singVal_err << (A.SingVals_.head(tr_rank)-svd.singularValues()).template lpNorm<2>() << std::endl;
    singVal_err.close();

    auto resU_m = A.U_.leftCols(svd.rank())-svd.matrixU();
    auto resU_p = A.U_.leftCols(svd.rank())+svd.matrixU();
    Eigen::saveMarket(resU_m.colwise().maxCoeff().array().min(resU_p.colwise().maxCoeff().array()),"results/l_sing_vecs_err.mtx");

    auto resV_m = A.V_.leftCols(svd.rank())-svd.matrixV();
    auto resV_p = A.V_.leftCols(svd.rank())+svd.matrixV();
    Eigen::saveMarket(resV_m.colwise().maxCoeff().array().min(resV_p.colwise().maxCoeff().array()),"results/r_sing_vecs_err.mtx");
}


int main(){
    //parsing data
    std::ifstream input("params.json");
    json data = json::parse(input);

    Eigen::setNbThreads(data["RunParams"].value("n_cores",1));
    std::cout << "N cores: " << Eigen::nbThreads() << std::endl;

    size_t rows, cols;
    rows = data["TestMatrix"].value("rows", 1000);
    cols = data["TestMatrix"].value("cols", 1000);

    DVector<double> sing_vals(std::min(rows,cols));
    //singular values decay
    for (int i = 0; i < sing_vals.size(); ++i){
        //sing_vals(i) = pow(0.997f, i); //fast-decaying
        sing_vals(i) = 1/std::log(i+2); //slow-decaying
    }
    TestMatrix A(rows, cols, sing_vals);
    int tr_rank = data["RunParams"].value("truncation_param",3);
    int seed = data["RunParams"].value("seed",7050);

    if(data["RunParams"].value("method","rsi") == "rsi"){
        performance_metrics<RandomizedSVD<DMatrix<double>,IterationPolicy::SubspaceIterations>>(A,tr_rank);
    }else if(data["RunParams"].value("method","rsi_ext") == "rsi_ext"){
        performance_metrics<RandomizedSVD<DMatrix<double>,IterationPolicy::ExtendedSubspaceIterations>>(A,tr_rank);
    }
    else if(data["RunParams"].value("method","rbki") == "rbki"){
        performance_metrics<RandomizedSVD<DMatrix<double>,IterationPolicy::BlockKrylovIterations>>(A,tr_rank);
    }
    else if(data["RunParams"].value("method","rbki_ext") == "rbki_ext"){
        performance_metrics<RandomizedSVD<DMatrix<double>,IterationPolicy::ExtendedBlockKrylovIterations>>(A,tr_rank);
    }else{
        const auto start{std::chrono::steady_clock::now()};
        Eigen::JacobiSVD<DMatrix<double>> svd(A.matrix(), Eigen::ComputeThinU | Eigen::ComputeThinV);
        const auto end{std::chrono::steady_clock::now()};

        std::ofstream exe_times("results/exe_times.csv");
        exe_times << (std::chrono::duration<double>{end - start}).count() << std::endl;
        exe_times.close();
    }
    return 0;
}
