//
// Created by Marco Galliani on 30/10/24.
//

#ifndef REVD_H
#define REVD_H

#include <utility>
#include <memory>
#include <tuple>
#include <limits>
#include <type_traits>

namespace fdapde{
namespace core{
//Interface for the Approximation strategy
template<typename MatrixType>
class REVDStrategy{
protected:
    unsigned int seed_=fdapde::random_seed;
    double tol_=1e-3;
    //storage of the decomposition
    DMatrix<double> U_;
    DVector<double> Lambda_;
public:
    REVDStrategy()=default;
    REVDStrategy(unsigned int seed, double tol) : seed_(seed), tol_(tol){}
    virtual void compute(const MatrixType &A, int rank, int max_iter) = 0;
    //setters
    void setTol(double tol){ tol_=tol;}
    void setSeed(unsigned int seed){ seed_=seed;}
    //getters
    int rank() const{ return Lambda_.size();}
    DMatrix<double> matrixU() const{ return U_;}
    DVector<double> eigenValues() const{ return Lambda_;}
    //destructor
    virtual ~REVDStrategy() = default;
};

template<typename MatrixType>
class NysRSI : public REVDStrategy<MatrixType>{
public:
    NysRSI()=default;
    NysRSI(unsigned int seed, double tol) : REVDStrategy<MatrixType>(seed,tol){}
    void compute(const MatrixType &A, int rank, int max_iter) override{
        //params init
        int max_rank = A.rows(); //equal to A.cols()
        int block_sz = std::min(2*rank,max_rank); //default setting
        max_iter = std::min(max_iter, max_rank);
        double shift = A.trace()*std::numeric_limits<double>::epsilon();
        //factor init
        DMatrix<double> Y = fdapde::internals::GaussianMatrix(A.rows(), block_sz, this->seed_);
        DMatrix<double> X;
        DMatrix<double> F;
        Eigen::HouseholderQR<DMatrix<double>> qr;
        //error
        Eigen::JacobiSVD<DMatrix<double>> svd;
        DMatrix<double> E;
        double norm_A = A.norm(), res_err = norm_A;
        //iterations
        for(int i=0; res_err > this->tol_*norm_A && i<max_iter; ++i) {
            qr.compute(Y);
            X = qr.householderQ() * DMatrix<double>::Identity(A.rows(),block_sz);
            Y = A*X;
            //construct the factor
            Y += shift*DMatrix<double>::Identity(Y.rows(),Y.cols());
            Eigen::LLT<DMatrix<double>> chol(X.transpose()*Y);
            F = chol.matrixU().solve<Eigen::OnTheRight>(Y);
            //update the error
            svd.compute(F,Eigen::ComputeThinU | Eigen::ComputeThinV);
            E = A*svd.matrixU().leftCols(rank) - svd.matrixU().leftCols(rank)*(svd.singularValues().head(rank).array().pow(2)-shift).matrix().asDiagonal();
            res_err =  E.colwise().template lpNorm<2>().maxCoeff();
        }
        this->U_ = svd.matrixU().leftCols(rank);
        this->Lambda_ = (svd.singularValues().head(rank).array().pow(2)-shift).matrix();
        return;
    }
};

template<typename MatrixType>
class NysRBKI : public REVDStrategy<MatrixType>{
public:
    NysRBKI()=default;
    NysRBKI(unsigned int seed, double tol) : REVDStrategy<MatrixType>(seed,tol){}
    void compute(const MatrixType &A, int rank, int max_iter) override{
        //params init
        int max_rank = A.rows(); //equal to A.cols()
        int block_sz; //default setting
        if(A.rows()<1000){
            block_sz = 1;
        }else{
            block_sz = 10;
        }
        max_iter = std::min(max_iter,max_rank/block_sz-1);
        double shift = A.trace()*std::numeric_limits<double>::epsilon();
        //factor init
        DMatrix<double> X,Y,S,F;
        X.resize(A.rows(),max_rank); Y.resize(A.rows(),max_rank);
        S = DMatrix<double>::Zero(max_rank,max_rank);
        Eigen::HouseholderQR<DMatrix<double>> qr(fdapde::internals::GaussianMatrix(A.rows(),block_sz,this->seed_));
        X.leftCols(block_sz) = qr.householderQ()*DMatrix<double>::Identity(A.rows(),block_sz);
        Y.leftCols(block_sz) = A*X.leftCols(block_sz);
        //error
        Eigen::JacobiSVD<DMatrix<double>> svd;
        DMatrix<double> E;
        double norm_A=A.norm(), res_err=norm_A;
        //iterations
        int n_cols_X = block_sz;
        for(int i=0; i<max_iter && res_err>this->tol_*norm_A;i++,n_cols_X+=block_sz){
            X.middleCols((i+1)*block_sz,block_sz) = Y.middleCols(i*block_sz,block_sz) + shift*X.middleCols(i*block_sz,block_sz);
            //blocked column
            DMatrix<double> new_col = DMatrix<double>::Zero(X.rows(),(i+1)*block_sz);
            new_col.middleCols(std::max(i-1,0)*block_sz,block_sz) = X.middleCols(std::max(i-1,0)*block_sz,block_sz);
            new_col.middleCols(i*block_sz,block_sz) = X.middleCols(i*block_sz,block_sz);
            new_col = new_col.transpose()*X.middleCols((i+1)*block_sz,block_sz);
            //orthogonalisation
            auto new_block_qr = fdapde::internals::BCGS_plus(X.leftCols((i+1)*block_sz),X.middleCols((i+1)*block_sz,block_sz));
            X.middleCols((i+1)*block_sz,block_sz) = new_block_qr.first;
            //cholesky
            S.block(0,i*block_sz,(i+1)*block_sz,block_sz) = new_col;
            Eigen::LLT<DMatrix<double>> chol(S.block(0,0,(i+1)*block_sz,(i+1)*block_sz));
            S.block((i+1)*block_sz,i*block_sz,block_sz,block_sz) = new_block_qr.second;
            F = chol.matrixU().solve<Eigen::OnTheRight>(S.block(0,0,(i+2)*block_sz,(i+1)*block_sz));
            //update Y
            Y.middleCols((i+1)*block_sz,block_sz) = A*X.middleCols((i+1)*block_sz,block_sz);
            //update the error
            svd.compute(F, Eigen::ComputeThinU | Eigen::ComputeThinV);
            E = Y.leftCols((i+2)*block_sz)*svd.matrixU().leftCols(std::min(rank,(i+1)*block_sz)) - X.leftCols((i+2)*block_sz)*svd.matrixU().leftCols(std::min(rank,(i+1)*block_sz))*(svd.singularValues().head(std::min(rank,(i+1)*block_sz)).array().pow(2)-shift).matrix().asDiagonal();
            res_err =  E.colwise().template lpNorm<2>().maxCoeff();
        }
        rank = std::min((int)svd.singularValues().size(), rank);
        this->U_ = X.leftCols(n_cols_X)*svd.matrixU().leftCols(rank);
        this->Lambda_ = (svd.singularValues().head(rank).array().pow(2)-shift).matrix();
        return;
    }
};

template<typename MatrixType>
class REVD{
private:
    std::unique_ptr<REVDStrategy<MatrixType>> revd_strategy_;
    DMatrix<double> U_;
    DVector<double> Lambda_;
public:
    explicit REVD(std::unique_ptr<REVDStrategy<MatrixType>> &&strategy=std::make_unique<NysRSI<DMatrix<double>>>()): revd_strategy_(std::move(strategy)){}
    void compute(const MatrixType &A, int tr_rank, int max_iter=1e3){
        revd_strategy_->compute(A,tr_rank,max_iter);
        return;
    }
    //setters
    void setTol(double tol){ revd_strategy_->setTol(tol);}
    void setSeed(unsigned int seed){ revd_strategy_->setSeed(seed);}
    //getters
    int rank() const{ return revd_strategy_->rank();}
    DMatrix<double> matrixU() const{ return revd_strategy_->matrixU();}
    DVector<double> eigenValues() const{ return revd_strategy_->eigenValues();}
};

}//core
}//fdpade

#endif //REVD_H
