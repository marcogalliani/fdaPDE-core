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
//Interface for the EVD strategy
template<typename MatrixType>
class REVDStrategy{
protected:
    double tolerance_=1e-5;
    int max_iter_=50;
    unsigned int seed_=fdapde::random_seed;
    //storage of the decomposition
    DMatrix<double> U_;
    DVector<double> Lambda_;
public:
    REVDStrategy()=default;
    REVDStrategy(double tol, int max_iter, unsigned int seed) : tolerance_(tol), max_iter_(max_iter), seed_(seed){}
    //compute methods overloads
    virtual void compute(const MatrixType &A, int rank) = 0;
    virtual void compute(const MatrixType &A, int rank, int block_sz) = 0;
    //utility to perform deep copies
    virtual std::unique_ptr<REVDStrategy<MatrixType>> clone() const = 0;
    //setters
    void set_tol(double tol){ tolerance_=tol;}
    void set_max_iter_(int max_iter){ max_iter_=max_iter;}
    void set_seed(unsigned int seed){ seed_=seed;}
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
    NysRSI(double tol, int max_iter, unsigned int seed) : REVDStrategy<MatrixType>(tol,max_iter,seed){}
    void compute(const MatrixType &A, int rank) override{
        int max_rank = A.rows(); //equal to A.cols()
        //default setting for RSI, following Halko et al (2010).
        int block_sz = std::min(2*rank,max_rank);
        compute(A,rank,block_sz);
    }
    void compute(const MatrixType &A, int rank, int block_sz) override{
        double shift = A.diagonal().sum()*std::numeric_limits<double>::epsilon();
        //Matrix initialisation: X approximates the range of A
        //Y stores A*X at each iteration
        // F stores the Nystrom factor, F*F^T is a low-rank (Nystrom) approximation of A
        DMatrix<double> Y = fdapde::internals::GaussianMatrix(A.rows(), block_sz, this->seed_);
        DMatrix<double> X;
        DMatrix<double> F;
        Eigen::HouseholderQR<DMatrix<double>> qr;
        //Stopping criterion initialisation
        Eigen::JacobiSVD<DMatrix<double>> svd;
        DMatrix<double> E;
        double res_err = this->tolerance_+1;
        //Subspace Iterations
        for(int i=0; res_err > this->tolerance_ && i<this->max_iter_; ++i) {
            qr.compute(Y);
            X = qr.householderQ() * DMatrix<double>::Identity(A.rows(),block_sz);
            Y = A*X;
            //Nystrom factor computation
            Y += shift*DMatrix<double>::Identity(Y.rows(),Y.cols());
            Eigen::LLT<DMatrix<double>> chol(X.transpose()*Y);
            F = chol.matrixU().solve<Eigen::OnTheRight>(Y);
            //update the error (+SVD of F that will be used to build the EVD)
            svd.compute(F,Eigen::ComputeThinU | Eigen::ComputeThinV);
            E = A*svd.matrixU().leftCols(rank) - svd.matrixU().leftCols(rank)*(svd.singularValues().head(rank).array().pow(2)-shift).matrix().asDiagonal();
            res_err =  std::sqrt(2)*E.colwise().template lpNorm<2>().maxCoeff();
        }
        //Constructing the EVD
        this->U_ = svd.matrixU().leftCols(rank);
        this->Lambda_ = (svd.singularValues().head(rank).array().pow(2)-shift).matrix();
        return;
    }
    //Utility used by REVD to perform deep copies
    virtual std::unique_ptr<REVDStrategy<MatrixType>> clone() const override{
        return std::make_unique<NysRSI<MatrixType>>(*this);
    }
};

template<typename MatrixType>
class NysRBKI : public REVDStrategy<MatrixType>{
public:
    NysRBKI()=default;
    NysRBKI(double tol, int max_iter, unsigned int seed) : REVDStrategy<MatrixType>(tol,max_iter,seed){}
    void compute(const MatrixType &A, int rank) override{
        int block_sz = (A.rows()<=100)? 1 : 10;
        //default setting for RBKI, following Tropp et al (2023).
        compute(A,rank,block_sz);
    }
    void compute(const MatrixType &A, int rank, int block_sz) override{
        //adjust maximum number of iterations and Krylov Subspace maximum dimension
        int max_iter = std::min(this->max_iter_,
                                static_cast<int>(std::min(A.rows(),A.cols())/block_sz+1));
        int max_dim = (max_iter+1)*block_sz; //maximum dimension of the Krylov subspace
        double shift = A.diagonal().sum()*std::numeric_limits<double>::epsilon();
        //Matrix init: X Krylov subspace, F Nystrom factor, Y=A*X used to compute F
        DMatrix<double> X,Y,S,F;
        X.resize(A.rows(),max_dim); Y.resize(A.rows(),max_dim);
        S = DMatrix<double>::Zero(max_dim,max_dim);
        Eigen::HouseholderQR<DMatrix<double>> qr(fdapde::internals::GaussianMatrix(A.rows(),block_sz,this->seed_));
        X.leftCols(block_sz) = qr.householderQ()*DMatrix<double>::Identity(A.rows(),block_sz);
        Y.leftCols(block_sz) = A*X.leftCols(block_sz);
        //Stopping criterion initialisation
        Eigen::JacobiSVD<DMatrix<double>> svd;
        DMatrix<double> E;
        double res_err=this->tolerance_+1;
        //Block Krylov iterations
        int sizeX = block_sz;
        for(int i=0; i<max_iter && res_err>this->tolerance_;i++){
            //add block to X subspace
            X.middleCols(sizeX,block_sz) = Y.middleCols(i*block_sz,block_sz) + shift*X.middleCols(i*block_sz,block_sz);
            //add columns to S
            DMatrix<double> colS = DMatrix<double>::Zero(X.rows(),(i+1)*block_sz);
            colS.middleCols(std::max(i-1,0)*block_sz,block_sz) = X.middleCols(std::max(i-1,0)*block_sz,block_sz);
            colS.middleCols(i*block_sz,block_sz) = X.middleCols(i*block_sz,block_sz);
            colS = colS.transpose()*X.middleCols((i+1)*block_sz,block_sz);
            //orthogonalisation of the updated X subspace
            auto new_block_qr = fdapde::internals::BCGS_plus(X.leftCols((i+1)*block_sz),X.middleCols((i+1)*block_sz,block_sz));
            X.middleCols((i+1)*block_sz,block_sz) = new_block_qr.first;
            //constructing the Nystrom factor
            S.block(0,i*block_sz,(i+1)*block_sz,block_sz) = colS;
            Eigen::LLT<DMatrix<double>> chol(S.block(0,0,(i+1)*block_sz,(i+1)*block_sz));
            S.block((i+1)*block_sz,i*block_sz,block_sz,block_sz) = new_block_qr.second;
            F = chol.matrixU().solve<Eigen::OnTheRight>(S.block(0,0,(i+2)*block_sz,(i+1)*block_sz));
            //update the residual matrix
            Y.middleCols(sizeX,block_sz) = A*X.middleCols(sizeX,block_sz);
            //update X subspace dimension
            sizeX+=block_sz;
            //update the error
            svd.compute(F, Eigen::ComputeThinU | Eigen::ComputeThinV);
            E = Y.leftCols(sizeX)*svd.matrixU().leftCols(std::min(rank,(i+1)*block_sz)) - X.leftCols(sizeX)*svd.matrixU().leftCols(std::min(rank,(i+1)*block_sz))*(svd.singularValues().head(std::min(rank,(i+1)*block_sz)).array().pow(2)-shift).matrix().asDiagonal();
            res_err =  std::sqrt(2)*E.colwise().template lpNorm<2>().maxCoeff();
        }
        rank = std::min(static_cast<int>(svd.singularValues().size()), rank);
        this->U_ = X.leftCols(sizeX)*svd.matrixU().leftCols(rank);
        this->Lambda_ = (svd.singularValues().head(rank).array().pow(2)-shift).matrix();
        return;
    }
    virtual std::unique_ptr<REVDStrategy<MatrixType>> clone() const override{
        return std::make_unique<NysRBKI<MatrixType>>(*this);
    }
};

template<typename MatrixType>
class REVD{
private:
    std::unique_ptr<REVDStrategy<MatrixType>> revd_strategy_;
public:
    explicit REVD(std::unique_ptr<REVDStrategy<MatrixType>> &&strategy=std::make_unique<NysRSI<MatrixType>>()): revd_strategy_(std::move(strategy)){}
    //custom copy-operators: deep copies
    REVD(const REVD& other)
        : revd_strategy_(other.revd_strategy_ ? other.revd_strategy_->clone() : nullptr){}
    REVD& operator=(const REVD other){
        if (this != &other) {
            // Create a deep copy of the strategy
            revd_strategy_ = other.revd_strategy_ ? other.revd_strategy_->clone() : nullptr;
        }
        return *this;
    }
    //default block_sz setting
    void compute(const MatrixType &A, int tr_rank){
        revd_strategy_->compute(A,tr_rank);
        return;
    }
    //explicit block_sz setting for expert users
    void compute(const MatrixType &A, int tr_rank, int block_sz){
        revd_strategy_->compute(A,tr_rank,block_sz);
        return;
    }
    //setters
    void set_tol(double tol){ revd_strategy_->set_tol(tol);}
    void set_max_iter(int max_iter){ revd_strategy_->set_max_iter(max_iter);}
    void set_seed(unsigned int seed){ revd_strategy_->set_seed(seed);}
    //getters
    int rank() const{ return revd_strategy_->rank();}
    DMatrix<double> matrixU() const{ return revd_strategy_->matrixU();}
    DVector<double> eigenValues() const{ return revd_strategy_->eigenValues();}
};

//a trait to detect the usage of randomized evd
template <typename T>
struct is_rand_evd : std::false_type {};

template <typename T>
struct is_rand_evd<REVD<T>> : std::true_type {};

}//core
}//fdpade

#endif //REVD_H
