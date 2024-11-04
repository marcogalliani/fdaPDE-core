//
// Created by Marco Galliani on 28/10/24.
//

#ifndef RSVD_H
#define RSVD_H

#include <utility>
#include <memory>
#include <tuple>
#include <random>


namespace fdapde{
namespace internals{

// Definition of the gaussian matrix generator
template<typename random_engine=std::mt19937>
DMatrix<double> GaussianMatrix(size_t rows, size_t cols, unsigned int seed=fdapde::random_seed, double sigma = 1.0){
    random_engine rand_eng{seed};
    std::normal_distribution norm_distr{0.0,sigma};

    DMatrix<double> Values(rows,cols);
    //filling the vector with random values
    for(size_t i = 0; i < rows; ++i){
        for(size_t j = 0; j < cols; j++){
            Values(i,j) = norm_distr(rand_eng);
        }
    }
    return Values;
}

//Block Classical Gram-Schmidt+ algorithm
std::pair<DMatrix<double>,DMatrix<double>> BCGS_plus(const DMatrix<double> &X, const DMatrix<double> &new_block){
    DMatrix<double> orth_block;
    //orthogonalization w.r.t. previous blocks
    orth_block =
            (DMatrix<double>::Identity(new_block.rows(),new_block.rows()) - X*X.transpose()) * new_block;
    //orthogonalization of the block
    Eigen::HouseholderQR<DMatrix<double>> qr(orth_block);
    orth_block = qr.householderQ() * DMatrix<double>::Identity(new_block.rows(),new_block.cols());
    //orthogonalization w.r.t. previous blocks
    orth_block =
            (DMatrix<double>::Identity(new_block.rows(),new_block.rows()) - X*X.transpose()) * new_block;
    //orthogonalization of the block
    qr.compute(orth_block);
    orth_block = qr.householderQ() * DMatrix<double>::Identity(new_block.rows(),new_block.cols());

    return std::make_pair(orth_block,qr.matrixQR().triangularView<Eigen::Upper>().toDenseMatrix().topRows(new_block.cols()));
}
}//internals

namespace core{
//Interface for the approximation strategy
template<typename MatrixType>
class RSVDStrategy{
protected:
    unsigned int seed_=fdapde::random_seed;
    double tol_=1e-3;
    //storage of the decomposition
    DMatrix<double> U_,V_;
    DVector<double> Sigma_;
public:
    RSVDStrategy()=default;
    RSVDStrategy(unsigned int seed, double tol) : seed_(seed), tol_(tol){}
    virtual void compute(const MatrixType &A, int rank, int max_iter) = 0;
    //setter
    void setTol(double tol){ tol_=tol;}
    void setSeed(unsigned int seed){ seed_=seed;}
    //getters
    int rank() const{ return Sigma_.size();}
    DMatrix<double> matrixU() const{ return U_;}
    DMatrix<double> matrixV() const{ return V_;}
    DVector<double> singularValues() const{ return Sigma_;}
    //destructor
    virtual ~RSVDStrategy() = default;
};

template<typename MatrixType>
class RSI : public RSVDStrategy<MatrixType>{
public:
    RSI()=default;
    RSI(unsigned int seed, double tol) : RSVDStrategy<MatrixType>(seed,tol){}
    void compute(const MatrixType &A, int rank, int max_iter) override{
        //params init
        int max_rank = std::min(A.rows(),A.cols());
        int block_sz = std::min(2*rank,max_rank); //default setting
        max_iter = std::min(max_iter, max_rank);
        //Q,B init
        Eigen::HouseholderQR<DMatrix<double>> qr(A*fdapde::internals::GaussianMatrix(A.cols(), block_sz, this->seed_));
        DMatrix<double> Q = qr.householderQ()*DMatrix<double>::Identity(A.rows(),block_sz);
        DMatrix<double> B = A.transpose()*Q;
        //Subspace Iterations
        Eigen::JacobiSVD<DMatrix<double>> svd(B.transpose(), Eigen::ComputeThinU | Eigen::ComputeThinV);
        DMatrix<double> E = A*svd.matrixV().leftCols(rank)-Q*svd.matrixU().leftCols(rank)*svd.singularValues().head(rank).asDiagonal();
        double res_err = E.colwise().template lpNorm<2>().maxCoeff();
        double norm_A = A.norm();
        for(int i=0; res_err>this->tol_*norm_A && i< max_iter; i++){
            qr.compute(B);
            Q = qr.householderQ()*DMatrix<double>::Identity(A.cols(), block_sz);
            B = A*Q;
            qr.compute(B);
            Q = qr.householderQ()*DMatrix<double>::Identity(A.rows(), block_sz);
            B = A.transpose()*Q;
            //compute the residual error
            svd.compute(B.transpose(),Eigen::ComputeThinU | Eigen::ComputeThinV);
            E = A*svd.matrixV().leftCols(rank)-Q*svd.matrixU().leftCols(rank)*svd.singularValues().head(rank).asDiagonal();
            res_err = E.colwise().template lpNorm<2>().maxCoeff();
        }
        this->U_ = Q*svd.matrixU().leftCols(rank);
        this->V_ = svd.matrixV().leftCols(rank);
        this->Sigma_ = svd.singularValues().head(rank);
        return;
    }
};

template<typename MatrixType>
class RBKI : public RSVDStrategy<MatrixType>{
public:
    RBKI()=default;
    RBKI(unsigned int seed, double tol) : RSVDStrategy<MatrixType>(seed,tol){}
    void compute(const MatrixType &A, int rank, int max_iter) override{
        //params init
        int block_sz; //default setting for RBKI
        if(A.rows()<=100){
            block_sz = 1;
        }else{
            block_sz = 10;
        }
        int max_dim = std::ceil((double)std::min(A.rows(), A.cols())/(double)block_sz)*block_sz;
        max_iter = std::min(max_iter,max_dim/block_sz);
        //Q,B init
        DMatrix<double> Q(A.rows(), max_dim);
        Q.leftCols(block_sz) = A*fdapde::internals::GaussianMatrix(A.cols(), block_sz, this->seed_);
        Eigen::HouseholderQR<DMatrix<double>> qr(Q.leftCols(block_sz));
        Q.leftCols(block_sz) = qr.householderQ()*DMatrix<double>::Identity(A.rows(), block_sz);
        DMatrix<double> B(A.cols(), max_dim);
        B.leftCols(block_sz) = A.transpose()*Q.leftCols(block_sz);
        //Block Krylov Iterations
        Eigen::JacobiSVD<DMatrix<double>> svd(B.leftCols(block_sz).transpose(), Eigen::ComputeThinU | Eigen::ComputeThinV);
        DMatrix<double> E = A*svd.matrixV().leftCols(std::min(rank,block_sz)) - Q.leftCols(block_sz)*svd.matrixU().leftCols(std::min(rank,block_sz))*svd.singularValues().head(std::min(rank,block_sz)).asDiagonal();
        double res_err = E.colwise().template lpNorm<2>().maxCoeff();
        double norm_A = A.norm();
        int n_cols_Q = block_sz;
        for(int i=0; res_err > this->tol_*norm_A && i < max_iter; i++, n_cols_Q+=block_sz){
            //update range matrix
            Q.middleCols((i+1)*block_sz,block_sz) = A*B.middleCols(i*block_sz, block_sz);
            Q.middleCols((i+1)*block_sz,block_sz) = fdapde::internals::BCGS_plus(Q.leftCols((i+1)*block_sz), Q.middleCols((i+1)*block_sz,block_sz)).first;
            //update residual matrix
            B.middleCols((i+1)*block_sz,block_sz) = A.transpose()*Q.middleCols((i+1)*block_sz,block_sz);
            //update the error
            svd.compute(B.leftCols((i+2)*block_sz).transpose(), Eigen::ComputeThinU | Eigen::ComputeThinV);
            E = A*svd.matrixV().leftCols(std::min(rank,(i+2)*block_sz)) - Q.leftCols((i+2)*block_sz)*svd.matrixU().leftCols(std::min(rank,(i+2)*block_sz))*svd.singularValues().head(std::min(rank,(i+2)*block_sz)).asDiagonal();
            res_err = E.colwise().template lpNorm<2>().maxCoeff();
        }
        rank = std::min((int)svd.singularValues().size(), rank);
        this->U_ = Q.leftCols(n_cols_Q)*svd.matrixU().leftCols(rank);
        this->V_ = svd.matrixV().leftCols(rank);
        this->Sigma_ = svd.singularValues().head(rank);
        return;
    }
};

template<typename MatrixType>
class RSVD{
private:
    std::unique_ptr<RSVDStrategy<MatrixType>> rsvd_strategy_;
public:
    explicit RSVD(std::unique_ptr<RSVDStrategy<MatrixType>> &&strategy=std::make_unique<RSI<DMatrix<double>>>()): rsvd_strategy_(std::move(strategy)){}
    void compute(const MatrixType &A, int rank, int max_iter=1e3){
        rsvd_strategy_->compute(A,rank,max_iter);
        return;
    }
    //setters
    void setTol(double tol){ rsvd_strategy_->setTol(tol);}
    void setSeed(unsigned int seed){ rsvd_strategy_->setSeed(seed);}
    //getters
    int rank() const{ return rsvd_strategy_->rank();}
    DMatrix<double> matrixU() const{ return rsvd_strategy_->matrixU();}
    DMatrix<double> matrixV() const{ return rsvd_strategy_->matrixV();}
    DVector<double> singularValues() const{ return rsvd_strategy_->singularValues();}
};

}//core
}//fdpade

#endif //RSVD_H
