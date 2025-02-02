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

//Gaussian matrix generator
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
    Eigen::HouseholderQR<DMatrix<double>> qr;
    DMatrix<double> X_orth_space_proj = (DMatrix<double>::Identity(new_block.rows(),new_block.rows()) - X*X.transpose());
    //orthogonalization w.r.t. previous blocks
    DMatrix<double> orth_block = X_orth_space_proj * new_block;
    //reorthogonalization w.r.t. previous blocks
    orth_block = X_orth_space_proj * orth_block;
    //orthogonalization of the block
    qr.compute(orth_block);
    orth_block = qr.householderQ() * DMatrix<double>::Identity(new_block.rows(),new_block.cols());

    return std::make_pair(orth_block,qr.matrixQR().triangularView<Eigen::Upper>().toDenseMatrix().topRows(new_block.cols()));
}
}//internals

namespace core{
//Interface for the SVD strategy
template<typename MatrixType>
class RSVDStrategy{
protected:
    double tolerance_=1e-5;
    int max_iter_=50;
    unsigned int seed_=fdapde::random_seed;
    //storage of the decomposition
    DMatrix<double> U_,V_;
    DVector<double> Sigma_;
public:
    RSVDStrategy()=default;
    RSVDStrategy(double tol, int max_iter, unsigned int seed) :  tolerance_(tol), max_iter_(max_iter), seed_(seed){}
    //compute methods overloads
    virtual void compute(const MatrixType &A, int rank) = 0;
    virtual void compute(const MatrixType &A, int rank, int block_sz) = 0; //for expert users, allows to set block_sz
    //utility to perform deep copies
    virtual std::unique_ptr<RSVDStrategy<MatrixType>> clone() const = 0;
    //setters
    void set_tolerance(double tol){ tolerance_=tol;}
    void set_max_iter(int max_iter){ max_iter_=max_iter;}
    void set_seed(unsigned int seed){ seed_=seed;}
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
    RSI(double tol, int max_iter, unsigned int seed) : RSVDStrategy<MatrixType>(tol,max_iter,seed){}
    //overload for default setting of the block_sz parameter
    void compute(const MatrixType &A, int rank) override{
        int max_rank = std::min(A.rows(),A.cols());
        //default setting for RSI, following Halko et al (2010).
        int block_sz = std::min(2*rank,max_rank);
        //call to the actual implementation
        compute(A,rank,block_sz);
    }
    //actual implementation (public to let expert users set block_sz)
    void compute(const MatrixType &A, int rank, int block_sz) override{
        //Q,B init, with Q approximating the range of A
        //Q*B is a low-rank approximation of A
        Eigen::HouseholderQR<DMatrix<double>> qr(A*fdapde::internals::GaussianMatrix(A.cols(), block_sz, this->seed_));
        DMatrix<double> Q = qr.householderQ()*DMatrix<double>::Identity(A.rows(),block_sz);
        DMatrix<double> B = A.transpose()*Q;
        //Stopping criterion initialisation
        Eigen::JacobiSVD<DMatrix<double>> svd(B.transpose(), Eigen::ComputeThinU | Eigen::ComputeThinV);
        DMatrix<double> E = A*svd.matrixV().leftCols(rank)-Q*svd.matrixU().leftCols(rank)*svd.singularValues().head(rank).asDiagonal();
        double res_err = E.colwise().template lpNorm<2>().maxCoeff();
        //Subspace iterations: building (AA^T)^{q} A\Omega
        for(int i=0; res_err>this->tolerance_ && i< this->max_iter_; i++){
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
        //Constructing the SVD decomposition
        this->U_ = Q*svd.matrixU().leftCols(rank);
        this->V_ = svd.matrixV().leftCols(rank);
        this->Sigma_ = svd.singularValues().head(rank);
        return;
    }
    //Utility used by RSVD to perform deep copies
    std::unique_ptr<RSVDStrategy<MatrixType>> clone() const override{
        return std::make_unique<RSI<MatrixType>>(*this);
    };
};

template<typename MatrixType>
class GeneralizedRSI : public RSVDStrategy<MatrixType>{
public:
    GeneralizedRSI()=default;
    GeneralizedRSI(double tol, int max_iter, unsigned int seed) : RSVDStrategy<MatrixType>(tol,max_iter,seed){}
    //overload for default setting of the block_sz parameter
    void compute(const MatrixType &A, int rank) override{
        int max_rank = std::min(A.rows(),A.cols());
        //default setting for RSI, following Halko et al (2010).
        int block_sz = std::min(2*rank,max_rank);
        //call to the actual implementation
        compute(A,rank,block_sz);
    }
    //actual implementation (public to let expert users set block_sz)
    void compute(const MatrixType &A, int rank, int block_sz) override{
        //X,Y init, with X,Y approximating the range and corange of A, respectively
        Eigen::HouseholderQR<DMatrix<double>> qr(A*fdapde::internals::GaussianMatrix(A.cols(), block_sz, this->seed_));
        DMatrix<double> X = qr.householderQ()*DMatrix<double>::Identity(A.rows(),block_sz);
        DMatrix<double> Y;
        //Stopping criterion initialisation
        Eigen::JacobiSVD<DMatrix<double>> svd;
        DMatrix<double> E;
        double res_err = this->tolerance_+1;
        //Generalized Subspace Iterations: building (A^TA)^q
        for(int i=0; res_err>this->tolerance_ && i< 2*this->max_iter_; i++){
            if(i%2 == 0){
                Y = A.transpose() * X;
                qr.compute(Y);
                Y = qr.householderQ()*DMatrix<double>::Identity(Y.rows(),block_sz);
                DMatrix<double> T = qr.matrixQR().triangularView<Eigen::Upper>(); //X*T*Y^T low-rank approximation of A
                //error update
                svd.compute(T.topRows(block_sz).transpose(), Eigen::ComputeThinU | Eigen::ComputeThinV);
                E = A*Y*svd.matrixV().leftCols(rank) - X*svd.matrixU().leftCols(rank)*svd.singularValues().head(rank).asDiagonal();
            }else{
                X = A *Y;
                qr.compute(X);
                X = qr.householderQ()*DMatrix<double>::Identity(X.rows(),block_sz);
                DMatrix<double> T = qr.matrixQR().triangularView<Eigen::Upper>(); //X*T*Y^T low-rank approximation of A
                //error update
                svd.compute(T.topRows(block_sz), Eigen::ComputeThinU | Eigen::ComputeThinV);
                E = A.transpose()*X*svd.matrixU().leftCols(rank) - Y*svd.matrixV().leftCols(rank)*svd.singularValues().head(rank).asDiagonal();
            }
            res_err = E.colwise().template lpNorm<2>().maxCoeff();
        }
        //Constructing the SVD decomposition
        this->U_ = X*svd.matrixU().leftCols(rank);
        this->V_ = Y*svd.matrixV().leftCols(rank);
        this->Sigma_ = svd.singularValues().head(rank);
        return;
    }
    //Utility used by RSVD to perform deep copies
    std::unique_ptr<RSVDStrategy<MatrixType>> clone() const override{
        return std::make_unique<GeneralizedRSI<MatrixType>>(*this);
    };
};

template<typename MatrixType>
class RBKI : public RSVDStrategy<MatrixType>{
public:
    RBKI()=default;
    //overload for default setting of the block_sz parameter
    RBKI(double tol, int max_iter, unsigned int seed) : RSVDStrategy<MatrixType>(tol,max_iter,seed){}
    void compute(const MatrixType &A, int rank) override{
        //iterations are performed on the smaller dimension of A
        bool transposed = A.rows() > A.cols();
        const DMatrix<double> &A_view = transposed ? A.transpose() : A;
        //default setting for RBKI, following Tropp et al. (2023)
        int block_sz = (A_view.rows()<=100)? 1 : 10;
        compute(A,rank,block_sz);
    }
    //actual implementation (public to let expert users set block_sz)
    void compute(const MatrixType &A, int rank, int block_sz) override{
        //iterations are performed on the smaller dimension of A
        bool transposed = A.rows() > A.cols();
        const DMatrix<double> &A_view = transposed ? A.transpose() : A;
        //adjust maximum number of iterations and Krylov Subspace maximum dimension
        int max_iter = std::min(this->max_iter_,
                                static_cast<int>(std::min(A_view.rows(),A_view.cols())/block_sz+1)
                                );
        int max_dim = (max_iter+1)*block_sz; //max dim Krylov subspace
        //Q,B init, with Q containing the Krylov subspace
        //Q*B low-rank approximation of A
        DMatrix<double> Q(A_view.rows(), max_dim);
        Q.leftCols(block_sz) = A_view*fdapde::internals::GaussianMatrix(A_view.cols(), block_sz, this->seed_);
        Eigen::HouseholderQR<DMatrix<double>> qr(Q.leftCols(block_sz));
        Q.leftCols(block_sz) = qr.householderQ()*DMatrix<double>::Identity(A_view.rows(), block_sz);
        DMatrix<double> B(A_view.cols(), max_dim);
        B.leftCols(block_sz) = A_view.transpose()*Q.leftCols(block_sz);
        //Stopping criterion initialisation
        Eigen::JacobiSVD<DMatrix<double>> svd(B.leftCols(block_sz).transpose(), Eigen::ComputeThinU | Eigen::ComputeThinV);
        DMatrix<double> E = A_view*svd.matrixV().leftCols(std::min(rank,block_sz)) - Q.leftCols(block_sz)*svd.matrixU().leftCols(std::min(rank,block_sz))*svd.singularValues().head(std::min(rank,block_sz)).asDiagonal();
        double res_err = E.colwise().template lpNorm<2>().maxCoeff();
        //Block Krylov Iterations: building the Krylov subspace
        int n_cols_Q = block_sz;
        for(int i=0; res_err > this->tolerance_ && i < max_iter; i++, n_cols_Q+=block_sz){
            //update Krylov subspace
            Q.middleCols((i+1)*block_sz,block_sz) = A_view*B.middleCols(i*block_sz, block_sz);
            Q.middleCols((i+1)*block_sz,block_sz) = fdapde::internals::BCGS_plus(Q.leftCols((i+1)*block_sz), Q.middleCols((i+1)*block_sz,block_sz)).first;
            //update residual matrix
            B.middleCols((i+1)*block_sz,block_sz) = A_view.transpose()*Q.middleCols((i+1)*block_sz,block_sz);
            //update the error
            svd.compute(B.leftCols((i+2)*block_sz).transpose(), Eigen::ComputeThinU | Eigen::ComputeThinV);
            E = A_view*svd.matrixV().leftCols(std::min(rank,(i+2)*block_sz)) - Q.leftCols((i+2)*block_sz)*svd.matrixU().leftCols(std::min(rank,(i+2)*block_sz))*svd.singularValues().head(std::min(rank,(i+2)*block_sz)).asDiagonal();
            res_err = E.colwise().template lpNorm<2>().maxCoeff();
        }
        //Constructing the SVD decomposition
        rank = std::min(static_cast<int>(svd.singularValues().size()),rank);
        this->Sigma_ = svd.singularValues().head(rank);
        if(transposed){
            this->U_ = svd.matrixV().leftCols(rank);
            this->V_ = Q.leftCols(n_cols_Q)*svd.matrixU().leftCols(rank);
        }else{
            this->U_ = Q.leftCols(n_cols_Q)*svd.matrixU().leftCols(rank);
            this->V_ = svd.matrixV().leftCols(rank);
        }
        return;
    }
    //Utility used by RSVD to perform deep copies
    std::unique_ptr<RSVDStrategy<MatrixType>> clone() const override{
        return std::make_unique<RBKI<MatrixType>>(*this);
    };
};

template<typename MatrixType>
class GeneralizedRBKI : public RSVDStrategy<MatrixType>{
public:
    GeneralizedRBKI()=default;
    GeneralizedRBKI(double tol, int max_iter, unsigned int seed) : RSVDStrategy<MatrixType>(tol,max_iter,seed){}
    //overload for default setting of the block_sz parameter
    void compute(const MatrixType &A, int rank) override{
        //iterations are performed on the smaller dimension of A
        bool transposed = A.rows() > A.cols();
        const DMatrix<double> &A_view = transposed ? A.transpose() : A;
        //default setting for RBKI, following Tropp et al. (2023)
        int block_sz = (A_view.rows()<=100)? 1 : 10;
        compute(A,rank,block_sz);
    }
    //actual implementation (public to let expert users set block_sz)
    void compute(const MatrixType &A, int rank, int block_sz) override{
        //adjust maximum number of iterations a Krylov Subspace maximum dimension
        int max_iter = std::min(this->max_iter_,
                                static_cast<int>(std::min(A.rows(),A.cols())/block_sz+1)
                                );
        int max_dim = (max_iter+1)*block_sz; //maximum dimension of the Krylov subspace
        //Initialising matrix dimension: X,Y approximate the range and corange of A
        //Z,W are needed to evaluate the stopping criterion and avoid recomputing matrix products already computed
        DMatrix<double > X(A.rows(),max_dim), Z(A.rows(),max_dim);
        DMatrix<double> Y(A.cols(),max_dim), W(A.cols(),max_dim);
        //XTY^T low-rank approximation of A, where T is either R^T or S depending on the iteration
        DMatrix<double> R=DMatrix<double>::Zero(max_dim,max_dim), S=DMatrix<double>::Zero(max_dim+block_sz,max_dim);
        //Initialising matrices
        Eigen::HouseholderQR<DMatrix<double>> qr;
        X.leftCols(block_sz) = A * fdapde::internals::GaussianMatrix(A.cols(),block_sz,this->seed_);
        qr.compute(X.leftCols(block_sz));
        X.leftCols(block_sz) = qr.householderQ() * DMatrix<double>::Identity(A.rows(),block_sz);
        W.leftCols(block_sz) = A.transpose() * X.leftCols(block_sz);
        //Stopping criterion initialisation
        Eigen::JacobiSVD<DMatrix<double>> svd;
        DMatrix<double> E;
        double res_err = this->tolerance_+1;
        //Block Krylov Iterations: building the Krylov subspace
        int sizeX = block_sz, sizeY = 0;
        for(int i=0; res_err > this->tolerance_ && i < 2*max_iter; i++){
            if(i%2 == 0){
                //adding new block to Y subspace
                Y.middleCols(sizeY,block_sz) = W.middleCols(sizeY,block_sz);
                DMatrix<double> colR = Y.leftCols(sizeY).transpose() * Y.middleCols(sizeY,block_sz);
                //block orthogonalisation of the updated Y subspace
                auto Y_bcgs = fdapde::internals::BCGS_plus(Y.leftCols(sizeY), Y.middleCols(sizeY,block_sz));
                Y.middleCols(sizeY,block_sz) = Y_bcgs.first;
                //constructing the low-rank factorization: X*R^T*Y^T
                R.block(0,sizeY,colR.rows(),block_sz) = colR;
                R.block(colR.rows(),sizeY,block_sz,block_sz) = Y_bcgs.second.template triangularView<Eigen::Upper>();
                //updating Z to evaluate the stopping criterion
                Z.middleCols(sizeY,block_sz) = A * Y.middleCols(sizeY,block_sz);
                //updating Y subspace dimension
                sizeY += block_sz;
                //error update (+SVD of R^T that will be used to construct the approximate SVD)
                svd.compute(R.block(0,0,sizeY,sizeY).transpose(), Eigen::ComputeThinU | Eigen::ComputeThinV);
                E = Z.leftCols(sizeY)*svd.matrixV().leftCols(std::min(rank,sizeY)) - X.leftCols(sizeX)*(svd.matrixU().leftCols(std::min(rank,sizeY)))*svd.singularValues().head(std::min(rank,sizeY)).asDiagonal();
            }else{
                //adding new block to the X subspace
                X.middleCols(sizeX,block_sz) = Z.middleCols(sizeX-block_sz,block_sz);
                DMatrix<double> colS = X.leftCols(sizeX).transpose() * X.middleCols(sizeX,block_sz);
                //block orthogonalisation of the updated X subspace
                auto X_bcgs = fdapde::internals::BCGS_plus(X.leftCols(sizeX), X.middleCols(sizeX,block_sz));
                X.middleCols(sizeX,block_sz) = X_bcgs.first;
                //constructing the low-rank approximation X*S*Y^T
                S.block(0,sizeX-block_sz,colS.rows(),block_sz) = colS;
                S.block(colS.rows(), sizeX-block_sz,block_sz,block_sz) = X_bcgs.second;
                //updating W to evaluate the stopping criterion
                W.middleCols(sizeX,block_sz) = A.transpose() * X.middleCols(sizeX,block_sz);
                //updating X subspace dimension
                sizeX += block_sz;
                //error update (+SVD of S that will be used to construct the approximate SVD)
                svd.compute(S.block(0,0,sizeX,sizeX-block_sz), Eigen::ComputeThinU | Eigen::ComputeThinV);
                E = W.leftCols(sizeX)*svd.matrixU().leftCols(std::min(rank,sizeX-block_sz)) - Y.leftCols(sizeY)*svd.matrixV().leftCols(std::min(rank,sizeX-block_sz))*svd.singularValues().head(std::min(rank,sizeX-block_sz)).asDiagonal();
            }
            res_err =  E.colwise().template lpNorm<2>().maxCoeff();
        }
        //Constructing the SVD decomposition
        rank = std::min(static_cast<int>(svd.singularValues().size()), rank);
        this->U_ = X.leftCols(sizeX)*svd.matrixU().leftCols(rank);
        this->V_ = Y.leftCols(sizeY)*svd.matrixV().leftCols(rank);
        this->Sigma_ = svd.singularValues().head(rank);
        return;
    }
    //Utility used by RSVD to perform deep copies
    std::unique_ptr<RSVDStrategy<MatrixType>> clone() const override{
        return std::make_unique<GeneralizedRBKI<MatrixType>>(*this);
    };
};

template<typename MatrixType>
class RSVD{
private:
    //enforcing ownership over the computed decomposition
    std::unique_ptr<RSVDStrategy<MatrixType>> rsvd_strategy_;
public:
    explicit RSVD(std::unique_ptr<RSVDStrategy<MatrixType>> &&strategy=std::make_unique<RSI<MatrixType>>()): rsvd_strategy_(std::move(strategy)){}
    //custom copy-operators: deep copies
    RSVD(const RSVD& other)
        : rsvd_strategy_(other.rsvd_strategy_ ? other.rsvd_strategy_->clone() : nullptr){}
    RSVD& operator=(const RSVD other){
        if (this != &other) {
            // Create a deep copy of the strategy
            rsvd_strategy_ = other.rsvd_strategy_ ? other.rsvd_strategy_->clone() : nullptr;
        }
        return *this;
    }
    //default block_sz setting
    void compute(const MatrixType &A, int rank){
        rsvd_strategy_->compute(A,rank);
        return;
    }
    //explicit block_sz setting for expert users
    void compute(const MatrixType &A, int rank, int block_sz){
        rsvd_strategy_->compute(A,rank,block_sz);
        return;
    }
    //setters
    void set_tolerance(double tol){ rsvd_strategy_->set_tolerance(tol);}
    void set_max_iter(int max_iter){ rsvd_strategy_->set_max_iter(max_iter);}
    void set_seed(unsigned int seed){ rsvd_strategy_->set_seed(seed);}
    //getters
    int rank() const{ return rsvd_strategy_->rank();}
    DMatrix<double> matrixU() const{ return rsvd_strategy_->matrixU();}
    DMatrix<double> matrixV() const{ return rsvd_strategy_->matrixV();}
    DVector<double> singularValues() const{ return rsvd_strategy_->singularValues();}
};

//a trait to detect the usage of randomized svd
template <typename T>
struct is_rand_svd : std::false_type {};

template <typename T>
struct is_rand_svd<RSVD<T>> : std::true_type {};

}//core
}//fdpade

#endif //RSVD_H
