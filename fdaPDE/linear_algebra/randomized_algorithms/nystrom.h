//
// Created by Marco Galliani on 03/11/24.
//

#ifndef NYSTROM_APPROX_H
#define NYSTROM_APPROX_H

#include <memory>
#include <unordered_set>

namespace fdapde{
namespace core{

template<typename MatrixType>
class NysApproxStrategy{
protected:
    unsigned int seed_=fdapde::random_seed;
    double tol_=1e-3;
    //storage of the decomposition
    DMatrix<double> F_;
    double shift_;
public:
    NysApproxStrategy()=default;
    NysApproxStrategy(unsigned int seed , double tol) : seed_(seed), tol_(tol){}
    virtual void compute(const MatrixType &A, int block_sz, int max_iter) = 0;
    //setters
    void setTol(double tol){ tol_=tol;}
    void setSeed(unsigned int seed){ seed_=seed;}
    //getters
    int rank() const{ return F_.cols();}
    DMatrix<double> factor() const{ return F_;}
    double shift() const{ return shift_;}
    //destructor
    virtual ~NysApproxStrategy() = default;
};

template<typename MatrixType>
class RPChol : public NysApproxStrategy<MatrixType>{
public:
    RPChol()=default;
    RPChol(unsigned int seed, double tol) : NysApproxStrategy<MatrixType>(seed,tol){}
    void compute(const MatrixType &A, int block_sz, int max_iter) override{
        //params init
        max_iter = std::min(max_iter,(int)std::ceil((double)A.cols()/(double)block_sz));
        std::mt19937 rng{this->seed_};
        this->shift_ = std::numeric_limits<double>::epsilon()*A.trace();
        //factor init
        std::vector<int> ind_cols_A(A.cols());
        std::iota(ind_cols_A.begin(),ind_cols_A.end(),0);
        DVector<double> diag_res = A.diagonal();
        this->F_ = DMatrix<double>::Zero(A.rows(),max_iter*block_sz);
        //error
        double norm_A=A.norm(), reconstruction_err = norm_A;
        //iterations
        int n_cols_F = 0;
        while(n_cols_F<max_iter*block_sz && reconstruction_err>this->tol_*norm_A){
            //sampling
            std::discrete_distribution<int> sampling_distr(diag_res.begin(),diag_res.end());
            std::unordered_set<int> sampled_pivots(block_sz);
            for(int j=0; (int)sampled_pivots.size()<block_sz && j<2*block_sz; j++){
              sampled_pivots.insert(sampling_distr(rng));
            }
            std::vector<int> pivot_set(sampled_pivots.begin(),sampled_pivots.end()); //converting for Eigen slicing
            //building the F factor
            DMatrix<double> G = A(Eigen::all,pivot_set);
            G = G - this->F_.leftCols(n_cols_F) * this->F_(pivot_set,Eigen::all).leftCols(n_cols_F).transpose();
            double shift = std::numeric_limits<double>::epsilon()*G(pivot_set,Eigen::all).trace();
            Eigen::LLT<DMatrix<double>> chol(G(pivot_set,Eigen::all) + shift*DMatrix<double>::Identity(pivot_set.size(),pivot_set.size()));
            DMatrix<double> T = chol.matrixU().solve<Eigen::OnTheRight>(G);
            this->F_.middleCols(n_cols_F,pivot_set.size()) = T;
            //update the sampling distribution
            diag_res = (diag_res - T.rowwise().squaredNorm()).array().max(0);
            //update the error
            n_cols_F += pivot_set.size();
            reconstruction_err =  (A-this->F_.leftCols(n_cols_F)*this->F_.leftCols(n_cols_F).transpose()).norm();
        }
        this->F_ = this->F_.leftCols(n_cols_F);
        return;
    }
};

template<typename MatrixType>
class NystromApproximation{
private:
    std::unique_ptr<NysApproxStrategy<MatrixType>> nys_strategy_;
public:
    explicit NystromApproximation(std::unique_ptr<NysApproxStrategy<MatrixType>> &&strategy=std::make_unique<RPChol<DMatrix<double>>>()): nys_strategy_(std::move(strategy)){}
    void compute(const MatrixType &A, int block_sz, int max_iter=1e3){
        nys_strategy_->compute(A,block_sz, max_iter);
        return;
    }
    //setters
    void setTol(double tol){ nys_strategy_->setTol(tol);}
    void setSeed(unsigned int seed){ nys_strategy_->setSeed(seed);}
    //getters
    int rank() const{ return nys_strategy_->rank();}
    DMatrix<double> factor() const{ return nys_strategy_->factor();}
    double shift() const{ return nys_strategy_->shift();}
};

}//core
}//fdpade

#endif //NYSTROM_APPROX_H
