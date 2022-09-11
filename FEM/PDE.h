#ifndef __PDE_H__
#define __PDE_H__

#include "../utils/Symbols.h"
#include "../MESH/Mesh.h"
using fdaPDE::core::MESH::Mesh;
#include "operators/BilinearFormTraits.h"
using fdaPDE::core::FEM::is_parabolic;
#include "Assembler.h"
using fdaPDE::core::FEM::Assembler;
#include "operators/Identity.h"
using fdaPDE::core::FEM::Identity;
#include <type_traits>
#include <unordered_map>
#include <vector>

namespace fdaPDE{
namespace core{
namespace FEM{

  // forward declarations
  class FEMStandardSpaceTimeSolver;
  class FEMStandardSpaceSolver;

  // trait to select the space-only or the space-time variant of the PDE standard solver
  template <typename E> struct pde_standard_solver_selector {
    using type = typename std::conditional<is_parabolic<E>::value,
					   FEMStandardSpaceTimeSolver, FEMStandardSpaceSolver>::type;
  };

  // top level class to describe a partial differential equation.
  // N and M are the problem dimensions: these informations are strictly releated to the mesh used for domain discretization.
  // In particular N is the dimension of the problem domain, M is the local dimension of the manifold describing the domain.
  template <unsigned int M,  // local dimension of the mesh 
	    unsigned int N,  // dimension of the mesh embedding space
	    unsigned int R,  // order of the mesh
	    typename E,      // type of the BilinearFormExpr
	    typename Solver = typename pde_standard_solver_selector<E>::type>
  class PDE{
  private:
    const Mesh<M,N,R>& domain_; // problem domain
    E bilinearForm_; // the differential operator of the problem in its weak formulation
    DMatrix<double> forcingData_{}; // forcing data, a matrix is used to handle space-time problems
    DVector<double> initialCondition_{}; // initial condition, used in space-time problems only
  
    // memorize boundary data in a sparse structure, by storing the index of the boundary node and the relative boundary value.
    // a vector is used as mapped type to handle both space and space-time problems
    std::unordered_map<unsigned, DVector<double>> boundaryData_{};

    Solver solver_{}; // the solver used to solve the PDE
  public:
    // constructor, a DMatrix is accepted as forcingData to handle also space-time problems
    PDE(const Mesh<M,N,R>& domain, E bilinearForm, const DMatrix<double>& forcingData) :
      domain_(domain), bilinearForm_(bilinearForm), forcingData_(forcingData) {};

    // setters for boundary and initial conditions
    void setDirichletBC(const DMatrix<double>& data);
    //void setNeumannBC();
    void setInitialCondition(const DVector<double>& data) { initialCondition_ = data; };
  
    // getters
    const Mesh<M, N>& domain() const { return domain_; }
    E bilinearForm() const { return bilinearForm_; }
    const DMatrix<double>& forcingData() const { return forcingData_; }
    const DVector<double>& initialCondition() const { return initialCondition_; }
    const std::unordered_map<unsigned, DVector<double>>& boundaryData() const { return boundaryData_; };

    // solution informations produced by call to .solve()
    const DMatrix<double>  solution() const { return solver_.solution(); };
    const DMatrix<double>  force() const { return solver_.force(); }; // rhs of FEM linear system
    const SpMatrix<double> R1() const { return solver_.R1(); };
    const SpMatrix<double> R0() const { return solver_.R0(); };

    // computes matrices R1, R0 and forcing vector without solving the FEM linear system. Usefull for methods requiring just those quantites
    // without having the need to solve the pde
    template <typename B, typename I, typename... Args>
    void init(const B& base, const I& integrator, Args... args);
    
    // entry point for PDE solver. Solves the pde (i.e. after this call solution() will contain valid data)
    template <typename B, typename I, typename... Args>
    void solve(const B& base, const I& integrator, Args... args);
  };

  // argument deduction rule for PDE object
  template <unsigned int M, unsigned int N, unsigned int R, typename E>
  PDE(const Mesh<M,N,R>& domain, E bilinearForm, const DVector<double>& forcingData) -> PDE<M, N, R, decltype(bilinearForm)>;

#include "PDE.tpp"

}}}
#endif // __PDE_H__
