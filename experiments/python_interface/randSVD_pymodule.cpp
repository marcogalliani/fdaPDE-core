//
// Created by Marco Galliani on 13/12/24.
#include <Eigen/Dense>

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>

namespace py = pybind11;

#include "fdaPDE/linear_algebra.h"
using fdapde::core::RSVDStrategy;
using fdapde::core::REVDStrategy;

using fdapde::core::RSI;
using fdapde::core::RBKI;
using fdapde::core::GeneralizedRSI;
using fdapde::core::GeneralizedRBKI;

using fdapde::core::NysRSI;
using fdapde::core::NysRBKI;

template<typename MatrixType>
class PyRSVD : public RSVDStrategy<MatrixType> {
public:
    /* Inherit the constructors */
    using RSVDStrategy<MatrixType>::RSVDStrategy;

    /* Trampoline (need one for each virtual function) */
    void compute(const MatrixType &A, int rank, int max_iter) override{
        PYBIND11_OVERRIDE_PURE(
            void, /* Return type */
            RSVDStrategy<MatrixType>,      /* Parent class */
            compute,          /* Name of function in C++ (must match Python name) */
            A, rank, max_iter      /* Argument(s) */
        );
    }
};

PYBIND11_MODULE(randSVD,m){
    using d_RSVD = RSVDStrategy<DMatrix<double>>;
    py::class_<d_RSVD>(m, "RSVD")
        .def("matrixU", &d_RSVD::matrixU)
        .def("matrixV", &d_RSVD::matrixV)
        .def("singularValues", &d_RSVD::singularValues);

    using d_RSI = RSI<DMatrix<double>>;
    py::class_<d_RSI, d_RSVD>(m, "RSI")
        .def(py::init<unsigned int, double>())
        .def("compute", &d_RSI ::compute);

    using d_RBKI = RBKI<DMatrix<double>>;
    py::class_<d_RBKI , d_RSVD>(m, "RBKI")
        .def(py::init<unsigned int, double>())
        .def("compute", &d_RBKI ::compute);

    using d_GenRSI = GeneralizedRSI<DMatrix<double>>;
    py::class_<d_GenRSI, d_RSVD>(m, "GeneralizedRSI")
        .def(py::init<unsigned int, double>())
        .def("compute", &d_GenRSI ::compute);

    using d_GenRBKI = GeneralizedRBKI<DMatrix<double>>;
    py::class_<d_GenRBKI , d_RSVD>(m, "GeneralizedRBKI")
        .def(py::init<unsigned int, double>())
        .def("compute", &d_GenRBKI ::compute);

    using d_REVD = REVDStrategy<DMatrix<double>>;
    py::class_<d_REVD>(m, "REVD")
        .def("matrixU", &d_REVD::matrixU)
        .def("eigenValues", &d_REVD::eigenValues);

    using d_NysRSI = NysRSI<DMatrix<double>>;
    py::class_<d_NysRSI, d_REVD>(m, "NysRSI")
        .def(py::init<unsigned int, double>())
        .def("compute", &d_NysRSI ::compute);

    using d_NysRBKI = NysRBKI<DMatrix<double>>;
    py::class_<d_NysRBKI, d_REVD>(m, "NysRBKI")
        .def(py::init<unsigned int, double>())
        .def("compute", &d_NysRBKI ::compute);
}

