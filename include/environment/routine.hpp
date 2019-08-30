/**
    FBLAS: BLAS implementation for Intel FPGA
    Copyright (c) 2019 ETH-Zurich. All rights reserved.
    See LICENSE for license information.

    Host Api Implementation - Routine representation
*/


#ifndef ROUTINE_HPP
#define ROUTINE_HPP
#include <string>
#include <vector>
#include "../commons.hpp"
#include "../utils/ocl_utils.hpp"
#define CHECK_TRANS(TA,R) { if(TA != R.transA){std::cerr << "Wrong \"trans\" parameter for routine: "<<R.user_name  << std::endl;return;}}
#define CHECK_UPLO(U,R) { if(U != R.uplo) {std::cerr << "Wrong \"uplo\" parameter for routine: "<<R.user_name << std::endl;  return; }}
#define CHECK_INCX(INCX,R) { if(INCX != R.incx){std::cerr << "Wrong \"incx\" parameter for routine: "<<R.user_name  << " (expected "<<R.incx<<" instead of "<<INCX<<")"<<std::endl;return;}}
#define CHECK_INCY(INCY,R) { if(INCY != R.incy){std::cerr << "Wrong \"incy\" parameter for routine: "<<R.user_name  << std::endl;return;}}


/**
  Defines the Routine data structure used for the FBLAS Host API
  Not all the fields will be valid. It depends on the particular routine

*/


class Routine{
public:

    std::string blas_name;
    std::string user_name;


    bool double_precision;
    unsigned int width;
    unsigned int width_x;       //these two used in case of routine with 2D computational tiling
    unsigned int width_y;
    unsigned int tile_n_size;
    unsigned int tile_m_size;
    int incx;
    int incy;

    bool has2DComputationalTiling=false;
    bool systolic=false;

    unsigned int lda;
    unsigned int ldb;

    FblasTranspose transA;
    FblasTranspose transB;

    FblasOrder order;
    FblasUpLo uplo;
    FblasSide side;

    //for each kernel we have the corresponding queue
    //kernels are inserted in the proper order:
    // - first the main computational kernel
    // - then the data generators
    // - then the sink if any

    std::vector<cl::Kernel> kernels;
    std::vector<cl::CommandQueue> queues;


    std::vector<std::string> kernels_names;

};

#endif // ROUTINE_HPP
