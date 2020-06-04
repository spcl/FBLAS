/**
    FBLAS: BLAS implementation for Intel FPGA
    Copyright (c) 2019 ETH-Zurich. All rights reserved.
    See LICENSE for license information.

    Host API Environment
*/

#ifndef FBLAS_ENVIRONMENT_HPP
#define FBLAS_ENVIRONMENT_HPP

#include <iostream>
#include <string>
#include <fstream>
#include <rapidjson/document.h>
#include <unordered_map>
#include "utils/ocl_utils.hpp"
#include "environment/routine.hpp"
#include "commons.hpp"


#if !defined(CL_CHANNEL_1_INTELFPGA)
// include this header if channel macros are not defined in cl.hpp (versions >=19.0)
#include "CL/cl_ext_intelfpga.h"
#endif


class FBLASEnvironment{

public:

    /**
     * @brief FBLASEnvironment constructor: creates an empty FBLAS Environment
     */
    FBLASEnvironment(){}

    /**
     * @brief FBLASEnvironment creates an FBLAS Environment using the compiled bitstream and the JSON file produced by the code generator
     * @param binary_path path of the .aocx binary file
     * @param json_file path of the JSON file produced by the code generator
     */
    FBLASEnvironment(std::string binary_path, std::string json_file){
        //create program, context, devices
        IntelFPGAOCLUtils::initOpenCL(platform_,device_,context_,program_,binary_path);

        //load kernels
        FBLASEnvironment::parseJSON(json_file);
    }

    /**
     * @brief FBLASEnvironment copy constructor
     * @param e2
     */
    FBLASEnvironment(const FBLASEnvironment &e2)
    {
        context_=e2.context_;
        platform_=e2.platform_;
        program_=e2.program_;
        device_=e2.device_;
        routines_=e2.routines_;
    }

    /*
     * Routines
    */


    /*------------------------------------------
     *
     * LEVEL 1
     *
     * -------------------------------------------*/
    void scopy(std::string routine_name,  unsigned int N, cl::Buffer x, int incx,  cl::Buffer y, int incy,
               std::vector<cl::Event> * events_wait_list = nullptr, cl::Event * event = nullptr);
    void dcopy(std::string routine_name,  unsigned int N, cl::Buffer x, int incx,  cl::Buffer y, int incy,
               std::vector<cl::Event> * events_wait_list = nullptr, cl::Event * event = nullptr);

    /**
     * @brief sswap swap routine, single precision
     * @param routine_name user defined name of the routine as defined in the JSON file provided to the generator
     * @param N number of elements in input vectors
     * @param x vector x
     * @param incx vector stride
     * @param y vector y
     * @param incy vector stride
     * @param events_wait_list (optional) list of OpenCL events to wait before starting this routine
     * @param event (optional) for non blocking routine calls, specify this argument. The routine will
     *      return immediatelly and the even parameter will contain the OpenCL event related to this computation
     */
    void sswap(std::string routine_name, unsigned int N, cl::Buffer x, int incx,  cl::Buffer y, int incy,
               std::vector<cl::Event> * events_wait_list = nullptr, cl::Event * event = nullptr);

    /**
     * @brief dswap swap, double precision
     *  For parameter explanations check strsm
     */
    void dswap(std::string routine_name, unsigned int N, cl::Buffer x, int incx,  cl::Buffer y, int incy,
               std::vector<cl::Event> * events_wait_list = nullptr, cl::Event * event = nullptr);

    void saxpy(std::string routine_name,  unsigned int N, float alpha, cl::Buffer x, int incx,  cl::Buffer y, int incy,
               std::vector<cl::Event> * events_wait_list = nullptr, cl::Event * event = nullptr);

    void daxpy(std::string routine_name,  unsigned int N, double alpha, cl::Buffer x, int incx,  cl::Buffer y, int incy,
               std::vector<cl::Event> * events_wait_list = nullptr, cl::Event * event = nullptr);

    void sdot(std::string routine_name,  unsigned int N, cl::Buffer x, int incx,  cl::Buffer y, int incy, cl::Buffer res,
              std::vector<cl::Event> * events_wait_list = nullptr, cl::Event * event = nullptr);

    void ddot(std::string routine_name, unsigned int N, cl::Buffer x, int incx, cl::Buffer y, int  incy, cl::Buffer res,
              std::vector<cl::Event> * events_wait_list = nullptr, cl::Event * event = nullptr);

    void sscal(std::string routine_name, unsigned int N, float alpha, cl::Buffer x,  int incx, std::vector<cl::Event> * events_wait_list = nullptr,
               cl::Event * event = nullptr);

    void dscal(std::string routine_name, unsigned int N, double alpha, cl::Buffer x,  int incx, std::vector<cl::Event> * events_wait_list = nullptr,
               cl::Event * event = nullptr);

    void sasum(std::string routine_name, unsigned int N, cl::Buffer x, int incx, cl::Buffer res, std::vector<cl::Event> * events_wait_list = nullptr,
               cl::Event * event = nullptr);

    void dasum(std::string routine_name, unsigned int N, cl::Buffer x, int incx, cl::Buffer res, std::vector<cl::Event> * events_wait_list = nullptr,
               cl::Event * event = nullptr);

    /**
     * @brief isamax finds the index of the first element having maximum absolute value, single precision
     * @param routine_name user defined name of the routine as defined in the JSON file provided to the generator
     * @param N number of vector elements
     * @param x vector
     * @param incx access stride
     * @param res memory area in which saving the result (integer number)
     * @param events_wait_list (optional) list of OpenCL events to wait before starting this routine
     * @param event (optional) for non blocking routine calls, specify this argument. The routine will
     *      return immediatelly and the even parameter will contain the OpenCL event related to this computation
     */
    void isamax(std::string routine_name, unsigned int N, cl::Buffer x, int incx, cl::Buffer res, std::vector<cl::Event> * events_wait_list = nullptr,
               cl::Event * event = nullptr);

    /**
     * @brief idamax finds the index of the first element having maximum absolute value, double precision
     * Check isamax for parameters explanation
     */
    void idamax(std::string routine_name, unsigned int N, cl::Buffer x, int incx, cl::Buffer res, std::vector<cl::Event> * events_wait_list = nullptr,
               cl::Event * event = nullptr);

    void snrm2(std::string routine_name, unsigned int N, cl::Buffer x, int incx, cl::Buffer res, std::vector<cl::Event> * events_wait_list = nullptr,
               cl::Event * event = nullptr);

    void dnrm2(std::string routine_name, unsigned int N, cl::Buffer x, int incx, cl::Buffer res, std::vector<cl::Event> * events_wait_list = nullptr,
               cl::Event * event = nullptr);

    void srot(std::string routine_name, unsigned int N, cl::Buffer x, int incx, cl::Buffer y, int incy, float c, float s,
              std::vector<cl::Event> * events_wait_list = nullptr, cl::Event * event = nullptr);

    void drot(std::string routine_name, unsigned int N, cl::Buffer x, int incx, cl::Buffer y, int incy, double c, double s,
              std::vector<cl::Event> * events_wait_list = nullptr, cl::Event * event = nullptr);

    /**
     * @brief srotg construct givens plane rotation, single precision
     * @param routine_name user defined name of the routine as defined in the JSON file provided to the generator
     * @param sa input
     * @param sb input
     * @param c output (reference)
     * @param s output (reference)
     */
    void srotg(std::string routine_name, float sa, float sb , float &c, float &s);


    /**
     * @brief drotg construct givens plane rotation, double precision
     *  Please check srotg for parameters explanation
     */
    void drotg(std::string routine_name, double sa, double sb , double &c, double &s);



    /**
     * @brief srotm applies the Modified Givens rotations. Single precision
     * @param routine_name user defined name of the routine as defined in the JSON file provided to the generator
     * @param N, x, incx, y, incy, param routines parameters, please check BLAS documentation
     * @param events_wait_list (optional) list of OpenCL events to wait before starting this routine
     * @param event (optional) for non blocking routine calls, specify this argument. The routine will
     *      return immediatelly and the even parameter will contain the OpenCL event related to this computation
     */
    void srotm(std::string routine_name, unsigned int N, cl::Buffer x, int incx, cl::Buffer y, int incy, float param[5],
        std::vector<cl::Event> * events_wait_list = nullptr, cl::Event * event = nullptr);


    /**
     * @brief drotm rotm, double precision
        Please check srotm documentation for parameters explanation
     */
    void drotm(std::string routine_name, unsigned int N, cl::Buffer x, int incx, cl::Buffer y, int incy, double param[5],
        std::vector<cl::Event> * events_wait_list = nullptr, cl::Event * event = nullptr);


    /**
     * @brief srotmg setup modified givens rotations
     * @param routine_name routine_name user defined name of the routine as defined in the JSON file provided to the generator
     * @param sd1 input/output
     * @param sd2 input/output
     * @param sx1 input/output
     * @param sy1 input
     * @param param output
     */
    void srotmg(std::string routine_name, float &sd1, float &sd2, float &sx1, float sy1, float param[5]);

    /**
     * @brief drotmg setup modifiedgivens rotations
        Please check srotmg and BLAS documentation for parameter explanations
     */
    void drotmg(std::string routine_name, double &dd1, double &dd2, double &dx1, double dy1, double param[5]);

    /*------------------------------------------
     *
     * LEVEL 2
     *
     * -------------------------------------------*/


    /**
     * @brief sgemv gemv routine, single precision
     * @param routine_name user defined name of the routine as defined in the JSON file provided to the generator
     * @param trans specifies the operation to be performed
     * @param N number of rows of matrix A
     * @param M number of columns of matrix A
     * @param alpha, A, lda, x, incx,  beta, y, incy other routine parameters, check BLAS documentation
     * @param events_wait_list (optional) list of OpenCL events to wait before starting this routine
     * @param event (optional) for non blocking routine calls, specify this argument. The routine will
     *      return immediatelly and the even parameter will contain the OpenCL event related to this computation
     */
    void sgemv(std::string routine_name, FblasTranspose trans, unsigned int N, unsigned int M, float alpha, cl::Buffer A,
               unsigned int lda, cl::Buffer x, int incx, float beta, cl::Buffer y,  int incy, std::vector<cl::Event> * events_wait_list = nullptr, cl::Event * event = nullptr);

    /**
     * @brief dgemv gemv routine, double precision
     * Check sgemv for parameters explanations

     */
    void dgemv(std::string routine_name, FblasTranspose trans, unsigned int N, unsigned int M, double alpha, cl::Buffer A,
               unsigned int lda, cl::Buffer x, int incx,  double beta, cl::Buffer y, int incy, std::vector<cl::Event> * events_wait_list = nullptr, cl::Event * event = nullptr);

    /**
     * @brief strmv triangular matrix vector, single precision version
     * @param routine_name user defined name of the routine as defined in the JSON file provided to the generator
     * @param uplo,trans,diag,N,A,lda,x,incx routine parameters, check BLAS documentation
     * @param events_wait_list (optional) list of OpenCL events to wait before starting this routine
     * @param event (optional) for non blocking routine calls, specify this argument. The routine will
     *      return immediatelly and the event parameter will contain the OpenCL event related to this computation
     */
    void strmv(std::string routine_name, FblasUpLo uplo, FblasTranspose trans,FblasDiag diag, unsigned int N, cl::Buffer A,
               unsigned int lda, cl::Buffer x, int incx, std::vector<cl::Event> * events_wait_list = nullptr, cl::Event * event = nullptr);


    /**
     * @brief dtrmv trmv, double precision
     * For parameters explanation check strmv
     */
    void dtrmv(std::string routine_name, FblasUpLo uplo, FblasTranspose trans,FblasDiag diag, unsigned int N, cl::Buffer A,
               unsigned int lda, cl::Buffer x, int incx, std::vector<cl::Event> * events_wait_list = nullptr, cl::Event * event = nullptr);

    /**
     * @brief ssymv symmetric matrix vector multiply
     * @param routine_name user defined name of the routine as defined in the JSON file provided to the generator
     * @param uplo
     * @param N
     * @param alpha
     * @param A
     * @param lda
     * @param x
     * @param incx
     * @param beta
     * @param y
     * @param incy
     * @param event (optional) for non blocking routine calls, specify this argument. The routine will
     *      return immediatelly and the event parameter will contain the OpenCL event related to this computation
     */
    void ssymv(std::string routine_name, FblasUpLo uplo, unsigned int N, float alpha, cl::Buffer A, unsigned int lda,
               cl::Buffer x, int incx, float beta, cl::Buffer y,  int incy, std::vector<cl::Event> * events_wait_list = nullptr, cl::Event * event = nullptr);

    /**
     * @brief dsymv symmetic matrix vector multiply, double precision
     * Check ssymv for parameters explanation
     */
    void dsymv(std::string routine_name, FblasUpLo uplo, unsigned int N, double alpha, cl::Buffer A, unsigned int lda,
               cl::Buffer x, int incx, double beta, cl::Buffer y,  int incy, std::vector<cl::Event> * events_wait_list = nullptr, cl::Event * event = nullptr);


    void sger(std::string routine_name, unsigned int N, unsigned int M, float alpha, cl::Buffer x, int incx,  cl::Buffer y,
              int incy,  cl::Buffer A, unsigned int lda, std::vector<cl::Event> * events_wait_list = nullptr, cl::Event * event = nullptr);

    void dger(std::string routine_name, unsigned int N, unsigned int M, double alpha, cl::Buffer x, int incx,  cl::Buffer y,
              int incy,  cl::Buffer A, unsigned int lda, std::vector<cl::Event> * events_wait_list = nullptr, cl::Event * event = nullptr);

    /**
     * @brief ssyr symmetric rank 1 operation, single precision
     * @param routine_name user defined name of the routine as defined in the JSON file provided to the generator
     * @param uplo, N, alpha, x, incx, A, lda routine parameters, check BLAS documentation
     * @param events_wait_list (optional) list of OpenCL events to wait before starting this routine
     * @param event (optional) for non blocking routine calls, specify this argument. The routine will
     *      return immediatelly and the event parameter will contain the OpenCL event related to this computation
     */
    void ssyr(std::string routine_name, FblasUpLo uplo,  unsigned int N, float alpha, cl::Buffer x,int incx, cl::Buffer A, unsigned int lda,
              std::vector<cl::Event> * events_wait_list = nullptr, cl::Event * event = nullptr);

    /**
     * @brief dsyr symmetric rank 1 operation, double precision
        Check ssyr for parameters explnation
     */
    void dsyr(std::string routine_name, FblasUpLo uplo, unsigned int N, double alpha, cl::Buffer x,int incx, cl::Buffer A, unsigned int lda,
              std::vector<cl::Event> * events_wait_list = nullptr, cl::Event * event = nullptr);

    /**
     * @brief ssyr2 symmetric rank 2 operation, single precision
     * @param routine_name user defined name of the routine as defined in the JSON file provided to the generator
     * @param uplo, N, alpha, x, incx, y, incy, A, lda routine parameters, check BLAS documentation
     * @param event (optional) for non blocking routine calls, specify this argument. The routine will
     *      return immediatelly and the event parameter will contain the OpenCL event related to this computation
     */
    void ssyr2(std::string routine_name, FblasUpLo uplo, unsigned int N, float alpha, cl::Buffer x,int incx, cl::Buffer y, int incy, cl::Buffer A,
               unsigned int lda, std::vector<cl::Event> * events_wait_list = nullptr, cl::Event * event = nullptr);

    /**
     * @brief dsyr2 symmetric rank 2 operation, double precision
     * Check ssyr2 for parameters explanation
     */
    void dsyr2(std::string routine_name, FblasUpLo uplo, unsigned int N, double alpha, cl::Buffer x,int incx, cl::Buffer y, int incy, cl::Buffer A,
               unsigned int lda, std::vector<cl::Event> * events_wait_list = nullptr, cl::Event * event = nullptr);

    void strsv(std::string routine_name,FblasUpLo uplo, FblasTranspose trans, unsigned int N, cl::Buffer A, unsigned int lda,
               cl::Buffer x, int incx, std::vector<cl::Event> * events_wait_list = nullptr, cl::Event * event = nullptr);

    void dtrsv(std::string routine_name,FblasUpLo uplo, FblasTranspose trans, unsigned int N, cl::Buffer A, unsigned int lda,
               cl::Buffer x, int incx, std::vector<cl::Event> * events_wait_list = nullptr, cl::Event * event = nullptr);


    /*------------------------------------------
     *
     * LEVEL 3
     *
     * -------------------------------------------*/


    /**
     * @brief strsm trsm routine, single precision (for unspecified arguments, check BLAS documentation)
     * @param routine_name, user defined name of the routine as defined in the JSON file provided to the generator
     * @param side FBLAS_LEFT if solving AX=B, FBLAS_RIGHT for XA=B
     * @param transa FBLAS_NO_TRANS if A is not transposed, FBLAS_TRANS otherwise
     * @param uplo FBLAS_LOWER for A triangular lower, FBLAS_UPPER for upper triangular
     * @param N number of rows of matrix B
     * @param M number of columns of matrix B
     * @param alpha, A, lda, B, ldb routine parameters, check BLAS documentation
     * @param events_wait_list (optional) list of OpenCL events to wait before starting this routine
     * @param event (optional) for non blocking routine calls, specify this argument. The routine will
     *      return immediatelly and the even parameter will contain the OpenCL event related to this computation
     */

    void strsm(std::string routine_name,FblasSide side, FblasTranspose transa, FblasUpLo uplo, unsigned int N, unsigned int M,
               float alpha, cl::Buffer A, unsigned int lda, cl::Buffer B, unsigned int ldb,
               std::vector<cl::Event> * events_wait_list = nullptr, cl::Event * event = nullptr);

    /**
     * @brief dtrsm trsm routine, double precision
     *  For parameter explanations check strsm
     */
    void dtrsm(std::string routine_name,FblasSide side, FblasTranspose transa, FblasUpLo uplo, unsigned int N, unsigned int M,
               double alpha, cl::Buffer A, unsigned int lda, cl::Buffer B, unsigned int ldb,
               std::vector<cl::Event> * events_wait_list = nullptr, cl::Event * event = nullptr);

    /**
     * @brief sgemm gemm routine, performs matrix matrix multiplication and accumulation, single precision
     * @param routine_name user defined name of the routine as defined in the JSON file provided to the generator
     * @param transA specifies whether A is tranposed or not
     * @param transB specifies whether B is transposed or not
     * @param N number of rows of matrix A and C
     * @param M number of columns of matrix B and C
     * @param K number of columns of matrix A and number of rows of matrix B
     * @param alpha, A, lda, B, ldb, beta, C, ldc other routine parameters, check BLAS documentation
     * @param events_wait_list (optional) list of OpenCL events to wait before starting this routine
     * @param event (optional) for non blocking routine calls, specify this argument. The routine will
     *      return immediatelly and the even parameter will contain the OpenCL event related to this computation
     *
     */
    void sgemm(std::string routine_name, FblasTranspose transA, FblasTranspose transB, unsigned int N, unsigned int M, unsigned int K,
               float alpha, cl::Buffer A, unsigned int lda, cl::Buffer B, unsigned int ldb, float beta, cl::Buffer C, unsigned int ldc,
               std::vector<cl::Event> * events_wait_list = nullptr, cl::Event * event = nullptr);

    /**
     * @brief dgemm gemm routine, double precision
     * For parameters explanations check sgemm documentation
     */
    void dgemm(std::string routine_name, FblasTranspose transA, FblasTranspose transB, unsigned int N, unsigned int M, unsigned int K,
               double alpha, cl::Buffer A, unsigned int lda, cl::Buffer B, unsigned int ldb, double beta, cl::Buffer C, unsigned int ldc,
               std::vector<cl::Event> * events_wait_list = nullptr, cl::Event * event = nullptr);


    /**
     * @brief ssyrk syrk routine single precision, performs symmetric rank k update
     * @param routine_name user defined name of the routine as defined in the JSON file provided to the generator
     * @param uplo specifies wheter the matrix C is lower or upper triangular
     * @param trans specifies the operation to perform
     * @param N, K, alpha, A, lda, beta, C, ldc computation parameters, check BLAS documentation for further information
     * @param events_wait_list (optional) list of OpenCL events to wait before starting this routine
     * @param event (optional) for non blocking routine calls, specify this argument. The routine will
     *      return immediatelly and the even parameter will contain the OpenCL event related to this computation
     */
    void ssyrk(std::string routine_name, FblasUpLo uplo, FblasTranspose trans, unsigned int N, unsigned int K,
               float alpha, cl::Buffer A, unsigned int lda, float beta, cl::Buffer C, unsigned int ldc,
               std::vector<cl::Event> * events_wait_list = nullptr, cl::Event * event = nullptr);

    /**
     * @brief dsyrk syrk routine, double precision
     * Check ssyrk documentation for parameters explanation
     */
    void dsyrk(std::string routine_name, FblasUpLo uplo, FblasTranspose trans, unsigned int N, unsigned int K,
               double alpha, cl::Buffer A, unsigned int lda, double beta, cl::Buffer C, unsigned int ldc,
               std::vector<cl::Event> * events_wait_list = nullptr, cl::Event * event = nullptr);
    /**
     * @brief ssyr2k syr2k routine single precision, performs symmetric rank 2-k update
     * @param routine_name user defined name of the routine as defined in the JSON file provided to the generator
     * @param uplo specifies wheter the matrix C is lower or upper triangular
     * @param trans specifies the operation to perform
     * @param N, K, alpha, A, lda, B, ldb, beta, C, ldc computation parameters, check BLAS documentation for further information
     * @param events_wait_list (optional) list of OpenCL events to wait before starting this routine
     * @param event (optional) for non blocking routine calls, specify this argument. The routine will
     *      return immediatelly and the even parameter will contain the OpenCL event related to this computation
     */
    void ssyr2k(std::string routine_name, FblasUpLo uplo, FblasTranspose trans, unsigned int N, unsigned int K,
               float alpha, cl::Buffer A, unsigned int lda, cl::Buffer B, unsigned int ldb, float beta, cl::Buffer C, unsigned int ldc,
               std::vector<cl::Event> * events_wait_list = nullptr, cl::Event * event = nullptr);
    /**
     * @brief dsyr2k syr2k routine, double precision
     * Check dsyr2k documentation for parameters explanation
     */
    void dsyr2k(std::string routine_name, FblasUpLo uplo, FblasTranspose trans, unsigned int N, unsigned int K,
               double alpha, cl::Buffer A, unsigned int lda, cl::Buffer B, unsigned int ldb, double beta, cl::Buffer C, unsigned int ldc,
               std::vector<cl::Event> * events_wait_list = nullptr, cl::Event * event = nullptr);


    cl::Context get_context()
    {
        return context_;
    }

    cl::Device get_device()
    {
        return device_;
    }

private:

    /**
     * @brief parseJson load the JSON file containing the routine characteristics
     *
     * @param json_file
     */
    void parseJSON(std::string json_file);

    /**
     * @brief parseRoutine parse a routine descripted in a JSON object
     * @param routine
     */
    void parseRoutine(rapidjson::Value &routine);


    //Generic routines: the ones exposed to the user are just wrapper for more generic one

    //level 1

    template <typename T>
    void copy(std::string routine_name,  unsigned int N, cl::Buffer x, int incx,  cl::Buffer y, int incy,
               std::vector<cl::Event> * events_wait_list = nullptr, cl::Event * event = nullptr);

    template <typename T>
    void swap(std::string routine_name,  unsigned int N, cl::Buffer x, int incx,  cl::Buffer y, int incy,
               std::vector<cl::Event> * events_wait_list = nullptr, cl::Event * event = nullptr);

    template <typename T>
    void axpy(std::string routine_name,  unsigned int N, T alpha, cl::Buffer x, int incx,  cl::Buffer y, int incy,
               std::vector<cl::Event> * events_wait_list = nullptr, cl::Event * event = nullptr);


    template <typename T>
    void dot(std::string routine_name,  unsigned int N, cl::Buffer x, int incx,  cl::Buffer y, int incy, cl::Buffer res,
              std::vector<cl::Event> * events_wait_list = nullptr, cl::Event * event = nullptr);

    template <typename T>
    void scal(std::string routine_name, unsigned int N, T alpha, cl::Buffer x,  int incx, std::vector<cl::Event> * events_wait_list = nullptr,
               cl::Event * event = nullptr);


    template <typename T>
    void asum(std::string routine_name, unsigned int N, cl::Buffer x, int incx, cl::Buffer res, std::vector<cl::Event> * events_wait_list = nullptr,
               cl::Event * event = nullptr);

    template <typename T>
    void iamax(std::string routine_name, unsigned int N, cl::Buffer x, int incx, cl::Buffer res, std::vector<cl::Event> * events_wait_list = nullptr,
               cl::Event * event = nullptr);

    template <typename T>
    void nrm2(std::string routine_name, unsigned int N, cl::Buffer x, int incx, cl::Buffer res, std::vector<cl::Event> * events_wait_list = nullptr,
               cl::Event * event = nullptr);
    template <typename T>
    void rot(std::string routine_name, unsigned int N, cl::Buffer x, int incx, cl::Buffer y, int incy, T c, T s,
              std::vector<cl::Event> * events_wait_list = nullptr, cl::Event * event = nullptr);

    template <typename T>
    void rotg(std::string routine_name, T sa, T sb , T &c, T &s);


    template <typename T>
    void rotm(std::string routine_name, unsigned int N, cl::Buffer x, int incx, cl::Buffer y, int incy, T param[5],
        std::vector<cl::Event> * events_wait_list = nullptr, cl::Event * event = nullptr);

    template <typename T>
    void rotmg(std::string routine_name, T &dd1, T &dd2, T &dx1, T dy1, T param[5]);

    //--------------------------------------
    //Level 2
    //--------------------------------------
    template <typename T>
    void gemv(std::string routine_name, FblasTranspose trans, unsigned int N, unsigned int M, T alpha, cl::Buffer A, unsigned int lda,
              cl::Buffer x, int incx, T beta, cl::Buffer y, int incy,  std::vector<cl::Event> * events_wait_list = nullptr, cl::Event * event = nullptr);

    template <typename T>
    void ger(std::string routine_name, unsigned int N, unsigned int M, T alpha, cl::Buffer x, int incx,  cl::Buffer y,
              int incy,  cl::Buffer A, unsigned int lda, std::vector<cl::Event> * events_wait_list = nullptr, cl::Event * event = nullptr);

    template <typename T>
    void syr(std::string routine_name, FblasUpLo uplo, unsigned int N, T alpha, cl::Buffer x, int incx, cl::Buffer A,
             unsigned int lda, std::vector<cl::Event> *events_wait_list= nullptr, cl::Event *event= nullptr);

    template <typename T>
    void syr2(std::string routine_name, FblasUpLo uplo, unsigned int N, T alpha, cl::Buffer x, int incx, cl::Buffer y, int incy, cl::Buffer A,
             unsigned int lda, std::vector<cl::Event> *events_wait_list= nullptr, cl::Event *event= nullptr);

    template <typename T>
    void trsv(std::string routine_name,FblasUpLo uplo, FblasTranspose trans, unsigned int N, cl::Buffer A, unsigned int lda,
               cl::Buffer x, int incx, std::vector<cl::Event> * events_wait_list = nullptr, cl::Event * event = nullptr);

    template <typename T>
    void trmv(std::string routine_name, FblasUpLo uplo, FblasTranspose trans,FblasDiag diag, unsigned int N, cl::Buffer A,
              unsigned int lda, cl::Buffer x, int incx, std::vector<cl::Event> * events_wait_list = nullptr, cl::Event * event = nullptr);

    template <typename T>
    void symv(std::string routine_name, FblasUpLo uplo, unsigned int N, T alpha, cl::Buffer A, unsigned int lda,
               cl::Buffer x, int incx, T beta, cl::Buffer y,  int incy, std::vector<cl::Event> * events_wait_list = nullptr, cl::Event * event = nullptr);


    //-------------------------------------
    //Level 3
    //-------------------------------------

    template <typename T>
    void gemm(std::string routine_name, FblasTranspose transA, FblasTranspose transB, unsigned int N, unsigned int M, unsigned int K,
               T alpha, cl::Buffer A, unsigned int lda, cl::Buffer B, unsigned int ldb, T beta, cl::Buffer C, unsigned int ldc,
               std::vector<cl::Event> * events_wait_list = nullptr, cl::Event * event = nullptr);

    template <typename T>
    void trsm(std::string routine_name,FblasSide side, FblasTranspose transa, FblasUpLo uplo, unsigned int N, unsigned int M,
              T alpha, cl::Buffer A, unsigned int lda, cl::Buffer B, unsigned int ldb,
              std::vector<cl::Event> * events_wait_list = nullptr, cl::Event * event = nullptr);

    template <typename T>
    void syrk(std::string routine_name, FblasUpLo uplo, FblasTranspose trans, unsigned int N, unsigned int K,
              T alpha, cl::Buffer A, unsigned int lda, T beta, cl::Buffer C, unsigned int ldc,
              std::vector<cl::Event> * events_wait_list = nullptr, cl::Event * event = nullptr);

    template <typename T>
    void syr2k(std::string routine_name, FblasUpLo uplo, FblasTranspose trans, unsigned int N, unsigned int K,
                   T alpha, cl::Buffer A, unsigned int lda, cl::Buffer B, unsigned int ldb, T beta, cl::Buffer C, unsigned int ldc,
                   std::vector<cl::Event> * events_wait_list = nullptr, cl::Event * event = nullptr);




    // Variables

    //OpenCL environment

    cl::Context context_;
    cl::Platform platform_;
    cl::Program program_;
    cl::Device device_;

    //For systolic/autorun kernels we need to check wheter they are already in execution
    bool running_=false;

    //routine representation
    std::unordered_map<std::string, Routine> routines_;



};
//include the definition files
#include "environment/fblas_environment_level1.hpp"
#include "environment/fblas_environment_level2.hpp"
#include "environment/fblas_environment_level3.hpp"
#include "environment/fblas_environment_parsing.hpp"
#endif // FBLAS_ENVIRONMENT_HPP
