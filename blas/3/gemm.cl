/**
    FBLAS: BLAS implementation for Intel FPGA
    Copyright (c) 2019 ETH-Zurich. All rights reserved.
    See LICENSE for license information.


    GEMM: matrix matrix multiplication with accumulation
    This generic version works for all the four cases (A trans/no transp, B transp/no transp)
    Data arrive always in the same way, it will change the way in which it is read from helpers kernerls

    The implementation adopt a two level of tiling: the outermost for the memory (size MTILE x MTILE)
    and the innermost for the computation (CTILE_ROWS x CTILE_COLS).
    The computation is performed by computing outer products. So for each computational tile
    the kernel will receive the corresponding (portion of ) column of A, the row of B
    and will update properly the C elements.

    The kernel computes the matrix C in tiles by rows, Row streamed.
    The results are sent in the channel CHANNEL_MATRIX_OUT.
    Matrix A arrive through channel CHANNEL_MATRIX_A, matrix B through channel
    CHANNEL_MATRIX_B. Input  data must be padded to zeros according to the
    reference tile size MTILE.    

    Result is streamed in an output channel, tile by tile as soon as it is available.

    Check the kernel documentation for further information

*/

#pragma OPENCL EXTENSION cl_intel_channels : enable

//FBLAS_PARAMETERS_START

//#define DOUBLE_PRECISION		//enable if dgemm

//namings
#define KERNEL_NAME gemm_v1
#define CHANNEL_MATRIX_A channel_matrix_a
#define CHANNEL_MATRIX_B channel_matrix_b
#define CHANNEL_MATRIX_OUT channel_matrix_out

//tilings
#define MTILE 256   //suppose for the moment squared tiles
#define CTILE_ROWS 8    //computational tile
#define CTILE_COLS 8
//architecture
#define __STRATIX_10__

//FBLAS_PARAMETERS_END

#include <commons.h>

channel TYPE_T CHANNEL_MATRIX_A __attribute__((depth(CTILE_ROWS)));
channel TYPE_T CHANNEL_MATRIX_B __attribute__((depth(CTILE_COLS)));
channel TYPE_T CHANNEL_MATRIX_OUT __attribute__((depth(CTILE_COLS)));


/**
  Computes the GEMM. A and B are received through input channels.
    - Matrix A: for each outer tile (MTILE size), inner blocks are received one after the other
            The entire outer tile-row (MTILE x K) is resent a number of times equal to the number of
            outer tiles in matrix B (check helpers/read_matrix_a_notrans_gemm.cl for an example)
    - Matrix B: for each outer tile, inner blocks are received one of the other. Each
        outer tile row is sent multiple times (check helpers/read_matrx_b_notrans_gemm.cl for an example)

*/
__kernel void KERNEL_NAME(const int N, const int M, const int K, const TYPE_T alpha)

{

    const int OuterBlocksN = 1 + (int)((N-1) / MTILE);
    const int OuterBlocksM = 1 + (int)((M-1) / MTILE);
    const int InnerBlocksN = MTILE / CTILE_ROWS;
    const int InnerBlocksM = MTILE / CTILE_COLS;
    __local TYPE_T __attribute__((memory,doublepump)) localC[MTILE/CTILE_ROWS][MTILE/CTILE_COLS][CTILE_ROWS][CTILE_COLS];

    //GEMM with outer product

    //#pragma unroll
    for(int ti=0;ti<OuterBlocksN;ti++)	    //outer tile
    {
        for(int tj=0;tj<OuterBlocksM;tj++) //outer tile
        {

            for(int k=0;k<K;k++)
            {
                TYPE_T localA[CTILE_ROWS];
                TYPE_T localB[CTILE_COLS];
                #pragma unroll 1
                for(int tti=0;tti<InnerBlocksN;tti++)   //inner tile
                {
                   // #pragma unroll
                   // for(int i=0;i<CTILE_ROWS;i++)
                   //     localA[i]=read_channel_intel(CHANNEL_MATRIX_A);
                    #pragma unroll 1
                    for(int ttj=0;ttj<InnerBlocksM; ttj++)	//inner tile
                    {
                        if(ttj==0)
                        {
                            #pragma unroll
                            for(int i=0;i<CTILE_ROWS;i++)
                              localA[i]=read_channel_intel(CHANNEL_MATRIX_A);
                        }
                        #pragma unroll
                        for(int i=0;i<CTILE_COLS;i++)
                            localB[i]=read_channel_intel(CHANNEL_MATRIX_B);

                        //to unroll
                        #pragma unroll
                        for(int i=0;i<CTILE_ROWS;i++)
                        {
                            TYPE_T tmpa=alpha*localA[i];
                            #pragma unroll
                            for(int j=0; j<CTILE_COLS;j++)
                            {

                                const TYPE_T prev=(k==0)?0:localC[tti][ttj][i][j];
                                localC[tti][ttj][i][j]=prev+tmpa*localB[j];
                            }
                        }
                    }
                }
            }

            //prevent unrolls on this, for the sake of saving BRAMs
            #pragma unroll 1
            for(int ii=0;ii<MTILE/CTILE_ROWS;ii++)
                #pragma unroll 1
                for(int jj=0;jj<MTILE/CTILE_COLS;jj++)
                {
                    #pragma unroll 1
                    for(int iii=0;iii<CTILE_ROWS;iii++)
                            #pragma unroll
                            for(int jjj=0;jjj<CTILE_COLS;jjj++)
                            {
                                    int ind_i=ti*MTILE+ii*CTILE_ROWS+iii;
                                    int ind_j=tj*MTILE+jj*CTILE_COLS+jjj;
                                    write_channel_intel(CHANNEL_MATRIX_OUT,localC[ii][jj][iii][jjj]);
                             }
                }
        }
    }


}



