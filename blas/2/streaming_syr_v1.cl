/**
    FBLAS: BLAS implementation for Intel FPGA
    Copyright (c) 2019 ETH-Zurich. All rights reserved.
    See LICENSE for license information.

    SYR   performs the symmetric rank 1 operation

        A := alpha*x*x**T + A,

    where alpha is a real scalar, x is an n element vector and A is an
    n by n symmetric matrix.

    Data is received from three different channels (CHANNEL_VECTOR_X,
    CHANNEL_VECTOR_X_TRANS and CHANNEL_MATRIX_A).
    If A is a triangular lower matrix, it must arrives in tiles by row
    and Row Streamed. If A is an upper triangular matrix, it must arrives
    in tiles by cols and Col streamed.
    A is sent in packed format (that is, only the interesting elements are transmitted, padded to the Tile Size).

    Result is streamed in an output channel, tile by tile as soon as it is available.

    Check the kernel documentation for further information

*/

#pragma OPENCL EXTENSION cl_intel_channels : enable

//FBLAS_PARAMETERS_START

//#define DOUBLE_PRECISION		//enable if dsyr
#define W 32 				//width

//namings
#define KERNEL_NAME streaming_syr
#define CHANNEL_VECTOR_X channel_x
#define CHANNEL_VECTOR_X_TRANS channel_x_trans
#define CHANNEL_MATRIX_A channel_matrix
#define CHANNEL_MATRIX_OUT channel_matrix_out
#define TILE_N 512

//architecture
#define __STRATIX_10__

//FBLAS_PARAMETERS_END
#include <commons.h>

channel TYPE_T CHANNEL_VECTOR_X  __attribute__((depth(W)));
channel TYPE_T CHANNEL_VECTOR_X_TRANS  __attribute__((depth(W)));
channel TYPE_T CHANNEL_MATRIX_A  __attribute__((depth(W)));
channel TYPE_T CHANNEL_MATRIX_OUT  __attribute__((depth(W)));


/*
    In this version
    - A is row streamed, tiles by rows, lower triangular
    - or A is column streamed, tiles by cols, upper triangular
*/
__kernel void KERNEL_NAME(const TYPE_T alpha, const int N)
{


    //loops for buffering x and y are unrolled
    const int reading_x_outer_loop_limit=(int)(TILE_N/W);
    const int BlocksN=1+(int)((N-1)/TILE_N);


    TYPE_T local_A[W];
    TYPE_T local_x[TILE_N];
    TYPE_T local_x_trans[TILE_N];

    /**
        For each tile row ti
        - we need the ti-th block of x
        - we need the blocks 0,...,ti-1 of x (act as x transposed)
    */

    for(int ti=0; ti<BlocksN;ti++)
    {
        //here we reuse a block of x
        for(int i=0;i<reading_x_outer_loop_limit;i++)
        {
            #pragma unroll
            for(int ii=0;ii<W;ii++)
            {
                local_x[i*W+ii]=read_channel_intel(CHANNEL_VECTOR_X);
            }
        }
        //operates on the lower part of the matrix
        for(int tj=0;tj<=ti;tj++)
        {
            for(int i=0;i<reading_x_outer_loop_limit;i++) //buffer x transposed
            {
                #pragma unroll
                for(int ii=0;ii<W;ii++)
                {
                    local_x_trans[i*W+ii]=read_channel_intel(CHANNEL_VECTOR_X_TRANS);
                }
            }
            for(int i=0;i<TILE_N;i++)
            {
                const int i_idx=ti*TILE_N+i;
                //receive the row of A
                const int reading_A_limit=(tj<ti)?((int)(TILE_N)/W):ceilf(((float)(i+1))/W);
                for(int j=0;j<reading_A_limit;j++)
                {

                    TYPE_T tmp=alpha*local_x[i];
                    #pragma unroll
                    for(int jj=0;jj<W;jj++)
                    {

                        //compute and send out the data
                        TYPE_T a=read_channel_intel(CHANNEL_MATRIX_A);
                        local_A[jj]=tmp*local_x_trans[j*W+jj]+a;

                        //padding data set to zero
                        if(tj*TILE_N+j*W+jj>i_idx)
                            local_A[jj]=0;
                        write_channel_intel(CHANNEL_MATRIX_OUT,local_A[jj]);
                    }
                }
            }
        }
    }
}


