/**
    FBLAS: BLAS implementation for Intel FPGA
    Copyright (c) 2019 ETH-Zurich. All rights reserved.
    See LICENSE for license information.

    SYR2  performs the symmetric rank 2 operation

    A := alpha*x*y**T + alpha*y*x**T + A,

    where alpha is a scalar, x and y are n element vectors and A is an n
    by n symmetric matrix.

    Data is received from five different channels (CHANNEL_VECTOR_X,
    CHANNEL_VECTOR_X_TRANS, CHANNEL_VECTOR_Y, CHANNEL_VECTOR_Y_TRANS and CHANNEL_MATRIX_A).
    If A is a triangular lower matrix, it can arrives in tiles by row
    and Row Streamed. If A is an upper triangular matrix, it can arrives
    in tiles by cols and Col streamed.
    A is sent in packed format  (that is, only the interesting elements are transmitted, padded to the Tile Size).

    Result is streamed in an output channel, tile by tile as soon as it is available.

    Check the kernel documentation for further information

*/

#pragma OPENCL EXTENSION cl_intel_channels : enable

//FBLAS_PARAMETERS_START

//#define DOUBLE_PRECISION      //enable if dsyr2
#define W 32                    //width

//namings
#define KERNEL_NAME streaming_syr2
#define CHANNEL_VECTOR_X channel_x
#define CHANNEL_VECTOR_X_TRANS channel_x_trans
#define CHANNEL_VECTOR_Y channel_y
#define CHANNEL_VECTOR_Y_TRANS channel_y_trans
#define CHANNEL_MATRIX_A channel_matrix
#define CHANNEL_MATRIX_OUT channel_matrix_out
#define TILE_N 128

//architecture
#define __STRATIX_10__

//FBLAS_PARAMETERS_END
#include <commons.h>

channel TYPE_T CHANNEL_VECTOR_X  __attribute__((depth(TILE_N)));
channel TYPE_T CHANNEL_VECTOR_X_TRANS  __attribute__((depth(TILE_N)));
channel TYPE_T CHANNEL_VECTOR_Y  __attribute__((depth(TILE_N)));
channel TYPE_T CHANNEL_VECTOR_Y_TRANS  __attribute__((depth(TILE_N)));
channel TYPE_T CHANNEL_MATRIX_A  __attribute__((depth(W)));
channel TYPE_T CHANNEL_MATRIX_OUT  __attribute__((depth(W)));


/*
    - A is row streamed, tiles by rows, lower triangular
    - A is col streamed, tiles by cols, upper triangular
*/
__kernel void KERNEL_NAME(const TYPE_T alpha, const int N)
{

    const int reading_x_outer_loop_limit=(int)(TILE_N/W);
    const int BlocksN=1+(int)((N-1)/TILE_N);

    TYPE_T local_A[W];
    TYPE_T local_x[TILE_N];
    TYPE_T local_x_trans[TILE_N];
    TYPE_T local_y[TILE_N];
    TYPE_T local_y_trans[TILE_N];

    /**
        For each tile row ti
        - we need the ti-th block of x and y
        - we need the blocks 0,...,ti-1 of x and y (acts as x- and y-transposed)
    */

    for(int ti=0; ti<BlocksN;ti++)
    {
        //here we reuse a block of x and y
        for(int i=0;i<reading_x_outer_loop_limit;i++)
        {
            #pragma unroll
            for(int ii=0;ii<W;ii++)
            {
                local_x[i*W+ii]=read_channel_intel(CHANNEL_VECTOR_X);
                local_y[i*W+ii]=read_channel_intel(CHANNEL_VECTOR_Y);
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
                    local_y_trans[i*W+ii]=read_channel_intel(CHANNEL_VECTOR_Y_TRANS);
                }
            }
            for(int i=0;i<TILE_N;i++)
            {
                const int i_idx=ti*TILE_N+i;
                //receive the row of A
                const int reading_A_limit=(tj<ti)?((int)(TILE_N)/W):ceilf(((float)(i+1))/W);
                for(int j=0;j<reading_A_limit;j++)
                {

                    TYPE_T tmp1=alpha*local_x[i];
                    TYPE_T tmp2=alpha*local_y[i];
                    #pragma unroll
                    for(int jj=0;jj<W;jj++)
                    {
                        //compute and send out the data
                        //compute in any case to unroll, then set to zero in unnecessary
                        TYPE_T a=read_channel_intel(CHANNEL_MATRIX_A);
                        local_A[jj]=tmp1*local_x_trans[j*W+jj]+tmp2*local_y_trans[j*W+jj]+a;

                        if(tj*TILE_N+j*W+jj>i_idx)
                            local_A[jj]=0;
                        write_channel_intel(CHANNEL_MATRIX_OUT,local_A[jj]);
                    }
                }
            }
        }
    }
}


