/**
    FBLAS: BLAS implementation for Intel FPGA
    Copyright (c) 2019 ETH-Zurich. All rights reserved.
    See LICENSE for license information.

    A := alpha*x*y**T + A

    A arrives in tiles by rows, Row Streamed.

    Data is received from three different channels (CHANNEL_VECTOR_X, CHANNEL_VECTOR_Y
    and CHANNEL_MATRIX A). Input data must be padded with zeros according to
    the reference tile sizes (TILE_N and TILE_M).

    Result is streamed in an output channel, tile by tile as soon as it is available,
    respecting the same order of arrival of the input matrix

    Check the kernel documentation for further information
    
*/



#pragma OPENCL EXTENSION cl_intel_channels : enable

//FBLAS_PARAMETERS_START

//#define DOUBLE_PRECISION		//enable if dger
#define W 16 				//width

//namings
#define KERNEL_NAME streaming_ger
#define CHANNEL_VECTOR_X channel_x
#define CHANNEL_VECTOR_Y channel_y
#define CHANNEL_MATRIX_A channel_matrix
#define CHANNEL_MATRIX_OUT channel_matrix_out
#define TILE_N 512
#define TILE_M 512
//architecture
#define __STRATIX_10__

//FBLAS_PARAMETERS_END

#include <commons.h>
channel TYPE_T CHANNEL_VECTOR_X __attribute__((depth(W)));
channel TYPE_T CHANNEL_VECTOR_Y __attribute__((depth(W)));
channel TYPE_T CHANNEL_MATRIX_A __attribute__((depth(W)));
channel TYPE_T CHANNEL_MATRIX_OUT __attribute__((depth(W)));



/*
    This versions:
    - A by rows, tiles of size Tile_N x Tile_M by rows, elements in row order. y must be sent entirely N/Tile_N times
        Reuse is done over x by keeping a blocks of size TILE_N and over y by keeping a blocks of size TILE_M

*/

__kernel void KERNEL_NAME(const TYPE_T alpha, const int N, const int M )
{


    //loops for buffering x and y are unrolled
    const int reading_x_outer_loop_limit=(int)(TILE_N/W);
    const int reading_y_outer_loop_limit=(int)(TILE_M/W);

    const int BlocksN=1+(int)((N-1)/TILE_N); //ceiling for padded data
    const int BlocksM=1+(int)((M-1)/TILE_M);
    const int computing_outer_loop_limit=(int)(TILE_M/W);

    TYPE_T local_A[W];
    TYPE_T local_x[TILE_N];
    TYPE_T local_y[TILE_M];
    for(int ti=0; ti<BlocksN;ti++)
    {
        //here we reuse a block of x
        for(int i=0;i<reading_x_outer_loop_limit;i++)
        {
            #pragma unroll
            for(int ii=0;ii<W;ii++)
                local_x[i*W+ii]=read_channel_intel(CHANNEL_VECTOR_X);
        }
        for(int tj=0;tj<BlocksM;tj++)
        {
            //read y
            for(int k=0;k<reading_y_outer_loop_limit;k++)
            {
                #pragma unroll
                for(int kk=0;kk<W;kk++)
                    local_y[k*W+kk]=read_channel_intel(CHANNEL_VECTOR_Y);
            }
            for(int i=0;i<TILE_N;i++)
            {
                TYPE_T tmp=alpha*local_x[i];
                for(int j=0;j<computing_outer_loop_limit;j++)
                {

                    #pragma unroll
                    for(int jj=0;jj<W;jj++)
                    {
                        //compute and send out the data
                        local_A[jj]=tmp*local_y[j*W+jj]+read_channel_intel(CHANNEL_MATRIX_A);
                        write_channel_intel(CHANNEL_MATRIX_OUT,local_A[jj]);
                    }
                }
            }
        }
    }
}
