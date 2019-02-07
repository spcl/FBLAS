/**
    FBLAS: BLAS implementation for Intel FPGA
    Copyright (c) 2019 ETH-Zurich. All rights reserved.
    See LICENSE for license information.

    GEMV_V2  performs one of the matrix-vector operations

    -   y := alpha*A*x + beta*y,  
            where the NxM matrix A is received in tiles streamed by rows, where each 
            tile is ColumnStreamed. x is an M-elements vector, while y
            is an N-element vector

    -   or  y := alpha*A**T*x + beta*y,
            where the NxM matrix A is received in tiles streamed by columns,
            each tile is Row Streamed. x is an N-element vector, while y
            is an M-element vector

    Data is received from three different channels (CHANNEL_VECTOR_X, CHANNEL_VECTOR_Y
    and CHANNEL_MATRIX A). Input data must be padded with zeros according to 
    the reference tile sizes (TILE_N and TILE_M).

    Result is streamed in an output channel as soon as it is available.

    Check the kernel documentation for further information
    
*/

#pragma OPENCL EXTENSION cl_intel_channels : enable

//FBLAS_PARAMETERS_START

//#define DOUBLE_PRECISION			//enable for double precision
#define W 16 					//width

//namings

#define KERNEL_NAME streaming_gemv
#define CHANNEL_VECTOR_X channel_x
#define CHANNEL_VECTOR_Y channel_y
#define CHANNEL_MATRIX_A channel_matrix
#define CHANNEL_VECTOR_OUT channel_sink
#define TILE_N 512
#define TILE_M 512
//architecture
#define __STRATIX_10__

//FBLAS_PARAMETERS_END

#include <commons.h>
#if (TILE_N>TILE_M)
    #define MAX_TILE_SIZE TILE_N
#else
    #define MAX_TILE_SIZE TILE_M
#endif

channel TYPE_T CHANNEL_VECTOR_X __attribute__((depth(W)));
channel TYPE_T CHANNEL_VECTOR_Y __attribute__((depth(W)));
channel TYPE_T CHANNEL_MATRIX_A __attribute__((depth(W)));
channel TYPE_T CHANNEL_VECTOR_OUT __attribute__((depth(W)));


/**
    This version is meant for the following cases:
    - A is rowStreamed and Transposed. A is received by tiles by columns. For this case:
        - row_streamed must be set to 1
	- vector x is composed by N elements, y is composed by M elements
	- blocks of y are composed by Tile_M elements and they will be reused
	- blocks of x are composed by Tile_N elements, they will be not reused
	- x must be re-sent entirely M/TILE_M times

    - A is columnStreamed and Not Transposed. A is received by tiles in row ordering:
        - row_streamed must be set to 0
	- vector x is composed by M elements, vector y is composed by N elements
	- block of y are composed by TILE_N elements and they will be reused
	- block of x are composed by TILE_N elemenets, not reused
	- x must be re-sent N/TILE_N times
*/

__kernel void KERNEL_NAME(int row_streamed, const int N, const int M, const TYPE_T alpha, const TYPE_T beta)
{

    int len_x,tile_x;
    int len_y,tile_y;
    int BlocksY, BlocksX;
    //chose the loop limits
    if(row_streamed==1)
    {
        len_x = N;
        len_y = M;
        tile_x=TILE_N;
        tile_y=TILE_M;
        BlocksY=1 + (int)((M-1)/TILE_M);
        BlocksX=1 + (int)((N-1)/TILE_N);
    }
    else
    {	//in this case A is non transposed
        len_x = M;
        len_y = N;
        tile_x=TILE_M;
        tile_y=TILE_N;
        BlocksY=1 + (int)((N-1)/TILE_N);
        BlocksX=1 + (int)((M-1)/TILE_M);
    }


    //In this case each element of x will be multiplied for all the tile_y elements of A
    const int computing_outer_loop_limit=(int)(tile_y/W);
    const int reading_y_outer_loop_limit=(int)(tile_y/W);


    for(int ti=0;ti<BlocksY;ti++)
    {
        //Reuse over y
        TYPE_T local_y[MAX_TILE_SIZE];

        for(int i=0;i<reading_y_outer_loop_limit;i++)
        {
            if(beta == 0)
            {	//if beta is equal to zero we don't need to read from CHANNEL_Y
                #pragma unroll
                for(int j=0;j<W;j++)
                    local_y[i*W+j] = 0;
            }
            else
            {
                #pragma unroll
                for(int j=0;j<W;j++)
                    local_y[i*W+j]=beta*read_channel_intel(CHANNEL_VECTOR_Y);
            }
        }

        for(int tj=0;tj<BlocksX;tj++)
        {
            for(int i=0;i<tile_x;i++)
            {
                TYPE_T temp=alpha*read_channel_intel(CHANNEL_VECTOR_X);


                //here we read one row/column of A and multiply it for the same value of x
                for(int jj=0;jj<computing_outer_loop_limit;jj++)
                {
                    //receive elemnts of a: decoupling this form the computation loop
                    //maybe useful in case the sender of A does not perform unrolled writes into the channel
                    TYPE_T local_A[W];
                    #pragma unroll
                    for(int j=0;j<W;j++)
                            local_A[j]=read_channel_intel(CHANNEL_MATRIX_A);

                    //updates all y

                    #pragma unroll
                    for(int j=0;j<W;j++)
                        local_y[jj*W+j]+=local_A[j]*temp;

                 }
            }
        }

        //now we can send this block of y (avoid the unroll since the tile size can be large)

        for(int i=0;i<tile_y;i++)
            write_channel_intel(CHANNEL_VECTOR_OUT,local_y[i]);
	}

}

