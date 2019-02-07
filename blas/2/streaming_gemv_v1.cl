/**
    FBLAS: BLAS implementation for Intel FPGA
    Copyright (c) 2019 ETH-Zurich. All rights reserved.
    See LICENSE for license information.

    GEMV_V1  performs one of the matrix-vector operations

    -   y := alpha*A*x + beta*y,  
            where the NxM matrix A is received in tiles streamed by rows, where each 
            tile is Row Streamed. x is an M-elements vector, while y
            is an N-element vector

    -   or  y := alpha*A**T*x + beta*y,
            where the NxM matrix A is received in tiles streamed by columns,
            each tile is Column Streamed. x is an N-element vector, while y
            is an M-element vector

    Data is received from three different channels (CHANNEL_VECTOR_X, CHANNEL_VECTOR_Y
    and CHANNEL_MATRIX A). Input data must be padded with zeros according to 
    the reference tile sizes (TILE_N and TILE_M).

    Result is streamed in an output channel as soon as it is available.

    Check the kernel documentation for further information
    
*/

#pragma OPENCL EXTENSION cl_intel_channels : enable

//FBLAS_PARAMETERS_START
//#define DOUBLE_PRECISION  //enable if double precision version
#define W 16 			//width

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
channel TYPE_T CHANNEL_VECTOR_OUT __attribute__((depth(1)));

/**
    This version is meant for the following cases:
    - A is RowStreamed and NonTransposed, Tiles received by rows. In this case:
            - row_streamed must be set to 1
            - x is a vector of M elements, y is a vector of N elements
            - blocks of TILE_N elements of y are reused
            - also block of TILE_M elements of x are reused. The entire vector x must be resent N/TILE_N times (i.e. len_y/tile_y)
    - A is ColStreamed and Transposed, Tiles received by cols:
            - row_streamed must be set to 0
            - x is a vector of N elements, while y is a vector of M elements
            - blocks of y are of TILE_M elements
            - the entire x must be resent M/TILE_M times. Reuse will be applied also to it

    Matrix and vector must be padded to the respective tiling sizes
*/
__kernel void KERNEL_NAME(int row_streamed, const int N, const int M, const TYPE_T alpha, const TYPE_T beta)
{

    int len_x,tile_x;
    int len_y,tile_y;
    int BlocksX, BlocksY;
    //chose the loop limits
    if(row_streamed == 1)
    {
        len_x = M;
        len_y = N;
        tile_x=TILE_M;
        tile_y=TILE_N;
        BlocksY=1+(int)((N-1)/TILE_N); //ceiling
        BlocksX=1+(int)((M-1)/TILE_M);
    }
    else
    {	//in this case A is transposed
        len_x = N;
        len_y = M;
        tile_x=TILE_N;
        tile_y=TILE_M;
        BlocksY=1+(int)((M-1)/TILE_M);
        BlocksX=1+(int)((N-1)/TILE_N);
    }

    #ifdef DOUBLE_PRECISION
    TYPE_T shift_reg[SHIFT_REG+1]; //shift register

    for(int i=0;i<SHIFT_REG+1;i++)
       shift_reg[i]=0;
    #endif


    //The computation is performed by receiving A in tiles by row (A non transposed) or column (A transposed).
    //In this way, the result is computed by 'accumulating' over y elements
    //One block of y is computed for each row-tile (or column-tile) of A and using the entire x

    const int computing_outer_loop_limit=(int)(tile_x/W);
    const int reading_y_outer_loop_limit=(int)(tile_y/W);

    //Please note: the order in which tiles arrive, will determine the computation
    //(i.e. do not assume that you will receive the tiles one row after the other...maybe they can arrive column by column)

    TYPE_T local_y[MAX_TILE_SIZE];
    TYPE_T local_x[MAX_TILE_SIZE];
    for(int ti=0;ti<BlocksY;ti++)
    {
        #pragma ivdep array(local_y)
        for(int tj=0;tj<BlocksX;tj++)
        {

            #pragma ivdep array(local_x)
            for(int i=0;i<tile_y;i++)
            {

                TYPE_T acc_o=0;
                TYPE_T acc_i=0;
                TYPE_T prev;


                //here we read one element from A and one element from X and we use it
                //For X we buffer it at the first iteration
                #pragma ivdep array(local_x)
                for(int jj=0;jj<computing_outer_loop_limit;jj++)
                {
                    if(tj==0 && jj==0) //read y if this is the first iteration over the blocks of X
                    {
                        if(beta==0)
                            prev=0;
                        else
                           prev=beta*read_channel_intel(CHANNEL_VECTOR_Y);
                    }
                    if(tj!=0)
                        prev=local_y[i];

                    if(i==0) //receive x
                    {
                        #pragma unroll
                        for(int j=0;j<W;j++)
                           local_x[jj*W+j]=read_channel_intel(CHANNEL_VECTOR_X);
                    }
                    acc_i=0;
                    //read (a block of W elements) of the row of A

                    //receive elemnts of a: decoupling this from the computation loop
                    //maybe usefulein case the sender of A does not perform unrolled writes into the channel
                    TYPE_T local_A[W];
                    #pragma unroll
                    for(int j=0;j<W;j++)
                        local_A[j]=read_channel_intel(CHANNEL_MATRIX_A);

                    #pragma unroll
                    for(int j=0;j<W;j++)
                        acc_i+=local_A[j]*local_x[jj*W+j];

                    #ifdef DOUBLE_PRECISION
                        shift_reg[SHIFT_REG] = shift_reg[0]+alpha*acc_i;
                        //Shift every element of shift register
                        #pragma unroll
                        for(int j = 0; j < SHIFT_REG; ++j)
                            shift_reg[j] = shift_reg[j + 1];
                    #else
                        acc_o+=alpha*acc_i;
                    #endif

                }
                #ifdef DOUBLE_PRECISION
                    //reconstruct the result using the partial results in shift register
                    #pragma unroll
                    for(int i=0;i<SHIFT_REG;i++)
                    {
                        acc_o+=shift_reg[i];
                        shift_reg[i]=0;
                    }
                #endif
                TYPE_T result =  prev+ acc_o;
                local_y[i] = result;

                //output y if we reached the end of the matrix
                //y is output one element at a time
                if(tj==BlocksX-1)
                   write_channel_intel(CHANNEL_VECTOR_OUT,result);
            }
        }
    }
}
