/**
    FBLAS: BLAS implementation for Intel FPGA
    Copyright (c) 2019 ETH-Zurich. All rights reserved.
    See LICENSE for license information.


    TRSV  solves one of the systems of equations

       A*x = b,   or   A**T*x = b,

    where b and x are n element vectors and A is an n by n unit, or
    non-unit, upper or lower triangular matrix.

    No test for singularity or near-singularity is included in this
    routine.


    Data is received from two different channels, CHANNEL_VECTOR_X and
    CHANNEL_MATRIX_A. At the first iteration we receive the vector b
    from CHANNEL_VECTOR_X. The updates for vector x are sent through channel
    CHANNEL_VECTOR_OUT.

    This routine version for the following cases.
    If A is an upper triangular matrix, it must arrive in tiles by row
    and Row Streamed, in the reverse order.
    If A is a lower triangular matrix, it must arrive in tiles by cols
    and Col streamed, in the reverse order.
    A is sent in packed format (that is, only the interesting elements are transmitted, padded to the Tile Size).

    Check the kernel documentation for further information.

*/
#pragma OPENCL EXTENSION cl_intel_channels : enable

//FBLAS_PARAMETERS_START

//#define DOUBLE_PRECISION		//enable if dtrsv
#define W 32


//namings
#define KERNEL_NAME streaming_trsv
#define CHANNEL_VECTOR_X channel_x
#define CHANNEL_MATRIX_A channel_matrix
#define CHANNEL_VECTOR_OUT channel_out
#define TILE_N 512

//architecture
#define __STRATIX_10__

//FBLAS_PARAMETERS_END
#include <commons.h>

channel TYPE_T CHANNEL_VECTOR_X __attribute__((depth(TILE_N)));
channel TYPE_T CHANNEL_MATRIX_A __attribute__((depth(W)));
channel TYPE_T CHANNEL_VECTOR_OUT __attribute__((depth(TILE_N)));


/**
    In this case:
    - A is an upper triangular matrix, Non Transposed
        A is row streamed, tile by row. A is generate in the reverse order
        From the bottom to the top. Inside a row, elements are generated in the classic
        order, from left to right (it is helpful when reading from memory)
    - A is a lower triangular matrix, transposed
        A is column streamed, tiles by col. A is generated in the reverse order
        starting from the rightmost bottom tile
*/

__kernel void KERNEL_NAME(int N)
{
    const int BlocksN=1+(int)((N-1)/TILE_N);
    const int computing_outer_loop_limit=(int)(TILE_N/W);
    #ifdef DOUBLE_PRECISION
    TYPE_T shift_reg[SHIFT_REG+1]; //shift register

    for(int i=0;i<SHIFT_REG+1;i++)
       shift_reg[i]=0;
    #endif

    /*
        Pseudocode of the computation
        FOR i=N-1 to 0
            y[i]=x[i]
            FOR j=i+1 to N
                y[i]=y[i]-A[i][j]*y[j]
            y[i]=y[i]/A[i][i]
     */

    //in the following we will refer to y as the output and x the input
    for(int ti=BlocksN-1;ti>=0;ti--)
    {
        //Here we are going to produce the ti-th blocks of y
        TYPE_T local_y[TILE_N];
        for(int i=0;i<computing_outer_loop_limit;i++)
        {
            #pragma unroll
            for(int j=0;j<W;j++)
                local_y[i*W+j]=0;
        }

        //upper diagonal blocks in reverse order
        for(int tj=BlocksN-1;tj>=ti;tj--)
        {
            //receive a block of x
            TYPE_T local_x[TILE_N];
            for(int j=0;j<computing_outer_loop_limit;j++)
            {
                #pragma unroll
                for(int jj=0;jj<W;jj++)
                {
                    local_x[j*W+jj]=read_channel_intel(CHANNEL_VECTOR_X);
                }
            }

            for(int i=TILE_N-1;i>=0; i--)
            {
                TYPE_T local_A[TILE_N];
                const int reading_A_limit=(tj>ti)?((int)(TILE_N/W)):ceilf(((float)(TILE_N-i))/W);
                //read the row of A
                #pragma unroll
                for(int j=TILE_N/W-reading_A_limit;j<TILE_N/W;j++)
                {
                    #pragma unroll
                    for(int jj=0;jj<W;jj++)
                        local_A[j*W+jj]=read_channel_intel(CHANNEL_MATRIX_A);
                }


                TYPE_T acc_i=0,acc_o=0;
                int diag_idx=0;
                for(int j=0;j<computing_outer_loop_limit;j++)
                {
                    acc_i=0;
                    #pragma unroll
                    for(int jj=0;jj<W;jj++)
                    {
                        if(tj>ti)   //use an already computed block of
                            acc_i-=local_A[j*W+jj]*local_x[j*W+jj];
                        else
                        {   //this is a diagonal block
                            if(j*W+jj>i)
                            {
                                acc_i-=local_A[j*W+jj]*local_y[j*W+jj];
                                diag_idx++;
                            }
                        }
                    }
                    #ifdef DOUBLE_PRECISION
                        shift_reg[SHIFT_REG] = shift_reg[0]+acc_i;
                        //Shift every element of shift register
                        #pragma unroll
                        for(int j = 0; j < SHIFT_REG; ++j)
                            shift_reg[j] = shift_reg[j + 1];
                    #else
                        acc_o+=acc_i;
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
                local_y[i]+=acc_o;
                if(tj==ti && ti*TILE_N+i < N ) //if we are in a diagonal block and this is a valid element
                {
                    local_y[i]+=local_x[i];
                    local_y[i]/=local_A[TILE_N-diag_idx-1];
                }

            }
        }
        //send the updated block of x
        for(int i=0;i<computing_outer_loop_limit;i++)
        {
            #pragma unroll
            for(int j=0;j<W;j++)
            {
                write_channel_intel(CHANNEL_VECTOR_OUT,local_y[i*W+j]);
                local_y[i*W+j]=0;
            }
        }
    }

}

