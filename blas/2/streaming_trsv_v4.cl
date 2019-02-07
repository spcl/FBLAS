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
    If A is a lower triangular matrix and transposed, it must arrive in tiles by col
    and Row Streamed, in the reverse order.
    If A is an upper triangular matrix non transposed, it must arrive in tiles by row
    and Col streamed, in the reverse order.
    A must arrive in packed format (that is, only the interesting elements are transmitted, padded to the Tile Size).

    Check the kernel documentation for further information.
*/


#pragma OPENCL EXTENSION cl_intel_channels : enable

//FBLAS_PARAMETERS_START

//#define DOUBLE_PRECISION		//enable if dtrsv
#define W 32                            //width


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
    This version handle two cases:
    - A row streamed, tiles by column in reverse order, A lower, A transposed
    - or A element streamed by columns, tiles  streamed by row, A upper triangular and non-transposed
*/

__kernel void KERNEL_NAME(int N)
{
const int BlocksN=1+(int)((N-1)/TILE_N);
    const int computing_outer_loop_limit=(int)(TILE_N/W);

    //in the following we will refer to y as the output and x the input (even if we are essentially overwriting x)
    TYPE_T local_y[TILE_N];
    for(int i=0;i<TILE_N;i++)
        local_y[i]=0;

    /*	Tiles arrive by columns
        In this case, since we are working on A transposed, for each row of the tile we will update all the corresponding y
    */

    for(int tj=BlocksN-1;tj>=0;tj--)
    {
        for(int ti=BlocksN-1; ti>=tj;ti--)  //lower diagonal blocks
        {
            TYPE_T local_x[TILE_N];
            for(int j=0;j<computing_outer_loop_limit;j++)
            {
                #pragma unroll
                for(int jj=0;jj<W;jj++)
                    local_x[j*W+jj]=read_channel_intel(CHANNEL_VECTOR_X);
            }

            for(int i=TILE_N-1;i>=0;i--)
            {
                TYPE_T local_A[TILE_N];
                //receive the row of A (partially if we are on a diagonal tile)
                const int reading_A_limit=(ti>tj)?((int)(TILE_N/W)):ceilf(((float)(i+1))/W);
                const int empty_A_blocks=TILE_N/W-reading_A_limit;
                for(int j=0;j<reading_A_limit;j++)
                {
                    #pragma unroll
                    for(int jj=0;jj<W;jj++)
                        local_A[j*W+jj]=read_channel_intel(CHANNEL_MATRIX_A);
                }



                if(tj==ti &&  ti*TILE_N+i < N)
                {
                    //we are in a diagonal tile: at each row we are computing a new y[i]
                    local_y[i]+=local_x[i];
                    //divide for the diagonal element (we have to find it)
                    int diag_idx=i;
                    local_y[i]/=local_A[diag_idx];
                }

                #pragma ivdep array(local_y)	//no dependencies over y
                for(int j=0;j<computing_outer_loop_limit;j++)
                {
                    #pragma unroll
                    for(int jj=0;jj<W;jj++)
                    {
                        if(ti>tj)
                        {
                            //in this case the received tile of A is fully populated
                            local_y[j*W+jj]-=local_A[j*W+jj]*local_x[i]; //use the value calculated at the beginning of this row tile
                        }
                        else
                        {
                            if(j*W+jj<i)
                            {
                                //in this case, the incoming tile of A is triangular
                                local_y[j*W+jj]-=local_A[j*W+jj]*local_y[i];
                            }
                        }
                    }
                }
            }
        }
        //we can send the updated block
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
